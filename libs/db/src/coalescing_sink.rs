use impeller2::types::{IntoLenPacket, LenPacket};
use std::time::{Duration, Instant};
use stellarator::io::AsyncWrite;

use crate::Error;

/// A write-coalescing packet sink that buffers small outgoing packets and
/// flushes them as a single TCP write when the buffer reaches a target size.
///
/// This dramatically reduces network overhead when streaming many small
/// updates (e.g. 100 components at 300 Hz → ~30 000 tiny packets/sec).
/// With a 1500-byte target those coalesce into ~600 well-packed writes/sec.
///
/// The receiver sees no difference – the impeller2 length-delimited framing
/// lets `LengthDelReader` parse individual packets from the coalesced byte
/// stream as usual.
pub struct CoalescingSink<'a, W: AsyncWrite> {
    writer: &'a W,
    buffer: Vec<u8>,
    target_size: usize,
    flush_interval: Duration,
    last_flush: Instant,
}

impl<'a, W: AsyncWrite> CoalescingSink<'a, W> {
    /// Create a new coalescing sink wrapping `writer`.
    ///
    /// * `target_size` – target number of bytes to accumulate before flushing.
    /// * `flush_interval` – maximum time data may sit in the buffer before
    ///   being flushed even if `target_size` has not been reached.
    pub fn new(writer: &'a W, target_size: usize, flush_interval: Duration) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(target_size),
            target_size,
            flush_interval,
            last_flush: Instant::now(),
        }
    }

    /// Buffer a packet.  If the buffer has reached `target_size` after
    /// appending, it is flushed immediately.
    pub async fn send(&mut self, packet: impl IntoLenPacket) -> Result<(), Error> {
        let pkt = packet.into_len_packet();
        self.buffer.extend_from_slice(&pkt.inner);

        if self.buffer.len() >= self.target_size {
            self.flush().await?;
        }
        Ok(())
    }

    /// Buffer a raw `LenPacket`, returning it afterwards so the caller can
    /// reuse the allocation (mirrors the `rent!` pattern used elsewhere).
    #[allow(dead_code)]
    pub async fn send_reusable(&mut self, pkt: LenPacket) -> Result<LenPacket, Error> {
        self.buffer.extend_from_slice(&pkt.inner);

        if self.buffer.len() >= self.target_size {
            self.flush().await?;
        }
        Ok(pkt)
    }

    /// Check whether the flush interval has elapsed and flush if so.
    /// Call this periodically (e.g. once per wake cycle) to avoid stale data
    /// sitting in the buffer during low-activity periods.
    pub async fn maybe_flush(&mut self) -> Result<(), Error> {
        if !self.buffer.is_empty() && self.last_flush.elapsed() >= self.flush_interval {
            self.flush().await?;
        }
        Ok(())
    }

    /// Flush any buffered data to the underlying writer immediately.
    pub async fn flush(&mut self) -> Result<(), Error> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let buf = std::mem::replace(&mut self.buffer, Vec::with_capacity(self.target_size));
        let (res, _buf) = self.writer.write_all(buf).await;
        self.last_flush = Instant::now();
        res.map_err(Error::from)
    }

    /// Returns the number of bytes currently buffered.
    #[cfg(test)]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stellarator::{
        io::SplitExt,
        net::{TcpListener, TcpStream},
    };

    #[test]
    fn test_coalescing_buffering() {
        stellarator::run(|| test_coalescing_buffering_inner())
    }

    async fn test_coalescing_buffering_inner() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn a task that accepts the connection and keeps it alive.
        stellarator::spawn(async move {
            let stream = listener.accept().await.unwrap();
            // Read all data (keep connection alive until dropped).
            let mut buf = vec![0u8; 64 * 1024];
            loop {
                let (rx, rx_buf) = stream.read(buf).await;
                buf = rx_buf;
                match rx {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
        });

        let client = TcpStream::connect(addr).await.unwrap();
        let (_, writer) = client.split();

        let target_size = 200;
        let mut sink = CoalescingSink::new(&writer, target_size, Duration::from_millis(5));

        // Each LenPacket::msg has 4 (len) + 1 (ty) + 2 (id) + 1 (req_id) = 8 bytes header.
        // An empty msg packet is 8 bytes on the wire.
        // We'll send packets that are just headers (~8 bytes each).

        // Send 3 small packets -- should NOT flush (3 * 8 = 24 < 200).
        for _ in 0..3 {
            let pkt = LenPacket::msg(1u16.to_le_bytes(), 0);
            sink.send(pkt).await.unwrap();
        }
        assert!(
            sink.buffered_len() > 0 && sink.buffered_len() < target_size,
            "buffer should have data but not have flushed yet: {} bytes",
            sink.buffered_len()
        );
        let after_3 = sink.buffered_len();

        // Send enough packets to exceed the target size.
        // We need (200 / 8) ≈ 25 packets total. Send 25 more.
        for _ in 0..25 {
            let pkt = LenPacket::msg(1u16.to_le_bytes(), 0);
            sink.send(pkt).await.unwrap();
        }
        // After exceeding target, the buffer should have been flushed.
        // The residual may be non-empty (partial batch) but less than target_size.
        assert!(
            sink.buffered_len() < target_size,
            "buffer should have flushed when it exceeded target: {} bytes",
            sink.buffered_len()
        );
        // And the total buffered should be less than what it would be without flushing.
        assert!(
            sink.buffered_len() < after_3 + 25 * 8,
            "buffered_len should be less than total sent without flushing"
        );

        // Test maybe_flush with elapsed time.
        let pkt = LenPacket::msg(1u16.to_le_bytes(), 0);
        sink.send(pkt).await.unwrap();
        assert!(sink.buffered_len() > 0);

        // Artificially expire the flush interval.
        sink.last_flush = Instant::now() - Duration::from_millis(100);
        sink.maybe_flush().await.unwrap();
        assert_eq!(
            sink.buffered_len(),
            0,
            "maybe_flush should have flushed after interval elapsed"
        );

        // Test that maybe_flush does NOT flush when interval hasn't elapsed.
        let pkt = LenPacket::msg(1u16.to_le_bytes(), 0);
        sink.send(pkt).await.unwrap();
        sink.last_flush = Instant::now(); // just flushed
        sink.maybe_flush().await.unwrap();
        assert!(
            sink.buffered_len() > 0,
            "maybe_flush should NOT have flushed when interval hasn't elapsed"
        );

        // Clean up: flush remaining.
        sink.flush().await.unwrap();
        assert_eq!(sink.buffered_len(), 0);
    }
}
