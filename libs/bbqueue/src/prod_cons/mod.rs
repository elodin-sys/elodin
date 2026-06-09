//! Producer and Consumer interfaces
//!
//! BBQueues can be used with one of two kinds of producer/consumer pairs:
//!
//! * **Framed**, where the consumer sees the exact chunks that were inserted by the
//!   producer. This uses a small length header to note the length of the inserted frame.
//!   This means that if the producer writes a 10 byte grant, a 20 byte grant, then a 30
//!   byte grant, the consumer will need to read three times to drain the queue, seeing the
//!   10, 20, and 30 byte chunks in order. This is useful when you are working with data that
//!   has logical "frames", for example for network packets.
//! * **Stream**, where the consumer may potentially see multiple pushed chunks at once, with
//!   no separation. This means that if the producer writes a 10 byte grant, a 20 byte grant,
//!   then a 30 byte grant, the consumer could potentially see all 60 bytes in a single read
//!   grant (if there is no wrap-around).
//!
//! You should NOT "mix and match" framed/stream consumers and producers. This will not cause
//! memory safety/UB issues, but will not work properly.

pub mod framed;
pub mod stream;
