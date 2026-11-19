//! The lockstep exchange of `main.py` as an `el.NativeStepHook`: each call sends its rows' state
//! frames to their controllers and writes back the commands they return.
//!
//! When the run records a witness, the hook also gets its channels as columns: it copies each
//! state frame into `controller.ingress`, each reply into `controller.reply`, and the reply's
//! values into `controller.response`. In a replay, `controller.reply` is read-only and holds the
//! recorded replies, so the hook reads those instead of talking to controllers.

use std::ffi::{CStr, c_char, c_void};
use std::io;
use std::net::{Ipv4Addr, UdpSocket};
use std::time::Duration;

/// The `ElodinColumn` layout of `el.NativeStepHook`, ABI version 1.
#[repr(C)]
pub struct ElodinColumn {
    pub name: *const c_char,
    pub data: *mut u8,
    pub row_bytes: u64,
    pub writable: u32,
}

/// The `ElodinStep` layout of `el.NativeStepHook`, ABI version 1.
#[repr(C)]
pub struct ElodinStep {
    pub abi_version: u32,
    pub tick: u64,
    pub timestamp: i64,
    pub rows: u64,
    pub row_start: u64,
    pub row_end: u64,
    pub columns: *const ElodinColumn,
    pub column_count: u64,
    pub state: *mut c_void,
    pub error: *mut c_char,
    pub error_capacity: u64,
}

const STATE_SIZE: usize = 1024;
const COMMAND_SIZE: usize = 104;
const HEADER_SIZE: usize = 16;
/// The reply's values, without its header and the controller's service time.
const RESPONSE_SIZE: usize = 80;

/// Batch fields in frame order, each with its row size; `command` is last.
const FIELDS: [(&str, usize); 8] = [
    ("position", 24),
    ("velocity", 24),
    ("object_position", 24),
    ("object_velocity", 24),
    ("sensor_frame", 128),
    ("diagnostic_bank", 256),
    ("object_signal", 128),
    ("command", 24),
];

/// The sockets to the controllers.
struct Link {
    port_base: u16,
    sender: UdpSocket,
    /// Where each row's controller replies.
    replies: Vec<UdpSocket>,
}

/// `ELODIN_MANY_AGENT_FAULT_MODE=bit-flip`: XOR one byte of one agent's response at one tick.
struct BitFlip {
    tick: u64,
    agent: usize,
    byte: usize,
    mask: u8,
}

struct Hook {
    /// The column of each of `FIELDS`.
    columns: [usize; FIELDS.len()],
    ingress: Option<usize>,
    reply: Option<usize>,
    response: Option<usize>,
    /// None in a replay.
    link: Option<Link>,
    fault: Option<BitFlip>,
}

/// The column named `name`, which must have rows of `row_bytes` bytes.
fn column(columns: &[ElodinColumn], name: &str, row_bytes: usize) -> Result<Option<usize>, String> {
    // Safety: Elodin passes columns with NUL-terminated names.
    let index = columns
        .iter()
        .position(|column| unsafe { CStr::from_ptr(column.name) }.to_bytes() == name.as_bytes());
    if let Some(index) = index
        && columns[index].row_bytes as usize != row_bytes
    {
        return Err(format!("{name:?} rows must be {row_bytes} bytes"));
    }
    Ok(index)
}

impl Hook {
    fn new(step: &ElodinStep) -> Result<Self, String> {
        // Safety: Elodin passes `column_count` columns.
        let columns =
            unsafe { std::slice::from_raw_parts(step.columns, step.column_count as usize) };
        let mut indices = [0; FIELDS.len()];
        for (index, (name, row_bytes)) in indices.iter_mut().zip(FIELDS) {
            *index = column(columns, name, row_bytes)?
                .ok_or(format!("the batch has no {name:?} field"))?;
        }
        if columns[indices[FIELDS.len() - 1]].writable == 0 {
            return Err("\"command\" must be a write field".into());
        }
        let reply = column(columns, "controller.reply", COMMAND_SIZE)?;
        let replaying = reply.is_some_and(|reply| columns[reply].writable == 0);
        Ok(Self {
            columns: indices,
            ingress: column(columns, "controller.ingress", STATE_SIZE)?,
            reply,
            response: column(columns, "controller.response", RESPONSE_SIZE)?,
            link: if replaying {
                None
            } else {
                Some(Link::new(step)?)
            },
            fault: BitFlip::from_env()?,
        })
    }

    /// `ControllerBridge.exchange` of `main.py` for rows `row_start..row_end`.
    fn exchange(&self, step: &ElodinStep) -> Result<(), String> {
        // Safety: each column holds `step.rows` rows of `row_bytes` bytes, and this call touches
        // only rows `row_start..row_end`, which no other call owns.
        let columns =
            unsafe { std::slice::from_raw_parts(step.columns, step.column_count as usize) };
        let rows = |column: usize, row: usize| {
            let column = &columns[column];
            let size = column.row_bytes as usize;
            unsafe { std::slice::from_raw_parts_mut(column.data.add(row * size), size) }
        };
        let (start, end) = (step.row_start as usize, step.row_end as usize);
        let mut frame = [0u8; STATE_SIZE];
        frame[..4].copy_from_slice(b"MALS");
        frame[4..6].copy_from_slice(&1u16.to_le_bytes());
        frame[8..16].copy_from_slice(&step.tick.to_le_bytes());
        for row in start..end {
            let mut offset = HEADER_SIZE;
            for &field in &self.columns[..FIELDS.len() - 1] {
                let values = rows(field, row);
                frame[offset..offset + values.len()].copy_from_slice(values);
                offset += values.len();
            }
            if let Some(ingress) = self.ingress {
                rows(ingress, row).copy_from_slice(&frame);
            }
            if let Some(link) = &self.link {
                let port = link.port_base + row as u16 * 2;
                link.sender
                    .send_to(&frame, (Ipv4Addr::LOCALHOST, port))
                    .map_err(|err| format!("cannot send to controller {row}: {err}"))?;
            }
        }
        let mut received = [0u8; COMMAND_SIZE];
        for row in start..end {
            let reply: &[u8] = match (&self.link, self.reply) {
                (Some(link), reply) => {
                    let Ok(size) = link.replies[row].recv(&mut received) else {
                        return Err("controller response timeout".into());
                    };
                    if size != COMMAND_SIZE {
                        return Err("invalid command packet size".into());
                    }
                    if let Some(reply) = reply {
                        rows(reply, row).copy_from_slice(&received);
                    }
                    &received
                }
                (None, Some(reply)) => rows(reply, row),
                (None, None) => unreachable!("a replay has the reply column"),
            };
            if &reply[..4] != b"MALC"
                || reply[4..8] != [1, 0, 0, 0]
                || reply[8..16] != step.tick.to_le_bytes()
            {
                return Err("invalid command packet header".into());
            }
            let mut response = [0u8; RESPONSE_SIZE];
            response.copy_from_slice(&reply[HEADER_SIZE..HEADER_SIZE + RESPONSE_SIZE]);
            if let Some(fault) = &self.fault
                && (fault.tick, fault.agent) == (step.tick, row)
            {
                response[fault.byte] ^= fault.mask;
            }
            if let Some(column) = self.response {
                rows(column, row).copy_from_slice(&response);
            }
            rows(self.columns[FIELDS.len() - 1], row).copy_from_slice(&response[..24]);
        }
        Ok(())
    }
}

impl Link {
    fn new(step: &ElodinStep) -> Result<Self, String> {
        let port_base = env_or("ELODIN_MANY_AGENT_PORT_BASE", 13000)?;
        let timeout = Duration::from_secs_f64(env_or("ELODIN_MANY_AGENT_TIMEOUT_S", 0.25)?);
        let replies = (0..step.rows as u16)
            .map(|row| {
                let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, port_base + row * 2 + 1))?;
                socket.set_read_timeout(Some(timeout))?;
                Ok(socket)
            })
            .collect::<io::Result<_>>()
            .map_err(|err| format!("cannot bind the command sockets: {err}"))?;
        let sender = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|err| err.to_string())?;
        Ok(Self {
            port_base,
            sender,
            replies,
        })
    }
}

impl BitFlip {
    fn from_env() -> Result<Option<Self>, String> {
        if std::env::var("ELODIN_MANY_AGENT_FAULT_MODE").as_deref() != Ok("bit-flip") {
            return Ok(None);
        }
        let byte = env_or("ELODIN_MANY_AGENT_FAULT_BYTE", 3)?;
        if byte >= RESPONSE_SIZE {
            return Err(format!(
                "ELODIN_MANY_AGENT_FAULT_BYTE must be below {RESPONSE_SIZE}"
            ));
        }
        Ok(Some(Self {
            tick: env_or("ELODIN_MANY_AGENT_FAULT_TICK", 4096)?,
            agent: env_or("ELODIN_MANY_AGENT_FAULT_AGENT", 7)?,
            byte,
            mask: env_or("ELODIN_MANY_AGENT_FAULT_MASK", 1)?,
        }))
    }
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{name} is not valid: {value:?}")),
        Err(_) => Ok(default),
    }
}

fn fail(step: &ElodinStep, message: &str) -> i32 {
    let len = message.len().min(step.error_capacity as usize - 1);
    // Safety: `error` holds `error_capacity` bytes, and `len` leaves room for the NUL.
    unsafe {
        std::ptr::copy_nonoverlapping(message.as_ptr(), step.error.cast(), len);
        *step.error.add(len) = 0;
    }
    1
}

/// # Safety
///
/// Called by `el.NativeStepHook` with a valid step and state slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn many_agent_step_init(
    step: *const ElodinStep,
    state: *mut *mut c_void,
) -> i32 {
    let step = unsafe { &*step };
    match Hook::new(step) {
        Ok(hook) => {
            unsafe { *state = Box::into_raw(Box::new(hook)).cast() };
            0
        }
        Err(err) => fail(step, &err),
    }
}

/// # Safety
///
/// Called by `el.NativeStepHook` with a valid step whose state `many_agent_step_init` made.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn many_agent_step(step: *const ElodinStep) -> i32 {
    let step = unsafe { &*step };
    let hook = unsafe { &*step.state.cast::<Hook>() };
    match hook.exchange(step) {
        Ok(()) => 0,
        Err(err) => fail(step, &err),
    }
}

/// # Safety
///
/// Called by `el.NativeStepHook` once, with the state `many_agent_step_init` made.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn many_agent_step_destroy(state: *mut c_void) {
    drop(unsafe { Box::from_raw(state.cast::<Hook>()) });
}
