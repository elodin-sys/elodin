use std::env;
use std::io;
use std::net::{Ipv4Addr, UdpSocket};

const STATE_SIZE: usize = 1024;
const COMMAND_SIZE: usize = 104;

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_f64(bytes: &[u8], index: usize) -> f64 {
    let offset = 16 + index * 8;
    f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn write_f64(bytes: &mut [u8], index: usize, value: f64) {
    let offset = 16 + index * 8;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: many-agent-controller STATE_PORT COMMAND_PORT");
        std::process::exit(2);
    }
    let state_port: u16 = args[1].parse().expect("invalid state port");
    let command_port: u16 = args[2].parse().expect("invalid command port");
    let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, state_port))?;
    let command_address = (Ipv4Addr::LOCALHOST, command_port);
    let mut state = [0u8; STATE_SIZE];
    loop {
        let (size, _) = socket.recv_from(&mut state)?;
        if size != STATE_SIZE
            || &state[0..4] != b"MALS"
            || read_u16(&state, 4) != 1
            || read_u16(&state, 6) != 0
        {
            continue;
        }
        let tick = read_u64(&state, 8);
        let mut command = [0.0; 3];
        for axis in 0..3 {
            let position_error = read_f64(&state, 6 + axis) - read_f64(&state, axis);
            let velocity_error = read_f64(&state, 9 + axis) - read_f64(&state, 3 + axis);
            command[axis] = (1.4 * position_error + 0.45 * velocity_error).clamp(-12.0, 12.0);
        }
        let mut response = [0u8; COMMAND_SIZE];
        response[0..4].copy_from_slice(b"MALC");
        response[4..6].copy_from_slice(&1u16.to_le_bytes());
        response[6..8].copy_from_slice(&0u16.to_le_bytes());
        response[8..16].copy_from_slice(&tick.to_le_bytes());
        for (axis, value) in command.into_iter().enumerate() {
            write_f64(&mut response, axis, value);
        }
        socket.send_to(&response, command_address)?;
    }
}
