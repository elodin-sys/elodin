use std::io::{self, Read, Write};

pub struct StdinProxy {
    pub stdin: io::Stdin,
    stdin_tx: thingbuf::mpsc::blocking::Sender<Vec<u8>>,
}

pub struct StdoutProxy {
    pub stdout: io::Stdout,
    stdout_rx: thingbuf::mpsc::blocking::Receiver<Vec<u8>>,
}

pub fn pair() -> (
    StdinProxy,
    StdoutProxy,
    thingbuf::mpsc::blocking::Sender<Vec<u8>>,
    thingbuf::mpsc::blocking::Receiver<Vec<u8>>,
) {
    let (stdout_tx, stdout_rx) = thingbuf::mpsc::blocking::channel::<Vec<u8>>(32);
    let (stdin_tx, stdin_rx) = thingbuf::mpsc::blocking::channel::<Vec<u8>>(32);
    (
        StdinProxy {
            stdin: io::stdin(),
            stdin_tx,
        },
        StdoutProxy {
            stdout: io::stdout(),
            stdout_rx,
        },
        stdout_tx,
        stdin_rx,
    )
}

impl StdinProxy {
    pub fn spawn(self) {
        std::thread::spawn(move || self.run());
    }

    pub fn run(self) {
        loop {
            let mut slot = self.stdin_tx.send_ref().unwrap();
            slot.clear();
            slot.resize(1024, 0);
            let mut stdin = self.stdin.lock();
            let n = stdin.read(&mut slot).unwrap();
            slot.truncate(n);
        }
    }
}

impl StdoutProxy {
    pub fn spawn(self) {
        std::thread::spawn(move || self.run());
    }
    pub fn run(self) {
        loop {
            let slot = self.stdout_rx.recv().unwrap();
            let mut stdout = self.stdout.lock();
            stdout.write_all(&slot).unwrap();
        }
    }
}
