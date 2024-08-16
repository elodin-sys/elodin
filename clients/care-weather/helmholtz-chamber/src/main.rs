use std::io::{BufRead, BufReader};

use roci::Metadatatize;
use roci::{
    conduit::Query,
    drivers::{os_sleep_driver, Driver, Hz},
    tokio, Componentize, Decomponentize, System,
};

pub struct Chamber<const H: usize> {
    port: Box<dyn serialport::SerialPort>,
}

impl<const H: usize> Chamber<H> {
    pub fn new(mut port: Box<dyn serialport::SerialPort>) -> Self {
        writeln!(port, "#plot c").unwrap();
        writeln!(port, "#plot kv --keys \"MX,MY,MZ\" -p 100").unwrap();
        writeln!(port, "print=100").unwrap();
        Self { port }
    }
}

#[derive(Default, Componentize, Decomponentize, Debug, Metadatatize)]
#[roci(entity_id = 0)]
pub struct World {
    mag_ref: [f64; 3],
    chamber_mag_reading: [f64; 3],
}

impl<const H: usize> System for Chamber<H> {
    type World = World;

    type Driver = Hz<HZ>;

    fn update(&mut self, world: &mut Self::World) {
        let [x, y, z] = world.mag_ref;
        let mut buf = BufReader::new(&mut self.port);
        let mut line = String::new();
        while buf.read_line(&mut line).is_ok() {
            if line.get(..2) == Some("MX") {
                for (i, key) in line.split(':').skip(1).enumerate() {
                    let num = key.split('\t').next().unwrap().trim();
                    let x: f64 = num.parse().unwrap();
                    world.chamber_mag_reading[i] = x;
                }
            }
            line.clear();
        }
        println!("target={},{},{}", x, y, z);
        println!("reading={:?}", world.chamber_mag_reading);
        writeln!(self.port, "target={},{},{}", x, y, z).unwrap();
    }
}

const HZ: usize = 10;

fn main() {
    let (tx, rx) = tokio::tcp_connect::<Hz<HZ>>(
        "127.0.0.1:2240".parse().unwrap(),
        &[Query::with_id("mag_ref")],
        World::metadata().filter(|x| x.component_name() == "chamber_mag_reading"),
    );
    let chamber = Chamber::<HZ>::new(
        serialport::new("/dev/tty.usbmodem1201", 115200)
            .open()
            .unwrap(),
    );

    os_sleep_driver(rx.pipe(chamber).pipe(tx)).run();
}
