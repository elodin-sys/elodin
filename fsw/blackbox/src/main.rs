use std::io::Read;

use zerocopy::FromBytes;

#[derive(zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
#[repr(C)]
pub struct Record {
    pub ts: u32, // in milliseconds
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let input_file = args.next().expect("missing input file");
    let mut input_file = std::fs::File::open(input_file).expect("failed to open input file");

    let mut input = Vec::default();
    input_file.read_to_end(&mut input).unwrap();
    let mut input = input.as_slice();

    println!(
        "ts,mag_x,mag_y,mag_z,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,mag_temp,mag_sample"
    );
    while let Ok((record, remaining)) = Record::read_from_prefix(input) {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            record.ts,
            record.mag[0],
            record.mag[1],
            record.mag[2],
            record.gyro[0],
            record.gyro[1],
            record.gyro[2],
            record.accel[0],
            record.accel[1],
            record.accel[2],
            record.mag_temp,
            record.mag_sample
        );
        input = remaining;
    }
}
