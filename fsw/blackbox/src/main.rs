use std::io::Read;

use blackbox::Record;
use zerocopy::FromBytes;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let input_file = args.next().expect("missing input file");
    let mut input_file = std::fs::File::open(input_file).expect("failed to open input file");

    let mut input = Vec::default();
    input_file.read_to_end(&mut input).unwrap();
    let mut input = input.as_slice();

    println!("baro,baro_temp,vin,vbat,aux_current,rtc_vbat,cpu_temp");
    while let Ok((record, remaining)) = Record::read_from_prefix(input) {
        println!(
            "{},{},{},{},{},{},{}",
            record.baro,
            record.baro_temp,
            record.vin,
            record.vbat,
            record.aux_current,
            record.rtc_vbat,
            record.cpu_temp
        );
        input = remaining;
    }
}
