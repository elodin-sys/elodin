use impeller2::{
    com_de::Decomponentize,
    table::VTableBuilder,
    types::{ComponentId, EntityId, *},
};
use impeller2_wkt::{Metadata, SetComponentMetadata, SetEntityMetadata, VTableMsg};
use zerocopy::{FromBytes, Immutable, IntoBytes};

const ENTITY_ID: EntityId = EntityId(1);
const VTABLE_ID: [u8; 3] = [1, 0, 0];
const TIME_COMPONENT_ID: ComponentId = ComponentId::new("time");
const MAG_COMPONENT_ID: ComponentId = ComponentId::new("mag");
const GYRO_COMPONENT_ID: ComponentId = ComponentId::new("gyro");
const ACCEL_COMPONENT_ID: ComponentId = ComponentId::new("accel");
const TEMP_COMPONENT_ID: ComponentId = ComponentId::new("temp");
const PRESSURE_COMPONENT_ID: ComponentId = ComponentId::new("pressure");
const HUMIDITY_COMPONENT_ID: ComponentId = ComponentId::new("humidity");

const COMPONENT_NAMES: &[&str] = &[
    "time", "mag", "gyro", "accel", "temp", "pressure", "humidity",
];

#[repr(C)]
#[derive(Debug, Default, PartialEq, Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct SensorData {
    time: i64,
    mag: [f32; 3],
    gyro: [f32; 3],
    accel: [f32; 3],
    temp: f32,
    pressure: f32,
    humidity: f32,
}

impl Decomponentize for SensorData {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        _entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
    ) {
        match component_id {
            TIME_COMPONENT_ID => self.time = value.into(),
            MAG_COMPONENT_ID => self.mag = value.into(),
            GYRO_COMPONENT_ID => self.gyro = value.into(),
            ACCEL_COMPONENT_ID => self.accel = value.into(),
            TEMP_COMPONENT_ID => self.temp = value.into(),
            PRESSURE_COMPONENT_ID => self.pressure = value.into(),
            HUMIDITY_COMPONENT_ID => self.humidity = value.into(),
            _ => {}
        }
    }
}

fn main() -> Result<(), impeller2::error::Error> {
    let entity_ids = [ENTITY_ID];
    let mut vtable = VTableBuilder::<Vec<_>, Vec<_>>::default();
    vtable
        .column(TIME_COMPONENT_ID, PrimType::I64, [], entity_ids)?
        .column(MAG_COMPONENT_ID, PrimType::F32, [3], entity_ids)?
        .column(GYRO_COMPONENT_ID, PrimType::F32, [3], entity_ids)?
        .column(ACCEL_COMPONENT_ID, PrimType::F32, [3], entity_ids)?
        .column(TEMP_COMPONENT_ID, PrimType::F32, [], entity_ids)?
        .column(PRESSURE_COMPONENT_ID, PrimType::F32, [], entity_ids)?
        .column(HUMIDITY_COMPONENT_ID, PrimType::F32, [], entity_ids)?;
    let vtable = vtable.build();
    let data = SensorData {
        time: 1,
        mag: [1.0, 2.0, 3.0],
        gyro: [1.0, 2.0, 3.0],
        accel: [1.0, 2.0, 3.0],
        temp: 10.0,
        pressure: 20.0,
        humidity: 30.0,
    };
    let mut sink = SensorData::default();
    vtable.parse_table(data.as_bytes(), &mut sink)?;
    assert_eq!(data, sink);

    let msg = VTableMsg {
        id: VTABLE_ID,
        vtable,
    };
    let vtable_msg = msg.to_len_packet().inner;
    let data_len = std::mem::size_of::<SensorData>();

    let mut table = LenPacket::table(VTABLE_ID, data_len);
    table.extend_from_slice(SensorData::default().as_bytes());
    let sensor_data = table.inner;
    let sensor_data_header = &sensor_data[..16];

    let component_metadata = COMPONENT_NAMES
        .iter()
        .map(|name| SetComponentMetadata {
            component_id: ComponentId::new(name),
            name: name.to_string(),
            metadata: Metadata::default(),
            asset: false,
        })
        .map(|msg| msg.to_len_packet().inner)
        .flatten()
        .collect::<Vec<_>>();

    let entity_metadata = SetEntityMetadata {
        entity_id: ENTITY_ID,
        name: "Vehicle".to_string(),
        metadata: Metadata::default(),
    };
    let entity_metadata = entity_metadata.to_len_packet().inner;

    let init_msg = [
        vtable_msg.as_slice(),
        sensor_data.as_slice(),
        entity_metadata.as_slice(),
        component_metadata.as_slice(),
    ]
    .concat();

    print_c_array("init_msg", &init_msg);
    print_c_array("sensor_data_header", sensor_data_header);
    println!("#define {} {}", "SENSOR_DATA_LEN", data_len);
    Ok(())
}

fn print_c_array(name: &str, data: &[u8]) {
    print!("unsigned char {}[] = {{", name);
    for i in 0..data.len() {
        if i % 12 == 0 {
            println!();
            print!("  ");
        }
        print!("0x{:02x}, ", data[i]);
    }
    println!();
    println!("}};");
    println!("unsigned int {}_len = {};", name, data.len());
}
