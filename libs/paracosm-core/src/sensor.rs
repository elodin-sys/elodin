use crate::{FromState, Time};

pub trait Sensor<T, S> {
    fn sense(&self, time: Time, state: &S);
}

macro_rules! impl_sensor {
    ($($ty:tt),+) => {
        #[allow(non_snake_case)]
        impl<F, $($ty,)*  S> Sensor<($($ty, )*), S> for F
        where
            F: Fn($($ty, )*),
            $($ty: FromState<S>, )*
        {

            fn sense(&self, time: Time, state: &S) {
                $(
                    let $ty = $ty::from_state(time, &state);
                )*
                (self)($($ty,)*)
            }
        }
    };
}

impl_sensor!(T1);
impl_sensor!(T1, T2);
impl_sensor!(T1, T2, T3);
impl_sensor!(T1, T2, T3, T4);
impl_sensor!(T1, T2, T3, T4, T5);
impl_sensor!(T1, T2, T3, T4, T5, T6);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_sensor!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

// #[cfg(feature = "rerun")]
// pub mod rerun {
//     use crate::{BodyPos, BodyVel, Time};
//     use rerun::{
//         components::Vec3D, external::re_log_types::DataTableError, time::Timeline, RecordingStream,
//     };

//     pub fn vel_sensor(stream: RecordingStream) -> impl Fn(Time, BodyPos, BodyVel) {
//         move |Time(time), BodyPos(pos), BodyVel(vel)| {
//             let time = (time * 1000.0) as i64;
//             rerun::MsgSender::new("vel")
//                 .with_component(&[rerun::components::Arrow3D {
//                     origin: Vec3D::new(pos.x as f32, pos.y as f32, pos.z as f32),
//                     vector: Vec3D::new(vel.x as f32, vel.y as f32, vel.z as f32),
//                 }])
//                 .map_err(Error::MsgSender)
//                 .unwrap()
//                 .with_time(Timeline::default(), time)
//                 .send(&stream)
//                 .map_err(Error::DataTable)
//                 .unwrap();
//         }
//     }

//     pub fn time_pos_sensor(stream: RecordingStream) -> impl Fn(Time, BodyPos) {
//         move |Time(time), BodyPos(pos)| {
//             let time = (time * 1000.0) as i64;
//             rerun::MsgSender::new("pos")
//                 .with_component(&[rerun::components::Point3D::new(
//                     pos.x as f32,
//                     pos.y as f32,
//                     pos.z as f32,
//                 )])
//                 .map_err(Error::MsgSender)
//                 .unwrap()
//                 .with_time(Timeline::default(), time)
//                 .send(&stream)
//                 .map_err(Error::DataTable)
//                 .unwrap();
//         }
//     }

//     pub fn pos_sensor(stream: RecordingStream) -> impl Fn(BodyPos) {
//         move |BodyPos(pos)| {
//             rerun::MsgSender::new("total_pos")
//                 .with_component(&[rerun::components::Point3D::new(
//                     pos.x as f32,
//                     pos.y as f32,
//                     pos.z as f32,
//                 )])
//                 .map_err(Error::MsgSender)
//                 .unwrap()
//                 .with_timeless(true)
//                 .send(&stream)
//                 .map_err(Error::DataTable)
//                 .unwrap();
//         }
//     }

//     #[derive(Debug)]
//     pub enum Error {
//         MsgSender(rerun::MsgSenderError),
//         DataTable(DataTableError),
//     }
// }
