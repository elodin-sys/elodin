pub use hifitime::Epoch;
use sys::MAG_SetDefaults;

mod coef;
pub mod sys {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(rustdoc::broken_intra_doc_links)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct MagneticModel {
    model: Box<sys::MAGtype_MagneticModel>,
    timed_model: Box<sys::MAGtype_MagneticModel>,
    ellipsoid: Ellipsoid,
}

unsafe impl Send for MagneticModel {}
unsafe impl Sync for MagneticModel {}

impl Default for MagneticModel {
    fn default() -> Self {
        Self {
            model: coef::magnetic_model(),
            timed_model: coef::magnetic_model(),
            ellipsoid: Ellipsoid::default(),
        }
    }
}

pub struct Date {
    inner: sys::MAGtype_Date,
}

impl Date {
    pub fn new(year: i32, month: i32, day: i32, decimal_year: f64) -> Self {
        Self {
            inner: sys::MAGtype_Date {
                Year: year,
                Month: month,
                Day: day,
                DecimalYear: decimal_year,
            },
        }
    }
}

impl From<Epoch> for Date {
    fn from(val: Epoch) -> Self {
        let (y, m, d, _, _, _, _) = val.to_gregorian_utc();
        let wmm_epoch = Epoch::from_gregorian_utc(2020, 1, 1, 0, 0, 0, 0);
        let decimal_year = (val - wmm_epoch).to_seconds() / hifitime::SECONDS_PER_YEAR;
        Date::new(y, m as i32, d as i32, decimal_year)
    }
}

pub struct GeodeticCoords {
    coords: sys::MAGtype_CoordGeodetic,
}

impl GeodeticCoords {
    pub fn new(
        phi: f64,
        lambda: f64,
        height_above_ellipsoid: f64,
        height_above_geoid: f64,
        use_geoid: bool,
    ) -> Self {
        Self {
            coords: sys::MAGtype_CoordGeodetic {
                lambda,
                phi,
                HeightAboveEllipsoid: height_above_ellipsoid,
                HeightAboveGeoid: height_above_geoid,
                UseGeoid: if use_geoid { 1 } else { 0 },
            },
        }
    }

    pub fn with_elliposid_height(phi: f64, lambda: f64, height: f64) -> Self {
        Self::new(phi, lambda, height, 0.0, false)
    }

    pub fn with_geoid_height(phi: f64, lambda: f64, height: f64) -> Self {
        Self::new(phi, lambda, 0.0, height, true)
    }
}

#[derive(Debug)]
pub struct Elements {
    pub decl: f64,
    pub incl: f64,
    pub f: f64,
    pub h: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub gv: f64,
    pub decl_dot: f64,
    pub incl_dot: f64,
    pub f_dot: f64,
    pub h_dot: f64,
    pub x_dot: f64,
    pub y_dot: f64,
    pub z_dot: f64,
    pub gv_dot: f64,
}

impl From<sys::MAGtype_GeoMagneticElements> for Elements {
    fn from(sys: sys::MAGtype_GeoMagneticElements) -> Self {
        Elements {
            decl: sys.Decl,
            incl: sys.Incl,
            f: sys.F,
            h: sys.H,
            x: sys.X,
            y: sys.Y,
            z: sys.Z,
            gv: sys.GV,
            decl_dot: sys.Decldot,
            incl_dot: sys.Incldot,
            f_dot: sys.Fdot,
            h_dot: sys.Hdot,
            x_dot: sys.Xdot,
            y_dot: sys.Ydot,
            z_dot: sys.Zdot,
            gv_dot: sys.GVdot,
        }
    }
}

impl Elements {
    pub fn b_field(&self) -> [f64; 3] {
        [self.x * 1e-9, self.y * 1e-9, self.z * 1e-9]
    }
}

pub struct ErrorBars(pub Elements);

impl MagneticModel {
    pub fn calculate_field(
        &mut self,
        date: impl Into<Date>,
        geodetic: GeodeticCoords,
    ) -> (Elements, ErrorBars) {
        let date = date.into();
        let mut sphere_coords = sys::MAGtype_CoordSpherical::default();
        let mut elements = sys::MAGtype_GeoMagneticElements::default();
        let mut error_bars = sys::MAGtype_GeoMagneticElements::default();
        unsafe {
            sys::MAG_GeodeticToSpherical(self.ellipsoid.model, geodetic.coords, &mut sphere_coords);
            sys::MAG_TimelyModifyMagneticModel(
                date.inner,
                self.model.as_mut(),
                self.timed_model.as_mut(),
            );
            sys::MAG_Geomag(
                self.ellipsoid.model,
                sphere_coords,
                geodetic.coords,
                self.timed_model.as_mut(),
                &mut elements,
            );
            sys::MAG_CalculateGridVariation(geodetic.coords, &mut elements);
            sys::MAG_WMMErrorCalc(elements.H, &mut error_bars);
        }
        (elements.into(), ErrorBars(error_bars.into()))
    }
}

pub struct Ellipsoid {
    pub model: sys::MAGtype_Ellipsoid,
}

impl Default for Ellipsoid {
    fn default() -> Self {
        let mut geoid = sys::MAGtype_Geoid::default();
        let mut ellipsoid = sys::MAGtype_Ellipsoid::default();
        unsafe { MAG_SetDefaults(&mut ellipsoid as *mut _, &mut geoid as *mut _) };
        Self { model: ellipsoid }
    }
}

pub struct Geoid {
    pub model: sys::MAGtype_Geoid,
}

impl Default for Geoid {
    fn default() -> Self {
        let mut geoid = sys::MAGtype_Geoid::default();
        let mut ellipsoid = sys::MAGtype_Ellipsoid::default();
        unsafe { MAG_SetDefaults(&mut ellipsoid as *mut _, &mut geoid as *mut _) };
        Self { model: geoid }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_get_field_strength() {
        let mut model = MagneticModel::default();
        //let date = Date::new(2024, 1, 1, 4.0);
        let epoch = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
        let geodetic = GeodeticCoords::with_geoid_height(37.760283, -122.388016, 0.0);
        let (elements, _) = model.calculate_field(epoch, geodetic);
        println!("elements: {:#?}", elements);
        assert_relative_eq!(
            elements.b_field()[..],
            [22_383.7e-9, 5_187.5e-9, 41_730.9e-9],
            epsilon = 1e-5
        );
    }
}
