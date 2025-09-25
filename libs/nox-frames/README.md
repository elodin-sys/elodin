# nox-frames

## Description
`nox_frames` is a Rust crate from the Elodin workspace. It provides lightweight and safe types to represent coordinate frames and transformations (poses, rotations, geodetic conversions). The goal is to manipulate vectors and poses while avoiding confusion between frames (for example `Body` vs `World`), and to remain compatible with constrained environments (`no_std`).

In other words: `nox_frames` helps you keep track of what coordinate system your vectors live in, and makes invalid operations fail at compile-time.

### What is a quaternion?
A quaternion is a mathematical structure (four components: w, x, y, z) used to represent 3D rotations. Unlike Euler angles, it avoids gimbal lock and enables smooth interpolations (slerp). In `nox_frames`, quaternions encode orientation between frames.

### What is a pose?
A pose describes the complete spatial state of an object:
- its position (a 3D vector),
- its orientation (a rotation, often represented as a quaternion).
In `nox_frames`, a pose is expressed as a rigid transformation `Pose3<From, To>`, defining how to move from one frame to another.

## Glossary
- `Frame`: a tag that identifies a coordinate system, for example: *Body*, *World*, *ECEF*, *NED*, etc.
- `Pose / DCM`: a transform between two frames (rotation matrix or equivalent).
- `ECI / GCRF`: inertial celestial frame (Earth-centered, not rotating).
- `ECEF / ITRF`: Earth-fixed terrestrial frame (rotates with the Earth).
- `NED`: local tangent frame at a site (North, East, Down axes).
- `IERS`: data tables that provide Earth rotation corrections used in transforms.

## Frame hierarchy

Many transforms in `nox_frames` follow this typical chain of frames:
```text
GCRF / ECI (inertial, space)
     ↓ time-dependent
ECEF / ITRF (Earth-fixed)
     ↓ site-dependent
NED (local North-East-Down)
```
- `ECI (GCRF)`: for space and orbital objects, inertial frame not rotating with Earth.  
- `ECEF (ITRF)`: for Earth-fixed coordinates, rotating with the planet.  
- `NED`: for a local site on Earth (navigation, drones, vehicles).

## Some examples

### Avoiding Body ↔ World ambiguity (type safety)
One of the main goals of `nox_frames` is making mistakes impossible at compile-time.
If you try to apply a transform to a vector expressed in the wrong frame, it won’t even compile.
```rust
use std::marker::PhantomData;
use std::ops::Mul;

// Define two frame tags
struct Body;
struct World;

// A simple typed 3D vector
#[derive(Clone, Copy)]
struct Vec3<F> { x: f32, y: f32, z: f32, _f: PhantomData<F> }
impl<F> Vec3<F> {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _f: PhantomData }
    }
}

// A pose representing a transform between frames
struct Pose3<From, To> { _p: PhantomData<(From, To)> }
impl<From, To> Pose3<From, To> {
    fn identity() -> Self { Self { _p: PhantomData } }
}

// Only allow: Pose3<Body, World> * Vec3<Body> → Vec3<World>
impl Mul<Vec3<Body>> for Pose3<Body, World> {
    type Output = Vec3<World>;
    fn mul(self, v: Vec3<Body>) -> Self::Output {
        Vec3::<World>::new(v.x, v.y, v.z)
    }
}

//
// Example usage
//

// A point expressed in the World frame
let p_w: Vec3<World> = Vec3::new(1.0, 0.0, 0.0);

// A transform Body→World
let t_bw: Pose3<Body, World> = Pose3::identity();

// ❌ Wrong: applying Body→World to a World point
// This line does NOT compile because the frames don't match.
// Uncomment it to see the compiler error:
//
// let _p_bad = t_bw * p_w;

// ✅ Correct: apply Body→World to a Body point
let p_b: Vec3<Body> = Vec3::new(1.0, 0.0, 0.0);
let p_w2: Vec3<World> = t_bw * p_b;
assert!(p_w2.x >= 1.0);
```

### Compose time-aware transforms (ECI → ECEF → local NED)
Transforms can depend on time (Earth rotation, nutation, etc.) and on a site (latitude/longitude).
Here we compose the inertial → Earth-fixed transform (ECI→ECEF) with a local site transform (ECEF→NED).
The result maps a vector expressed in ECI directly into the local NED frame.
```rust
use hifitime::Epoch;
use nox::{ReprMonad, tensor};
use nox_frames::earth::{eci_to_ecef, ecef_to_ned};

let epoch = Epoch::from_gregorian_utc(2019, 1, 4, 12, 0, 0, 0);

// Site latitude/longitude in radians (example values)
let lat = 40.29959_f64.to_radians();
let long = -111.72822_f64.to_radians();

// Time-aware ECI→ECEF, and site-local ECEF→NED
let t_eci_ecef = eci_to_ecef(epoch);
let t_ecef_ned = ecef_to_ned(lat, long);

// Compose to get ECI→NED (types ensure the order is correct)
let t_eci_ned = t_ecef_ned * t_eci_ecef;

// Example state expressed in ECI (GCRF)
let x_eci = tensor![-2981784.0, 5207055.0, 3161595.0];

// Apply the composite transform to get NED coordinates
let x_ned = t_eci_ned.dot(&x_eci);

// Check the result.
assert_eq!(x_ned, tensor![16172.3316007721, -4730268.58039587, -4860497.726850228]);
```

### Sun direction in ECI, mapped to ECEF
Sometimes you need a direction defined in space (for example Sun vector in ECI) and want to express it in Earth-fixed coordinates (ECEF). This is a typical mix of celestial and terrestrial frames.
```rust
use hifitime::Epoch;
use nox::tensor;
use nox_frames::earth::{sun_vec, eci_to_ecef};

let epoch = Epoch::from_gregorian_utc(2019, 1, 4, 12, 0, 0, 0);

// Unit-like direction of the Sun in ECI (GCRF)
let s_eci = sun_vec(epoch);

// Map ECI→ECEF at this time
let t_eci_ecef = eci_to_ecef(epoch);
let s_ecef = t_eci_ecef.dot(&s_eci);

// Check the result.
assert_eq!(s_ecef, tensor![0.9222945252607837, 0.024367834113164066, -0.3857188319678194]);
```
