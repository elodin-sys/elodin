+++
title = "Coordinate Systems"
description = "Coordinate Systems in Elodin"
draft = false
weight = 102
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 2
+++


The default coordinate systems used in Elodin align with the editor's representation of the (X, Y, Z)
axes as (red, green, blue) respectively. Also, all coordinate systems are right-handed.

## Configuring the World Frame

When creating a world in Elodin, you must explicitly specify the coordinate frame convention. This ensures clarity and prevents ambiguity about the coordinate system being used:

```python
import elodin as el

# Create a world with ENU (East-North-Up) frame for terrestrial vehicles
world = el.World(frame=el.Frame.ENU)

# Or use NED (North-East-Down) for aviation applications
world = el.World(frame=el.Frame.NED)

# Or use ECI for orbital mechanics
world = el.World(frame=el.Frame.ECI)
```

### Supported Frame Conventions

| Frame | +X Axis | +Y Axis | +Z Axis | Gravity* | Use Case |
|-------|---------|---------|---------|----------|----------|
| **ENU** | East | North | Up | [0, 0, -9.81] | Terrestrial vehicles (drones, cars, robots) |
| **NED** | North | East | Down | [0, 0, 9.81] | Aviation, marine navigation |
| **ECEF** | 0° Long | 90° E | N Pole | Position-dependent | GPS, Earth surface mapping |
| **ECI** | Vernal Equinox | 90° E | N Pole | Position-dependent | Low Earth orbit |
| **GCRF** | J2000 | 90° E | N Pole | N/A | Deep space, inertial reference |

*For frames with constant gravity vectors. GCRF has no constant gravity.

### Local Geodetic World Frames

{% image(href="/assets/coordinates") %}Elodin Default Coordinate System{% end %}

Local geodetic world frames are commonly used for terrestrial vehicles. Elodin supports two conventions:

#### ENU (East-North-Up)

The [East, North, Up (ENU)](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_east,_north,_up_(ENU)_coordinates)
coordinate system where:
- +X: East
- +Y: North
- +Z: Up

This frame is commonly used for robotics, ground vehicles, and general simulation work.

#### NED (North-East-Down)

The [North, East, Down (NED)](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates)
coordinate system where:
- +X: North
- +Y: East
- +Z: Down

This frame is the standard in aviation and marine applications. Note that gravity points in the +Z direction in NED.

### Geocentric World Frame

A geodetic world frame is represented using either the [Earth-Centered, Earth-Fixed (ECEF)](https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
or the [Earth-Centered Inertial (ECI)](https://en.wikipedia.org/wiki/Earth-centered_inertial) coordinate system. ECEF is useful for
Earth-relative positioning (e.g. GPS, mapping) while ECI is useful for Earth-orbiting space vehicles. The main difference between
ECEF and ECI is in how they handle the rotation of the Earth. In ECEF, the Earth is fixed and the coordinate system rotates with
the Earth. In ECI, the coordinate system is fixed with respect to the stars and does not rotate with the Earth.

#### ECEF

ECEF is loosely defined as:
- +X: along the equatorial plane towards the prime meridian (0° longitude)
- +Y: along the equatorial plane towards 90°E longitude
- +Z: perpendicular to the equatorial plane towards the North Pole

However, the terms "equatorial plane", "prime meridian", "North Pole", and "origin" are not precisely defined in the ECEF
coordinate system. The [International Terrestrial Reference Frame](https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame) provides a more precise definition of the ECEF coordinate system:
- Origin: center of mass for the whole Earth (including oceans and atmosphere)
- +Z: direction of the IERS Reference Pole (IRP)
- +X: intersection of the IERS Reference Meridian (IRM) and the plane passing through the origin and normal to the IRP
- +Y: completes the right-handed coordinate system

#### ECI

ECI is loosely defined as:
- +X: along the equatorial plane and permanently fixed in a direction relative to the celestial sphere
- +Y: along the equatorial plane pointing 90° East about the celestial equator
- +Z: perpendicular to the equatorial plane towards the North Pole

Similar to ECEF, ECI itself is not a precise coordinate system. [GCRS](https://en.wikipedia.org/wiki/Barycentric_and_geocentric_celestial_reference_systems) is a specific realization of the ECI coordinate system where:
- Origin: center of mass for the whole Earth (including oceans and atmosphere)
- +Z: points towards the Earth's north celestial pole as it was at noon on January 1, 2000
- +X: points towards the vernal equinox (also called the first point of Aries) as it was at noon on January 1, 2000, in the Terrestrial Time scale
- +Y: completes the right-handed coordinate system

{% alert(kind="info") %}
"J2000" refers to 4 related but different frames: J2000 proper is barycentric, not earth centric; ICRF which is barycentric & offset from
J2000 by 0.1 arc seconds; "J2000 earth-centric" is aligned with "J2000", but centered on earth; finally GCRS/GCRF is signed with ICRF but
centered on Earth. We typically use ICRF as the default ECI frame in Elodin examples. References:
[(1)](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf)
[(2)](https://space.stackexchange.com/questions/26259/what-is-the-difference-between-gcrs-and-j2000-frames)
[(3)](https://en.wikipedia.org/wiki/Barycentric_and_geocentric_celestial_reference_systems)
{% end %}

### Body Frame

For vehicles, the body frame is represented as a right-handed coordinate system with:
- +X: along the body centerline towards the nose
- +Y: left when looking forward
- +Z: up when looking forward

This is because vehicle attitude is typically defined in terms of roll, pitch, and yaw angles in that order. A consequence of this is that a positive yaw angle corresponds to a counterclockwise rotation and a positive pitch angle corresponds to a nose-down rotation.

{% alert(kind="notice") %}
The world frame can be aligned with the body frame by considering the (+X, +Y) axes in the world frame as (forward, left) respectively with the (+Z) axis pointing up.
{% end %}
