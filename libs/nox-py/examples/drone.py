import jax
import elodin as el
from jax import numpy as np

TIME_STEP = 1.0 / 120.0


def get_knot_interval(a: jax.Array, b: jax.Array, alpha: jax.Array) -> jax.Array:
    r = a - b
    return np.dot(r, r) ** 0.5 * alpha


def remap(a, b, c, d, u) -> jax.Array:
    return lerp(c, d, (u - a) / (b - a))


def lerp(a, b, t) -> jax.Array:
    return a + (b - a) * t


def get_spline_point(t: jax.Array, points: jax.Array, alpha: jax.Array) -> jax.Array:
    k0 = 0
    k1 = get_knot_interval(points[0], points[1], alpha)
    k2 = get_knot_interval(points[1], points[2], alpha) + k1
    k3 = get_knot_interval(points[2], points[3], alpha) + k2
    u = lerp(k1, k2, t)
    a1 = remap(k0, k1, points[0], points[1], u)
    a2 = remap(k1, k2, points[1], points[2], u)
    a3 = remap(k2, k3, points[2], points[3], u)
    b1 = remap(k0, k2, a1, a2, u)
    b2 = remap(k1, k3, a2, a3, u)
    return remap(k1, k2, b1, b2, u)


points = np.array(
    [
        [34490.0, -14100.0, 11200.0],
        [34660.0, -13210.0, 11200.0],
        [35280.0, -10270.0, 11200.0],
        [38000.0, -11120.0, 11360.0],
        [40690.0, -12270.0, 11510.0],
        [42120.0, -12640.0, 11670.0],
        [44620.0, -12670.0, 11970.0],
        [45870.0, -12670.0, 12090.0],
        [51710.492557, -13782.772977, 12810.0],
    ]
)[:, [0, 2, 1]]


@el.map
def spline(_: el.WorldPos, t: el.Time) -> el.WorldPos:
    t = (t / 1.0) % (len(points) - 3)
    i = t.astype("int32")
    points_subset = np.array(
        [
            points[i],
            points[i + 1],
            points[i + 2],
            points[i + 3],
        ]
    )

    def spline(t):
        return get_spline_point(t, points_subset, np.array(0.5))

    grad = jax.jacfwd(spline)
    ang = rot_between(np.array([1.0, 0.0, 0.0]), grad(t - i))
    return el.WorldPos.from_linear(spline(t - i)) + el.WorldPos.from_angular(ang)


def rot_between(body_axis, r):
    a = np.cross(body_axis, r)
    w = 1 + np.dot(body_axis, r)
    return el.Quaternion(np.array([a[0], a[1], a[2], w])).normalize()


@el.dataclass
class Camera(el.Archetype):
    camera: el.Camera
    time: el.Time


world = el.World()
drone = world.spawn(
    [
        el.Body(world_pos=el.SpatialTransform.from_linear(points[0])),
        world.glb("https://storage.googleapis.com/elodin-marketing/models/drone.glb"),
        Camera(np.array([0]), np.array(0.0)),
    ],
    name="Drone",
)
world.spawn(
    el.Panel.viewport(
        track_entity=drone,
        track_rotation=False,
        active=True,
        show_grid=True,
    ),
    name="Viewport",
)
sys = el.advance_time(TIME_STEP) | spline
world.run(sys)
