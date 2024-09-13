import filterpy.common
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter, unscented_transform
from jax.numpy.linalg import cholesky
from numpy.random import randn

jax.config.update("jax_enable_x64", True)


def default_params() -> tuple[float, float, float]:
    kappa = 0
    alpha = 0.5
    beta = 2
    return kappa, alpha, beta


def lam(alpha, n, kappa):
    return alpha**2 * (n + kappa) - n


def sigma_points(x_0, lam, sigma) -> jax.Array:
    chi = [x_0]
    n = x_0.size
    sigma = jnp.atleast_2d(sigma)

    U = cholesky((n + lam) * sigma, upper=True)
    for i in range(n):
        chi.append(x_0 + U[i])
    for i in range(n):
        chi.append(x_0 - U[i])
    return jnp.array(chi)


def weight_mean(lam, n):
    w_0 = jnp.array([lam / (n + lam)])
    return jnp.concat((w_0, w_i(lam, n)))


def weight_covar(lam, n, alpha, beta):
    w_0 = lam / (n + lam) + (1 - alpha**2 + beta)
    w_0 = jnp.array([w_0])
    return jnp.concat((w_0, w_i(lam, n)))


def w_i(lam, n):
    w = 1 / (2 * (n + lam))
    return jnp.repeat(w, 2 * n)


def predict(sigma_points, prop_fn, mean_weights, covar_weights, prop_covar):
    x = jax.vmap(prop_fn)(sigma_points)
    # print(f"x = {x.__repr__()}")
    x_hat = jnp.dot(mean_weights, x)
    # x_hat = jnp.sum(jax.vmap(lambda w, x: w * x)(mean_weights, x), axis = 0)
    # print(f"x_hat = {x_hat}")
    covar = prop_covar
    for i in range(x.shape[0]):
        covar += covar_weights[i] * jnp.outer(x[i] - x_hat, x[i] - x_hat)
    # covar = jnp.sum(jax.vmap(lambda w, x: w * jnp.outer(x - x_hat, x - x_hat))(covar_weights, x), axis = 0) + prop_covar
    # print(f"covar = {covar}")
    return x, x_hat, covar


def innovate(sigma_points, measure_fn, mean_weights, covar_weights, noise_covar, x, x_hat):
    z = jax.vmap(measure_fn)(x)
    z_hat = jnp.dot(mean_weights, z)
    measure_covar = (
        jnp.sum(
            jax.vmap(lambda w, z: w * jnp.outer(z - z_hat, z - z_hat))(covar_weights, z),
            axis=0,
        )
        + noise_covar
    )
    cross_covar = jnp.sum(
        jax.vmap(lambda w, z, x: w * jnp.outer(x - x_hat, z - z_hat))(covar_weights, z, x),
        axis=0,
    )
    # cross_covar = 0.0
    # for i in range(x.shape[0]):
    #     dx = numpy.subtract(x[i], numpy.array(x_hat))
    #     dz = numpy.subtract(z[i], z_hat)
    #     #print(f"dx = {dx}", f"dz = {dz}")
    #     cross_covar += covar_weights[i] * numpy.outer(dx, dz)
    # print(f"z ={z}")
    # print(f"x ={x}")
    # print(f"z_hat = {z_hat} x_hat = {x_hat}")
    print(f"z_s = {z} z_hat = {z_hat} measure_covar = {measure_covar}")
    return z_hat, measure_covar, cross_covar


def ukf(x, alpha, kappa, beta, mean_covar, prop_fn, prop_covar, measure_fn, noise_covar, z):
    n = x.size
    l = lam(alpha, n, kappa)  # noqa: E741
    s_points = sigma_points(x, l, mean_covar)
    print(f"sigma_points = {s_points}")
    mean_weights = weight_mean(l, n)
    covar_weights = weight_covar(l, n, alpha, beta)
    # print(f"covar_weights = {covar_weights.__repr__()}")
    # print(f"mean_weights = {mean_weights.__repr__()}")
    x_s, x_hat, mean_covar = predict(s_points, prop_fn, mean_weights, covar_weights, prop_covar)
    print(f"x_s = {x_s} x_hat = {x_hat} mean_covar = {mean_covar}")
    z_hat, measure_covar, cross_covar = innovate(
        s_points, measure_fn, mean_weights, covar_weights, noise_covar, x_s, x_hat
    )
    cross_covar = jnp.atleast_2d(cross_covar)
    # print(f"x = {x_s}")
    print(f"cross_covar = {cross_covar.shape}")
    # print(f"measure = {measure_covar}")
    measure_covar_inv = la.inv(measure_covar)
    # print(f"measure_covar= {measure_covar}")
    # print(f"measure_covar_inv = {measure_covar_inv}")
    gain = cross_covar @ measure_covar_inv
    # print(f"gain = {gain}")
    y = z - z_hat
    # print(f"y = {y}")
    new_x = x_hat + jnp.dot(gain, y)
    mean_covar = mean_covar - gain @ (measure_covar @ gain.T)
    return new_x, mean_covar


def test_filterpy():
    alpha = 0.1
    beta = 2
    kappa = -1

    dt = 0.1

    def f(x):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        F = jnp.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        return jnp.dot(F, x)

    def h(x):
        # measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        return jnp.array([x[0], x[2]])

    x = jnp.array([-1.0, 1.0, -1.0, 1.0])
    P = jnp.eye(4) * 0.02
    z_std = 0.1
    noise_covar = jnp.diag(jnp.array([z_std**2, z_std**2]))
    prop_covar = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
    print("prop_covar = ", prop_covar.__repr__())
    # zs = jnp.array([[i+randn()*z_std, i+randn()*z_std] for i in range(50)])
    # print("zs = ", zs.__repr__())

    zs = jnp.array(
        [
            [5.17725854e-02, -6.26320583e-03],
            [7.76510642e-01, 9.41957633e-01],
            [2.00841988e00, 2.03743906e00],
            [2.99844618e00, 3.11297309e00],
            [4.02493837e00, 4.18458527e00],
            [5.15192102e00, 4.94829940e00],
            [5.97820965e00, 6.19408601e00],
            [6.90700638e00, 7.09993104e00],
            [8.16174727e00, 7.84922020e00],
            [9.07054319e00, 9.02949139e00],
            [1.00943927e01, 1.02153963e01],
            [1.10528857e01, 1.09655620e01],
            [1.20433517e01, 1.19606005e01],
            [1.30130301e01, 1.30389662e01],
            [1.39492112e01, 1.38780037e01],
            [1.50232252e01, 1.50704141e01],
            [1.59498538e01, 1.60893516e01],
            [1.70097415e01, 1.70585561e01],
            [1.81292609e01, 1.80656253e01],
            [1.90783022e01, 1.89139529e01],
            [1.99490761e01, 1.99682328e01],
            [2.10253265e01, 2.09926241e01],
            [2.18166124e01, 2.19475433e01],
            [2.29619247e01, 2.29313189e01],
            [2.40366414e01, 2.40207406e01],
            [2.50164997e01, 2.50594340e01],
            [2.60602065e01, 2.59104916e01],
            [2.68926856e01, 2.68682419e01],
            [2.81448564e01, 2.81699908e01],
            [2.89114209e01, 2.90161936e01],
            [2.99632302e01, 3.01334351e01],
            [3.09547757e01, 3.09778803e01],
            [3.20168683e01, 3.19300419e01],
            [3.28656686e01, 3.30364708e01],
            [3.39697008e01, 3.40282794e01],
            [3.50369500e01, 3.51215329e01],
            [3.59710004e01, 3.61372108e01],
            [3.70591558e01, 3.69247502e01],
            [3.80522440e01, 3.78498751e01],
            [3.90272805e01, 3.90100329e01],
            [4.00377740e01, 3.98368033e01],
            [4.08131455e01, 4.09728212e01],
            [4.18214855e01, 4.19909894e01],
            [4.31312654e01, 4.29949226e01],
            [4.40398607e01, 4.38911788e01],
            [4.50978163e01, 4.49942054e01],
            [4.61407950e01, 4.59709971e01],
            [4.69322204e01, 4.71080633e01],
            [4.80521531e01, 4.81292422e01],
            [4.91282949e01, 4.90346729e01],
        ]
    )

    points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)
    kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=lambda x, _: f(x), hx=h, points=points)
    kf.x = jnp.copy(x)
    kf.P *= 0.02
    print("filterpy.P = ", kf.P)
    kf.R = np.asarray(noise_covar)
    kf.Q = prop_covar

    for z in zs:
        kf.predict()
        kf.update(np.copy(z))
        x, P = ukf(x, alpha, kappa, beta, P, f, prop_covar, h, noise_covar, z)
        print("filterpy.x_prior = ", kf.x_prior)
        print("filterpy.P_prior = ", kf.P_prior)
        print("filterpy.S = ", kf.S)
        print("filterpy.SI = ", kf.SI)
        print("filterpy.K = ", kf.K.__repr__())
        print("filterpy.Q = ", kf.Q)
        print("filterpy.y = ", kf.y)
        print(f"new_x = {x}")
        print("filterpy.x = ", kf.x)
        print(f"new_mean_covar = {P}")
        print("filterpy.P= ", kf.P)
        assert jnp.isclose(x, kf.x, rtol=1e-7).all()
        assert jnp.isclose(P, kf.P, rtol=1e-7).all()


orbit_sub_dt = 1
orbit_dt = 2 * 60
G = 6.6743e-11  # gravitational constant
M = 5.972e24  # mass of the Earth


def orbit_prop_step(state):
    r = state[:3]
    v = state[3:]
    norm = la.norm(r)
    a = G * M * r / (norm * norm * norm)
    v = v + a * orbit_sub_dt
    r = r + v * orbit_sub_dt
    return jnp.concat((r, v))


def orbit_prop(state):
    for _ in range(orbit_dt // orbit_sub_dt):
        state = orbit_prop_step(state)
    return state


z_std = 7
earth_radius = 6378.1 * 1000
altitude = 400 * 1000
radius = earth_radius + altitude
velocity = jnp.sqrt(G * M / radius)


def gen_orbit_data(z_std=z_std):
    zs = []
    r = jnp.array([1.0, 0.0, 0.0]) * radius
    v = jnp.array([0.0, 1.0, 0.0]) * velocity
    state = jnp.concat((r, v))
    for _ in range(20):
        zs.append(state + randn(6) * z_std)
        state = orbit_prop(state)
    return zs


def _test_simple_orbit():
    # state = [x, v]
    alpha = 1
    beta = -3
    kappa = 2
    noise_covar = jnp.eye(6) * z_std**2
    r = jnp.array([1.0, 0.0, 0.0]) * radius
    v = jnp.array([0.0, 1.0, 0.0]) * velocity
    state = jnp.concat((r, v))
    prop_covar = filterpy.common.Q_discrete_white_noise(
        dim=2, dt=orbit_dt, var=0.001**2, block_size=3
    )
    P = jnp.eye(6) * z_std
    count = 20
    zs = gen_orbit_data(count)
    truth_data = gen_orbit_data(count)
    points = MerweScaledSigmaPoints(6, alpha=alpha, beta=beta, kappa=kappa)
    kf = UnscentedKalmanFilter(
        dim_x=6,
        dim_z=6,
        dt=orbit_dt,
        fx=lambda x, _: orbit_prop(x),
        hx=lambda s: s,
        points=points,
    )
    kf.x = jnp.copy(state)
    kf.P = P
    print("filterpy.P = ", kf.P)
    kf.R = noise_covar
    kf.Q = prop_covar

    for z, truth_data in zip(zs, truth_data):
        state, P = ukf(
            state,
            alpha,
            kappa,
            beta,
            P,
            orbit_prop,
            prop_covar,
            lambda s: s,
            noise_covar,
            z,
        )
        kf.predict()
        kf.update(np.copy(z))
        assert jnp.isclose(state, kf.x, rtol=1e-7).all()
        print(f"state = {state}")
        print(f"truth_data = {truth_data}")
        print(f"oP = {P}")
        print(f"fP = {kf.P}")
        print(f"filterpy.x = {kf.x}")


l = lam(1, 3, 2)  # noqa: E741
print("lambda = ", l)
print(weight_mean(l, 3))
print(weight_covar(l, 3, 1, 2))


print(
    unscented_transform(
        jnp.array([[1, 2], [2, 4], [5, 4]]),
        jnp.array([0.4, 0.1, 0.1]),
        jnp.array([0.4, 0.1, 0.1]),
    )
)


points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)
kf = UnscentedKalmanFilter(
    dim_x=2, dim_z=2, dt=0.1, fx=lambda x, _: x, hx=lambda x: x, points=points
)
kf.Wc = jnp.array([0.4, 0.1, 0.1])


x_hat = jnp.array([1.0, 2.0])
z_hat = jnp.array([2.0, 3.0])
points_x = jnp.array([[1.0, 2.0], [2.0, 4.0], [5.0, 4.0]])
points_z = jnp.array([[2.0, 3.0], [3.0, 5.0], [6.0, 5.0]])
print("cross_var = ", kf.cross_variance(x_hat, z_hat, points_x, points_z))
