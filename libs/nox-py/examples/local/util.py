import jax
import jax.numpy as jnp

def second_order_butterworth(
    signal: jax.Array, f_sampling: int, f_cutoff: int, method: str = "forward"
) -> jax.Array:
    "https://stackoverflow.com/questions/20924868/calculate-coefficients-of-2nd-order-butterworth-low-pass-filter"
    if method == "forward_backward":
        signal = second_order_butterworth(signal, f_sampling, f_cutoff, "forward")
        return second_order_butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError
    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = jax.lax.scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))
    if method == "backward":
        signal = jnp.flip(signal, axis=0)
    return signal
