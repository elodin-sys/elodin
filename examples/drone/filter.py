import jax
from jax import numpy as jnp
import math
from jax.typing import ArrayLike


# Discrete-time LPF (exponentially weighted moving average)
class LPF:
    def __init__(self, cutoff_freq: ArrayLike, sample_freq: float):
        assert sample_freq > 0

        dt = 1 / sample_freq
        rc = 1 / (2 * math.pi * jnp.array(cutoff_freq))
        # treat 0 Hz as no filtering
        rc = jnp.nan_to_num(rc, posinf=0)
        self.alpha = dt / (rc + dt)

    def apply(self, y_n1: jax.Array, x_n: jax.Array) -> jax.Array:
        return y_n1 + self.alpha * (x_n - y_n1)

    def freq_response(
        self, freq: jax.typing.ArrayLike, sample_freq: float
    ) -> jax.Array:
        alpha = self.alpha
        phi = jnp.exp(-1j * 2 * jnp.pi * freq / sample_freq)
        return 1 / jnp.abs(1 - alpha * (1 - phi))


# Discrete-time second order recursive linear filter
class BiquadLPF:
    def __init__(self, cutoff_freq: float, sample_freq: float):
        assert cutoff_freq > 0
        assert sample_freq > 0

        Q = 1 / math.sqrt(2)
        omega = 2 * math.pi * cutoff_freq / sample_freq
        alpha = math.sin(omega) / (2 * Q)
        a0 = 1 + alpha

        b0 = (1 - math.cos(omega)) / 2
        b1 = 1 - math.cos(omega)
        b2 = b0
        a1 = -2 * math.cos(omega)
        a2 = 1 - alpha

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        self.coefs = jnp.array([b0, b1, b2, a1, a2])

    # delay is [x_n-1, x_n-2, y_n-1, y_n-2]
    # returns updated delay, where y is output[2]
    def apply(self, delay: jax.Array, x_n: jax.Array) -> jax.Array:
        # assert that the shape of delay is (4, <shape of x_n>):
        assert delay.shape == (4, *x_n.shape)
        b0, b1, b2, a1, a2 = self.coefs
        x_n1, x_n2, y_n1, y_n2 = delay
        y_n = b0 * x_n + b1 * x_n1 + b2 * x_n2 - a1 * y_n1 - a2 * y_n2
        return jnp.array([x_n, x_n1, y_n, y_n1])

    def freq_response(
        self, freq: jax.typing.ArrayLike, sample_freq: float
    ) -> jax.Array:
        b0, b1, b2, a1, a2 = self.coefs
        phi = (jnp.sin(jnp.pi * freq * 2 / (2 * sample_freq))) ** 2
        r = (
            (b0 + b1 + b2) ** 2
            - 4 * (b0 * b1 + 4 * b0 * b2 + b1 * b2) * phi
            + 16 * b0 * b2 * phi * phi
        ) / (
            (1 + a1 + a2) ** 2 - 4 * (a1 + 4 * a2 + a1 * a2) * phi + 16 * a2 * phi * phi
        )
        return r**0.5


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    CUTOFF_FREQ = 40  # hz
    SAMPLE_FREQ = 1200  # hz

    signal_freq = CUTOFF_FREQ / 4  # low frequency wave (below LPF cutoff)
    noise_freq = CUTOFF_FREQ * 10  # high frequency wave (above LPF cutoff)

    print(f"cut-off frequency = {CUTOFF_FREQ} Hz")
    print(f"sample frequency  = {SAMPLE_FREQ} Hz")
    print(f"signal frequency  = {signal_freq} Hz")
    print(f"noise frequency   = {noise_freq} Hz")

    samples = int(16 * SAMPLE_FREQ / CUTOFF_FREQ)
    lpf = LPF(CUTOFF_FREQ, SAMPLE_FREQ)
    biquad_lpf = BiquadLPF(CUTOFF_FREQ, SAMPLE_FREQ)

    def sine(t, freq):
        return jnp.sin(t * 2 * jnp.pi * freq)

    dt = 1.0 / SAMPLE_FREQ
    ts = jnp.arange(0, samples, 1) * dt
    signal = sine(jnp.array(ts), signal_freq)
    noise = sine(jnp.array(ts), noise_freq)
    raw = signal + 0.3 * noise

    lpf_delay = jnp.float64(0.0)
    biquiad_lpf_delay = jnp.zeros(4)
    lpf_filtered = []
    biquad_lpf_filtered = []
    for raw_sample in raw:
        lpf_delay = lpf.apply(lpf_delay, jnp.float64(raw_sample))
        lpf_filtered.append(lpf_delay)
        biquiad_lpf_delay = biquad_lpf.apply(biquiad_lpf_delay, jnp.float64(raw_sample))
        biquad_lpf_filtered.append(biquiad_lpf_delay[2].astype("float"))

    # Signal plot
    fig = plt.figure(num="Low Pass Filter")

    p1 = plt.subplot(2, 1, 1)
    p1.plot(ts, raw, color="orange", label="raw", zorder=-1)
    p1.plot(ts, signal, color="green", label="signal")
    p1.plot(ts, lpf_filtered, color="red", label="LPF filtered")
    p1.plot(ts, biquad_lpf_filtered, color="blue", label="biquad LPF filtered")
    p1.set_xlabel("time")
    p1.legend()

    # FFT plot
    fft_x = jnp.linspace(0.0, 1.0 / (2.0 * dt), int(samples / 2))
    fft_raw = jnp.fft.fft(jnp.array(raw))
    fft_lpf_filtered = jnp.fft.fft(jnp.array(lpf_filtered))
    fft_biquad_lpf_filtered = jnp.fft.fft(jnp.array(biquad_lpf_filtered))
    fft_response = biquad_lpf.freq_response(fft_x, SAMPLE_FREQ)

    p2 = plt.subplot(2, 1, 2)
    p2.plot(
        fft_x,
        2.0 / samples * jnp.abs(fft_raw[: samples // 2]),
        color="orange",
        label="raw FFT",
    )
    p2.plot(
        fft_x,
        2.0 / samples * jnp.abs(fft_lpf_filtered[: samples // 2]),
        color="red",
        label="LPF filtered FFT",
    )
    p2.plot(
        fft_x,
        2.0 / samples * jnp.abs(fft_biquad_lpf_filtered[: samples // 2]),
        color="blue",
        label="biquad LPF filtered FFT",
    )
    p2.plot(fft_x, fft_response, color="grey", label="LPF response")
    p2.set_xlabel("frequency")
    p2.set_xscale("log")
    p2.xaxis.set_major_formatter(ScalarFormatter())
    p2.legend()

    fig.tight_layout()
    plt.show()
