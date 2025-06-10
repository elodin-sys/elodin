from typing import Any, Callable, Tuple

import jax
import jax.typing

class UKFState:
    x_hat: jax.typing.ArrayLike
    covar: jax.typing.ArrayLike
    prop_covar: jax.typing.ArrayLike
    noise_covar: jax.typing.ArrayLike

    def __init__(
        self,
        x_hat: jax.typing.ArrayLike,
        covar: jax.typing.ArrayLike,
        prop_covar: jax.typing.ArrayLike,
        noise_covar: jax.typing.ArrayLike,
        alpha: float,
        beta: float,
        kappa: float
    ) -> None: ...

    def update(
        self,
        z: jax.typing.ArrayLike,
        prop_fn: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
        measure_fn: Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike], jax.typing.ArrayLike]
    ) -> None: ...

def unscented_transform(
    points: jax.typing.ArrayLike,
    mean_weights: jax.typing.ArrayLike,
    covar_weights: jax.typing.ArrayLike
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]: ...

def cross_covar(
    x_hat: jax.typing.ArrayLike,
    z_hat: jax.typing.ArrayLike,
    points_x: jax.typing.ArrayLike,
    points_z: jax.typing.ArrayLike,
    covar_weights: jax.typing.ArrayLike
) -> jax.typing.ArrayLike: ...

def predict(
    sigma_points: jax.typing.ArrayLike,
    prop_fn: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
    mean_weights: jax.typing.ArrayLike,
    covar_weights: jax.typing.ArrayLike,
    prop_covar: jax.typing.ArrayLike
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike]: ...

def innovate(
    x_points: jax.typing.ArrayLike,
    z: jax.typing.ArrayLike,
    measure_fn: Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike], jax.typing.ArrayLike],
    mean_weights: jax.typing.ArrayLike,
    covar_weights: jax.typing.ArrayLike,
    noise_covar: jax.typing.ArrayLike
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike]: ...
