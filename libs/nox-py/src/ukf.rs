use nox_ecs::nox::{Dyn, Noxpr, Op, Tensor};
use pyo3::prelude::*;
use roci_adcs::ukf;
use roci_adcs::ukf::{UncheckedMerweConfig, UncheckedState};

use crate::Error;

#[pyfunction]
pub fn unscented_transform(
    points: Py<PyAny>,
    mean_weights: Py<PyAny>,
    covar_weights: Py<PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>), Error> {
    let points: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(points));
    let mean_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(mean_weights));
    let covar_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(covar_weights));
    let (x_hat, covar) = ukf::unscented_transform(&points, &mean_weights, &covar_weights);
    let x_hat = x_hat.inner().to_jax()?;
    let covar = covar.inner().to_jax()?;
    Ok((x_hat, covar))
}

#[pyfunction]
pub fn cross_covar(
    x_hat: Py<PyAny>,
    z_hat: Py<PyAny>,
    points_x: Py<PyAny>,
    points_z: Py<PyAny>,
    covar_weights: Py<PyAny>,
) -> Result<Py<PyAny>, Error> {
    let x_hat: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(x_hat));
    let z_hat: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(z_hat));
    let points_x: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(points_x));
    let points_z: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(points_z));
    let covar_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(covar_weights));
    let cross_covar = ukf::cross_covar(&x_hat, &z_hat, points_x, points_z, covar_weights);
    cross_covar.inner().to_jax().map_err(Error::from)
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn predict(
    sigma_points: Py<PyAny>,
    prop_fn: Py<PyAny>,
    mean_weights: Py<PyAny>,
    covar_weights: Py<PyAny>,
    prop_covar: Py<PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>, Py<PyAny>), Error> {
    let sigma_points: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(sigma_points));
    let mean_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(mean_weights));
    let covar_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(covar_weights));
    let prop_covar: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(prop_covar));

    let prop_fn_wrapper = |x: Tensor<f64, Dyn, Op>| -> Tensor<f64, Dyn, Op> {
        let py_x = x.inner().to_jax().unwrap();
        let result = Python::attach(|py| prop_fn.call1(py, (py_x,)).unwrap());
        Tensor::from_inner(Noxpr::jax(result))
    };

    let (points, x_hat, covar) = ukf::predict(
        sigma_points,
        prop_fn_wrapper,
        &mean_weights,
        &covar_weights,
        &prop_covar,
    );

    let py_points = points.inner().to_jax()?;
    let py_x_hat = x_hat.inner().to_jax()?;
    let py_covar = covar.inner().to_jax()?;

    Ok((py_points, py_x_hat, py_covar))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn innovate(
    x_points: Py<PyAny>,
    z: Py<PyAny>,
    measure_fn: Py<PyAny>,
    mean_weights: Py<PyAny>,
    covar_weights: Py<PyAny>,
    noise_covar: Py<PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>, Py<PyAny>), Error> {
    let x_points: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(x_points));
    let z: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(z));
    let mean_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(mean_weights));
    let covar_weights: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(covar_weights));
    let noise_covar: Tensor<f64, (Dyn, Dyn), Op> = Tensor::from_inner(Noxpr::jax(noise_covar));

    let measure_fn_wrapper =
        |x: Tensor<f64, Dyn, Op>, z: Tensor<f64, Dyn, Op>| -> Tensor<f64, Dyn, Op> {
            let py_x = x.inner().to_jax().unwrap();
            let py_z = z.inner().to_jax().unwrap();
            let result = Python::attach(|py| measure_fn.call1(py, (py_x, py_z)).unwrap());
            Tensor::from_inner(Noxpr::jax(result))
        };

    let (points, z_hat, measure_covar) = ukf::innovate(
        &x_points,
        &z,
        measure_fn_wrapper,
        &mean_weights,
        &covar_weights,
        &noise_covar,
    );

    let py_points = points.inner().to_jax()?;
    let py_z_hat = z_hat.inner().to_jax()?;
    let py_measure_covar = measure_covar.inner().to_jax()?;

    Ok((py_points, py_z_hat, py_measure_covar))
}
#[pyclass]
pub struct UKFState {
    #[pyo3(get, set)]
    x_hat: Py<PyAny>,
    #[pyo3(get, set)]
    covar: Py<PyAny>,
    #[pyo3(get, set)]
    prop_covar: Py<PyAny>,
    #[pyo3(get, set)]
    noise_covar: Py<PyAny>,
    config: UncheckedMerweConfig,
}

#[pymethods]
impl UKFState {
    #[new]
    fn new(
        x_hat: Py<PyAny>,
        covar: Py<PyAny>,
        prop_covar: Py<PyAny>,
        noise_covar: Py<PyAny>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> PyResult<Self> {
        let x_shape: Vec<usize> = Python::attach(|py| {
            Ok::<_, crate::Error>(x_hat.getattr(py, "shape")?.extract::<Vec<usize>>(py)?)
        })?;
        let n = x_shape[0];
        let config = UncheckedMerweConfig::new(n, alpha, beta, kappa);

        Ok(UKFState {
            config,
            x_hat,
            covar,
            prop_covar,
            noise_covar,
        })
    }

    fn update(
        &mut self,
        z: Py<PyAny>,
        prop_fn: Py<PyAny>,
        measure_fn: Py<PyAny>,
    ) -> Result<(), crate::Error> {
        let z: Tensor<f64, Dyn, Op> = Tensor::from_inner(Noxpr::jax(z));

        let prop_fn_wrapper = |x: Tensor<f64, Dyn, Op>| -> Tensor<f64, Dyn, Op> {
            let py_x = x.inner().to_jax().unwrap();
            let result = Python::attach(|py| prop_fn.call1(py, (py_x,)).unwrap());
            Tensor::from_inner(Noxpr::jax(result))
        };

        let measure_fn_wrapper =
            |x: Tensor<f64, Dyn, Op>, z: Tensor<f64, Dyn, Op>| -> Tensor<f64, Dyn, Op> {
                let py_x = x.inner().to_jax().unwrap();
                let py_z = z.inner().to_jax().unwrap();
                let result = Python::attach(|py| measure_fn.call1(py, (py_x, py_z)).unwrap());
                Tensor::from_inner(Noxpr::jax(result))
            };

        let x_hat: Tensor<f64, Dyn, Op> =
            Python::attach(|py| Tensor::from_inner(Noxpr::jax(self.x_hat.clone_ref(py))));
        let covar: Tensor<f64, (Dyn, Dyn), Op> =
            Python::attach(|py| Tensor::from_inner(Noxpr::jax(self.covar.clone_ref(py))));
        let prop_covar: Tensor<f64, (Dyn, Dyn), Op> =
            Python::attach(|py| Tensor::from_inner(Noxpr::jax(self.prop_covar.clone_ref(py))));
        let noise_covar: Tensor<f64, (Dyn, Dyn), Op> =
            Python::attach(|py| Tensor::from_inner(Noxpr::jax(self.noise_covar.clone_ref(py))));

        let state = UncheckedState {
            x_hat,
            covar,
            prop_covar,
            noise_covar,
        }
        .update::<Dyn>(self.config, z, prop_fn_wrapper, measure_fn_wrapper)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.x_hat = state.x_hat.inner().to_jax()?;
        self.covar = state.covar.inner().to_jax()?;
        self.prop_covar = state.prop_covar.inner().to_jax()?;
        self.noise_covar = state.noise_covar.inner().to_jax()?;

        Ok(())
    }
}

pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child = PyModule::new(parent_module.py(), "ukf")?;
    child.add_function(wrap_pyfunction!(unscented_transform, &child)?)?;
    child.add_function(wrap_pyfunction!(cross_covar, &child)?)?;
    child.add_function(wrap_pyfunction!(predict, &child)?)?;
    child.add_function(wrap_pyfunction!(innovate, &child)?)?;
    child.add_class::<UKFState>()?;
    parent_module.add_submodule(&child)
}
