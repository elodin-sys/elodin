use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

static PARAM_SPEC: LazyLock<Mutex<Option<ParamSpecData>>> = LazyLock::new(|| Mutex::new(None));

const CONTEXT_ENV: &str = "ELODIN_MONTE_CARLO_CONTEXT";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParamSpecData {
    params: HashMap<String, ParamData>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParamData {
    type_name: String,
    default: Value,
    min: Option<Value>,
    max: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContextData {
    run_id: Option<String>,
    seed: Option<u64>,
    db_path: Option<String>,
    db_addr: Option<String>,
    cache_dir: Option<String>,
    run_dir: Option<String>,
    #[serde(default)]
    params: HashMap<String, Value>,
    #[serde(default)]
    slots: HashMap<String, Value>,
}

#[pyclass(name = "Param")]
#[derive(Clone)]
pub struct PyParam {
    data: ParamData,
}

#[pymethods]
impl PyParam {
    #[new]
    #[pyo3(signature = (type_, default = None, min = None, max = None))]
    fn new(
        type_: &Bound<'_, PyAny>,
        default: Option<&Bound<'_, PyAny>>,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let type_name = if let Ok(type_obj) = type_.downcast::<PyType>() {
            type_obj
                .getattr("__name__")?
                .extract::<String>()
                .unwrap_or_else(|_| "object".to_string())
        } else if let Ok(raw) = type_.extract::<String>() {
            raw
        } else {
            type_.repr()?.extract::<String>()?
        };
        let default = match default {
            Some(value) => py_to_json(value)?,
            None => Value::Null,
        };
        Ok(Self {
            data: ParamData {
                type_name,
                default,
                min: min.map(py_to_json).transpose()?,
                max: max.map(py_to_json).transpose()?,
            },
        })
    }
}

#[pyclass(name = "ParamsSpec")]
#[derive(Clone)]
pub struct PyParamsSpec {
    data: ParamSpecData,
}

#[pymethods]
impl PyParamsSpec {
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.data)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pyclass(name = "Params")]
pub struct PyParams {
    run_id: Option<String>,
    seed: Option<u64>,
    db_path: Option<String>,
    db_addr: Option<String>,
    cache_dir: Option<String>,
    run_dir: Option<String>,
    params: HashMap<String, Value>,
    slots: HashMap<String, Value>,
}

#[pymethods]
impl PyParams {
    #[getter]
    fn run_id(&self) -> Option<String> {
        self.run_id.clone()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.seed
    }

    #[getter]
    fn db_path(&self) -> Option<String> {
        self.db_path.clone()
    }

    #[getter]
    fn db_addr(&self) -> Option<String> {
        self.db_addr.clone()
    }

    #[getter]
    fn cache_dir(&self) -> Option<String> {
        self.cache_dir.clone()
    }

    #[getter]
    fn run_dir(&self) -> Option<String> {
        self.run_dir.clone()
    }

    #[pyo3(signature = (key, default = None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.params.get(key) {
            return json_to_py(py, value);
        }
        match default {
            Some(value) => Ok(value.clone().unbind()),
            None => Ok(py.None()),
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.params
            .get(key)
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
            .and_then(|value| json_to_py(py, value))
    }

    fn as_overrides_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        json_map_to_dict(py, &self.params)
    }

    fn slots(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        json_map_to_dict(py, &self.slots)
    }
}

#[pyfunction]
#[pyo3(signature = (**kwargs))]
fn params_spec(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyParamsSpec> {
    let mut params = HashMap::new();
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            let param = value.extract::<PyRef<'_, PyParam>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "params_spec value for `{key}` must be el.monte_carlo.Param"
                ))
            })?;
            params.insert(key, param.data.clone());
        }
    }
    let data = ParamSpecData { params };
    *PARAM_SPEC.lock().expect("param spec mutex poisoned") = Some(data.clone());
    Ok(PyParamsSpec { data })
}

#[pyfunction]
#[pyo3(signature = (spec = None))]
fn params(spec: Option<&PyParamsSpec>) -> PyResult<PyParams> {
    let spec_data = spec.map(|spec| spec.data.clone()).or_else(|| {
        PARAM_SPEC
            .lock()
            .expect("param spec mutex poisoned")
            .clone()
    });
    let mut values = HashMap::new();
    if let Some(spec) = spec_data {
        for (name, param) in spec.params {
            values.insert(name, param.default);
        }
    }

    let context = match std::env::var(CONTEXT_ENV) {
        Ok(path) => {
            let contents = fs::read_to_string(&path).map_err(|err| {
                PyRuntimeError::new_err(format!("failed to read {CONTEXT_ENV}={path}: {err}"))
            })?;
            Some(
                serde_json::from_str::<ContextData>(&contents)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        }
        Err(_) => None,
    };

    if let Some(context) = context {
        values.extend(context.params.clone());
        Ok(PyParams {
            run_id: context.run_id,
            seed: context.seed,
            db_path: context.db_path,
            db_addr: context.db_addr,
            cache_dir: context.cache_dir,
            run_dir: context.run_dir,
            params: values,
            slots: context.slots,
        })
    } else {
        Ok(PyParams {
            run_id: None,
            seed: None,
            db_path: None,
            db_addr: None,
            cache_dir: None,
            run_dir: None,
            params: values,
            slots: HashMap::new(),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (**kwargs))]
fn result(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let Some(kwargs) = kwargs else {
        return Ok(());
    };
    let run_dir = params(None)?.run_dir.ok_or_else(|| {
        PyRuntimeError::new_err("result() requires ELODIN_MONTE_CARLO_CONTEXT with run_dir")
    })?;
    let mut output = HashMap::new();
    for (key, value) in kwargs.iter() {
        output.insert(key.extract::<String>()?, py_to_json(&value)?);
    }
    let path = PathBuf::from(run_dir).join("result.json");
    let json = serde_json::to_string_pretty(&output)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    fs::write(&path, json)
        .map_err(|err| PyRuntimeError::new_err(format!("write {path:?}: {err}")))?;
    Ok(())
}

#[pyfunction]
pub fn spec_json() -> PyResult<String> {
    let data = PARAM_SPEC
        .lock()
        .expect("param spec mutex poisoned")
        .clone();
    serde_json::to_string_pretty(&data.unwrap_or(ParamSpecData {
        params: HashMap::new(),
    }))
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child = PyModule::new(parent_module.py(), "monte_carlo")?;
    child.add_class::<PyParam>()?;
    child.add_class::<PyParams>()?;
    child.add_class::<PyParamsSpec>()?;
    child.add_function(wrap_pyfunction!(params_spec, &child)?)?;
    child.add_function(wrap_pyfunction!(params, &child)?)?;
    child.add_function(wrap_pyfunction!(result, &child)?)?;
    child.add_function(wrap_pyfunction!(spec_json, &child)?)?;
    parent_module.add_submodule(&child)
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = value.downcast::<PyBool>() {
        return Ok(Value::Bool(value.is_true()));
    }
    if let Ok(value) = value.downcast::<PyInt>() {
        return Ok(Value::Number(value.extract::<i64>()?.into()));
    }
    if let Ok(value) = value.downcast::<PyFloat>() {
        let value = value.extract::<f64>()?;
        return serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("float values must be finite"));
    }
    if let Ok(value) = value.downcast::<PyString>() {
        return Ok(Value::String(value.extract::<String>()?));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let mut values = Vec::with_capacity(list.len());
        for item in list.iter() {
            values.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(values));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            map.insert(key.extract::<String>()?, py_to_json(&value)?);
        }
        return Ok(Value::Object(map));
    }
    Err(PyTypeError::new_err(format!(
        "value is not JSON serializable: {}",
        value.get_type().name()?
    )))
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any().unbind()),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            } else if let Some(value) = value.as_u64() {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            } else if let Some(value) = value.as_f64() {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        Value::String(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        Value::Array(values) => {
            let items = values
                .iter()
                .map(|value| json_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, json_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn json_map_to_dict(py: Python<'_>, map: &HashMap<String, Value>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in map {
        dict.set_item(key, json_to_py(py, value)?)?;
    }
    Ok(dict.unbind())
}
