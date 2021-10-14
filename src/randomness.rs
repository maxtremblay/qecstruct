use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro512StarStar;

pub type RandomNumberGenerator = Xoshiro512StarStar;

#[pyclass(name = "Rng", module = "qecstruct")]
#[pyo3(text_signature = "(seed=None)")]
pub struct PyRng {
    pub(crate) inner: RandomNumberGenerator,
}

impl From<RandomNumberGenerator> for PyRng {
    fn from(inner: RandomNumberGenerator) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRng {
    #[new]
    #[args(seed = "None")]
    pub fn new(seed: Option<u64>) -> Self {
        PyRng::from(match seed {
            Some(seed) => RandomNumberGenerator::seed_from_u64(seed),
            None => RandomNumberGenerator::from_entropy(),
        })
    }

    #[pyo3(text_signature = "(self)")]
    pub fn jump(&mut self) -> Self {
        let other = Self {
            inner: self.inner.clone(),
        };
        self.inner.jump();
        other
    }

    #[pyo3(text_signature = "(self)")]
    pub fn long_jump(&mut self) -> Self {
        let other = Self {
            inner: self.inner.clone(),
        };
        self.inner.long_jump();
        other
    }

    #[pyo3(text_signature = "(self)")]
    #[args(range = "None")]
    pub fn rand_int(&mut self, range: Option<(u64, u64)>) -> u64 {
        match range {
            Some(range) => self.inner.gen_range(range.0..range.1),
            None => self.inner.gen(),
        }
    }

    #[pyo3(text_signature = "(self)")]
    #[args(range = "None")]
    pub fn rand_float(&mut self, range: Option<(f64, f64)>) -> f64 {
        match range {
            Some(range) => self.inner.gen_range(range.0..range.1),
            None => self.inner.gen(),
        }
    }

    #[pyo3(text_signature = "(self)")]
    #[args(probability = "0.5")]
    pub fn rand_bool(&mut self, probability: f64) -> bool {
        self.inner.gen_bool(probability)
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => serde_pickle::from_slice(s.as_bytes())
                .map(|inner| {
                    self.inner = inner;
                })
                .map_err(|error| PyValueError::new_err(error.to_string())),
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serde_pickle::to_vec(&(&self.inner), true).unwrap()).to_object(py))
    }
}
