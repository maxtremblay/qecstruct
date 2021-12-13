#![allow(non_snake_case)]

use pauli::{Pauli, I, X, Y, Z};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::PyObjectProtocol;
use pyo3::ToPyObject;

#[pyclass(name = "Pauli")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyPauli {
    pub(crate) inner: Pauli,
}

impl From<Pauli> for PyPauli {
    fn from(inner: Pauli) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPauli {
    /// The identity single-qubit Pauli operator.
    #[staticmethod]
    #[pyo3(text_signature = "")]
    pub fn I() -> Self {
        Self { inner: I }
    }

    /// The X single-qubit Pauli operator.
    #[staticmethod]
    #[pyo3(text_signature = "")]
    pub fn X() -> Self {
        Self { inner: X }
    }

    /// The Y single-qubit Pauli operator.
    #[staticmethod]
    #[pyo3(text_signature = "")]
    pub fn Y() -> Self {
        Self { inner: Y }
    }

    /// The Z single-qubit Pauli operator.
    #[staticmethod]
    #[pyo3(text_signature = "")]
    pub fn Z() -> Self {
        Self { inner: Z }
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

#[pyproto]
impl PyObjectProtocol for PyPauli {
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn __richcmp__(&self, other: PyRef<Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            _ => Err(PyNotImplementedError::new_err("comparison not implemented")),
        }
    }
}
