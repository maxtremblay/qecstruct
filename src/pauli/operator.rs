use super::PyPauli;
use pauli::PauliOperator;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use pyo3::{PyObjectProtocol, PySequenceProtocol};

#[pyclass(name = "PauliOperator", module="qecstruct")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pyo3(text_signature = "(length, positions = [], paulis = [])")]
pub struct PyPauliOperator {
    pub(crate) inner: PauliOperator,
}

impl From<PauliOperator> for PyPauliOperator {
    fn from(inner: PauliOperator) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPauliOperator {
    #[new]
    #[args(length, positions = "Vec::new()", paulis = "Vec::new()")]
    pub fn new(length: usize, positions: Vec<usize>, paulis: Vec<PyPauli>) -> PyResult<Self> {
        let paulis = paulis.iter().map(|pauli| pauli.inner).collect();
        match PauliOperator::try_new(length, positions, paulis) {
            Ok(inner) => Ok(Self { inner }),
            Err(error) => Err(PyValueError::new_err(error.to_string())),
        }
    }

    #[pyo3(text_signature = "(self, other)")]
    pub fn commutes_with(&self, other: &Self) -> bool {
        self.inner.commutes_with(&other.inner)
    }

    #[pyo3(text_signature = "(self, other)")]
    pub fn anticommutes_with(&self, other: &Self) -> bool {
        self.inner.anticommutes_with(&other.inner)
    }

    #[pyo3(text_signature = "(self)")]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(text_signature = "(self)")]
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    #[pyo3(text_signature = "(self)")]
    pub fn non_trivial_positions(&self) -> Vec<usize> {
        self.inner.non_trivial_positions().to_owned()
    }

    #[pyo3(text_signature = "(self, other)")]
    pub fn apply(&self, other: &Self) -> PyResult<Self> {
        match self.inner.multiply_with(&other.inner) {
            Ok(inner) => Ok(inner.into()),
            Err(error) => Err(PyValueError::new_err(error.to_string())),
        }
    }

    #[pyo3(text_signature = "(self)")]
    pub fn x_part(&self) -> Self {
        self.inner.x_part().into()
    }

    #[pyo3(text_signature = "(self)")]
    pub fn z_part(&self) -> Self {
        self.inner.z_part().into()
    }

    #[pyo3(text_signature = "(self)")]
    pub fn partition_x_and_z(&self) -> (Self, Self) {
        (self.x_part(), self.z_part())
    }

    #[pyo3(text_signature = "(self, position)")]
    pub fn get(&self, position: usize) -> PyResult<PyPauli> {
        match self.inner.get(position) {
            Some(pauli) => Ok(pauli.into()),
            None => Err(PyIndexError::new_err("position out of bound")),
        }
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
        Ok(PyBytes::new(py, &serde_pickle::to_vec(&self.inner, true).unwrap()).to_object(py))
    }
}

#[pyproto]
impl PyObjectProtocol for PyPauliOperator {
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

#[pyproto]
impl PySequenceProtocol for PyPauliOperator {
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
