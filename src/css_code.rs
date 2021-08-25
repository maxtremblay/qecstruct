use crate::pauli::PyPauliOperator;
use crate::sparse::{PyBinaryMatrix, PyBinaryVector};
use crate::PyLinearCode;
use ldpc::quantum::CssCode;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::PyObjectProtocol;
use pyo3::PySequenceProtocol;
use pyo3::ToPyObject;

pub(crate) fn hypergraph_product(
    first_code: &PyLinearCode,
    second_code: &PyLinearCode,
) -> PyCssCode {
    PyCssCode {
        inner: CssCode::hypergraph_product(&first_code.inner, &second_code.inner),
    }
}

pub(crate) fn steane_code() -> PyCssCode {
    PyCssCode {
        inner: CssCode::steane_code(),
    }
}

pub(crate) fn shor_code() -> PyCssCode {
    PyCssCode {
        inner: CssCode::shor_code(),
    }
}
/// An implementation of quantum CSS codes optimized for LDPC codes.
///
/// A code is defined from a pair of orthogonal linear codes.
///
/// Parameters
/// ----------
/// x_code : pyqec.classical.LinearCode
///     The code for which the parity check matrix generator the X stabilizers
///     and the generator matrix generates the Z logical operators.
/// z_code : pyqec.classical.LinearCode
///     The code for which the parity check matrix generator the Z stabilizers
///     and the generator matrix generates the X logical operators.
#[pyclass(name = "CssCode", module = "qecstruct")]
#[pyo3(text_signature = "(x_code, z_code)")]
pub struct PyCssCode {
    pub(crate) inner: CssCode,
}

impl From<CssCode> for PyCssCode {
    fn from(inner: CssCode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCssCode {
    #[new]
    #[args(x_code, z_code)]
    pub fn new(
        x_code: &PyLinearCode,
        z_code: &PyLinearCode,
    ) -> PyResult<Self> {
        match CssCode::try_new(&x_code.inner, &z_code.inner) {
            Ok(inner) => Ok(Self { inner }),
            Err(error) => Err(PyValueError::new_err(error.to_string())),
        }
    }

    /// Returns the X stabilizer generators represented as a binary matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn x_stabs_binary(&self) -> PyBinaryMatrix {
        self.inner.x_stabs_binary().clone().into()
    }

    /// Returns the Z stabilizer generators represented as a binary matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn z_stabs_binary(&self) -> PyBinaryMatrix {
        self.inner.z_stabs_binary().clone().into()
    }

    /// Returns the X logical generators represented as a binary matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn x_logicals_binary(&self) -> PyBinaryMatrix {
        self.inner.x_logicals_binary().clone().into()
    }

    /// Returns the Z logical generators represented as a binary matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn z_logicals_binary(&self) -> PyBinaryMatrix {
        self.inner.z_logicals_binary().clone().into()
    }

    /// The number of qubits in the code.
    ///
    ///     >>> len(code) == code.length()
    ///     true
    #[pyo3(text_signature = "(self)")]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// The number of X stablizer generators in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_x_stabs(&self) -> usize {
        self.inner.num_z_stabs()
    }

    /// The number of Z stablizer generators in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_z_stabs(&self) -> usize {
        self.inner.num_z_stabs()
    }

    /// The number of X logical operator generators in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_x_logicals(&self) -> usize {
        self.inner.num_x_logicals()
    }

    /// The number of Z logical operator generators in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_z_logicals(&self) -> usize {
        self.inner.num_z_logicals()
    }

    /// The syndrome of a given operator.
    ///
    /// Parameters
    /// ----------
    /// operator: pyqec.quantum.PyPauliOperator
    ///     The operator.
    ///
    /// Returns
    /// -------
    /// (BinaryVector, BinaryVector)
    ///     The X and Z syndromes. The X syndrome is the syndrome
    ///     measure by the X stabilizers corresponding to the Z part
    ///     of the error.
    #[pyo3(text_signature = "(self, operator)")]
    pub fn syndrome_of(
        &self,
        operator: &PyPauliOperator,
    ) -> PyResult<(PyBinaryVector, PyBinaryVector)> {
        let syndrome = self.inner.syndrome_of(&operator.inner);
        Ok((syndrome.x.into(), syndrome.z.into()))
    }

    #[pyo3(text_signature = "(self, operator)")]
    pub fn has_logical(&self, operator: &PyPauliOperator) -> bool {
        self.inner.has_logical(&operator.inner)
    }

    #[pyo3(text_signature = "(self, operator)")]
    pub fn has_stabilizer(&self, operator: &PyPauliOperator) -> bool {
        self.inner.has_logical(&operator.inner)
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
impl PyObjectProtocol for PyCssCode {
    fn __repr__(&self) -> String {
        format!(
            "X stabilizers:\n{}Z stabilizers:\n{}",
            self.inner.x_stabs_binary(),
            self.inner.z_stabs_binary(),
        )
    }
}

#[pyproto]
impl PySequenceProtocol for PyCssCode {
    fn __len__(&self) -> usize {
        self.length()
    }
}
