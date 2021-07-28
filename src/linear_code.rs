use crate::randomness::get_rng_with_seed;
use crate::sparse::{PyBinaryMatrix, PyBinaryVector};
use ldpc::classical::LinearCode;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::PyObjectProtocol;
use pyo3::PySequenceProtocol;
use pyo3::ToPyObject;

pub(crate) fn random_regular_code(
    num_bits: usize,
    num_checks: usize,
    bit_degree: usize,
    check_degree: usize,
    random_seed: Option<u64>,
    tag: Option<String>,
) -> PyResult<PyLinearCode> {
    let tag = tag.unwrap_or("".to_string());
    let mut rng = get_rng_with_seed(random_seed);
    LinearCode::random_regular_code()
        .num_bits(num_bits)
        .num_checks(num_checks)
        .bit_degree(bit_degree)
        .check_degree(check_degree)
        .sample_with(&mut rng)
        .map(|code| PyLinearCode { inner: code, tag })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}


pub(crate) fn hamming_code(tag: Option<String>) -> PyLinearCode {
    PyLinearCode {
        inner: LinearCode::hamming_code(),
        tag: tag.unwrap_or("".to_string()),
    }
}

pub(crate) fn repetition_code(length: usize, tag: Option<String>) -> PyLinearCode {
    PyLinearCode {
        inner: LinearCode::repetition_code(length),
        tag: tag.unwrap_or("".to_string()),
    }
}

/// An implementation of linear codes optimized for LDPC codes.
///
/// A code can be defined from either a parity check matrix `H`
/// or a generator matrix `G`.
/// These matrices need to be orthogonal, that is `H G^T = 0`.
///
/// Parameters
/// ----------
/// parity_check_matrix : Optional[pyqec.sparse.BinaryMatrix]
///     The parity check matrix of the code.
///     Most be orthogonal to the generator matrix.
///     If omited, one is computed from the generator matrix.
/// generator_matrix : Optional[pyqec.sparse.BinaryMatrix]
///     The generator matrix of the code.
///     Most be orthogonal to the parity check matrix.
///     If omited, one is computed from the parity check matrix.
/// tag : Optional[String]
///     A label for the code used to save data
///     and make automatic legend in plots.
///     If omited, the empty string is used a default tag.
///
/// Example
/// -------
/// This example shows 2 ways to define the Hamming code.
/// They both required you import the following.
///
///     >>> from pyqec.sparse import BinaryMatrix
///     >>> from pyqec.classical import LinearCode
///
/// You can build a linear code from a parity check matrix
///     
///     >>> matrix = BinaryMatrix(7, [[0, 1, 2, 4], [0, 1, 3, 5], [0, 2, 3, 6]])
///     >>> code_pcm = LinearCode(parity_check_matrix=pcm)
///
/// or from a generator matrix
///
///     >>> matrix = BinaryMatrix(7, [[0, 4, 5, 6], [1, 4, 5], [2, 4, 6], [3, 5, 6]])
///     >>> code_gm = LinearCode(generator_matrix=matrix)
///
/// Note
/// ----
/// Use the `==` if you want to know if 2 codes
/// have exactly the same parity check matrix, generator matrix and tags.
/// However, since there is freedom in the choice of the matrices and tag
/// for the same code, use **has_same_codespace** method
/// if you want to know if 2 codes define the same codespace even
/// if they may have different parity check matrices or generator matrices.
///
///     >>> code_pcm == code_gm
///     False
///     >>> code_pcm.has_same_codespace(code_gm)
///     True
#[pyclass(name = "LinearCode", module="qecstruct")]
#[pyo3(text_signature = "(parity_check_matrix=None, generator_matrix=None, tag=None)")]
pub struct PyLinearCode {
    pub(crate) inner: LinearCode,
    tag: String,
}

impl From<LinearCode> for PyLinearCode {
    fn from(inner: LinearCode) -> Self {
        Self {
            inner,
            tag: String::from(""),
        }
    }
}

#[pymethods]
impl PyLinearCode {
    #[new]
    #[args(parity_check_matrix = "None", generator_matrix = "None", tag = "None")]
    pub fn new(
        parity_check_matrix: Option<PyBinaryMatrix>,
        generator_matrix: Option<PyBinaryMatrix>,
        tag: Option<String>,
    ) -> PyResult<Self> {
        let tag = tag.unwrap_or("".to_string());
        match (parity_check_matrix, generator_matrix) {
            (Some(h), Some(g)) => h.dot_with_matrix(&g.transposed()).and_then(|product| {
                if product.is_zero() {
                    Ok(Self {
                        inner: LinearCode::from_parity_check_matrix(h.inner),
                        tag,
                    })
                } else {
                    Err(PyValueError::new_err("matrices are not orthogonal"))
                }
            }),
            (Some(h), None) => Ok(Self {
                inner: LinearCode::from_parity_check_matrix(h.inner),
                tag,
            }),
            (None, Some(g)) => Ok(Self {
                inner: LinearCode::from_parity_check_matrix(g.inner),
                tag,
            }),
            (None, None) => Ok(Self {
                inner: LinearCode::empty(),
                tag,
            }),
        }
    }


    /// The tag of the code.
    #[pyo3(text_signature = "(self)")]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// The parity check matrix of the code.
    #[pyo3(text_signature = "(self)")]
    pub fn parity_check_matrix(&self) -> PyBinaryMatrix {
        self.inner.parity_check_matrix().clone().into()
    }

    /// The parity check matrix of the code.
    #[pyo3(text_signature = "(self)")]
    pub fn generator_matrix(&self) -> PyBinaryMatrix {
        self.inner.generator_matrix().clone().into()
    }

    /// The number of bits in the code.
    ///
    ///     >>> len(code) == code.length()
    ///     true
    #[pyo3(text_signature = "(self)")]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// The number of encoded qubits.
    #[pyo3(text_signature = "(self)")]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The weight of the smallest non trivial codeword.
    ///
    /// Returns
    /// -------
    /// The minimal distance of the code if
    /// the dimension is at least 1 or -1
    /// if the dimension is 0.
    ///
    /// Caution
    /// -------
    /// This function execution time scale exponentially
    /// with the dimension of the code.
    /// Use at your own risk!
    #[pyo3(text_signature = "(self)")]
    pub fn minimal_distance(&self) -> i64 {
        self.inner
            .minimal_distance()
            .map(|d| d as i64)
            .unwrap_or(-1)
    }

    /// The number of checks in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_checks(&self) -> usize {
        self.inner.num_checks()
    }

    /// The number of codeword generators in the code.
    #[pyo3(text_signature = "(self)")]
    pub fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// The syndrome of a given message.
    ///
    /// Parameters
    /// ----------
    /// message: Seq[int]
    ///     The positions with value 1 in the message.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     The positions where `H y` is 1 where `H` is
    ///     the parity check matrix of the code and `y`
    ///     the input message.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a position in the message is greater or equal to the length of the code.
    #[pyo3(text_signature = "(self, message)")]
    pub fn syndrome_of(&self, message: &PyBinaryVector) -> PyResult<PyBinaryVector> {
        Ok(self.inner.syndrome_of(&message.inner).into())
    }

    /// Checks if the given message is a codeword of the code.
    ///
    /// Parameters
    /// ----------
    /// message: Seq[int]
    ///     The positions with value 1 in the message.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the message has the right length and a zero syndrome
    ///     or False otherwise.
    #[pyo3(text_signature = "(self, message)")]
    pub fn has_codeword(&self, message: &PyBinaryVector) -> bool {
        self.inner.has_codeword(&message.inner)
    }

    /// Checks if the other code defined the same codespace
    /// as this code.
    ///
    /// Parameters
    /// ----------
    /// other: LinearCode
    ///     The code to compare.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if other codewords are exactly the same
    ///     as this code codewords.
    #[pyo3(text_signature = "(self, other)")]
    pub fn has_same_codespace(&self, other: &Self) -> bool {
        self.inner.has_same_codespace(&other.inner)
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => serde_pickle::from_slice(s.as_bytes())
                .map(|(inner, tag)| {
                    self.inner = inner;
                    self.tag = tag;
                })
                .map_err(|error| PyValueError::new_err(error.to_string())),
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(
            py,
            &serde_pickle::to_vec(&(&self.inner, &self.tag), true).unwrap(),
        )
        .to_object(py))
    }
}

#[pyproto]
impl PyObjectProtocol for PyLinearCode {
    fn __repr__(&self) -> String {
        let mut display = if self.tag != "" {
            format!("Tag = {}\n", self.tag)
        } else {
            String::new()
        };
        display.push_str(&format!(
            "Parity check matrix:\n{}\nGenerator matrix:\n{}",
            self.inner.parity_check_matrix(),
            self.inner.generator_matrix(),
        ));
        display
    }
}

#[pyproto]
impl PySequenceProtocol for PyLinearCode {
    fn __len__(&self) -> usize {
        self.length()
    }
}
