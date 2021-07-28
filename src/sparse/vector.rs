use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use pyo3::{PyIterProtocol, PyNumberProtocol, PyObjectProtocol, PySequenceProtocol};
use sparse_bin_mat::SparseBinVec;
use super::PyBinaryMatrix;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A sparse binary vector.
///
/// Parameters
/// ----------
/// length : Int
///     The number of elements in the vector.
///     Must be a non-negative integer.
/// positions : Seq[Int]
///     The positions of entries with value 1.
///
/// Example
/// -------
///     >>> from pyqec.sparse import BinaryVector, to_dense
///     >>> vector = BinaryVector(3, [0, 2])
///     >>> to_dense(vector)
///     array([1, 0, 1], dtype=int32)
///
/// Raises
/// ------
/// ValueError
///     If the length is negative or if a position is out of bound.
#[pyclass(name = "BinaryVector", module="pyqec.pyqec")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PyBinaryVector {
    pub(crate) inner: SparseBinVec,
}

impl From<SparseBinVec> for PyBinaryVector {
    fn from(vector: SparseBinVec) -> Self {
        Self { inner: vector }
    }
}

#[pymethods]
impl PyBinaryVector {
    #[new]
    #[args(length = "0", non_trivial_positions = "Vec::new()")]
    fn new(length: usize, non_trivial_positions: Vec<usize>) -> PyResult<Self> {
        SparseBinVec::try_new(length, non_trivial_positions)
            .map(|vector| Self::from(vector))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Constructs a vector of the given length filled with zeros.
    #[staticmethod]
    #[pyo3(text_signature = "(length)")]
    pub fn zeros(length: usize) -> Self {
        Self::from(SparseBinVec::zeros(length))
    }

    /// Constructs a vector of length 0.
    ///
    /// This is useful as a placeholder since it allocates a minimal
    /// amount of memory.
    #[staticmethod]
    #[pyo3(text_signature = "()")]
    pub fn empty() -> Self {
        Self::from(SparseBinVec::empty())
    }

    /// Returns the number of elements in the vector.
    #[pyo3(text_signature = "(self)")]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of elements with value 1.
    #[pyo3(text_signature = "(self)")]
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }
    
    /// Checks if the length of the vector is 0.
    #[pyo3(text_signature = "(self)")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Checks if all elements are 0.
    #[pyo3(text_signature = "(self)")]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Checks if the element at the given position is zero.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     The position is out of bound.
    #[pyo3(text_signature = "(self, position)")]
    pub fn is_zero_at(&self, position: usize) -> PyResult<bool> {
        self.inner.is_zero_at(position).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "invalid position {} for vector of length {}",
                position,
                self.inner.len()
            ))
        })
    }

    /// Checks if the element at the given position is one.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     The position is out of bound.
    #[pyo3(text_signature = "(self, position)")]
    pub fn is_one_at(&self, index: usize) -> PyResult<bool> {
        self.inner.is_one_at(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "invalid position {} for vector of length {}",
                index,
                self.inner.len()
            ))
        })
    }

    /// Returns the value of the element at the given position.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     The position is out of bound.
    #[pyo3(text_signature = "(self, position)")]
    pub fn element(&self, position: usize) -> PyResult<u8> {
        self.inner.get(position).map(|el| el.into()).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "invalid position {} for vector of length {}",
                position,
                self.len()
            ))
        })
    }

    /// Index the given value in the list of non-trivial positions.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryVector
    ///     >>> vector = BinaryVector(5, [0, 2, 4])
    ///     >>> vector.non_trivial_position(0)
    ///     0
    ///     >>> vector.non_trivial_position(1)
    ///     2
    ///     >>> vector.non_trivial_position(2)
    ///     4
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     The index is greater or equal to the weight.
    #[pyo3(text_signature = "(self, index)")]
    pub fn non_trivial_position(&self, index: usize) -> PyResult<usize> {
        self.non_trivial_positions()
            .get(index)
            .cloned()
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "invalid index {} for vector of weight {}",
                    index,
                    self.weight(),
                ))
            })
    }

    /// Concatenates self with other.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryVector
    ///     >>> left = BinaryVector(5, [0, 2, 4])
    ///     >>> right = BinaryVector(4, [1, 3])
    ///     >>> left.concat(right)
    ///     [0, 2, 4, 6, 8]
    #[pyo3(text_signature = "(self, other)")]
    pub fn concat(&self, other: PyRef<Self>) -> Self {
        self.inner.concat(&other.inner).into()
    }

    /// Computes the dot product between two vectors.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryVector
    ///     >>> left = BinaryVector(5, [0, 2, 4])
    ///     >>> right = BinaryVector(5, [1, 3])
    ///     >>> left.dot_with_vector(right)
    ///     0
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the vectors have different length.
    #[pyo3(text_signature = "(self, other)")]
    pub fn dot_with_vector(&self, other: PyRef<Self>) -> PyResult<u8> {
        self.inner
            .dot_with(&other.inner)
            .map(|result| result.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Computes the dot product between self and a matrix.
    ///
    /// This assume that the vector is in row shape
    /// and compute `v * M`.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryVector, PyBinaryMatrix
    ///     >>> vector = BinaryVector(3, [0, 2])
    ///     >>> matrix = PyBinaryMatrix(4, [[0, 3]], [0, 1, 2], [1, 3]])
    ///     >>> vector.dot_with_matrix(matrix)
    ///     [0, 1]
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the vector length is the not the same as the matrix number of rows.
    #[pyo3(text_signature = "(self, matrix)")]
    pub fn dot_with_matrix(&self, other: PyRef<PyBinaryMatrix>) -> PyResult<Self> {
        other.dot_with_vector(self)
    }

    /// Computes the bitwise xor sum of two vectors.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryVector
    ///     >>> left = BinaryVector(5, [0, 2, 4])
    ///     >>> right = BinaryVector(5, [1, 2, 3])
    ///     >>> left.bitwise_xor(right)
    ///     [0, 1, 3, 4]
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the vectors have different lengths.
    #[pyo3(text_signature = "(self, other)")]
    pub fn bitwise_xor(&self, other: PyRef<Self>) -> PyResult<Self> {
        self.inner
            .bitwise_xor_with(&other.inner)
            .map(|vector| vector.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
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

impl PyBinaryVector {
    pub fn non_trivial_positions(&self) -> &[usize] {
        self.inner.as_slice()
    }
}

#[pyproto]
impl PyObjectProtocol for PyBinaryVector {
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn __richcmp__(&self, other: PyRef<Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(&self.inner == &other.inner),
            CompareOp::Ne => Ok(&self.inner != &other.inner),
            _ => Err(PyNotImplementedError::new_err("not implemented")),
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

#[pyproto]
impl PyNumberProtocol for PyBinaryVector {
    fn __add__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<Self> {
        lhs.inner
            .bitwise_xor_with(&rhs.inner)
            .map(|vector| vector.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyproto]
impl PyIterProtocol for PyBinaryVector {
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<Iter>> {
        let iter = Iter {
            vector: Py::new(slf.py(), slf.clone())?,
            index: 0,
        };
        Py::new(slf.py(), iter)
    }
}

#[pyclass]
pub struct Iter {
    vector: Py<PyBinaryVector>,
    index: usize,
}

#[pyproto]
impl PyIterProtocol for Iter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<usize> {
        let value = slf
            .vector
            .try_borrow(slf.py())
            .ok()?
            .non_trivial_positions()
            .get(slf.index)
            .cloned();
        slf.index += 1;
        value
    }
}


#[pyproto]
impl PySequenceProtocol for PyBinaryVector {
    fn __len__(&self) -> usize {
        self.len()
    }
}
