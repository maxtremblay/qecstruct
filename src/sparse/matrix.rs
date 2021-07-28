use crate::sparse::PyBinaryVector;
use bincode::{deserialize, serialize};
use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use pyo3::{PyIterProtocol, PyNumberProtocol, PyObjectProtocol};
use sparse_bin_mat::SparseBinMat;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A sparse binary matrix with efficient row access.
///
/// Parameters
/// ----------
/// num_columns : Int
///     The number of columns in the matrix.
///     Must be a non-negative integer.
/// rows : Seq[Seq[Int]]
///     Each sequence in the outer sequence represents a row of the matrix.
///     The inner sequences contain the positions (columns) of entries with value 1
///     in the corresponding row.
///
/// Example
/// -------
///     >>> from pyqec.sparse import BinaryMatrix, to_dense
///     >>> matrix = BinaryMatrix(3, [[0, 2], [1], [0, 1]])
///     >>> to_dense(matrix)
///     array([[1, 0, 1],
///            [0, 1, 0],
///            [1, 1, 0]], dtype=int32)
///
/// Raises
/// ------
/// If the number of columns is negative or
/// if a position in a row is out of bound.
#[pyclass(name = "BinaryMatrix", module = "qecstruct")]
#[derive(Debug, Clone)]
pub struct PyBinaryMatrix {
    pub(crate) inner: SparseBinMat,
}

impl From<SparseBinMat> for PyBinaryMatrix {
    fn from(inner: SparseBinMat) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBinaryMatrix {
    #[new]
    #[args(num_columns = "0", rows = "Vec::new()")]
    pub fn new(num_columns: usize, rows: Vec<Vec<usize>>) -> PyResult<Self> {
        let matrix = SparseBinMat::try_new(num_columns, rows)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self::from(matrix))
    }

    /// An identity matrix of the given length.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix, to_dense
    ///     >>> matrix = BinaryMatrix.identity(3)
    ///     >>> to_dense(matrix)
    ///     array([[1, 0, 0],
    ///            [0, 1, 0],
    ///            [0, 0, 1]], dtype=int32)
    #[staticmethod]
    #[pyo3(text_signature = "(length)")]
    pub fn identity(length: usize) -> Self {
        Self::from(SparseBinMat::identity(length))
    }

    /// A matrix filled with zeros.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix, to_dense
    ///     >>> matrix = BinaryMatrix.zeros(2, 3)
    ///     >>> to_dense(matrix)
    ///     array([[0, 0, 0],
    ///            [0, 0, 0]], dtype=int32)
    #[staticmethod]
    #[pyo3(text_signature = "(num_rows, num_columns)")]
    pub fn zeros(num_rows: usize, num_columns: usize) -> Self {
        Self::from(SparseBinMat::zeros(num_rows, num_columns))
    }

    /// An empty matrix.
    ///
    /// Mostly useful as a placeholder since
    /// it allocate a minimal amount of memory.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix.empty()
    ///     >>> matrix.num_rows()
    ///     0
    ///     >>> matrix.num_columns()
    ///     0
    #[staticmethod]
    pub fn empty() -> Self {
        Self::from(SparseBinMat::empty())
    }

    /// Returns the number of columns in the matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn num_columns(&self) -> usize {
        self.inner.number_of_columns()
    }

    /// Returns the number of rows in the matrix.
    #[pyo3(text_signature = "(self)")]
    pub fn num_rows(&self) -> usize {
        self.inner.number_of_rows()
    }

    /// Returns a tuple of the numbers of rows and columns.  
    #[pyo3(text_signature = "(self)")]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.dimension()
    }

    /// Returns the number of elements with value 0.
    #[pyo3(text_signature = "(self)")]
    pub fn num_zeros(&self) -> usize {
        self.inner.number_of_zeros()
    }

    /// Returns the number of elements with value 1.
    #[pyo3(text_signature = "(self)")]
    pub fn num_ones(&self) -> usize {
        self.inner.number_of_ones()
    }

    /// Checks if the matrix has shape (0, 0).
    #[pyo3(text_signature = "(self)")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Checks if all the elements have value 0.
    #[pyo3(text_signature = "(self)")]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Computes the number of linearly independent rows (or columns)
    /// of the matrix.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 3], [0, 2], [0, 2, 3]])
    ///     >>> matrix.rank()
    ///     3
    #[pyo3(text_signature = "(self)")]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns the transpose of the matrix.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(3, [[0, 2], [1], [0, 1]])
    ///     >>> matrix.transpose()
    ///     [0, 2]
    ///     [1, 2]
    ///     [0]
    #[pyo3(text_signature = "(self)")]
    pub fn transposed(&self) -> Self {
        self.inner.transposed().into()
    }

    /// Performs Gaussian elimination to return
    /// the matrix in echelon form.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 3], [0, 2], [0, 2, 3]])
    ///     >>> matrix.echelon_form()
    ///     [0, 1, 2]
    ///     [1, 3]
    ///     [3]
    #[pyo3(text_signature = "(self)")]
    pub fn echelon_form(&self) -> Self {
        self.inner.echelon_form().into()
    }

    /// Returns an orthogonal matrix where the rows
    /// generate the nullspace of self.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3]])
    ///     >>> matrix.nullspace()
    ///     [1, 2]
    ///     [0, 1, 3]
    #[pyo3(text_signature = "(self)")]
    pub fn nullspace(&self) -> Self {
        self.inner.nullspace().into()
    }

    /// Check if the given element has value 0.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   The row or column is out of bound.
    #[pyo3(text_signature = "(self, row, column)")]
    pub fn is_zero_at(&self, row: usize, column: usize) -> PyResult<bool> {
        self.inner.is_zero_at(row, column).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "row {} or column {} is out of bound for {} x {} matrix",
                row,
                column,
                self.num_rows(),
                self.num_columns()
            ))
        })
    }

    /// Check if the given element has value 1.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   The row or column is out of bound.
    #[pyo3(text_signature = "(self, row, column)")]
    pub fn is_one_at(&self, row: usize, column: usize) -> PyResult<bool> {
        self.inner.is_one_at(row, column).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "row {} or column {} is out of bound for {} x {} matrix",
                row,
                column,
                self.num_rows(),
                self.num_columns()
            ))
        })
    }

    /// Returns the horizontal concatenation of self and other matrix.
    ///
    /// If the matrices have a different number of rows,
    /// the smallest one is padded with zeros.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3]])
    ///     >>> matrix.horizontal_concat_with(BinaryMatrix.identity(3))
    ///     [0, 1, 2, 4]
    ///     [1, 2, 3, 5]
    ///     [6]
    #[pyo3(text_signature = "(self, other)")]
    pub fn horizontal_concat_with(&self, other: &Self) -> Self {
        self.inner.horizontal_concat_with(&other.inner).into()
    }

    /// Returns the vertical concatenation of self and other matrix.
    ///
    /// If the matrices have a different number of columns,
    /// the smallest one is padded with zeros.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3]])
    ///     >>> matrix.vertical_concat_with(BinaryMatrix.identity(3))
    ///     [0, 1, 2]
    ///     [1, 2, 3]
    ///     [0]
    ///     [1]
    ///     [2]
    #[pyo3(text_signature = "(self, other)")]
    pub fn vertical_concat_with(&self, other: &Self) -> Self {
        self.inner.vertical_concat_with(&other.inner).into()
    }

    /// Returns the dot product between self and the given vector.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix, BinaryVector
    ///     >>> matrix = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3], [1, 2]])
    ///     >>> vector = BinaryVector(4, [1, 3])
    ///     >>> matrix.dot_with_vector(vector)
    ///     [0, 2]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     The vector length is not the same as the matrix number of columns.
    #[pyo3(text_signature = "(self, vector)")]
    pub fn dot_with_vector(&self, vector: &PyBinaryVector) -> PyResult<PyBinaryVector> {
        self.inner
            .dot_with_vector(&vector.inner)
            .map(|result| result.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Returns the dot product between self and the given matrix.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix, BinaryVector
    ///     >>> matrix1 = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3], [1, 2]])
    ///     >>> matrix2 = BinaryMatrix(3, [[0, 1], [1, 2], [0, 1], [1, 2])
    ///     >>> matrix.dot_with_matrix(matrix2)
    ///     [1, 2]
    ///     [0, 1]
    ///     [0, 2]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Number of columns of self is not the same as the number of rows of the other matrix.
    #[pyo3(text_signature = "(self, matrix)")]
    pub fn dot_with_matrix(&self, matrix: &PyBinaryMatrix) -> PyResult<PyBinaryMatrix> {
        self.inner
            .dot_with_matrix(&matrix.inner)
            .map(|result| result.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Returns the bitwise xor sum of self and other.
    ///
    /// Example
    /// -------
    ///     >>> from pyqec.sparse import BinaryMatrix, BinaryVector
    ///     >>> matrix1 = BinaryMatrix(4, [[0, 1, 2], [1, 2, 3], [1, 2]])
    ///     >>> matrix2 = BinaryMatrix(4, [[0, 1], [1, 2], [0, 1, 3]])
    ///     >>> matrix.bitwise_xor(matrix2)
    ///     [2]
    ///     [3]
    ///     [0, 2, 3]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     The shapes of the matrices are different.
    #[pyo3(text_signature = "(self, other)")]
    pub fn bitwise_xor(&self, other: &PyBinaryMatrix) -> PyResult<PyBinaryMatrix> {
        self.inner
            .bitwise_xor_with(&other.inner)
            .map(|result| result.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Returns the element at the given row and column.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   The row or column is out of bound.
    #[pyo3(text_signature = "(self, row, column)")]
    pub fn element(&self, row: usize, column: usize) -> PyResult<u8> {
        self.inner.get(row, column).map(|el| el.into()).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "invalid indices {:?} for {} x {} matrix",
                (row, column),
                self.num_rows(),
                self.num_columns()
            ))
        })
    }

    /// Returns the given row as a BinaryVector.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   The row is out of bound.
    #[pyo3(text_signature = "(self, row)")]
    pub fn row(&self, row: usize) -> PyResult<PyBinaryVector> {
        self.inner
            .row(row)
            .map(|row| row.to_owned().into())
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "invalid row {} for {} x {} matrix",
                    row,
                    self.num_rows(),
                    self.num_columns()
                ))
            })
    }

    // Returns an iterator throught all rows.
    //
    // Example
    // -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(3, [[0, 2], [1], [0, 1]])
    ///     >>> for row in matrix.rows():
    ///     ...    print(row)
    ///     [0, 2]
    ///     [1]
    ///     [0, 1]
    #[pyo3(text_signature = "(self)")]
    pub fn rows(&self) -> PyRows {
        PyRows {
            matrix: self.clone(),
            row_index: 0,
        }
    }

    // Returns an iterator throught all elements with value 1.
    //
    // Example
    // -------
    ///     >>> from pyqec.sparse import BinaryMatrix
    ///     >>> matrix = BinaryMatrix(3, [[0, 2], [1], [0, 1]])
    ///     >>> for elem in matrix.non_trivial_elements():
    ///     ...    print(elem)
    ///     (0, 0)
    ///     (0, 2)
    ///     (1, 1)
    ///     (2, 0)
    ///     (2, 1)
    #[pyo3(text_signature = "(self)")]
    pub fn non_trivial_elements(&self) -> PyElements {
        PyElements {
            matrix: self.clone(),
            row_index: 0,
            column_index: 0,
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }
}

#[pyproto]
impl PyObjectProtocol for PyBinaryMatrix {
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
impl PyNumberProtocol for PyBinaryMatrix {
    fn __add__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<Self> {
        lhs.inner
            .bitwise_xor_with(&rhs.inner)
            .map(|matrix| matrix.into())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyclass]
pub struct PyRows {
    matrix: PyBinaryMatrix,
    row_index: usize,
}

#[pyproto]
impl PyIterProtocol for PyRows {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyBinaryVector> {
        let row = slf
            .matrix
            .inner
            .row(slf.row_index)
            .map(|row| row.to_owned().into());
        slf.row_index += 1;
        row
    }
}

#[pyclass]
pub struct PyElements {
    matrix: PyBinaryMatrix,
    row_index: usize,
    column_index: usize,
}

impl PyElements {
    fn next_element(&mut self) -> Option<(usize, usize)> {
        self.matrix
            .inner
            .row(self.row_index)
            .and_then(|row| row.as_slice().get(self.column_index).cloned())
            .map(|column| (self.row_index, column))
    }

    fn move_to_next_row(&mut self) {
        self.row_index += 1;
        self.column_index = 0;
    }

    fn move_to_next_column(&mut self) {
        self.column_index += 1;
    }

    fn is_done(&self) -> bool {
        self.row_index >= self.matrix.num_rows()
    }
}

#[pyproto]
impl PyIterProtocol for PyElements {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<(usize, usize)> {
        while !slf.is_done() {
            match slf.next_element() {
                Some(element) => {
                    slf.move_to_next_column();
                    return Some(element);
                }
                None => {
                    slf.move_to_next_row();
                }
            }
        }
        None
    }
}
