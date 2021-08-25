use pyo3::prelude::*;

mod linear_code;
use linear_code::{hamming_code, random_regular_code, repetition_code, PyLinearCode};

mod css_code;
use css_code::{hypergraph_product, shor_code, steane_code, PyCssCode};

mod noise;
use noise::PyBinarySymmetricChannel;

mod pauli;
use crate::pauli::{PyPauli, PyPauliOperator};

mod randomness;

mod sparse;
use sparse::{PyBinaryMatrix, PyBinaryVector};

/// Sparse data structure for classical and quantum error correction.
#[pymodule]
fn qecstruct(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyLinearCode>()?;
    module.add_class::<PyBinarySymmetricChannel>()?;
    module.add_class::<PyBinaryMatrix>()?;
    module.add_class::<PyBinaryVector>()?;
    module.add_class::<PyPauli>()?;
    module.add_class::<PyPauliOperator>()?;
    module.add_class::<PyCssCode>()?;

    /// Samples a random regular codes.
    ///
    /// Parameters
    /// ----------
    /// num_bits: int
    ///     The number of bits in the code.
    /// num_checks: int, default = 3
    ///     The number of checks in the code.
    /// bit_degree: int
    ///     The number of checks connected to each bit.
    /// check_degree: int
    ///     The number of bits connected to each check.
    /// random_seed: Optional[int]
    ///     A seed to feed the random number generator.
    ///     By default, the rng is initialize from entropy.
    ///
    /// Returns
    /// -------
    /// LinearCode
    ///     A random linear code with the given parameters.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `block_size * bit_degree != number_of_checks * check_degree`.
    #[pyfn(module)]
    #[pyo3(
        name = "random_regular_code",
        text_signature = "(num_bits=4, num_checks=3, bit_degree=3, check_degree=4, random_seed=None)"
    )]
    fn py_random_regular_code(
        num_bits: usize,
        num_checks: usize,
        bit_degree: usize,
        check_degree: usize,
        random_seed: Option<u64>,
    ) -> PyResult<PyLinearCode> {
        random_regular_code(num_bits, num_checks, bit_degree, check_degree, random_seed)
    }

    /// Returns an instance of the Hamming code.
    #[pyfn(module)]
    #[pyo3(name = "hamming_code", text_signature = "")]
    pub fn py_hamming_code() -> PyLinearCode {
        hamming_code()
    }

    /// Returns an instance of the repetition code.
    ///
    /// Arguments
    /// ---------
    /// length : Int
    ///     The number of bits.
    #[pyfn(module)]
    #[pyo3(name = "repetition_code", text_signature = "(length)")]
    pub fn py_repetition_code(length: usize) -> PyLinearCode {
        repetition_code(length)
    }

    /// Returns an instance of the Steane code.
    ///
    #[pyfn(module)]
    #[pyo3(name = "steane_code", text_signature = "")]
    pub fn py_steane_code() -> PyCssCode {
        steane_code()
    }

    /// Returns an instance of the 9-qubit Shor code.
    #[pyfn(module)]
    #[pyo3(name = "shor_code", text_signature = "")]
    pub fn py_shor_code() -> PyCssCode {
        shor_code()
    }

    /// Returns the hypergraph product of two linear codes.
    ///
    /// Arguments
    /// ---------
    /// first_code : pyqec.classical.LinearCode
    /// second_code : pyqec.classical.LinearCode
    #[pyfn(module)]
    #[pyo3(name = "hypergraph_product", text_signature = "")]
    pub fn py_hypergraph_product(
        first_code: &PyLinearCode,
        second_code: &PyLinearCode,
    ) -> PyCssCode {
        hypergraph_product(first_code, second_code)
    }

    Ok(())
}
