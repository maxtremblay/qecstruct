use crate::randomness::{get_rng_with_seed, RandomNumberGenerator};
use crate::sparse::PyBinaryVector;
use crate::pauli::PyPauliOperator;
use bincode::{deserialize, serialize};
use ldpc::noise_model::{DepolarizingNoise, BinarySymmetricChannel, NoiseModel, Probability};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::PyObjectProtocol;
use pyo3::ToPyObject;

/// An implementation of a binary symmetric channel.
///
/// A binary symmetric channel flips the value
/// of each bits according to a given error probability.
#[pyclass(name = "BinarySymmetricChannel", module="qecstruct")]
pub struct PyBinarySymmetricChannel {
    channel: BinarySymmetricChannel,
    probability: f64,
    rng: RandomNumberGenerator,
}

#[pymethods]
impl PyBinarySymmetricChannel {
    #[new]
    #[args(probability = "0.0", rng_seed = "None")]
    pub fn new(probability: f64, rng_seed: Option<u64>) -> PyResult<PyBinarySymmetricChannel> {
        let prob_wrapper = Probability::try_new(probability).ok_or(PyValueError::new_err(
            format!("{} is not a valid probability", probability,),
        ))?;
        let channel = BinarySymmetricChannel::with_probability(prob_wrapper);
        let rng = get_rng_with_seed(rng_seed);
        Ok(PyBinarySymmetricChannel {
            channel,
            probability,
            rng,
        })
    }

    #[pyo3(text_signature = "(self, length)")]
    fn sample_error_of_length(&mut self, length: usize) -> PyBinaryVector {
        self.channel
            .sample_error_of_length(length, &mut self.rng)
            .into()
    }

    #[pyo3(text_signature = "(self)")]
    fn error_probability(&self) -> f64 {
        self.probability
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let (channel, probability, rng) = deserialize(s.as_bytes()).unwrap();
                self.channel = channel;
                self.probability = probability;
                self.rng = rng;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(
            py,
            &serialize(&(&self.channel, &self.probability, &self.rng)).unwrap(),
        )
        .to_object(py))
    }
}

#[pyproto]
impl PyObjectProtocol for PyBinarySymmetricChannel {
    fn __repr__(&self) -> String {
        format!("BSC({})", self.error_probability())
    }
}


/// An implementation of a depolarizing noise channel.
///
/// A depolarizing noise channel apply one of {X, Y, Z}
/// with probability p and identity with probability 1 - p.
#[pyclass(name = "DepolarizingNoise", module="qecstruct")]
pub struct PyDepolarizingNoise {
    channel: DepolarizingNoise,
    probability: f64,
    rng: RandomNumberGenerator,
}

#[pymethods]
impl PyDepolarizingNoise {
    #[new]
    #[args(probability = "0.0", rng_seed = "None")]
    pub fn new(probability: f64, rng_seed: Option<u64>) -> PyResult<PyDepolarizingNoise> {
        let prob_wrapper = Probability::try_new(probability).ok_or(PyValueError::new_err(
            format!("{} is not a valid probability", probability,),
        ))?;
        let channel = DepolarizingNoise::with_probability(prob_wrapper);
        let rng = get_rng_with_seed(rng_seed);
        Ok(PyDepolarizingNoise {
            channel,
            probability,
            rng,
        })
    }

    #[pyo3(text_signature = "(self, length)")]
    fn sample_error_of_length(&mut self, length: usize) -> PyPauliOperator {
        self.channel
            .sample_error_of_length(length, &mut self.rng)
            .into()
    }

    #[pyo3(text_signature = "(self)")]
    fn error_probability(&self) -> f64 {
        self.probability
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let (channel, probability, rng) = deserialize(s.as_bytes()).unwrap();
                self.channel = channel;
                self.probability = probability;
                self.rng = rng;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(
            py,
            &serialize(&(&self.channel, &self.probability, &self.rng)).unwrap(),
        )
        .to_object(py))
    }
}

#[pyproto]
impl PyObjectProtocol for PyDepolarizingNoise {
    fn __repr__(&self) -> String {
        format!("Depolarizing({})", self.error_probability())
    }
}
