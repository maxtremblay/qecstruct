use crate::{PyBinaryVector, PyLinearCode};
use ldpc::classical::decoders::FlipDecoder;
use ldpc::classical::LinearCode;
use pyo3::prelude::*;

#[pyclass(name = "FlipDecoder", module="qecstruct")]
pub struct PyFlipDecoder {
    pub(crate) inner: FlipDecoder<LinearCode>,
    tag: String,
}

#[pymethods]
impl PyFlipDecoder {
    #[new]
    #[args(tag = "String::from(\"FLIP\")")]
    pub fn new(code: &PyLinearCode, tag: String) -> PyFlipDecoder {
        PyFlipDecoder {
            inner: FlipDecoder::new(code.inner.clone()),
            tag,
        }
    }

    pub fn decode(&self, message: &PyBinaryVector) -> PyResult<PyBinaryVector> {
        Ok(self.inner.decode(&message.inner).into())
    }

    pub fn tag(&self) -> &str {
        self.tag.as_str()
    }

    pub fn to_json(&self) -> String {
        String::from("Flip decoder")
    }
}
