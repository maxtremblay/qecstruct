use pyo3::prelude::*;
use rand::SeedableRng;
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
        let other = Self { inner: self.inner.clone() };
        self.inner.jump();
        other
    }

    #[pyo3(text_signature = "(self)")]
    pub fn long_jump(&mut self) -> Self {
        let other = Self { inner: self.inner.clone() };
        self.inner.long_jump();
        other
    }
}
