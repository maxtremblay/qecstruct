use rand::SeedableRng;
use rand_xoshiro::Xoshiro512StarStar;

pub type RandomNumberGenerator = Xoshiro512StarStar;

pub fn get_rng_with_seed(seed: Option<u64>) -> RandomNumberGenerator {
    match seed {
        Some(seed) => Xoshiro512StarStar::seed_from_u64(seed),
        None => Xoshiro512StarStar::from_entropy(),
    }
}
