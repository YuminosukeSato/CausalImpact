//! Sampling from conjugate distributions used in the Gibbs sampler.

use rand::Rng;
use rand_distr::{Gamma, StandardNormal};

pub fn sample_inv_gamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let gamma = Gamma::new(shape, 1.0 / scale).expect("Invalid Gamma parameters");
    let x: f64 = rng.sample(gamma);
    1.0 / x
}

pub fn sample_normal<R: Rng>(rng: &mut R, mean: f64, variance: f64) -> f64 {
    let std = variance.sqrt();
    let z: f64 = rng.sample(StandardNormal);
    mean + std * z
}

pub fn sample_mvnormal<R: Rng>(rng: &mut R, mean: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let lower_cholesky = cholesky(cov);
    let z: Vec<f64> = mean.iter().map(|_| rng.sample(StandardNormal)).collect();

    let mut result = mean.to_vec();
    for (i, value) in result.iter_mut().enumerate() {
        *value += lower_cholesky[i]
            .iter()
            .zip(z.iter())
            .take(i + 1)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
    }
    result
}

fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a.len();
    let mut lower_cholesky = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let sum = lower_cholesky[i]
                .iter()
                .zip(lower_cholesky[j].iter())
                .take(j)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
            if i == j {
                let diagonal = a[i][i] - sum;
                lower_cholesky[i][j] = if diagonal > 0.0 {
                    diagonal.sqrt()
                } else {
                    1e-12
                };
            } else {
                lower_cholesky[i][j] = (a[i][j] - sum) / lower_cholesky[j][j];
            }
        }
    }
    lower_cholesky
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_inv_gamma_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x = sample_inv_gamma(&mut rng, 2.0, 1.0);
            assert!(x > 0.0, "InvGamma sample must be positive");
        }
    }

    #[test]
    fn test_normal_mean_convergence() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10_000;
        let mean = 5.0;
        let var = 2.0;
        let samples: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, mean, var)).collect();
        let sample_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!(
            (sample_mean - mean).abs() < 0.1,
            "Mean should converge to {mean}"
        );
    }

    #[test]
    fn test_mvnormal_dimension() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = vec![1.0, 2.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sample = sample_mvnormal(&mut rng, &mean, &cov);
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky(&a);
        assert!((l[0][0] - 1.0).abs() < 1e-12);
        assert!((l[1][1] - 1.0).abs() < 1e-12);
        assert!((l[1][0]).abs() < 1e-12);
    }
}
