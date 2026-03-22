//! State-space model definition for Local Level + Regression.
//!
//! Observation equation:  y_t = Z_t * α_t + ε_t,  ε_t ~ N(0, σ²_obs)
//! State transition:      α_t = T * α_{t-1} + R * η_t,  η_t ~ N(0, Q)
//!
//! For Local Level:
//!   α_t = [μ_t]   (scalar state: trend level)
//!   Z_t = 1       (observation loads directly on level)
//!   T   = 1       (random walk)
//!   R   = 1
//!   Q   = σ²_level

pub struct StateSpaceModel {
    x: Vec<Vec<f64>>,
}

impl StateSpaceModel {
    pub fn new(_y: Vec<f64>, x: Vec<Vec<f64>>) -> Self {
        Self { x }
    }

    #[inline]
    pub fn num_covariates(&self) -> usize {
        self.x.len()
    }

    #[inline]
    pub fn covariates(&self) -> &[Vec<f64>] {
        &self.x
    }

    #[inline]
    pub fn observe(&self, t: usize, mu: f64, beta: &[f64]) -> f64 {
        mu + self
            .x
            .iter()
            .zip(beta.iter())
            .map(|(x_col, beta_value)| x_col[t] * beta_value)
            .sum::<f64>()
    }

    #[inline]
    pub fn x_at(&self, j: usize, t: usize) -> f64 {
        self.x[j][t]
    }
}
