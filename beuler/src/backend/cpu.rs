//! CPU backend: `Vec<f64>` storage with rayon-parallel kernels.
//!
//! Small vectors run serially — rayon's fork/join overhead dwarfs the work
//! below a few thousand elements — and switch to parallel iterators past
//! [`PAR_THRESHOLD`].

use super::Backend;
use rayon::prelude::*;

/// Element count above which kernels parallelise. Tuned roughly to where rayon
/// overhead is repaid; adjust for your core count and memory bandwidth.
pub const PAR_THRESHOLD: usize = 8_192;

/// CPU backend. Zero-sized; holds no device state.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Vector = Vec<f64>;

    #[inline]
    fn zeros(&self, n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    #[inline]
    fn len(&self, v: &Vec<f64>) -> usize {
        v.len()
    }

    #[inline]
    fn from_host(&self, data: &[f64]) -> Vec<f64> {
        data.to_vec()
    }

    #[inline]
    fn to_host(&self, v: &Vec<f64>) -> Vec<f64> {
        v.clone()
    }

    #[inline]
    fn copy(&self, dst: &mut Vec<f64>, src: &Vec<f64>) {
        debug_assert_eq!(dst.len(), src.len());
        dst.copy_from_slice(src);
    }

    #[inline]
    fn fill(&self, x: &mut Vec<f64>, value: f64) {
        x.iter_mut().for_each(|xi| *xi = value);
    }

    #[inline]
    fn scale(&self, x: &mut Vec<f64>, alpha: f64) {
        if x.len() >= PAR_THRESHOLD {
            x.par_iter_mut().for_each(|xi| *xi *= alpha);
        } else {
            x.iter_mut().for_each(|xi| *xi *= alpha);
        }
    }

    #[inline]
    fn axpy(&self, alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        debug_assert_eq!(x.len(), y.len());
        if y.len() >= PAR_THRESHOLD {
            y.par_iter_mut()
                .zip(x.par_iter())
                .for_each(|(yi, xi)| *yi += alpha * *xi);
        } else {
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi += alpha * *xi;
            }
        }
    }

    #[inline]
    fn axpby(&self, alpha: f64, x: &Vec<f64>, beta: f64, y: &mut Vec<f64>) {
        debug_assert_eq!(x.len(), y.len());
        if y.len() >= PAR_THRESHOLD {
            y.par_iter_mut()
                .zip(x.par_iter())
                .for_each(|(yi, xi)| *yi = alpha * *xi + beta * *yi);
        } else {
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = alpha * *xi + beta * *yi;
            }
        }
    }

    #[inline]
    fn dot(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        if x.len() >= PAR_THRESHOLD {
            x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum()
        } else {
            x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        }
    }
}
