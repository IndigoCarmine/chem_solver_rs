//! The compute backend abstraction.
//!
//! A [`Backend`] owns device-resident vectors ([`Backend::Vector`]) and exposes
//! the handful of BLAS-1 primitives the solver needs. The solver never touches
//! raw storage: every operation goes through this trait, so the same Newton /
//! BiCGStab code runs unchanged on the CPU ([`CpuBackend`], `f64`, parallelised
//! with rayon) or on the GPU ([`GpuBackend`], `f32`, via wgpu compute shaders).
//!
//! # Why only BLAS-1?
//! Matrix-free Newton–Krylov reduces every backward Euler step to vector
//! updates plus *evaluations of `f`*. The Jacobian is never assembled, so the
//! backend needs no matrix type and no linear-algebra routines beyond
//! `axpy`/`dot`/`scale`. This is what keeps memory at `O(N)` (see the crate
//! docs) and keeps backends cheap to implement.

mod cpu;
pub use cpu::CpuBackend;

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

/// GPU expression DSL types and helpers, gated on the `gpu` feature.
///
/// Use [`GpuEquation::build`] to compile an [`Expr`] closure into a WGSL
/// compute shader that fully runs on the GPU.
#[cfg(feature = "gpu")]
pub mod gpu_eq {
    pub use super::gpu::{
        abs, cos, exp, generate_eval_wgsl, ln, pow, sin, sqrt, BoundaryCondition, Expr,
        GpuEquation, StateRef,
    };
}

/// A compute backend that owns vectors and implements BLAS-1 kernels.
///
/// All scalars at the API boundary are `f64`. A backend whose device works in
/// a narrower type (e.g. the wgpu backend, which is `f32`) converts internally;
/// see [`GpuBackend`] for the precision caveat.
pub trait Backend: Sized {
    /// Device-resident vector handle. Cloning may be expensive (a device copy).
    type Vector;

    /// Allocate a zero-filled vector of length `n`.
    fn zeros(&self, n: usize) -> Self::Vector;

    /// Length of a vector.
    fn len(&self, v: &Self::Vector) -> usize;

    /// Upload host data into a fresh device vector.
    #[allow(clippy::wrong_self_convention)]
    fn from_host(&self, data: &[f64]) -> Self::Vector;

    /// Download a device vector back to host memory.
    fn to_host(&self, v: &Self::Vector) -> Vec<f64>;

    /// `dst <- src` (lengths must match).
    fn copy(&self, dst: &mut Self::Vector, src: &Self::Vector);

    /// `x <- value` for every element.
    fn fill(&self, x: &mut Self::Vector, value: f64);

    /// `x <- alpha * x`.
    fn scale(&self, x: &mut Self::Vector, alpha: f64);

    /// `y <- y + alpha * x`.
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector);

    /// `y <- alpha * x + beta * y`.
    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector);

    /// Euclidean inner product `x · y`.
    fn dot(&self, x: &Self::Vector, y: &Self::Vector) -> f64;

    /// Euclidean norm `||x||_2`. Default: `sqrt(dot(x, x))`.
    fn norm2(&self, x: &Self::Vector) -> f64 {
        self.dot(x, x).sqrt()
    }
}

/// Which backend to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU, `f64`, rayon-parallel.
    Cpu,
    /// GPU, `f32`, wgpu (requires the `gpu` feature and capable hardware).
    Gpu,
}

/// Recommend a backend for a problem of dimension `dim`.
///
/// The heuristic is deliberately conservative: per-step work in an implicit
/// solver is dominated by a few Krylov iterations, each a handful of `O(N)`
/// kernels. GPU launch + host/device transfer overhead only amortises once `N`
/// is large, so small and medium systems stay on the CPU. Tune the threshold
/// for your hardware, or override with an explicit backend.
///
/// Without the `gpu` feature this always returns [`BackendKind::Cpu`].
pub fn recommend_backend(dim: usize) -> BackendKind {
    #[cfg(feature = "gpu")]
    {
        const GPU_MIN_DIM: usize = 50_000;
        if dim >= GPU_MIN_DIM && gpu::is_available() {
            return BackendKind::Gpu;
        }
    }
    let _ = dim;
    BackendKind::Cpu
}
