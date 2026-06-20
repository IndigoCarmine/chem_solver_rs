//! # beuler — memory-efficient backward Euler ODE solver
//!
//! Solves `dy/dt = f(t, y)` with the **implicit (backward) Euler** method.
//!
//! ## Design highlights
//!
//! | Property | Detail |
//! |---|---|
//! | Memory | O(N) — no Jacobian matrix stored |
//! | Linear solver | Matrix-free BiCGStab (~6·N vectors) |
//! | Nonlinear solver | Newton–Krylov with finite-difference JVP |
//! | Allocation | Zero heap allocation inside the time loop |
//! | CPU backend | `f64`, rayon-parallel above 8 192 elements |
//! | GPU backend | `f32`, wgpu compute shaders (**experimental**) |
//!
//! ## Quick start
//!
//! ```rust
//! use beuler::{BackwardEuler, CpuBackend, OdeProblem, Backend};
//!
//! struct Decay;
//! impl OdeProblem<CpuBackend> for Decay {
//!     fn dim(&self) -> usize { 1 }
//!     fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
//!         out[0] = -y[0];
//!     }
//! }
//!
//! let stepper = BackwardEuler::new(0.1);
//! let traj = stepper.integrate(&CpuBackend, &Decay, 0.0, &[1.0], 10).unwrap();
//! let (t_end, y_end) = traj.last().unwrap();
//! println!("y({t_end}) ≈ {:.4}  (exact {:.4})", y_end[0], (-t_end).exp());
//! ```
//!
//! ## GPU note
//!
//! The GPU backend uses `f32` internally (WGSL has no `f64`). Enable it with
//! `--features gpu`. It is **experimental** and untested against real hardware
//! in this release. Prefer `CpuBackend` for production use.

pub mod backend;
pub mod error;
pub mod linsolve;
pub mod newton;
pub mod problem;
pub mod stepper;

#[cfg(feature = "gpu")]
pub use backend::gpu_eq;
#[cfg(feature = "gpu")]
pub use backend::GpuBackend;
pub use backend::{recommend_backend, Backend, BackendKind, CpuBackend};

pub use error::{Result, SolveError};
pub use newton::{DenseNewtonWorkspace, NewtonParams, NewtonWorkspace};
pub use problem::{DenseJacobian, OdeProblem};
pub use stepper::BackwardEuler;
