//! Error types for the solver stack.

use std::fmt;

/// Errors that can arise while taking a step or solving the inner systems.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// Newton iteration failed to reach `tol` within `max_iter` steps.
    NewtonDiverged { iters: usize, residual: f64 },
    /// The inner (linear) solver broke down or did not converge.
    LinearSolverFailed { iters: usize, residual: f64 },
    /// A singular matrix was encountered in the dense direct solver.
    SingularMatrix { pivot: usize },
    /// State / problem dimension disagreement.
    DimensionMismatch { expected: usize, found: usize },
    /// A non-finite value (NaN / inf) appeared during iteration.
    NonFinite,
    /// GPU backend error (only reachable with the `gpu` feature).
    Backend(String),
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::NewtonDiverged { iters, residual } => write!(
                f,
                "Newton iteration did not converge after {iters} iterations (residual = {residual:.3e})"
            ),
            SolveError::LinearSolverFailed { iters, residual } => write!(
                f,
                "linear solver failed after {iters} iterations (residual = {residual:.3e})"
            ),
            SolveError::SingularMatrix { pivot } => {
                write!(f, "singular matrix: zero pivot at column {pivot}")
            }
            SolveError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            SolveError::NonFinite => write!(f, "non-finite value encountered during iteration"),
            SolveError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for SolveError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, SolveError>;
