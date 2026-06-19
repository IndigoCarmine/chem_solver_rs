//! The user-facing problem definition.

use crate::backend::Backend;

/// A first-order system `dy/dt = f(t, y)` whose state lives on backend `B`.
///
/// Implement [`eval`](OdeProblem::eval) and you get the full matrix-free
/// solver. Optionally override [`jac_vec`](OdeProblem::jac_vec) with an analytic
/// Jacobian–vector product for speed and accuracy; the default approximates it
/// by a finite difference, which costs one extra `f` evaluation per product.
pub trait OdeProblem<B: Backend> {
    /// Number of state components.
    fn dim(&self) -> usize;

    /// Write `f(t, y)` into `out`.
    ///
    /// This is the hot path: avoid heap allocation here. Express the dynamics
    /// through `backend` operations (and, on the CPU backend, you may also read
    /// the slice directly via [`Backend::to_host`] for prototyping, at the cost
    /// of a copy).
    fn eval(&self, backend: &B, t: f64, y: &B::Vector, out: &mut B::Vector);

    /// Jacobian–vector product `out <- J_f(t, y) · v`.
    ///
    /// Default: directional finite difference
    /// `J·v ≈ (f(t, y + εv) − f(t, y)) / ε`, with `ε` scaled à la Brown–Saad
    /// from `‖y‖` and `‖v‖`. `fy` must equal `f(t, y)` (the caller already has
    /// it during a Newton step), and `scratch` is borrowed work space of length
    /// `dim` so this allocates nothing.
    #[allow(clippy::too_many_arguments)]
    fn jac_vec(
        &self,
        backend: &B,
        t: f64,
        y: &B::Vector,
        v: &B::Vector,
        fy: &B::Vector,
        out: &mut B::Vector,
        scratch: &mut B::Vector,
    ) {
        let nv = backend.norm2(v);
        if nv == 0.0 {
            backend.fill(out, 0.0);
            return;
        }
        let ny = backend.norm2(y);
        // Step that balances truncation vs. rounding error for f32/f64 storage.
        let eps = 1e-7 * (1.0 + ny) / nv;

        // scratch <- y + eps * v
        backend.copy(scratch, y);
        backend.axpy(eps, v, scratch);

        // out <- f(t, y + eps*v)
        self.eval(backend, t, scratch, out);

        // out <- (out - fy) / eps
        backend.axpy(-1.0, fy, out);
        backend.scale(out, 1.0 / eps);
    }
}

/// Opt-in: provide a dense Jacobian for the direct (LU) solve path.
///
/// Only worthwhile for small `dim` (roughly `≤ 500`), where a direct solve is
/// more robust than the iterative one. The dense path is host/`f64` oriented.
pub trait DenseJacobian<B: Backend>: OdeProblem<B> {
    /// Fill `jac` (row-major, `dim × dim`) with `J_f(t, y)`, where `y` is given
    /// as a host slice.
    fn jacobian(&self, t: f64, y: &[f64], jac: &mut [f64]);
}
