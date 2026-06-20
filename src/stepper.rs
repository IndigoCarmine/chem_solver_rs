//! The top-level `BackwardEuler` stepper.

use crate::backend::Backend;
use crate::error::Result;
use crate::newton::{newton_be, NewtonParams, NewtonWorkspace};
use crate::problem::OdeProblem;

/// Backward Euler stepper with a fixed step size `h`.
pub struct BackwardEuler {
    /// Step size.
    pub h: f64,
    /// Newton solver parameters.
    pub newton: NewtonParams,
}

impl BackwardEuler {
    /// Create a stepper with step size `h` and default Newton parameters.
    pub fn new(h: f64) -> Self {
        Self {
            h,
            newton: NewtonParams::default(),
        }
    }

    /// Advance `y` by one step from `t0` to `t0 + h` in-place.
    ///
    /// `ws` is reused across calls; create it once with
    /// `NewtonWorkspace::new(backend, dim)`.
    pub fn step<B, P>(
        &self,
        backend: &B,
        problem: &P,
        t0: f64,
        y: &mut B::Vector,
        ws: &mut NewtonWorkspace<B>,
    ) -> Result<()>
    where
        B: Backend,
        P: OdeProblem<B>,
    {
        // y_n buffer: one allocation per standalone step() call (acceptable).
        let mut y_n = backend.zeros(backend.len(y));
        backend.copy(&mut y_n, y);
        newton_be(
            backend,
            problem,
            t0 + self.h,
            self.h,
            &y_n,
            y,
            ws,
            &self.newton,
        )?;
        Ok(())
    }

    /// Integrate for `n_steps` steps, calling `on_step(t, y_host)` after each.
    ///
    /// All device allocations happen before the loop; no heap allocation occurs
    /// during stepping. Returns the final state as a host `Vec<f64>`.
    pub fn integrate_with<B, P, F>(
        &self,
        backend: &B,
        problem: &P,
        t0: f64,
        y0: &[f64],
        n_steps: usize,
        mut on_step: F,
    ) -> Result<Vec<f64>>
    where
        B: Backend,
        P: OdeProblem<B>,
        F: FnMut(f64, &[f64]),
    {
        let mut y = backend.from_host(y0);
        // Pre-allocate y_n once; reused every step.
        let mut y_n = backend.zeros(y0.len());
        let mut ws = NewtonWorkspace::new(backend, y0.len());

        for i in 0..n_steps {
            let t = t0 + i as f64 * self.h;
            backend.copy(&mut y_n, &y);
            newton_be(
                backend,
                problem,
                t + self.h,
                self.h,
                &y_n,
                &mut y,
                &mut ws,
                &self.newton,
            )?;
            let y_host = backend.to_host(&y);
            on_step(t + self.h, &y_host);
        }
        Ok(backend.to_host(&y))
    }

    /// Integrate for `n_steps` steps, collecting the full trajectory.
    ///
    /// Convenience wrapper around [`integrate_with`](Self::integrate_with).
    /// Suitable for small problems; for large `N` prefer `integrate_with` to
    /// avoid storing all snapshots in memory.
    pub fn integrate<B, P>(
        &self,
        backend: &B,
        problem: &P,
        t0: f64,
        y0: &[f64],
        n_steps: usize,
    ) -> Result<Vec<(f64, Vec<f64>)>>
    where
        B: Backend,
        P: OdeProblem<B>,
    {
        let mut traj = Vec::with_capacity(n_steps + 1);
        traj.push((t0, y0.to_vec()));
        self.integrate_with(backend, problem, t0, y0, n_steps, |t, y| {
            traj.push((t, y.to_vec()));
        })?;
        Ok(traj)
    }
}
