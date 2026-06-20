//! Newton iteration for the backward Euler residual `G(y) = y − y_n − h·f(t₁,y)`.

use crate::backend::{Backend, CpuBackend};
use crate::error::{Result, SolveError};
use crate::linsolve::{bicgstab, lu_solve, BiCgStabWork};
use crate::problem::{DenseJacobian, OdeProblem};

/// Tuning knobs for the Newton solver.
#[derive(Clone, Debug)]
pub struct NewtonParams {
    /// Convergence tolerance on `‖G(y)‖`. Default: `1e-8`.
    pub tol: f64,
    /// Maximum Newton iterations. Default: `50`.
    pub max_iter: usize,
    /// Relative tolerance passed to the inner BiCGStab. Default: `1e-3`
    /// (inexact-Newton forcing term).
    pub lin_tol: f64,
    /// Maximum BiCGStab iterations. Default: `200`.
    pub lin_max_iter: usize,
}

impl Default for NewtonParams {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 50,
            lin_tol: 1e-3,
            lin_max_iter: 200,
        }
    }
}

/// Preallocated vectors for one Newton solve. Create once per problem size.
pub struct NewtonWorkspace<B: Backend> {
    pub fy: B::Vector,
    pub g: B::Vector,
    pub delta: B::Vector,
    pub scratch: B::Vector,
    pub lin: BiCgStabWork<B>,
}

impl<B: Backend> NewtonWorkspace<B> {
    pub fn new(backend: &B, n: usize) -> Self {
        Self {
            fy: backend.zeros(n),
            g: backend.zeros(n),
            delta: backend.zeros(n),
            scratch: backend.zeros(n),
            lin: BiCgStabWork::new(backend, n),
        }
    }
}

/// Solve one backward Euler step: find `y` such that `G(y) = y − y_n − h·f(t₁,y) = 0`.
///
/// On entry `y` holds the initial Newton guess (typically `y_n`); on success it
/// holds the converged solution. Returns the number of Newton iterations.
#[allow(clippy::too_many_arguments)]
pub fn newton_be<B, P>(
    backend: &B,
    problem: &P,
    t1: f64,
    h: f64,
    y_n: &B::Vector,
    y: &mut B::Vector,
    ws: &mut NewtonWorkspace<B>,
    params: &NewtonParams,
) -> Result<usize>
where
    B: Backend,
    P: OdeProblem<B>,
{
    for k in 0..params.max_iter {
        problem.eval(backend, t1, y, &mut ws.fy); // fy = f(t₁, y)

        // g = y − y_n − h·fy
        backend.copy(&mut ws.g, y);
        backend.axpy(-1.0, y_n, &mut ws.g);
        backend.axpy(-h, &ws.fy, &mut ws.g);

        let gnorm = backend.norm2(&ws.g);
        if !gnorm.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if gnorm <= params.tol {
            return Ok(k);
        }

        backend.scale(&mut ws.g, -1.0); // b = −g
        backend.fill(&mut ws.delta, 0.0); // initial guess 0

        // Disjoint-field borrow: closure gets fy/scratch, bicgstab gets g/delta/lin.
        let NewtonWorkspace {
            fy,
            g,
            delta,
            scratch,
            lin,
        } = &mut *ws;
        let y_ref: &B::Vector = y;
        let apply = |v: &B::Vector, out: &mut B::Vector| {
            // out = J_f·v  (finite-difference JVP)
            problem.jac_vec(backend, t1, y_ref, v, fy, out, scratch);
            // out = v − h·(J_f·v)  =  (I − h·J_f)·v
            backend.axpby(1.0, v, -h, out);
        };
        bicgstab(
            backend,
            apply,
            g,
            delta,
            lin,
            params.lin_tol,
            params.lin_max_iter,
        )?;

        backend.axpy(1.0, delta, y); // y += delta
    }

    // Final residual for the error message.
    problem.eval(backend, t1, y, &mut ws.fy);
    backend.copy(&mut ws.g, y);
    backend.axpy(-1.0, y_n, &mut ws.g);
    backend.axpy(-h, &ws.fy, &mut ws.g);
    Err(SolveError::NewtonDiverged {
        iters: params.max_iter,
        residual: backend.norm2(&ws.g),
    })
}

// ── Dense-Jacobian path (CPU only, small dim) ────────────────────────────────

/// Workspace for the dense (LU) Newton path.
pub struct DenseNewtonWorkspace {
    /// Row-major `n×n` Jacobian buffer (overwritten by LU each iteration).
    pub jac: Vec<f64>,
    /// RHS / solution buffer (`−G` on entry, `delta` on exit).
    pub rhs: Vec<f64>,
    /// `f(t, y)` buffer.
    pub fy: Vec<f64>,
    pub n: usize,
}

impl DenseNewtonWorkspace {
    pub fn new(n: usize) -> Self {
        Self {
            jac: vec![0.0; n * n],
            rhs: vec![0.0; n],
            fy: vec![0.0; n],
            n,
        }
    }
}

/// Newton iteration using a dense Jacobian and LU factorisation.
///
/// `y_n` and `y` are host (`Vec<f64>`) slices — this path is CPU-only and
/// intended for `dim ≲ 500`. Returns the number of Newton iterations.
pub fn newton_be_dense<P>(
    problem: &P,
    t1: f64,
    h: f64,
    y_n: &[f64],
    y: &mut Vec<f64>,
    ws: &mut DenseNewtonWorkspace,
    params: &NewtonParams,
) -> Result<usize>
where
    P: DenseJacobian<CpuBackend>,
{
    let backend = CpuBackend;
    let n = ws.n;

    for k in 0..params.max_iter {
        problem.eval(&backend, t1, y, &mut ws.fy);

        // Build G and compute its norm.
        let mut gnorm2 = 0.0f64;
        for i in 0..n {
            let g = y[i] - y_n[i] - h * ws.fy[i];
            ws.rhs[i] = -g;
            gnorm2 += g * g;
        }
        let gnorm = gnorm2.sqrt();
        if !gnorm.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if gnorm <= params.tol {
            return Ok(k);
        }

        // Build M = I − h·J.
        problem.jacobian(t1, y, &mut ws.jac);
        for i in 0..n {
            for j in 0..n {
                let jij = ws.jac[i * n + j];
                ws.jac[i * n + j] = if i == j { 1.0 - h * jij } else { -h * jij };
            }
        }

        // Solve M·delta = −G  (rhs holds −G; overwritten with delta).
        lu_solve(&mut ws.jac, n, &mut ws.rhs)?;

        for (yi, &di) in y.iter_mut().zip(ws.rhs.iter()) {
            *yi += di;
        }
    }

    // Final residual.
    problem.eval(&backend, t1, y, &mut ws.fy);
    let mut gnorm2 = 0.0f64;
    for i in 0..n {
        let g = y[i] - y_n[i] - h * ws.fy[i];
        gnorm2 += g * g;
    }
    Err(SolveError::NewtonDiverged {
        iters: params.max_iter,
        residual: gnorm2.sqrt(),
    })
}
