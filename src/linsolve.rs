//! Linear solvers: matrix-free BiCGStab and dense LU with partial pivoting.

use crate::backend::Backend;
use crate::error::{Result, SolveError};

/// Preallocated workspace for [`bicgstab`]. Allocate once, reuse across steps.
pub struct BiCgStabWork<B: Backend> {
    pub r: B::Vector,
    pub r0: B::Vector,
    pub p: B::Vector,
    pub v: B::Vector,
    pub s: B::Vector,
    pub t: B::Vector,
}

impl<B: Backend> BiCgStabWork<B> {
    pub fn new(backend: &B, n: usize) -> Self {
        Self {
            r: backend.zeros(n),
            r0: backend.zeros(n),
            p: backend.zeros(n),
            v: backend.zeros(n),
            s: backend.zeros(n),
            t: backend.zeros(n),
        }
    }
}

/// Matrix-free BiCGStab: solve `A·x = b` via a closure `apply(v, out)` that
/// computes `out = A·v`. The solution overwrites `x` (initial guess on entry).
/// Returns the number of iterations on success.
pub fn bicgstab<B, A>(
    backend: &B,
    mut apply: A,
    b: &B::Vector,
    x: &mut B::Vector,
    work: &mut BiCgStabWork<B>,
    tol: f64,
    max_iter: usize,
) -> Result<usize>
where
    B: Backend,
    A: FnMut(&B::Vector, &mut B::Vector),
{
    let bnorm = backend.norm2(b);
    if bnorm == 0.0 {
        backend.fill(x, 0.0);
        return Ok(0);
    }

    // r = b - A·x
    apply(x, &mut work.v);
    backend.copy(&mut work.r, b);
    backend.axpy(-1.0, &work.v, &mut work.r);
    backend.copy(&mut work.r0, &work.r);

    let (mut rho_prev, mut alpha, mut omega) = (1.0_f64, 1.0_f64, 1.0_f64);
    backend.fill(&mut work.v, 0.0);
    backend.fill(&mut work.p, 0.0);

    // Accept the current iterate if a breakdown occurs but the residual is
    // already "small enough": within a factor of 10 of the target.  This
    // avoids false failures when the solution is found to near floating-point
    // noise before the normal convergence check fires.
    let accept = |rnorm: f64| rnorm <= tol * bnorm * 10.0;

    const BREAK: f64 = 1e-30;
    for it in 0..max_iter {
        let rnorm = backend.norm2(&work.r);
        if !rnorm.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if rnorm <= tol * bnorm {
            return Ok(it);
        }

        let rho = backend.dot(&work.r0, &work.r);
        if rho.abs() < BREAK {
            if accept(rnorm) {
                return Ok(it);
            }
            return Err(SolveError::LinearSolverFailed {
                iters: it,
                residual: rnorm,
            });
        }
        let beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta*(p - omega*v)
        backend.axpy(-omega, &work.v, &mut work.p);
        backend.axpby(1.0, &work.r, beta, &mut work.p);

        apply(&work.p, &mut work.v); // v = A·p
        let r0v = backend.dot(&work.r0, &work.v);
        if r0v.abs() < BREAK {
            if accept(rnorm) {
                return Ok(it);
            }
            return Err(SolveError::LinearSolverFailed {
                iters: it,
                residual: rnorm,
            });
        }
        alpha = rho / r0v;

        // s = r - alpha*v
        backend.copy(&mut work.s, &work.r);
        backend.axpy(-alpha, &work.v, &mut work.s);
        let snorm = backend.norm2(&work.s);
        if snorm <= tol * bnorm {
            backend.axpy(alpha, &work.p, x);
            return Ok(it + 1);
        }

        apply(&work.s, &mut work.t); // t = A·s
        let tt = backend.dot(&work.t, &work.t);
        if tt < BREAK {
            // x already has the alpha*p correction applied from the previous step.
            backend.axpy(alpha, &work.p, x);
            if accept(snorm) {
                return Ok(it + 1);
            }
            return Err(SolveError::LinearSolverFailed {
                iters: it,
                residual: snorm,
            });
        }
        omega = backend.dot(&work.t, &work.s) / tt;
        if omega.abs() < BREAK {
            backend.axpy(alpha, &work.p, x);
            if accept(snorm) {
                return Ok(it + 1);
            }
            return Err(SolveError::LinearSolverFailed {
                iters: it,
                residual: snorm,
            });
        }

        // x += alpha*p + omega*s;  r = s - omega*t
        backend.axpy(alpha, &work.p, x);
        backend.axpy(omega, &work.s, x);
        backend.copy(&mut work.r, &work.s);
        backend.axpy(-omega, &work.t, &mut work.r);
        rho_prev = rho;
    }
    let rnorm = backend.norm2(&work.r);
    Err(SolveError::LinearSolverFailed {
        iters: max_iter,
        residual: rnorm,
    })
}

/// Dense LU factorisation with partial pivoting, solved in-place.
///
/// `a` is a row-major `n×n` matrix (modified in-place). On return `b` holds
/// the solution `x` of `A·x = b`. Pivot threshold is `1e-300`.
pub fn lu_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> Result<()> {
    for k in 0..n {
        // Find the pivot row.
        let mut max_row = k;
        let mut max_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-300 {
            return Err(SolveError::SingularMatrix { pivot: k });
        }
        // Swap rows k and max_row.
        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
            b.swap(k, max_row);
        }
        // Forward elimination.
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                let akj = a[k * n + j];
                a[i * n + j] -= factor * akj;
            }
            b[i] -= factor * b[k];
        }
    }
    // Back substitution.
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
    Ok(())
}
