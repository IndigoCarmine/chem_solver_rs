//! Integration tests: first-order convergence and A-stability.

use beuler::newton::newton_be_dense;
use beuler::{
    BackwardEuler, CpuBackend, DenseJacobian, DenseNewtonWorkspace, NewtonParams, OdeProblem,
};

// ── Test problem: dy/dt = -y ────────────────────────────────────────────────

struct DecayUnit;

impl OdeProblem<CpuBackend> for DecayUnit {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        out[0] = -y[0];
    }
}

impl DenseJacobian<CpuBackend> for DecayUnit {
    fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
        jac[0] = -1.0; // 1×1 matrix
    }
}

// ── Helper: integrate dy/dt = -y to t=1 with n steps, return |y(1) - e^{-1}|

fn decay_error(n: usize) -> f64 {
    let h = 1.0 / n as f64;
    let stepper = BackwardEuler::new(h);
    let traj = stepper
        .integrate(&CpuBackend, &DecayUnit, 0.0, &[1.0], n)
        .unwrap();
    let (_, y) = traj.last().unwrap();
    (y[0] - (-1.0_f64).exp()).abs()
}

// ── 1. First-order convergence ───────────────────────────────────────────────

#[test]
fn first_order_convergence() {
    let e_coarse = decay_error(100);
    let e_fine = decay_error(200);

    // Global error for backward Euler is O(h), so halving h should roughly
    // halve the error. We accept ratios in [1.7, 2.3] for safety.
    let ratio = e_coarse / e_fine;
    assert!(
        ratio > 1.7 && ratio < 2.3,
        "expected error ratio ≈ 2, got {ratio:.3} (coarse={e_coarse:.3e}, fine={e_fine:.3e})"
    );

    // Absolute error with h=1/100 must be below 5e-3.
    assert!(e_coarse < 5e-3, "error at h=0.01 too large: {e_coarse:.3e}");
}

// ── 2. A-stability: stiff problem λ=1000, h=0.1 (λh = 100 ≫ 1) ─────────────

#[test]
fn a_stability_stiff() {
    struct StiffDecay;
    impl OdeProblem<CpuBackend> for StiffDecay {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
            out[0] = -1000.0 * y[0];
        }
    }

    // h=0.1, λh=100. Forward Euler would blow up; backward Euler must stay bounded.
    let mut stepper = BackwardEuler::new(0.1);
    stepper.newton.lin_max_iter = 500;

    let traj = stepper
        .integrate(&CpuBackend, &StiffDecay, 0.0, &[1.0], 50)
        .unwrap();

    let mut prev = 1.0f64;
    for (_, y) in traj.iter().skip(1) {
        let cur = y[0];
        // Solution must be finite and decreasing toward 0.
        assert!(cur.is_finite(), "non-finite value: {cur}");
        assert!(cur >= 0.0, "solution went negative: {cur}");
        assert!(
            cur < prev + 1e-12,
            "solution not monotone: prev={prev:.3e} cur={cur:.3e}"
        );
        prev = cur;
    }

    // After 50 steps (t=5), |y| should be tiny.
    let (_, y_final) = traj.last().unwrap();
    assert!(
        y_final[0] < 1e-10,
        "stiff solution did not decay: y(5) = {:.3e}",
        y_final[0]
    );
}

// ── 3. Dense path agrees with matrix-free on a 2×2 linear system ────────────

struct Linear2x2;

// dy/dt = A·y  with  A = [[-1, 0.5], [0, -2]]
impl OdeProblem<CpuBackend> for Linear2x2 {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        out[0] = -y[0] + 0.5 * y[1];
        out[1] = -2.0 * y[1];
    }
}

impl DenseJacobian<CpuBackend> for Linear2x2 {
    fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
        // Row-major 2×2: [[-1, 0.5], [0, -2]]
        jac[0] = -1.0;
        jac[1] = 0.5;
        jac[2] = 0.0;
        jac[3] = -2.0;
    }
}

#[test]
fn dense_matches_matrix_free() {
    let h = 0.1_f64;
    let y0 = vec![1.0_f64, 1.0];
    let t1 = h;

    // Matrix-free path
    let stepper = BackwardEuler::new(h);
    let traj_mf = stepper
        .integrate(&CpuBackend, &Linear2x2, 0.0, &y0, 1)
        .unwrap();
    let (_, y_mf) = traj_mf.last().unwrap();

    // Dense path
    let params = NewtonParams::default();
    let mut ws_dense = DenseNewtonWorkspace::new(2);
    let mut y_dense = y0.clone();
    newton_be_dense(&Linear2x2, t1, h, &y0, &mut y_dense, &mut ws_dense, &params).unwrap();

    // Both paths should agree to within a tight tolerance.
    let diff = ((y_mf[0] - y_dense[0]).powi(2) + (y_mf[1] - y_dense[1]).powi(2)).sqrt();
    assert!(
        diff < 1e-10,
        "matrix-free and dense paths disagree: mf={y_mf:?} dense={y_dense:?} diff={diff:.3e}"
    );
}
