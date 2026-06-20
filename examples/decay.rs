//! Scalar exponential decay: `dy/dt = -λy`, `y(0) = 1`.
//!
//! Demonstrates basic usage and first-order convergence against the exact
//! solution `y(t) = e^{-λt}`.

use beuler::{BackwardEuler, CpuBackend, OdeProblem};

struct Decay {
    lambda: f64,
}

impl OdeProblem<CpuBackend> for Decay {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        out[0] = -self.lambda * y[0];
    }
}

fn main() {
    let lambda = 2.0;
    let problem = Decay { lambda };
    let backend = CpuBackend;

    println!("dy/dt = -{lambda}·y,  y(0)=1  →  exact: e^(-{lambda}·t)\n");
    println!(
        "{:>8}  {:>14}  {:>14}  {:>12}",
        "t", "y_numerical", "y_exact", "error"
    );

    // Integrate to t=2 with h=0.1
    let stepper = BackwardEuler::new(0.1);
    let traj = stepper
        .integrate(&backend, &problem, 0.0, &[1.0], 20)
        .unwrap();

    for (t, y) in &traj {
        let exact = (-lambda * t).exp();
        println!(
            "{:8.4}  {:14.9}  {:14.9}  {:12.3e}",
            t,
            y[0],
            exact,
            (y[0] - exact).abs()
        );
    }

    // First-order convergence check
    println!("\nConvergence check at t=1:");
    println!("{:>6}  {:>12}  {:>10}", "h", "error", "ratio");
    let mut prev_err: Option<f64> = None;
    for &n in &[10usize, 20, 50, 100, 200] {
        let h = 1.0 / n as f64;
        let st = BackwardEuler::new(h);
        let tr = st.integrate(&backend, &problem, 0.0, &[1.0], n).unwrap();
        let (_, y_final) = tr.last().unwrap();
        let err = (y_final[0] - (-lambda).exp()).abs();
        let ratio = prev_err.map_or(f64::NAN, |p| p / err);
        println!("{:6.4}  {:12.3e}  {:10.3}", h, err, ratio);
        prev_err = Some(err);
    }
}
