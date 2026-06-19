//! 1-D heat equation on [0,1] with Dirichlet boundaries (zero at both ends).
//!
//! `∂u/∂t = α ∂²u/∂x²`,  `u(0,t) = u(1,t) = 0`,  `u(x,0) = sin(πx)`.
//!
//! Exact solution: `u(x,t) = e^{-α π² t} sin(πx)`.
//!
//! This example uses N=100_000 grid points to show that the matrix-free
//! Newton–BiCGStab solver scales to large problems with O(N) memory.
//! Rayon parallelises the BLAS-1 kernels automatically above 8 192 elements.

use beuler::{BackwardEuler, CpuBackend, OdeProblem};
use std::time::Instant;

const N: usize = 100_000;
const ALPHA: f64 = 1e-4; // diffusivity: keeps condition number manageable
const DX: f64 = 1.0 / (N as f64 + 1.0);
const COEFF: f64 = ALPHA / (DX * DX);

struct Heat1D;

impl OdeProblem<CpuBackend> for Heat1D {
    fn dim(&self) -> usize {
        N
    }

    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        // Interior: standard second-order central difference.
        // Boundary: y[-1] = y[N] = 0 (Dirichlet).
        out[0] = COEFF * (-2.0 * y[0] + y[1]);
        for i in 1..(N - 1) {
            out[i] = COEFF * (y[i - 1] - 2.0 * y[i] + y[i + 1]);
        }
        out[N - 1] = COEFF * (y[N - 2] - 2.0 * y[N - 1]);
    }
}

fn main() {
    let problem = Heat1D;
    let backend = CpuBackend;

    // Initial condition u(x,0) = sin(πx)
    let y0: Vec<f64> = (1..=N)
        .map(|i| (std::f64::consts::PI * i as f64 * DX).sin())
        .collect();

    // With α=1e-4, dx≈1e-5: λ_max ≈ 4α/dx² = 4e6.
    // h=1e-7 gives λ_max·h ≈ 0.4  →  condition number ≈ 1.4  →  BiCGStab converges fast.
    let h = 1e-7_f64;
    let n_steps = 5;

    let mut stepper = BackwardEuler::new(h);
    stepper.newton.lin_max_iter = 50;

    println!("1-D heat equation  N={N}  α={ALPHA}  h={h}  steps={n_steps}");
    println!("Memory per vector: {:.1} MB", N as f64 * 8.0 / 1e6);
    println!(
        "Workspace: ~10 vectors × {:.1} MB ≈ {:.0} MB total\n",
        N as f64 * 8.0 / 1e6,
        10.0 * N as f64 * 8.0 / 1e6
    );

    println!("{:>12}  {:>14}  {:>10}", "t", "max|u|", "elapsed_ms");

    let t_total = Instant::now();
    let _ = stepper
        .integrate_with(&backend, &problem, 0.0, &y0, n_steps, |t, y| {
            let max_u = y.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "{:12.2e}  {:14.9}  {:10.1}",
                t,
                max_u,
                t_total.elapsed().as_secs_f64() * 1000.0
            );
        })
        .expect("integration failed");

    // Exact solution at final time
    let t_final = h * n_steps as f64;
    let decay = (-ALPHA * std::f64::consts::PI * std::f64::consts::PI * t_final).exp();
    println!("\nExact max|u| at t={t_final:.2e}: {decay:.9}  (= e^(-α π² t))");
    println!(
        "Total elapsed: {:.1} ms",
        t_total.elapsed().as_secs_f64() * 1000.0
    );
}
