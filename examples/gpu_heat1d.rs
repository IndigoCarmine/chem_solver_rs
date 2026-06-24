//! 1-D heat equation on the GPU using the expression DSL.
//!
//! Solves `du/dt = alpha * Laplacian(u)` on a uniform grid of `n` points with
//! Dirichlet boundary conditions `u(0,t) = u(1,t) = 0` and initial condition
//! `u(x,0) = sin(pi*x)`.  The exact solution is `u(x,t) = sin(pi*x) * exp(-alpha*pi^2*t)`.
//!
//! All BLAS-1 operations AND the right-hand side evaluation run on the GPU;
//! no per-step CPU work is done once the shader is compiled.
//!
use chem_solver_rs::{
    gpu_eq::{BoundaryCondition, Expr, GpuEquation},
    BackwardEuler, GpuBackend, NewtonParams,
    visualize::TrajectoryPlayer,
};

fn main() {
    let backend = GpuBackend::new().expect("no GPU adapter available");

    // Grid: n interior points on (0, 1); boundary values fixed at 0.
    let n: usize = 1_000;
    let alpha = 1e-2_f64;
    let dx = 1.0 / (n as f64 + 1.0);
    // Laplacian coefficient (f32 — GPU precision)
    let coeff = (alpha / (dx * dx)) as f32;

    // Compile the ODE right-hand side into a WGSL shader once.
    let problem = GpuEquation::build(&backend, n, BoundaryCondition::Dirichlet(0.0), |x| {
        let c = Expr::from(coeff);
        c * (x.at(-1) - x.at(0) * 2.0_f32 + x.at(1))
    });

    // println!(generate_eval_wgsl(&problem, BoundaryCondition::Dirichlet(0.0)).unwrap());

    // Initial condition u(x, 0) = sin(pi * x)
    let y0: Vec<f64> = (0..n)
        .map(|i| {
            let xi = (i as f64 + 1.0) * dx;
            (std::f64::consts::PI * xi).sin()
        })
        .collect();

    let h = 1e-6_f64;
    let n_steps = 10000;

    let mut stepper = BackwardEuler::new(h);
    // GPU backend works in f32 (~7 significant digits). The Newton residual
    // cannot converge below ~1e-6 for a 1000-element f32 system, so we relax
    // the tolerance to match f32 precision.
    stepper.newton = NewtonParams {
        tol: 1e-5,
        ..NewtonParams::default()
    };

    let traj = stepper
        .integrate(&backend, &problem, 0.0, &y0, n_steps)
        .expect("integration failed");

    let (t_end, y_end) = traj.last().unwrap();
    let max_u = y_end.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exact_max = (-alpha * std::f64::consts::PI.powi(2) * t_end).exp();

    println!("t = {t_end:.6}  max|u| = {max_u:.6}  (exact {exact_max:.6})");

    // ── Animated GUI (only when compiled with --features visualize) ──────────

    // Physical x-coordinates for the interior grid points.
    let xs: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * dx).collect();

    TrajectoryPlayer::new(traj)
        .with_title("GPU Heat Equation — 1-D")
        .with_x_values(xs)
        .with_labels("x", "u(x, t)")
        .with_fps(30.0)
        .play()
        .expect("GUI error");
    
}
