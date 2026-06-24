//! Nonlinear reaction-diffusion chain.
//!
//! Equation (for each i = 0..N-1):
//!   dx_i/dt = k1·(x_{i-1}·x_1 − x_i) + k2·(x_{i+1} − x_i)
//!
//! x_1 = x[0] is the **first species** and acts as a nonlinear catalyst:
//!   • k1·(x_{i-1}·x_1 − x_i)  — production from the upstream neighbour
//!     catalysed by x_1, minus linear degradation.
//!   • k2·(x_{i+1} − x_i)      — linear back-flow from the downstream neighbour.
//!
//! Boundary conditions (Dirichlet):
//!   x_{-1} = 1.0  (constant left source feeding x_0)
//!   x_N    = 0.0  (right sink)
//!
//! Initial condition: x_0 = 1.0, x_i = 0 for i > 0.
//!
//! This example uses the CPU backend (recommended for N < 5 000).
//! The GpuEquation DSL *can* express this equation via `x.abs_at(0)` — see
//! the commented block at the bottom — but the current GPU backend downloads
//! state vectors to host for every BiCGStab dot product, so for moderate N
//! the CPU is faster until an on-device reduction is implemented.
//!
//! Run:  cargo run --example chain_reaction --release

use chem_solver_rs::{
    visualize::TrajectoryPlayer,
    BackwardEuler, CpuBackend, OdeProblem,
};

const N: usize = 500;
const K1: f64 = 2.0;
const K2: f64 = 0.5;
const LEFT_BC: f64 = 1.0;

struct ChainReaction;

impl OdeProblem<CpuBackend> for ChainReaction {
    fn dim(&self) -> usize { N }

    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        let x1 = y[0]; // x_1: first species, nonlinear catalyst
        for i in 0..N {
            let xl = if i == 0 { LEFT_BC } else { y[i - 1] };
            let xr = if i == N - 1 { 0.0 } else { y[i + 1] };
            out[i] = K1 * (xl * x1 - y[i]) + K2 * (xr - y[i]);
        }
    }

    /// Analytic JVP: avoids finite-difference noise for the quadratic term.
    fn jac_vec(
        &self,
        _b: &CpuBackend,
        _t: f64,
        y: &Vec<f64>,
        v: &Vec<f64>,
        _fy: &Vec<f64>,
        out: &mut Vec<f64>,
        _scratch: &mut Vec<f64>,
    ) {
        let x1 = y[0];
        let v1 = v[0];
        for i in 0..N {
            let xl = if i == 0 { 0.0 } else { y[i - 1] }; // d(x_{i-1})/dy_j, j=i-1
            let vl = if i == 0 { 0.0 } else { v[i - 1] };
            let vr = if i == N - 1 { 0.0 } else { v[i + 1] };
            // d/dy [k1*(x_{i-1}*x1 - x_i) + k2*(x_{i+1}-x_i)]
            //  = k1*(v_{i-1}*x1 + x_{i-1}*v1 - v_i) + k2*(v_{i+1} - v_i)
            out[i] = K1 * (vl * x1 + xl * v1 - v[i]) + K2 * (vr - v[i]);
        }
    }
}

fn main() {
    let h = 0.02_f64;           // h·(k1+k2) = 0.02·2.5 = 0.05  (well within BE stability)
    let t_end = 20.0_f64;
    let n_steps = (t_end / h).round() as usize;
    let save_every = (n_steps / 800).max(1);

    let mut y0 = vec![0.0_f64; N];
    y0[0] = 3.0;

    let xs: Vec<f64> = (0..N).map(|i| i as f64 / (N - 1) as f64).collect();

    let mut frames: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n_steps / save_every + 1);
    frames.push((0.0, y0.clone()));

    let stepper = BackwardEuler::new(h);
    let mut step = 0usize;

    println!("N={N}  k1={K1}  k2={K2}  t_end={t_end}  steps={n_steps}  frames≈{}", n_steps / save_every);

    let t0 = std::time::Instant::now();
    stepper
        .integrate_with(&CpuBackend, &ChainReaction, 0.0, &y0, n_steps, |t, y| {
            step += 1;
            if step % save_every == 0 {
                frames.push((t, y.to_vec()));
            }
        })
        .expect("integration failed");

    let (t_last, y_last) = frames.last().unwrap();
    let max_x = y_last.iter().cloned().fold(0.0_f64, f64::max);
    println!("Done in {:.2?} — {} frames.  t={t_last:.1}  max(x)={max_x:.4}", t0.elapsed(), frames.len());

    TrajectoryPlayer::new(frames)
        .with_title(format!("Chain Reaction  N={N}  k1={K1}  k2={K2}"))
        .with_x_values(xs)
        .with_labels("index  i/N", "concentration  xᵢ")
        .with_fps(50.0)
        .play()
        .expect("GUI error");
}

// ── GPU version (requires on-device dot reduction for practical performance) ──
//
// The GPU DSL can express this equation thanks to `StateRef::abs_at(u32)`,
// which was added specifically for this use case (reading a fixed global index
// from every thread without boundary-condition wrapping):
//
//   use chem_solver_rs::{gpu_eq::{BoundaryCondition, Expr, GpuEquation}, GpuBackend};
//
//   let backend = GpuBackend::new().unwrap();
//   let problem = GpuEquation::build(
//       &backend, N, BoundaryCondition::Dirichlet(1.0 /* left BC */), |x| {
//           let x1 = x.abs_at(0);                   // y[0] — fixed global read
//           let k1 = Expr::from(K1 as f32);
//           let k2 = Expr::from(K2 as f32);
//           k1 * (x.at(-1) * x1 - x.at(0)) + k2 * (x.at(1) - x.at(0))
//       });
//   // Then pass &backend and &problem to BackwardEuler::integrate_with as usual.
