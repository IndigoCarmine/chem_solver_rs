//! Demonstrates `recommend_backend`: selects CPU or GPU based on problem size.

use beuler::{recommend_backend, BackendKind, BackwardEuler, CpuBackend, OdeProblem};

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

fn solve_cpu(dim: usize) {
    let backend = CpuBackend;
    let problem = Decay { lambda: 1.0 };
    let stepper = BackwardEuler::new(0.1);
    // Use dim to set the initial condition size (all 1.0s for demo).
    let y0 = vec![1.0; dim];
    let final_y = stepper
        .integrate_with(&backend, &problem, 0.0, &y0, 10, |_, _| {})
        .unwrap();
    println!(
        "  y[0] at t=1.0: {:.6}  (exact {:.6})",
        final_y[0],
        (-1.0_f64).exp()
    );
}

fn main() {
    for &dim in &[100usize, 1_000, 10_000, 100_000] {
        let kind = recommend_backend(dim);
        println!("dim={dim:>7}  →  backend: {kind:?}");

        match kind {
            BackendKind::Cpu => solve_cpu(dim),
            BackendKind::Gpu => {
                // GPU path: only reachable when `--features gpu` is enabled
                // and hardware is present (threshold: dim ≥ 50_000).
                #[cfg(feature = "gpu")]
                {
                    println!("  (GPU path: experimental, f32 precision)");
                    // GPU example would go here; omitted — arbitrary f is CPU-only in v0.1.
                }
                #[cfg(not(feature = "gpu"))]
                {
                    // recommend_backend always returns Cpu without the gpu feature.
                    unreachable!("gpu feature not enabled");
                }
            }
        }
    }
}
