//! Forced oscillation of a 1-D mass-spring chain.
//!
//! One endpoint is driven sinusoidally; the other is clamped to zero.
//!
//!   u(0, t) = A · sin(ω t)   (forced left end)
//!   u(L, t) = 0              (clamped right end)
//!
//! The 2N-dimensional state vector is [q₀ … q_{N-1}, v₀ … v_{N-1}]
//!   q_i = transverse displacement of mass i
//!   v_i = dq_i / dt
//!
//! Equations of motion (unit mass per point):
//!   dq_i/dt  = v_i
//!   dv_i/dt  = (c/Δx)² · (q_{i-1} − 2 q_i + q_{i+1})
//!
//! We set ω = 4 rad/s (above the 1st resonance at ω₁=πc/L≈3.14), so the
//! forcing creates genuine travelling waves rather than an evanescent response.
//!
//! Run (text summary):
//!   cargo run --example forced_chain --release
//!
//! Run with animated GUI:
//!   cargo run --example forced_chain --features visualize --release

use chem_solver_rs::{BackwardEuler, CpuBackend, OdeProblem, visualize::TrajectoryPlayer};

const N: usize = 200;
const L: f64 = 1.0;
const DX: f64 = L / (N as f64 + 1.0);
/// Wave speed.
const C: f64 = 1.0;
/// Coupling (k/m = (c/Δx)²).
const KM: f64 = (C / DX) * (C / DX);
/// Forcing angular frequency (rad/s).  Period T = 2π/ω ≈ 1.57 s.
/// Above the 1st mode (ω₁ = πc ≈ 3.14), so waves propagate.
const OMEGA: f64 = 4.0;
const AMPLITUDE: f64 = 1.0;

// ── ODE problem ───────────────────────────────────────────────────────────

struct ForcedChain;

impl OdeProblem<CpuBackend> for ForcedChain {
    fn dim(&self) -> usize {
        2 * N
    }

    fn eval(&self, _b: &CpuBackend, t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        let q = &y[..N];
        let v = &y[N..];
        let q_bc = AMPLITUDE * (OMEGA * t).sin();

        // dq/dt = v
        out[..N].copy_from_slice(v);

        // dv/dt = KM · (q_{i-1} − 2 q_i + q_{i+1}),  boundaries applied here.
        for i in 0..N {
            let ql = if i == 0 { q_bc } else { q[i - 1] };
            let qr = if i == N - 1 { 0.0 } else { q[i + 1] };
            out[N + i] = KM * (ql - 2.0 * q[i] + qr);
        }
    }

    /// Analytic Jacobian–vector product: J · [v_q, v_v] = [v_v, K_free · v_q].
    ///
    /// The left boundary is *time-varying but not state-dependent*, so the
    /// Jacobian of f w.r.t. y has no boundary contribution — K_free is the
    /// purely interior tridiagonal operator.
    fn jac_vec(
        &self,
        _b: &CpuBackend,
        _t: f64,
        _y: &Vec<f64>,
        v: &Vec<f64>,
        _fy: &Vec<f64>,
        out: &mut Vec<f64>,
        _scratch: &mut Vec<f64>,
    ) {
        let vq = &v[..N];
        let vv = &v[N..];

        // Position part:  dq = vv
        out[..N].copy_from_slice(vv);

        // Velocity part:  dv = KM · tridiag(-1, 2, -1) · vq
        for i in 0..N {
            let ql = if i == 0 { 0.0 } else { vq[i - 1] }; // BC wall: contributes 0
            let qr = if i == N - 1 { 0.0 } else { vq[i + 1] };
            out[N + i] = KM * (ql - 2.0 * vq[i] + qr);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────

fn main() {
    let period = 2.0 * std::f64::consts::PI / OMEGA; // ≈ 1.5708 s
    let t_end = 20.0 * period;                        // 20 full oscillations
    // Keep h·ω_max < 1 so BiCGStab sees a well-conditioned system.
    // ω_max (chain) = 2c/Δx = 2C*(N+1)/L ≈ 2*201 = 402  →  h = 2e-3 → h·ω_max ≈ 0.8
    let h = 2e-3_f64;
    let n_steps = (t_end / h).round() as usize;
    // ≈ 1 000 frames: smooth enough for 50 fps playback of 20 periods
    let save_every = (n_steps / 1_000).max(1);

    println!(
        "N={N}  ω={OMEGA}  T≈{period:.3}  t_end={t_end:.1}  h={h:.0e}  \
         steps={n_steps}  saving 1/{save_every}={} frames",
        n_steps / save_every
    );

    let y0 = vec![0.0_f64; 2 * N]; // chain at rest

    // x-coords for display: include the two boundary points
    let xs: Vec<f64> = (0..=N + 1).map(|i| i as f64 * DX).collect();

    let mut frames: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n_steps / save_every + 1);
    frames.push((0.0, snapshot(0.0, &y0[..N])));

    let stepper = BackwardEuler::new(h);
    let mut step = 0usize;

    let t0 = std::time::Instant::now();
    stepper
        .integrate_with(&CpuBackend, &ForcedChain, 0.0, &y0, n_steps, |t, y| {
            step += 1;
            if step % save_every == 0 {
                frames.push((t, snapshot(t, &y[..N])));
            }
        })
        .expect("integration failed");

    println!("Done in {:.2?} — {} frames.", t0.elapsed(), frames.len());

    TrajectoryPlayer::new(frames)
        .with_title(format!(
            "Forced Chain  ·  N={N}  ·  c={C}  ·  ω={OMEGA}  ·  20 periods"
        ))
        .with_x_values(xs)
        .with_labels("x", "displacement  u(x, t)")
        .with_fps(50.0)
        .play()
        .expect("GUI error");
}

/// Frame = [left BC,  q_0 … q_{N-1},  right BC=0] (N+2 points, matches xs).
fn snapshot(t: f64, q: &[f64]) -> Vec<f64> {
    let mut frame = Vec::with_capacity(N + 2);
    frame.push(AMPLITUDE * (OMEGA * t).sin());
    frame.extend_from_slice(q);
    frame.push(0.0);
    frame
}
