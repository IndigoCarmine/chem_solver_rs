# beuler

**後退オイラー法（implicit Euler）による ODE ソルバー。行列フリー Newton–Krylov 法で O(N) メモリを実現し、CPU（rayon）と GPU（wgpu、experimental）の両バックエンドをサポートする Rust ライブラリです。**

A memory-efficient backward Euler ODE solver for stiff systems `dy/dt = f(t, y)`.  
Solves each step with **matrix-free Newton–Krylov (BiCGStab)** — no Jacobian is ever
assembled. Memory scales as `O(N)` regardless of system size.

[![CI](https://github.com/YOUR_USERNAME/beuler/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/beuler/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

---

## Features

- **O(N) memory** — Jacobian is never stored. Only ~10 vectors of length N are needed.
- **Matrix-free Newton–Krylov** — each step solves `G(y) = y − yₙ − h·f(t,y) = 0`
  by Newton iteration, using finite-difference Jacobian–vector products and BiCGStab.
- **Zero allocation in the time loop** — all workspace is pre-allocated before stepping.
- **Pluggable backends** via the `Backend` trait:
  - `CpuBackend` — `f64`, rayon-parallel above 8 192 elements.
  - `GpuBackend` — `f32`, wgpu compute shaders (*experimental*, see caveats below).
- **Dense LU fallback** — for small `dim ≲ 500` a direct `DenseJacobian` + LU path is
  available for robustness.
- **Auto backend selection** via `recommend_backend(dim)`.

---

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
beuler = "0.1"

# Optional: GPU support (requires wgpu-capable hardware, MSRV 1.87)
# beuler = { version = "0.1", features = ["gpu"] }
```

---

## Quick start

```rust
use beuler::{BackwardEuler, CpuBackend, OdeProblem};

// dy/dt = -y,  y(0) = 1  →  exact: e^{-t}
struct Decay;
impl OdeProblem<CpuBackend> for Decay {
    fn dim(&self) -> usize { 1 }
    fn eval(&self, _b: &CpuBackend, _t: f64, y: &Vec<f64>, out: &mut Vec<f64>) {
        out[0] = -y[0];
    }
}

fn main() {
    let stepper = BackwardEuler::new(0.1);          // fixed step h = 0.1
    let traj = stepper
        .integrate(&CpuBackend, &Decay, 0.0, &[1.0], 10)
        .unwrap();
    let (t, y) = traj.last().unwrap();
    println!("y({t}) ≈ {:.4}  exact {:.4}", y[0], (-t).exp());
}
```

See `examples/` for more:

| Example | Description |
|---|---|
| `decay` | Scalar exponential decay, convergence table |
| `heat1d` | 1-D heat equation, N=100 000, rayon parallelism |
| `auto_backend` | `recommend_backend` demo |

---

## Architecture

### Matrix-free Newton–Krylov

Each backward Euler step finds `y_{n+1}` satisfying

```
G(y) = y − yₙ − h·f(t_{n+1}, y) = 0
```

by Newton iteration. The Jacobian `J_G = I − h·J_f` is **never assembled**.
Instead, the inner linear system `J_G · δ = −G(y)` is solved by **BiCGStab**
using only matrix–vector products `J_G·v = v − h·(J_f·v)`, where `J_f·v` is
approximated by a single finite-difference evaluation:

```
J_f·v ≈ (f(t, y + εv) − f(t, y)) / ε,   ε = 1e-7·(1+‖y‖)/‖v‖
```

This costs **one extra `f` evaluation per BiCGStab iteration** and no matrix storage.

### Memory budget

| Quantity | Vectors of length N |
|---|---|
| State `y` | 1 |
| Newton workspace (`fy`, `g`, `δ`, `scratch`) | 4 |
| BiCGStab workspace (`r`, `r̃₀`, `p`, `v`, `s`, `t`) | 6 |
| **Total** | **≈ 11** |

For `f64` / CPU: `11 × N × 8` bytes — no matrix, no fill-in.

### Why is this robust? (Dissipativity)

For systems satisfying the **one-sided Lipschitz (dissipativity)** condition

```
⟨f(t,u) − f(t,v), u − v⟩ ≤ ν‖u−v‖²,   ν ≤ 0
```

backward Euler is an **unconditional contraction mapping** regardless of step size.
The matrix `I − h·J_f` is then well-conditioned (eigenvalues bounded away from
zero), which is why BiCGStab converges in few iterations even for very stiff systems.

The 1-D heat equation and scalar linear decay both satisfy this with `ν ≤ 0`,
and indeed BiCGStab typically converges in single-digit iterations for these problems.

### Backend abstraction

The `Backend` trait exposes only BLAS-1 primitives:

```
zeros / len / from_host / to_host / copy / fill / scale / axpy / axpby / dot / norm2
```

All solver code (Newton, BiCGStab, stepper) depends solely on this trait, making
backends interchangeable without touching the solver.

### Auto backend selection

`recommend_backend(dim)` returns `BackendKind::Gpu` only when the `gpu` feature is
enabled, hardware is available, and `dim ≥ 50_000`. Otherwise it returns
`BackendKind::Cpu`. The threshold is conservative; tune it for your hardware.

---

## GPU backend (experimental)

Enable with `--features gpu`. Caveats:

- **`f32` only.** WGSL storage buffers are 32-bit. For stiff problems this may be
  insufficient. Use `CpuBackend` (`f64`) for precision-sensitive work.
- **Reductions on host.** `dot` / `norm2` currently download the buffer and reduce
  in `f64`. Elementwise kernels (`axpy`, `scale`, …) run on device.
- **Untested against real hardware** in this release. The wgpu 29 API is targeted;
  `request_adapter` / `request_device` / `device.poll` signatures have shifted
  across wgpu versions — verify against your exact version if builds fail.
- **Arbitrary `f` on GPU is out of scope for v0.1.** The `OdeProblem::eval` closure
  runs on the CPU; only the BLAS-1 kernels are offloaded.

---

## Status & Roadmap

| Component | Status |
|---|---|
| CPU backend (`f64`, rayon) | ✅ working, tested |
| Matrix-free BiCGStab | ✅ working, tested |
| Newton–Krylov stepper | ✅ working, tested |
| Dense LU path (`dim ≲ 500`) | ✅ working, tested |
| GPU backend (`f32`, wgpu 29) | ⚗️ experimental — compiles, untested on hardware |
| f64 GPU path | 🗺 roadmap (CUDA via `cust`, or Vulkan `shaderFloat64`) |
| GMRES(m) inner solver | 🗺 roadmap (lower memory per iteration for some problems) |
| Adaptive step / order control | 🗺 roadmap |
| Sparse Jacobian (CSR) | 🗺 roadmap |

---

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your option.
