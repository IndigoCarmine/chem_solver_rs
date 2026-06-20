# beuler — Claude Code 引き継ぎ指示書

後退オイラー法 ODE ソルバー（Rust）を完成させ、GitHub に公開するための指示書。
**この `HANDOFF.md` の内容に従って残りを実装し、テストを通し、リポジトリを作成・push してほしい。**
同梱の `beuler/` には実装済みファイルが入っている（§4）。残タスクは §5、検証は §6、公開は §7。

---

## 0. ゴール

- 後退オイラー法（implicit Euler）で `dy/dt = f(t, y)` を解くライブラリ。
- 設計の主眼は **(1) メモリ効率最優先**、**(2) CPU(rayon) / GPU(wgpu) バックエンドの切り替え（自動選択あり）**。
- 公開可能な品質（README・ライセンス・CI・examples・tests）まで仕上げる。

## 1. 設計判断（**変更しないこと**。理由つき）

- **行列フリー Newton–Krylov**。各ステップで
  `G(y) = y − y_n − h·f(t_{n+1}, y) = 0` を Newton 法で解く。
  ヤコビアン `J` は **構築しない**。有限差分のヤコビアン–ベクトル積 `J·v` だけ使う。
  → 行列を持たないのでメモリ `O(N)`。これが「メモリ効率最優先」の核。
- 内部線形ソルバは **行列フリー BiCGStab**。一定メモリ（約 6N）で非対称系に対応。
  GMRES より省メモリなので方針に最も合う（GMRES(m) は roadmap）。
- **事前確保ワークスペース**。時間ステップのループ内では一切ヒープ確保しない。
- **Backend トレイト**で BLAS-1 のみ抽象化（`zeros/len/from_host/to_host/copy/fill/scale/axpy/axpby/dot/norm2`）。
  ソルバ（Newton/BiCGStab/stepper）はこのトレイトの上だけで動く＝バックエンド非依存。
  - `CpuBackend`: `Vec<f64>`、rayon 並列、しきい値（`PAR_THRESHOLD`）以下は逐次。
  - `GpuBackend`: wgpu、**f32**（WGSL に f64 型が無い）。**experimental**。境界で f64↔f32 変換。`dot/norm` は当面ホストで縮約。
- 小規模（`dim ≲ 500`）向けに **dense LU 直接法**も用意（`DenseJacobian` トレイト + `lu_solve`）。反復法より堅牢な場面用。
- **単調性についての補足（README の動機付けに書く）**: 元々の「f が単調」をベクトルに一般化すると
  **one-sided Lipschitz / 散逸性** `⟨f(t,u)−f(t,v), u−v⟩ ≤ ν‖u−v‖²`。
  `ν ≤ 0`（散逸的）なら後退オイラーは**無条件に縮小写像**で、`I − h·J_f` も良条件 →
  BiCGStab が少数回で収束する。これがこのソルバ構成が堅牢な理由。

## 2. バージョン / features

```toml
rayon = "1.10"                                   # CPU バックエンド
wgpu = { version = "29", optional = true }       # GPU（MSRV 1.87）
pollster = { version = "0.4", optional = true }
bytemuck = { version = "1", features = ["derive"], optional = true }

[features]
default = []
gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]
```
`Cargo.toml` は同梱済み。`wgpu` の最新は 29 系（2026 時点）、MSRV 1.87。

## 3. リポジトリ構成

```
beuler/
├── Cargo.toml                    [済]
├── README.md                     [要作成 §5.7]
├── LICENSE-MIT                   [要作成]
├── LICENSE-APACHE                [要作成]
├── .gitignore                    [要作成]
├── .github/workflows/ci.yml      [要作成]
├── src/
│   ├── lib.rs                    [要作成 §5.4]
│   ├── error.rs                  [済]
│   ├── backend/
│   │   ├── mod.rs                [済]  Backend トレイト・BackendKind・recommend_backend
│   │   ├── cpu.rs                [済]  CpuBackend (rayon, f64)
│   │   └── gpu.rs                [済]  GpuBackend (wgpu, f32, experimental, #[cfg(feature="gpu")])
│   ├── problem.rs                [済]  OdeProblem (有限差分 JVP) / DenseJacobian
│   ├── linsolve.rs               [要作成 §5.1]  bicgstab + lu_solve
│   ├── newton.rs                 [要作成 §5.2]  matrix-free 後退オイラー Newton
│   └── stepper.rs                [要作成 §5.3]  BackwardEuler
├── examples/
│   ├── decay.rs                  [要作成]
│   ├── heat1d.rs                 [要作成]
│   └── auto_backend.rs           [要作成]
└── tests/
    └── convergence.rs            [要作成 §5.6]
```

## 4. 同梱済みファイル（原則そのまま使用）

`Cargo.toml`, `src/error.rs`, `src/backend/mod.rs`, `src/backend/cpu.rs`,
`src/backend/gpu.rs`, `src/problem.rs`。
**コンパイルに必要な範囲でのみ修正可**。特に `gpu.rs` は wgpu 29 想定で書いたが
未コンパイル。`request_adapter`/`request_device` の戻り型や `device.poll` の引数
（`Maintain::Wait` vs `PollType::Wait`）はバージョン差があるので、`--features gpu` を
ビルドするなら実バージョンに合わせて調整すること。

参考: `Backend` トレイトの確定シグネチャ（`src/backend/mod.rs` 済）
```rust
fn zeros(&self, n: usize) -> Self::Vector;
fn len(&self, v: &Self::Vector) -> usize;
fn from_host(&self, data: &[f64]) -> Self::Vector;
fn to_host(&self, v: &Self::Vector) -> Vec<f64>;
fn copy(&self, dst: &mut Self::Vector, src: &Self::Vector);   // dst <- src
fn fill(&self, x: &mut Self::Vector, value: f64);             // x  <- value
fn scale(&self, x: &mut Self::Vector, alpha: f64);            // x  <- a*x
fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector);          // y <- y + a*x
fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector); // y <- a*x + b*y
fn dot(&self, x: &Self::Vector, y: &Self::Vector) -> f64;
fn norm2(&self, x: &Self::Vector) -> f64;                     // default sqrt(dot)
```
`OdeProblem::jac_vec`（既定の有限差分 JVP, 済）は
`out <- (f(t, y+εv) − fy)/ε`, `ε = 1e-7·(1+‖y‖)/‖v‖`, `‖v‖=0` なら `out=0`。
`scratch`（長さ dim）を借りてアロケーションしない。

---

## 5. 残タスクの仕様

### 5.1 `src/linsolve.rs`

`error::{Result, SolveError}`、`backend::Backend` を use。

**(a) `BiCgStabWork<B>`**: ベクトル6本 `r, r0, p, v, s, t` を保持。
`fn new(backend: &B, n: usize) -> Self` で全部 `backend.zeros(n)`。

**(b) 行列フリー BiCGStab**。`apply: FnMut(&V, &mut V)` が `out = A·v` を計算するクロージャ。
`A·x = b` を解き、解は `x`（初期推定 in / 解 out）に書く。収束判定 `‖r‖ ≤ tol·‖b‖`。

```rust
pub fn bicgstab<B, A>(
    backend: &B, mut apply: A,
    b: &B::Vector, x: &mut B::Vector,
    work: &mut BiCgStabWork<B>, tol: f64, max_iter: usize,
) -> Result<usize>
where B: Backend, A: FnMut(&B::Vector, &mut B::Vector)
{
    let bnorm = backend.norm2(b);
    if bnorm == 0.0 { backend.fill(x, 0.0); return Ok(0); }

    // r = b - A x
    apply(x, &mut work.v);
    backend.copy(&mut work.r, b);
    backend.axpy(-1.0, &work.v, &mut work.r);
    backend.copy(&mut work.r0, &work.r);

    let (mut rho_prev, mut alpha, mut omega) = (1.0_f64, 1.0_f64, 1.0_f64);
    backend.fill(&mut work.v, 0.0);
    backend.fill(&mut work.p, 0.0);

    const BREAK: f64 = 1e-30;
    for it in 0..max_iter {
        let rnorm = backend.norm2(&work.r);
        if !rnorm.is_finite() { return Err(SolveError::NonFinite); }
        if rnorm <= tol * bnorm { return Ok(it); }

        let rho = backend.dot(&work.r0, &work.r);
        if rho.abs() < BREAK { return Err(SolveError::LinearSolverFailed { iters: it, residual: rnorm }); }
        let beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta*(p - omega*v)
        backend.axpy(-omega, &work.v, &mut work.p);   // p -= omega*v
        backend.axpby(1.0, &work.r, beta, &mut work.p); // p  = r + beta*p

        apply(&work.p, &mut work.v);                  // v = A p
        let r0v = backend.dot(&work.r0, &work.v);
        if r0v.abs() < BREAK { return Err(SolveError::LinearSolverFailed { iters: it, residual: rnorm }); }
        alpha = rho / r0v;

        // s = r - alpha*v
        backend.copy(&mut work.s, &work.r);
        backend.axpy(-alpha, &work.v, &mut work.s);
        let snorm = backend.norm2(&work.s);
        if snorm <= tol * bnorm {
            backend.axpy(alpha, &work.p, x);          // x += alpha*p
            return Ok(it + 1);
        }

        apply(&work.s, &mut work.t);                  // t = A s
        let tt = backend.dot(&work.t, &work.t);
        if tt < BREAK { return Err(SolveError::LinearSolverFailed { iters: it, residual: snorm }); }
        omega = backend.dot(&work.t, &work.s) / tt;
        if omega.abs() < BREAK { return Err(SolveError::LinearSolverFailed { iters: it, residual: snorm }); }

        // x += alpha*p + omega*s ; r = s - omega*t
        backend.axpy(alpha, &work.p, x);
        backend.axpy(omega, &work.s, x);
        backend.copy(&mut work.r, &work.s);
        backend.axpy(-omega, &work.t, &mut work.r);
        rho_prev = rho;
    }
    let rnorm = backend.norm2(&work.r);
    Err(SolveError::LinearSolverFailed { iters: max_iter, residual: rnorm })
}
```
注: `apply(&work.p, &mut work.v)` のように **異なるフィールド**を同時に借りるのは
disjoint field borrow なので OK（メソッド経由にしないこと）。

**(c) dense LU**（partial pivot, 行優先, in-place, 解を `b` に上書き）。
```rust
pub fn lu_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> Result<()>
```
各列でピボット探索 → 行スワップ（`a` と `b` の対応行）→ 前進消去 → 後退代入。
ピボット `< 1e-300` なら `Err(SolveError::SingularMatrix { pivot: k })`。

### 5.2 `src/newton.rs`

```rust
pub struct NewtonParams { pub tol: f64, pub max_iter: usize, pub lin_tol: f64, pub lin_max_iter: usize }
// Default: tol 1e-8, max_iter 50, lin_tol 1e-3(inexact Newton), lin_max_iter 200

pub struct NewtonWorkspace<B: Backend> { fy: B::Vector, g: B::Vector, delta: B::Vector, scratch: B::Vector, lin: BiCgStabWork<B> }
// new(backend, n): 各 zeros(n) と BiCgStabWork::new(backend, n)
```

`y` は反復値（呼び出し側が初期推定 = `y_n` を入れて渡す。下の stepper で実施）。
1 ステップ分（`t1 = t_n + h` での解）を **その場で** 求める:

```rust
pub fn newton_be<B, P>(
    backend: &B, problem: &P, t1: f64, h: f64,
    y_n: &B::Vector, y: &mut B::Vector,
    ws: &mut NewtonWorkspace<B>, params: &NewtonParams,
) -> Result<usize>
where B: Backend, P: OdeProblem<B>
{
    for k in 0..params.max_iter {
        problem.eval(backend, t1, y, &mut ws.fy);        // fy = f(t1, y)
        // g = y - y_n - h*fy   (= 残差 G(y))
        backend.copy(&mut ws.g, y);
        backend.axpy(-1.0, y_n, &mut ws.g);
        backend.axpy(-h, &ws.fy, &mut ws.g);

        let gnorm = backend.norm2(&ws.g);
        if !gnorm.is_finite() { return Err(SolveError::NonFinite); }
        if gnorm <= params.tol { return Ok(k); }

        backend.scale(&mut ws.g, -1.0);                  // b = -g
        backend.fill(&mut ws.delta, 0.0);                // 初期推定 0

        // borrow split: クロージャが scratch/fy/y を、bicgstab が g/delta/lin を借りる
        let NewtonWorkspace { fy, g, delta, scratch, lin } = &mut *ws;
        let y_ref: &B::Vector = &*y;
        let apply = |v: &B::Vector, out: &mut B::Vector| {
            problem.jac_vec(backend, t1, y_ref, v, fy, out, scratch); // out = J_f·v
            backend.axpby(1.0, v, -h, out);                          // out = v - h*(J_f·v) = A·v
        };
        bicgstab(backend, apply, g, delta, lin, params.lin_tol, params.lin_max_iter)?;

        backend.axpy(1.0, delta, y);                     // y += delta
    }
    // 最終残差を計算して NewtonDiverged を返す
    problem.eval(backend, t1, y, &mut ws.fy);
    backend.copy(&mut ws.g, y);
    backend.axpy(-1.0, y_n, &mut ws.g);
    backend.axpy(-h, &ws.fy, &mut ws.g);
    Err(SolveError::NewtonDiverged { iters: params.max_iter, residual: backend.norm2(&ws.g) })
}
```
`A·v = v − h·(J_f·v)` は `I − h·J_f`。`fy` は現在の反復値での `f(t1,y)`（JVP の有限差分に使う）。

（任意）`DenseJacobian` 用の `newton_be_dense`: `M = I − h·J`(行優先) を組み、
`b = −G` をホストで作り、`lu_solve` で解いて更新。小規模・CPU 向け。example/test で 1 つ使う。

### 5.3 `src/stepper.rs`

```rust
pub struct BackwardEuler { pub h: f64, pub newton: NewtonParams }
impl BackwardEuler {
    pub fn new(h: f64) -> Self                                   // newton = Default
    // y を t1 = t0+h の解へ更新（in-place）。ws 再利用。
    pub fn step<B,P>(&self, backend:&B, problem:&P, t0:f64, y:&mut B::Vector, ws:&mut NewtonWorkspace<B>) -> Result<()>
    // メモリ軽量: 各ステップ後にコールバックへ (t, &[f64]) を渡す。軌跡を貯めない。
    pub fn integrate_with<B,P,F>(&self, backend:&B, problem:&P, t0:f64, y0:&[f64], n_steps:usize, mut on_step:F) -> Result<Vec<f64>>
        where F: FnMut(f64, &[f64]);  // 最終状態 host を返す
    // 利便性: 小規模向けに全軌跡 Vec<(f64, Vec<f64>)> を返す版
    pub fn integrate<B,P>(&self, backend:&B, problem:&P, t0:f64, y0:&[f64], n_steps:usize) -> Result<Vec<(f64, Vec<f64>)>>
}
```
`step` 内で `y_n` の控え（ws ではなく stepper 内のローカル or 追加バッファ）を持ち、
`newton_be(backend, problem, t0+h, h, &y_n, y, ws, &self.newton)` を呼ぶ。
`y_n` 用バッファも `integrate*` の冒頭で 1 回だけ確保（ループ内で確保しない）。

### 5.4 `src/lib.rs`

- クレートドキュメント（//! で §0,§1 の要約 + 簡単な使用例 + f32/GPU 注意）。
- `pub mod error; pub mod backend; pub mod problem; pub mod linsolve; pub mod newton; pub mod stepper;`
- re-export: `pub use backend::{Backend, BackendKind, CpuBackend, recommend_backend};`
  `#[cfg(feature="gpu")] pub use backend::GpuBackend;`
  `pub use problem::{OdeProblem, DenseJacobian};`
  `pub use stepper::BackwardEuler;`
  `pub use newton::{NewtonParams, NewtonWorkspace};`
  `pub use error::{SolveError, Result};`

### 5.5 examples（**CpuBackend 具象**で実装。`Vec<f64>` を直接添字アクセスして良い）

- `examples/decay.rs`: `dy/dt = -λy`, `y(0)=1`, 解析解 `e^{-λt}` と比較して誤差表示。
- `examples/heat1d.rs`: 1次元熱伝導 `f_i = α (y_{i-1} − 2y_i + y_{i+1})/dx²`,
  Dirichlet 境界（端は 0）。`N` を大きく（例 100_000）して **行列フリー＋rayon** が効くことを示す。
  剛性が高く後退オイラーが活きる題材。各ステップの Newton/BiCGStab 反復数を表示。
- `examples/auto_backend.rs`: `recommend_backend(dim)` を呼び、選ばれた種類を表示して小問題を解く。
  GPU 経路は `#[cfg(feature="gpu")]` でガード。
- 注: **任意の非線形 f を GPU で走らせるのは v0.1 のスコープ外**（GPU 経路は BLAS-1 カーネル / 将来の affine-f 用）。examples は CPU。

### 5.6 `tests/convergence.rs`（CpuBackend）

- **1次収束**: `dy/dt=-y`, `y(0)=1` を `h` と `h/2` で `t=1` まで積分し、
  グローバル誤差が概ね半減（比が 1.7〜2.3 程度）かつ `h=1/100` で `|y−e^{-1}|<5e-3`。
- **A-安定性（剛性）**: `λ=1000`, `h=0.1`（`λh≫1`）でも発散せず、単調減少して 0 に近づくこと。
  （前進オイラーなら発散する領域で安定を確認）
- 余裕があれば dense 経路 (`lu_solve`/`newton_be_dense`) と matrix-free の解が一致することを 2×2 線形系で確認。

### 5.7 リポジトリ整備

- **README.md（英語）**: 概要 / インストール / 使用例（`decay` 相当の最小コード）/
  アーキテクチャ（Backend 抽象, matrix-free Newton–BiCGStab, O(N) メモリ, 事前確保ワークスペース,
  CPU(rayon,f64)・GPU(wgpu,f32 *experimental*））/ **メモリ収支表**（状態+ワークスペースで約 10·N 本の `f64`、行列なし）/
  バックエンド自動選択ヒューリスティクスと注意（f32 精度・転送オーバーヘッド）/
  §1 の散逸性（one-sided Lipschitz）による堅牢性の説明 /
  Status & Roadmap（CPU=動作/テスト済、GPU=experimental・未実機検証・f32、
  将来: f64 GPU 経路〔CUDA via `cust` or Vulkan `shaderFloat64`〕, GMRES(m), 適応ステップ/次数, スパースヤコビアン CSR）/
  ライセンス。冒頭に日本語 1 段落の概要を添えてよい。
- **LICENSE-MIT** と **LICENSE-APACHE**: 標準テキスト（著作権表記は西暦 2026 と適切な氏名 or プレースホルダ）。デュアルライセンス。
- **.gitignore**: `/target`, `Cargo.lock`（ライブラリなので無視）, `**/*.rs.bk`, OS/エディタ系。
- **.github/workflows/ci.yml**: stable で `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --all`。GPU feature は実機 GPU が無い CI では実行不可なので、`cargo build --features gpu` の
  **型チェックのみ**（`--no-run` 相当、あるいはコンパイル確認）を別ジョブにし、失敗しても良い `continue-on-error: true` で扱う。

---

## 6. 検証（実装後に必ず実行）

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings   # ただし gpu は環境次第
cargo test --all
cargo run --example decay
cargo run --example heat1d
# GPU 実機がある場合のみ:
cargo build --features gpu
```
警告ゼロのクリーンな状態にすること。`cargo test` が緑になるのを確認してから §7 へ。

## 7. Git / GitHub 公開

```bash
cd beuler
git init
git add -A
git commit -m "Initial commit: memory-efficient backward Euler ODE solver (CPU rayon / GPU wgpu)"
# gh CLI がある場合:
gh repo create beuler --public --source=. --remote=origin --push
# 無い場合は GitHub で空リポジトリを作成後:
#   git remote add origin git@github.com:USERNAME/beuler.git
#   git branch -M main
#   git push -u origin main
```
`Cargo.toml` の `repository = "https://github.com/YOUR_USERNAME/beuler"` を実際の URL に更新すること。

## 8. 厳守する注意点（要約）

1. **行列を作らない**。メモリ `O(N)`。時間ループ内でヒープ確保しない（ワークスペース再利用）。
2. **GPU は f32・experimental・wgpu 29 想定**。`request_adapter`/`request_device`/`device.poll` の
   API はバージョン差があるので `--features gpu` を実機ビルドする際に要調整。README で明記。
3. **examples は CpuBackend 具象**で `Vec` 直接アクセス可。任意非線形 f の GPU 実行は v0.1 スコープ外。
4. 数値の妥当性は §5.6 のテスト（1次収束・A 安定）で担保。
