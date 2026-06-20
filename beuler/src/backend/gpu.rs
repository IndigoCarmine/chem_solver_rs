//! GPU backend via wgpu compute shaders. **Experimental.**
//!
//! ## Read this before relying on it
//! * **Precision is `f32`.** WGSL storage buffers are 32-bit floats — there is
//!   no `f64` in portable WGSL. For stiff problems this is often *not* enough.
//!   Treat this backend as suitable for large, non-stiff or precision-tolerant
//!   systems, and prefer [`CpuBackend`](super::CpuBackend) when you need `f64`.
//!   A true `f64` GPU path needs a different compute layer (CUDA via `cust`, or
//!   Vulkan with the `shaderFloat64` feature) — see the roadmap in the README.
//! * **Reductions run on the host.** `dot`/`norm2` currently download the
//!   buffer and sum in `f64`. The elementwise kernels (`axpy`, `scale`, …) run
//!   on device. An on-device tree reduction is the main TODO here.
//! * **Untested against hardware in this form.** The host glue targets
//!   `wgpu = 29`. Adapter/device request signatures and `Device::poll` have
//!   shifted across wgpu releases; verify against your exact version.
//!
//! All values cross the API boundary as `f64` (the [`Backend`] contract) and
//! are converted to/from `f32` at the buffer edge.

use super::Backend;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Fused elementwise kernel. One pipeline, mode-selected, to keep setup small.
const KERNEL_WGSL: &str = r#"
struct Params {
    n     : u32,
    mode  : u32,   // 0 fill, 1 scale, 2 axpy, 3 axpby, 4 copy
    alpha : f32,
    beta  : f32,
};
@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var<storage, read>       x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    switch p.mode {
        case 0u: { y[i] = p.alpha; }
        case 1u: { y[i] = p.alpha * y[i]; }
        case 2u: { y[i] = y[i] + p.alpha * x[i]; }
        case 3u: { y[i] = p.alpha * x[i] + p.beta * y[i]; }
        default: { y[i] = x[i]; }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    mode: u32,
    alpha: f32,
    beta: f32,
}

/// A device-resident `f32` vector plus its logical length.
pub struct GpuVec {
    buffer: wgpu::Buffer,
    len: usize,
}

impl Clone for GpuVec {
    fn clone(&self) -> Self {
        // Cloning would require a device copy; not needed by the solver's hot
        // path, so we forbid it loudly rather than silently allocate.
        panic!("GpuVec: clone is intentionally unsupported; reuse preallocated workspace vectors");
    }
}

/// GPU backend. Holds the device, queue, and the single compute pipeline.
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    dummy: wgpu::Buffer, // 1-element buffer bound as `x` when x is unused
}

/// Cheap probe: can we acquire any adapter at all?
pub fn is_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .is_ok()
}

impl GpuBackend {
    /// Initialise the backend (adapter → device → pipeline).
    ///
    /// Returns an error string if no adapter/device is available.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e:?}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("beuler-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .map_err(|e| format!("device request failed: {e:?}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beuler-kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(KERNEL_WGSL)),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("beuler-bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("beuler-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beuler-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beuler-dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            dummy,
        })
    }

    fn run(&self, params: Params, x: &wgpu::Buffer, y: &wgpu::Buffer) {
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("beuler-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beuler-bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("beuler-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (params.n + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Download an `f32` buffer to host. Used by reductions and `to_host`.
    fn read_f32(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beuler-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().expect("map channel").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Backend for GpuBackend {
    type Vector = GpuVec;

    fn zeros(&self, n: usize) -> GpuVec {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beuler-vec"),
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // create_buffer with mapped_at_creation:false is zero-initialised.
        GpuVec { buffer, len: n }
    }

    fn len(&self, v: &GpuVec) -> usize {
        v.len
    }

    fn from_host(&self, data: &[f64]) -> GpuVec {
        let as_f32: Vec<f32> = data.iter().map(|&d| d as f32).collect();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("beuler-vec-init"),
                contents: bytemuck::cast_slice(&as_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        GpuVec {
            buffer,
            len: data.len(),
        }
    }

    fn to_host(&self, v: &GpuVec) -> Vec<f64> {
        self.read_f32(&v.buffer, v.len)
            .into_iter()
            .map(|f| f as f64)
            .collect()
    }

    fn copy(&self, dst: &mut GpuVec, src: &GpuVec) {
        self.run(
            Params {
                n: dst.len as u32,
                mode: 4,
                alpha: 0.0,
                beta: 0.0,
            },
            &src.buffer,
            &dst.buffer,
        );
    }

    fn fill(&self, x: &mut GpuVec, value: f64) {
        self.run(
            Params {
                n: x.len as u32,
                mode: 0,
                alpha: value as f32,
                beta: 0.0,
            },
            &self.dummy,
            &x.buffer,
        );
    }

    fn scale(&self, x: &mut GpuVec, alpha: f64) {
        self.run(
            Params {
                n: x.len as u32,
                mode: 1,
                alpha: alpha as f32,
                beta: 0.0,
            },
            &self.dummy,
            &x.buffer,
        );
    }

    fn axpy(&self, alpha: f64, x: &GpuVec, y: &mut GpuVec) {
        self.run(
            Params {
                n: y.len as u32,
                mode: 2,
                alpha: alpha as f32,
                beta: 0.0,
            },
            &x.buffer,
            &y.buffer,
        );
    }

    fn axpby(&self, alpha: f64, x: &GpuVec, beta: f64, y: &mut GpuVec) {
        self.run(
            Params {
                n: y.len as u32,
                mode: 3,
                alpha: alpha as f32,
                beta: beta as f32,
            },
            &x.buffer,
            &y.buffer,
        );
    }

    fn dot(&self, x: &GpuVec, y: &GpuVec) -> f64 {
        // TODO: on-device reduction. For now, accumulate in f64 on the host.
        let xs = self.read_f32(&x.buffer, x.len);
        let ys = self.read_f32(&y.buffer, y.len);
        xs.iter().zip(ys).map(|(a, b)| *a as f64 * b as f64).sum()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GPU Expression DSL — Expr, StateRef, BoundaryCondition, GpuEquation
// ────────────────────────────────────────────────────────────────────────────

/// Expression tree for GPU-side `f(t, y)` evaluation.
///
/// Build with operator overloads (`+`, `-`, `*`, `/`, unary `-`) and helper
/// functions ([`sin`], [`cos`], [`exp`], …), then hand the closure to
/// [`GpuEquation::build`].
///
/// `f32` and `f64` scalars can be mixed freely with `Expr` values; `f64` is
/// narrowed to `f32` at the call site (matching the GPU precision).
#[derive(Clone, Debug)]
pub enum Expr {
    /// Scalar constant.
    Const(f32),
    /// Element `y[i + offset]`, boundary-padded by the chosen [`BoundaryCondition`].
    StateAt(i32),
    /// Current time `t` (as `f32`).
    Time,
    /// Thread index `i` (as `f32`).
    Index,
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    /// Natural logarithm (maps to WGSL `log`).
    Ln(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
}

impl From<f32> for Expr {
    fn from(v: f32) -> Self {
        Expr::Const(v)
    }
}
impl From<f64> for Expr {
    fn from(v: f64) -> Self {
        Expr::Const(v as f32)
    }
}

// ── arithmetic operator overloads ─────────────────────────────────────────

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl std::ops::$trait for Expr {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(self), Box::new(rhs))
            }
        }
        impl std::ops::$trait<f32> for Expr {
            type Output = Expr;
            fn $method(self, rhs: f32) -> Expr {
                Expr::$variant(Box::new(self), Box::new(Expr::Const(rhs)))
            }
        }
        impl std::ops::$trait<f64> for Expr {
            type Output = Expr;
            fn $method(self, rhs: f64) -> Expr {
                Expr::$variant(Box::new(self), Box::new(Expr::Const(rhs as f32)))
            }
        }
        impl std::ops::$trait<Expr> for f32 {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(Expr::Const(self)), Box::new(rhs))
            }
        }
        impl std::ops::$trait<Expr> for f64 {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(Expr::Const(self as f32)), Box::new(rhs))
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Sub, sub, Sub);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Div, div, Div);

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

// ── elementary functions ──────────────────────────────────────────────────

pub fn sin(x: Expr) -> Expr {
    Expr::Sin(Box::new(x))
}
pub fn cos(x: Expr) -> Expr {
    Expr::Cos(Box::new(x))
}
pub fn exp(x: Expr) -> Expr {
    Expr::Exp(Box::new(x))
}
pub fn ln(x: Expr) -> Expr {
    Expr::Ln(Box::new(x))
}
pub fn sqrt(x: Expr) -> Expr {
    Expr::Sqrt(Box::new(x))
}
pub fn abs(x: Expr) -> Expr {
    Expr::Abs(Box::new(x))
}
pub fn pow(base: Expr, e: Expr) -> Expr {
    Expr::Pow(Box::new(base), Box::new(e))
}

// ── StateRef ──────────────────────────────────────────────────────────────

/// State accessor provided to the closure in [`GpuEquation::build`].
pub struct StateRef;

impl StateRef {
    /// `y[i + offset]`, boundary-padded per the chosen [`BoundaryCondition`].
    pub fn at(&self, offset: i32) -> Expr {
        Expr::StateAt(offset)
    }
    /// Thread index `i` as `f32`.
    pub fn index(&self) -> Expr {
        Expr::Index
    }
    /// Current time `t` as `f32`.
    pub fn t(&self) -> Expr {
        Expr::Time
    }
}

// ── BoundaryCondition ────────────────────────────────────────────────────

/// How out-of-range stencil accesses are handled.
pub enum BoundaryCondition {
    /// Out-of-range reads return a constant (default `0.0`).
    Dirichlet(f32),
    /// Out-of-range reads wrap modulo `n`.
    Periodic,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Dirichlet(0.0)
    }
}

// ── WGSL code generation ─────────────────────────────────────────────────

fn offset_var(k: i32) -> String {
    match k.cmp(&0) {
        std::cmp::Ordering::Equal => "_s0".to_string(),
        std::cmp::Ordering::Greater => format!("_sp{k}"),
        std::cmp::Ordering::Less => format!("_sn{}", -k),
    }
}

fn const_to_wgsl(v: f32) -> String {
    if v.is_nan() {
        "0.0f".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "3.40282347e38f".to_string()
        } else {
            "-3.40282347e38f".to_string()
        }
    } else {
        // {:e} produces e.g. "1.5e3"; appending 'f' makes it a valid WGSL f32 literal.
        format!("{:e}f", v)
    }
}

fn collect_offsets(expr: &Expr, out: &mut std::collections::BTreeSet<i32>) {
    match expr {
        Expr::StateAt(k) => {
            out.insert(*k);
        }
        Expr::Const(_) | Expr::Time | Expr::Index => {}
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_offsets(a, out);
            collect_offsets(b, out);
        }
        Expr::Neg(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Sqrt(a)
        | Expr::Abs(a) => collect_offsets(a, out),
    }
}

fn expr_to_wgsl(expr: &Expr) -> String {
    match expr {
        Expr::Const(v) => const_to_wgsl(*v),
        Expr::StateAt(k) => offset_var(*k),
        Expr::Time => "p.t".to_string(),
        Expr::Index => "f32(i)".to_string(),
        Expr::Add(a, b) => format!("({} + {})", expr_to_wgsl(a), expr_to_wgsl(b)),
        Expr::Sub(a, b) => format!("({} - {})", expr_to_wgsl(a), expr_to_wgsl(b)),
        Expr::Mul(a, b) => format!("({} * {})", expr_to_wgsl(a), expr_to_wgsl(b)),
        Expr::Div(a, b) => format!("({} / {})", expr_to_wgsl(a), expr_to_wgsl(b)),
        Expr::Neg(a) => format!("(-{})", expr_to_wgsl(a)),
        Expr::Sin(a) => format!("sin({})", expr_to_wgsl(a)),
        Expr::Cos(a) => format!("cos({})", expr_to_wgsl(a)),
        Expr::Exp(a) => format!("exp({})", expr_to_wgsl(a)),
        Expr::Ln(a) => format!("log({})", expr_to_wgsl(a)),
        Expr::Sqrt(a) => format!("sqrt({})", expr_to_wgsl(a)),
        Expr::Abs(a) => format!("abs({})", expr_to_wgsl(a)),
        Expr::Pow(b, e) => format!("pow({}, {})", expr_to_wgsl(b), expr_to_wgsl(e)),
    }
}

/// Generate a complete WGSL compute shader that evaluates `expr` element-wise.
///
/// The shader entry point is `eval_main`. Bind group 0:
/// - binding 0: uniform `EvalParams { n: u32, t: f32, … }`
/// - binding 1: read-only `array<f32>` (input `y`)
/// - binding 2: read-write `array<f32>` (output)
pub fn generate_eval_wgsl(expr: &Expr, boundary: &BoundaryCondition) -> String {
    let mut offsets = std::collections::BTreeSet::new();
    collect_offsets(expr, &mut offsets);

    let mut prelude = String::new();
    for &k in &offsets {
        let var = offset_var(k);
        let line = match boundary {
            BoundaryCondition::Dirichlet(bc_val) => {
                let bc = const_to_wgsl(*bc_val);
                if k == 0 {
                    format!("    let {var} = y[i];\n")
                } else if k > 0 {
                    let ku = k as u32;
                    format!("    let {var} = select({bc}, y[i + {ku}u], i + {ku}u < p.n);\n")
                } else {
                    let ku = (-k) as u32;
                    format!("    let {var} = select({bc}, y[i - {ku}u], i >= {ku}u);\n")
                }
            }
            BoundaryCondition::Periodic => {
                if k == 0 {
                    format!("    let {var} = y[i];\n")
                } else if k > 0 {
                    let ku = k as u32;
                    format!("    let {var} = y[(i + {ku}u) % p.n];\n")
                } else {
                    let ku = (-k) as u32;
                    format!("    let {var} = y[(i + p.n - {ku}u) % p.n];\n")
                }
            }
        };
        prelude.push_str(&line);
    }

    let body = expr_to_wgsl(expr);
    format!(
        r#"struct EvalParams {{ n: u32, t: f32, _p0: u32, _p1: u32 }}
@group(0) @binding(0) var<uniform>             p:   EvalParams;
@group(0) @binding(1) var<storage, read>       y:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
{prelude}    out[i] = {body};
}}
"#
    )
}

// ── EvalParams — must mirror the WGSL struct layout exactly ─────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EvalParams {
    n: u32,
    t: f32,
    _p0: u32,
    _p1: u32,
}

// ── GpuEquation ──────────────────────────────────────────────────────────

/// A GPU-compiled ODE right-hand side.
///
/// At build time [`GpuEquation::build`] turns an [`Expr`] closure into a WGSL
/// compute shader and compiles it once. At solve time `eval` only writes 4
/// bytes (the current `t`) and dispatches the pre-compiled pipeline — no CPU
/// evaluation of `f` occurs.
///
/// Implements [`OdeProblem<GpuBackend>`](crate::problem::OdeProblem), so it
/// plugs directly into [`BackwardEuler`](crate::stepper::BackwardEuler).
pub struct GpuEquation {
    eval_pipeline: wgpu::ComputePipeline,
    eval_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    n: usize,
}

impl GpuEquation {
    /// Compile the closure `f(&StateRef) -> Expr` into a GPU compute shader.
    ///
    /// # Panics
    /// Panics if wgpu rejects the generated WGSL.
    pub fn build<F>(backend: &GpuBackend, n: usize, boundary: BoundaryCondition, f: F) -> Self
    where
        F: Fn(&StateRef) -> Expr,
    {
        let expr = f(&StateRef);
        let wgsl = generate_eval_wgsl(&expr, &boundary);

        let module = backend
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("beuler-eval"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
            });

        let eval_layout =
            backend
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("beuler-eval-bgl"),
                    entries: &[
                        bgl_entry(0, wgpu::BufferBindingType::Uniform),
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                    ],
                });

        let pipeline_layout =
            backend
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("beuler-eval-pl"),
                    bind_group_layouts: &[Some(&eval_layout)],
                    immediate_size: 0,
                });

        let eval_pipeline =
            backend
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("beuler-eval-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("eval_main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let params_buf = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beuler-eval-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let init = EvalParams {
            n: n as u32,
            t: 0.0,
            _p0: 0,
            _p1: 0,
        };
        backend
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&init));

        Self {
            eval_pipeline,
            eval_layout,
            params_buf,
            n,
        }
    }
}

impl crate::problem::OdeProblem<GpuBackend> for GpuEquation {
    fn dim(&self) -> usize {
        self.n
    }

    fn eval(&self, backend: &GpuBackend, t: f64, y: &GpuVec, out: &mut GpuVec) {
        // Update t in the pre-allocated uniform buffer (byte offset 4 = after n: u32).
        let t_f32 = t as f32;
        backend
            .queue
            .write_buffer(&self.params_buf, 4, bytemuck::bytes_of(&t_f32));

        let bind = backend
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("beuler-eval-bind"),
                layout: &self.eval_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out.buffer.as_entire_binding(),
                    },
                ],
            });

        let mut enc = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("beuler-eval-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.eval_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((self.n as u32 + 63) / 64, 1, 1);
        }
        backend.queue.submit(Some(enc.finish()));
    }
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_const_and_state() {
        let x = StateRef;
        let expr = Expr::Const(2.0_f32) * x.at(0);
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::Dirichlet(0.0));
        assert!(wgsl.contains("_s0"), "state at offset 0 in prelude");
        assert!(
            wgsl.contains("* _s0") || wgsl.contains("_s0 *"),
            "mul in body"
        );
    }

    #[test]
    fn wgsl_elementary_functions() {
        let x = StateRef;
        let expr = sin(x.at(0)) + exp(Expr::Time);
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::default());
        assert!(wgsl.contains("sin("), "sin");
        assert!(wgsl.contains("exp("), "exp");
        assert!(wgsl.contains("p.t"), "time variable");
    }

    #[test]
    fn wgsl_all_elementary() {
        let x = StateRef;
        let expr = cos(x.at(0)) + ln(x.at(0)) + sqrt(x.at(0)) + abs(x.at(0));
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::default());
        assert!(wgsl.contains("cos("), "cos");
        assert!(wgsl.contains("log("), "ln maps to log in WGSL");
        assert!(wgsl.contains("sqrt("), "sqrt");
        assert!(wgsl.contains("abs("), "abs");
    }

    #[test]
    fn wgsl_pow_and_index() {
        let x = StateRef;
        let expr = pow(x.at(0), Expr::from(2.0_f32)) + x.index();
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::default());
        assert!(wgsl.contains("pow("), "pow");
        assert!(wgsl.contains("f32(i)"), "index as f32");
    }

    #[test]
    fn wgsl_periodic_bc() {
        let x = StateRef;
        let expr = x.at(-1) - x.at(0) * 2.0_f32 + x.at(1);
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::Periodic);
        assert!(wgsl.contains("% p.n"), "modulo in periodic BC");
        assert!(!wgsl.contains("select("), "no select in periodic BC");
    }

    #[test]
    fn wgsl_dirichlet_bc() {
        let x = StateRef;
        let expr = x.at(-1) + x.at(1);
        let wgsl = generate_eval_wgsl(&expr, &BoundaryCondition::Dirichlet(0.0));
        assert!(wgsl.contains("select("), "select in dirichlet BC");
        assert!(!wgsl.contains("% p.n"), "no modulo in dirichlet BC");
    }

    #[test]
    fn operator_overloads_compile() {
        let x = StateRef;
        let _a = x.at(0) + x.at(1);
        let _b = x.at(0) - x.at(1);
        let _c = x.at(0) * 2.0_f32;
        let _d = x.at(0) * 2.0_f64;
        let _e = x.at(0) / 2.0_f32;
        let _f = x.at(0) / 2.0_f64;
        let _g = -x.at(0);
        let _h = 3.0_f32 + x.at(0);
        let _i = 3.0_f64 - x.at(0);
        let _j = 3.0_f32 * x.at(0);
        let _k = 3.0_f64 / x.at(0);
        let _l = Expr::from(1.0_f32) + Expr::from(2.0_f64);
    }
}
