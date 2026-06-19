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
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
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
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
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
        // NOTE: `poll` argument type has changed across wgpu versions
        // (Maintain::Wait vs PollType::Wait). Adjust if your version differs.
        let _ = self.device.poll(wgpu::Maintain::Wait);
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
