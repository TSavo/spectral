use spectral_core::rng::PathRng;
use spectral_gpu::GpuContext;

const PARITY_WGSL: &str = concat!(
    include_str!("../src/shaders/rng.wgsl"),
    r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var r = rng_new(43981u, 4660u); // 0xABCD, 0x1234
    for (var i = 0u; i < 16u; i = i + 1u) {
        out[i] = rng_next_u32(&r);
    }
}
"#
);

#[test]
fn wgsl_rng_matches_cpu() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // CPU reference.
    let mut cpu = PathRng::new(0xABCD, 0x1234);
    let cpu_draws: Vec<u32> = (0..16).map(|_| cpu.next_u32()).collect();

    // GPU draws.
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rng_parity"),
        source: wgpu::ShaderSource::Wgsl(PARITY_WGSL.into()),
    });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 16 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: 16 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rng"),
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() }],
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 16 * 4);
    ctx.queue.submit([enc.finish()]);
    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let gpu_draws: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();

    assert_eq!(gpu_draws, cpu_draws, "WGSL RNG must match CPU bit-for-bit");
    // Sanity: the first four are the frozen anchor.
    assert_eq!(&gpu_draws[..4], &[0xa1dd5847, 0x13248a2d, 0x37278def, 0xde1d5b62]);
}
