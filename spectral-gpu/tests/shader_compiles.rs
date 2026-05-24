use spectral_gpu::GpuContext;

#[test]
fn trace_shader_compiles() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let src = concat!(
        include_str!("../src/shaders/rng.wgsl"),
        include_str!("../src/shaders/trace.wgsl"),
    );
    let _m = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    // Poll to surface any async validation errors from the driver
    ctx.device.poll(wgpu::Maintain::Wait);
}
