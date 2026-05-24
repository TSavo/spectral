//! GPU mirror of the CPU spectral path tracer.

/// A headless wgpu device + queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Request a headless adapter/device. Returns None if no GPU is available
    /// (so tests skip cleanly rather than fail on a headless CI box).
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;
            Some(GpuContext { device, queue })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_initializes_or_skips() {
        match GpuContext::new() {
            Some(_) => { /* GPU present (this Mac has Metal) */ }
            None => eprintln!("no GPU adapter; skipping GPU tests"),
        }
    }
}
