//! Pack the CPU Scene + materials into flat, std430-friendly POD structs.

use bytemuck::{Pod, Zeroable};

/// A primitive for the GPU. `kind`: 0 = sphere, 1 = convex solid.
/// Spheres use center/radius; solids index a plane range in the planes buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPrimitive {
    pub center: [f32; 3],
    pub radius: f32,
    pub plane_start: u32,
    pub plane_count: u32,
    pub kind: u32,
    pub material: u32, // index into the materials buffer
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub reflectance: f32, // Lambertian albedo
    pub glass: u32,       // 0 = BK7, 1 = SF11, 2 = Water (matches Glass enum order)
    pub kind: u32,        // 0 = Lambertian, 1 = Dielectric
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPlane {
    pub normal: [f32; 3],
    pub d: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structs_are_std430_sized() {
        // std430 storage structs: sizes are multiples of their alignment (16 here).
        assert_eq!(std::mem::size_of::<GpuPrimitive>() % 16, 0);
        assert_eq!(std::mem::size_of::<GpuMaterial>() % 16, 0);
        assert_eq!(std::mem::size_of::<GpuPlane>(), 16);
    }
}
