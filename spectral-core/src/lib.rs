//! spectral-core: CPU spectral path tracer.

/// CIE 1931 XYZ tristimulus, in channel order [X, Y, Z].
pub type Xyz = [f32; 3];

/// Per-pixel running sum of XYZ contributions plus the sample count.
pub struct AccumBuffer {
    pub width: usize,
    pub height: usize,
    pub(crate) sum: Vec<Xyz>,
    pub(crate) samples: u32,
}

impl AccumBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        AccumBuffer { width, height, sum: vec![[0.0; 3]; width * height], samples: 0 }
    }
    pub fn reset(&mut self) {
        self.sum.iter_mut().for_each(|p| *p = [0.0; 3]);
        self.samples = 0;
    }
    /// Mean XYZ per pixel.
    pub fn mean(&self) -> Vec<Xyz> {
        let n = self.samples.max(1) as f32;
        self.sum.iter().map(|s| [s[0] / n, s[1] / n, s[2] / n]).collect()
    }
}

/// Backend-agnostic renderer. CPU now; GPU and wave solver attach later.
pub trait Renderer {
    fn accumulate(&mut self, samples_per_pixel: u32);
    fn buffer(&self) -> &AccumBuffer;
    fn reset(&mut self);
}

pub mod camera;
pub mod cie;
pub mod geom;
pub mod material;
pub mod optics;
pub mod rng;
pub mod scene;
pub mod sellmeier;
pub mod spectrum;
pub mod tracer;

#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
