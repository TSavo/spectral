//! CIE 1931 sensor: color-matching functions, illuminant SPDs, XYZ integration,
//! XYZ->linear sRGB, and tone mapping. Loaded from data/*.csv via include_str!.

use crate::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

const CMF_CSV: &str = include_str!("../data/cie_1931_2deg.csv");
const D65_CSV: &str = include_str!("../data/illuminant_d65.csv");
const A_CSV: &str = include_str!("../data/illuminant_a.csv");

#[derive(Clone, Copy)]
pub enum Illuminant {
    D65,
    A,
}

/// A tabulated curve sampled at uniform `step` from `start`.
struct Table {
    start: f32,
    step: f32,
    values: Vec<[f32; 3]>, // up to 3 columns; SPDs use only col 0
}

impl Table {
    fn parse(csv: &str, cols: usize) -> Self {
        let mut start = f32::NAN;
        let mut step = f32::NAN;
        let mut last_nm = f32::NAN;
        let mut values = Vec::new();
        for line in csv.lines().filter(|l| !l.trim().is_empty()) {
            let mut it = line.split(',');
            let nm: f32 = it.next().unwrap().trim().parse().unwrap();
            let mut row = [0.0; 3];
            for slot in row.iter_mut().take(cols) {
                *slot = it.next().unwrap().trim().parse().unwrap();
            }
            if start.is_nan() {
                start = nm;
            } else if step.is_nan() {
                step = nm - last_nm;
            }
            last_nm = nm;
            values.push(row);
        }
        Table { start, step, values }
    }

    fn sample(&self, nm: f32) -> [f32; 3] {
        let f = ((nm - self.start) / self.step).clamp(0.0, (self.values.len() - 1) as f32);
        let i = f.floor() as usize;
        let frac = f - i as f32;
        if i + 1 >= self.values.len() {
            return self.values[self.values.len() - 1];
        }
        let a = self.values[i];
        let b = self.values[i + 1];
        [
            a[0] + (b[0] - a[0]) * frac,
            a[1] + (b[1] - a[1]) * frac,
            a[2] + (b[2] - a[2]) * frac,
        ]
    }
}

pub struct Sensor {
    cmf: Table,
    d65: Table,
    a: Table,
}

impl Sensor {
    pub fn new() -> Self {
        Sensor {
            cmf: Table::parse(CMF_CSV, 3),
            d65: Table::parse(D65_CSV, 1),
            a: Table::parse(A_CSV, 1),
        }
    }

    pub fn cmf(&self, nm: f32) -> (f32, f32, f32) {
        let v = self.cmf.sample(nm);
        (v[0], v[1], v[2])
    }

    pub fn illuminant(&self, ill: Illuminant, nm: f32) -> f32 {
        match ill {
            Illuminant::D65 => self.d65.sample(nm)[0],
            Illuminant::A => self.a.sample(nm)[0],
        }
    }

    /// XYZ of a perfect (reflectance=1) white under `ill`, normalized so Y=1.
    pub fn integrate_illuminant(&self, ill: Illuminant) -> [f32; 3] {
        let step = 5.0_f32;
        let (mut x, mut y, mut z, mut k) = (0.0, 0.0, 0.0, 0.0);
        let mut nm = LAMBDA_MIN;
        while nm <= LAMBDA_MAX {
            let s = self.illuminant(ill, nm);
            let (xb, yb, zb) = self.cmf(nm);
            x += s * xb * step;
            y += s * yb * step;
            z += s * zb * step;
            k += s * yb * step;
            nm += step;
        }
        [x / k, y / k, z / k]
    }
}

impl Default for Sensor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn chromaticity(xyz: [f32; 3]) -> (f32, f32) {
    let sum = xyz[0] + xyz[1] + xyz[2];
    (xyz[0] / sum, xyz[1] / sum)
}

/// XYZ (D65) to linear sRGB via the standard matrix.
pub fn xyz_to_linear_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ]
}

/// Reinhard tone map + sRGB gamma encode, returning 8-bit channels.
pub fn tonemap_to_u8(linear: [f32; 3], exposure: f32) -> [u8; 3] {
    let mut out = [0u8; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let v = (linear[c] * exposure).max(0.0);
        let mapped = v / (1.0 + v);
        let gamma = if mapped <= 0.0031308 {
            12.92 * mapped
        } else {
            1.055 * mapped.powf(1.0 / 2.4) - 0.055
        };
        *slot = (gamma.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn d65_white_point_chromaticity() {
        let sensor = Sensor::new();
        let xyz = sensor.integrate_illuminant(Illuminant::D65);
        let (x, y) = chromaticity(xyz);
        assert_relative_eq!(x, 0.3127, epsilon = 2e-3);
        assert_relative_eq!(y, 0.3290, epsilon = 2e-3);
    }

    #[test]
    fn cmf_and_illuminant_sampling_in_band() {
        let sensor = Sensor::new();
        let (x, y, z) = sensor.cmf(555.0);
        assert!(x >= 0.0 && y > 0.0 && z >= 0.0);
        assert!(sensor.illuminant(Illuminant::D65, 555.0) > 0.0);
    }

    #[test]
    fn xyz_to_srgb_white_is_unit() {
        let rgb = xyz_to_linear_srgb([0.9505, 1.0, 1.0890]);
        assert_relative_eq!(rgb[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(rgb[1], 1.0, epsilon = 1e-2);
        assert_relative_eq!(rgb[2], 1.0, epsilon = 1e-2);
    }
}
