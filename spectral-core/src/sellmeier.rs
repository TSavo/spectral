//! Sellmeier index of refraction n(λ). λ in nanometers at the API boundary.

/// Glasses with published Sellmeier coefficients.
#[derive(Clone, Copy, Debug)]
pub enum Glass {
    Bk7,
    Sf11,
}

impl Glass {
    /// (B1,B2,B3, C1,C2,C3) with C in micrometers^2.
    fn coeffs(self) -> ([f64; 3], [f64; 3]) {
        match self {
            Glass::Bk7 => (
                [1.03961212, 0.231792344, 1.01046945],
                [0.00600069867, 0.0200179144, 103.560653],
            ),
            Glass::Sf11 => (
                [1.73759695, 0.313747346, 1.89878101],
                [0.013188707, 0.0623068142, 155.23629],
            ),
        }
    }

    /// Index of refraction at wavelength `lambda_nm`.
    pub fn n(self, lambda_nm: f32) -> f32 {
        let l_um = (lambda_nm as f64) / 1000.0; // nm -> micrometers
        let l2 = l_um * l_um;
        let (b, c) = self.coeffs();
        let n2 = 1.0
            + b.iter()
                .zip(c.iter())
                .map(|(bk, ck)| bk * l2 / (l2 - ck))
                .sum::<f64>();
        n2.sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bk7_matches_published_lines() {
        // F, d, C Fraunhofer lines. Tolerance 5e-4 absolute index units.
        assert_relative_eq!(Glass::Bk7.n(486.1), 1.52238, epsilon = 5e-4);
        assert_relative_eq!(Glass::Bk7.n(587.6), 1.51680, epsilon = 5e-4);
        assert_relative_eq!(Glass::Bk7.n(656.3), 1.51432, epsilon = 5e-4);
    }

    #[test]
    fn dispersion_is_normal() {
        assert!(Glass::Bk7.n(450.0) > Glass::Bk7.n(650.0));
        assert!(Glass::Sf11.n(450.0) > Glass::Sf11.n(650.0));
    }

    #[test]
    fn sf11_disperses_more_than_bk7() {
        let spread = |g: Glass| g.n(450.0) - g.n(650.0);
        assert!(spread(Glass::Sf11) > spread(Glass::Bk7));
    }
}
