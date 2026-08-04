// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Generates a stereo convolution function (single coeff set, two input channels).
///
/// Parameters include SIMD intrinsics as closures and step/alignment constants.
/// The `$attr` fragment captures `#[doc]` and `#[target_feature]` attributes.
#[macro_export]
macro_rules! impl_convolve_stereo {
    (
        $(#[$attr:meta])*
        $fn_name:ident,
        $step_dbl:expr, $half_step:expr, $align:expr,
        $zero:expr,
        $loadu:expr,
        $fmadd:expr,
        $add:expr,
        $reduce:expr
    ) => {
        $(#[$attr])*
        pub unsafe fn $fn_name(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            debug_assert!(
                (coeffs as usize).is_multiple_of($align),
                "coeffs must be {}-byte aligned",
                $align
            );
            let mut sum_l0 = $zero();
            let mut sum_l1 = $zero();
            let mut sum_r0 = $zero();
            let mut sum_r1 = $zero();
            let mut i = 0;

            while i + $step_dbl <= taps {
                let h0 = $loadu(coeffs.add(i));
                let x0_l = $loadu(input_l.add(i));
                let x0_r = $loadu(input_r.add(i));
                sum_l0 = $fmadd(h0, x0_l, sum_l0);
                sum_r0 = $fmadd(h0, x0_r, sum_r0);

                let h1 = $loadu(coeffs.add(i + $half_step));
                let x1_l = $loadu(input_l.add(i + $half_step));
                let x1_r = $loadu(input_r.add(i + $half_step));
                sum_l1 = $fmadd(h1, x1_l, sum_l1);
                sum_r1 = $fmadd(h1, x1_r, sum_r1);

                i += $step_dbl;
            }

            while i + $half_step <= taps {
                let h = $loadu(coeffs.add(i));
                let x_l = $loadu(input_l.add(i));
                let x_r = $loadu(input_r.add(i));
                sum_l0 = $fmadd(h, x_l, sum_l0);
                sum_r0 = $fmadd(h, x_r, sum_r0);
                i += $half_step;
            }

            let sum_l = $add(sum_l0, sum_l1);
            let sum_r = $add(sum_r0, sum_r1);
            let mut out_l = $reduce(sum_l);
            let mut out_r = $reduce(sum_r);

            while i < taps {
                let h = *coeffs.add(i);
                out_l += h * *input_l.add(i);
                out_r += h * *input_r.add(i);
                i += 1;
            }

            (out_l, out_r)
        }
    };
}

/// Generates a stereo dual convolution function (two coeff sets, two input channels).
///
/// Loads input samples once and applies them to both coefficient sets.
/// Uses double unrolling for both architectures.
#[macro_export]
macro_rules! impl_convolve_stereo_dual {
    (
        $(#[$attr:meta])*
        $fn_name:ident,
        $step_dbl:expr, $half_step:expr, $align:expr,
        $zero:expr,
        $loadu:expr,
        $fmadd:expr,
        $add:expr,
        $reduce:expr
    ) => {
        $(#[$attr])*
        pub unsafe fn $fn_name(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            debug_assert!(
                (coeffs0 as usize).is_multiple_of($align),
                "coeffs0 must be {}-byte aligned",
                $align
            );
            debug_assert!(
                (coeffs1 as usize).is_multiple_of($align),
                "coeffs1 must be {}-byte aligned",
                $align
            );
            let mut sum0_l0 = $zero();
            let mut sum0_r0 = $zero();
            let mut sum0_l1 = $zero();
            let mut sum0_r1 = $zero();

            let mut sum1_l0 = $zero();
            let mut sum1_r0 = $zero();
            let mut sum1_l1 = $zero();
            let mut sum1_r1 = $zero();

            let mut i = 0;

            while i + $step_dbl <= taps {
                let x0_l = $loadu(input_l.add(i));
                let x0_r = $loadu(input_r.add(i));

                let h0_0 = $loadu(coeffs0.add(i));
                sum0_l0 = $fmadd(h0_0, x0_l, sum0_l0);
                sum0_r0 = $fmadd(h0_0, x0_r, sum0_r0);

                let h1_0 = $loadu(coeffs1.add(i));
                sum1_l0 = $fmadd(h1_0, x0_l, sum1_l0);
                sum1_r0 = $fmadd(h1_0, x0_r, sum1_r0);

                let x1_l = $loadu(input_l.add(i + $half_step));
                let x1_r = $loadu(input_r.add(i + $half_step));

                let h0_1 = $loadu(coeffs0.add(i + $half_step));
                sum0_l1 = $fmadd(h0_1, x1_l, sum0_l1);
                sum0_r1 = $fmadd(h0_1, x1_r, sum0_r1);

                let h1_1 = $loadu(coeffs1.add(i + $half_step));
                sum1_l1 = $fmadd(h1_1, x1_l, sum1_l1);
                sum1_r1 = $fmadd(h1_1, x1_r, sum1_r1);

                i += $step_dbl;
            }

            while i + $half_step <= taps {
                let x_l = $loadu(input_l.add(i));
                let x_r = $loadu(input_r.add(i));

                let h0 = $loadu(coeffs0.add(i));
                sum0_l0 = $fmadd(h0, x_l, sum0_l0);
                sum0_r0 = $fmadd(h0, x_r, sum0_r0);

                let h1 = $loadu(coeffs1.add(i));
                sum1_l0 = $fmadd(h1, x_l, sum1_l0);
                sum1_r0 = $fmadd(h1, x_r, sum1_r0);

                i += $half_step;
            }

            let sum0_l = $add(sum0_l0, sum0_l1);
            let sum0_r = $add(sum0_r0, sum0_r1);
            let sum1_l = $add(sum1_l0, sum1_l1);
            let sum1_r = $add(sum1_r0, sum1_r1);

            let mut out0_l = $reduce(sum0_l);
            let mut out0_r = $reduce(sum0_r);
            let mut out1_l = $reduce(sum1_l);
            let mut out1_r = $reduce(sum1_r);

            while i < taps {
                let h0 = *coeffs0.add(i);
                let h1 = *coeffs1.add(i);
                let xl = *input_l.add(i);
                let xr = *input_r.add(i);
                out0_l += h0 * xl;
                out0_r += h0 * xr;
                out1_l += h1 * xl;
                out1_r += h1 * xr;
                i += 1;
            }

            ((out0_l, out0_r), (out1_l, out1_r))
        }
    };
}

/// Generates a mono dual convolution function (two coeff sets, one input channel).
///
/// Loads input samples once and applies them to both coefficient sets.
/// Uses double unrolling for both architectures.
#[macro_export]
macro_rules! impl_convolve_mono_dual {
    (
        $(#[$attr:meta])*
        $fn_name:ident,
        $step_dbl:expr, $half_step:expr, $align:expr,
        $zero:expr,
        $loadu:expr,
        $fmadd:expr,
        $add:expr,
        $reduce:expr
    ) => {
        $(#[$attr])*
        pub unsafe fn $fn_name(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            debug_assert!(
                (coeffs0 as usize).is_multiple_of($align),
                "coeffs0 must be {}-byte aligned",
                $align
            );
            debug_assert!(
                (coeffs1 as usize).is_multiple_of($align),
                "coeffs1 must be {}-byte aligned",
                $align
            );
            let mut sum0_0 = $zero();
            let mut sum0_1 = $zero();
            let mut sum1_0 = $zero();
            let mut sum1_1 = $zero();
            let mut i = 0;

            while i + $step_dbl <= taps {
                let x0 = $loadu(input.add(i));

                let h0_0 = $loadu(coeffs0.add(i));
                sum0_0 = $fmadd(h0_0, x0, sum0_0);

                let h1_0 = $loadu(coeffs1.add(i));
                sum1_0 = $fmadd(h1_0, x0, sum1_0);

                let x1 = $loadu(input.add(i + $half_step));

                let h0_1 = $loadu(coeffs0.add(i + $half_step));
                sum0_1 = $fmadd(h0_1, x1, sum0_1);

                let h1_1 = $loadu(coeffs1.add(i + $half_step));
                sum1_1 = $fmadd(h1_1, x1, sum1_1);

                i += $step_dbl;
            }

            while i + $half_step <= taps {
                let x = $loadu(input.add(i));

                let h0 = $loadu(coeffs0.add(i));
                sum0_0 = $fmadd(h0, x, sum0_0);

                let h1 = $loadu(coeffs1.add(i));
                sum1_0 = $fmadd(h1, x, sum1_0);

                i += $half_step;
            }

            let sum0 = $add(sum0_0, sum0_1);
            let sum1 = $add(sum1_0, sum1_1);

            let mut out0 = $reduce(sum0);
            let mut out1 = $reduce(sum1);

            while i < taps {
                let xl = *input.add(i);
                out0 += *coeffs0.add(i) * xl;
                out1 += *coeffs1.add(i) * xl;
                i += 1;
            }

            (out0, out1)
        }
    };
}

/// Generates a mono convolution function (single coeff set, one input channel).
///
/// Uses double unrolling for both architectures.
#[macro_export]
macro_rules! impl_convolve_mono {
    (
        $(#[$attr:meta])*
        $fn_name:ident,
        $step_dbl:expr, $half_step:expr, $align:expr,
        $zero:expr,
        $loadu:expr,
        $fmadd:expr,
        $add:expr,
        $reduce:expr
    ) => {
        $(#[$attr])*
        pub unsafe fn $fn_name(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            debug_assert!(
                (coeffs as usize).is_multiple_of($align),
                "coeffs must be {}-byte aligned",
                $align
            );
            let mut sum0 = $zero();
            let mut sum1 = $zero();
            let mut i = 0;

            while i + $step_dbl <= taps {
                let h0 = $loadu(coeffs.add(i));
                let x0 = $loadu(input.add(i));
                sum0 = $fmadd(h0, x0, sum0);

                let h1 = $loadu(coeffs.add(i + $half_step));
                let x1 = $loadu(input.add(i + $half_step));
                sum1 = $fmadd(h1, x1, sum1);

                i += $step_dbl;
            }

            while i + $half_step <= taps {
                let h = $loadu(coeffs.add(i));
                let x = $loadu(input.add(i));
                sum0 = $fmadd(h, x, sum0);
                i += $half_step;
            }

            let sum = $add(sum0, sum1);
            let mut out = $reduce(sum);

            while i < taps {
                let h = *coeffs.add(i);
                out += h * *input.add(i);
                i += 1;
            }

            out
        }
    };
}
