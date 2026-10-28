//! Tunable search parameters, exposed as UCI spin options for SPSA.
//!
//! Every parameter is declared once in the `tunables!` table below:
//! UCI name, field, default, min, max, scale and the smallest SPSA step.
//! The UCI value is an integer; the search uses `value / scale` (so
//! `CPuct 150` means 1.5). Adding a parameter = one line in the table and
//! using `params.<field>` in the search.
//!
//! Exploration term of PUCT:
//!
//! ```text
//! U          = cpuct(N) · gini_scale · N^CPuctExponent · P / (1 + n)
//! cpuct(N)   = CPuct + CPuctFactor · ln((N + CPuctBase) / CPuctBase)
//! gini_scale = clamp(CPuctGiniBase − CPuctGiniFactor · ln(gini + 0.001), 0, CPuctGiniMax)
//! gini       = 1 − Σ prior²   (of the node's children)
//! ```
//!
//! Value of an unvisited child (first-play urgency), from the view of the
//! side to move at the parent:
//!
//! ```text
//! FPU = FpuParentWeight · Q(parent) − FpuReduction · √(prior mass of the visited children)
//! ```
//!
//! with separate `...Root` values at the root. Priors are
//! softmax(logit / PolicyTemp), `PolicyTempRoot` at the root.
//!
//! Defaults: cpuct 2.15 growing logarithmically (factor 2.82, base 18368),
//! exponent 0.5 = √N, no gini scaling; FPU = parent Q minus 0.44 · √(visited
//! prior mass) in the tree, minus 1.0 · √(...) at the root; policy
//! temperature 1.61. The original search (constant cpuct 1.5, unvisited
//! children at Q = 0, temperature 1) is CPuct 150, CPuctFactor 0, all FPU
//! values 0, both temperatures 100.
//!
//! The non-standard UCI command `spsa` prints the *current* values (after
//! any `setoption`) in OpenBench's SPSA input format
//! (`name, int, value, min, max, c_end, r_end`), ready to paste into a
//! tune: `c_end` is 7 % of the value (at least the table's minimum step).
//! After tuning, put the new defaults into the table.

/// OpenBench SPSA learning rate at the end of the tune (its usual default).
const R_END: f64 = 0.002;

macro_rules! tunables {
    ($( $(#[$doc:meta])* $uci:literal => $field:ident : default $def:literal, min $min:literal, max $max:literal, scale $scale:literal, step $c:literal; )*) => {
        /// Search parameters in search units (UCI value / scale).
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct SearchParams {
            $( $(#[$doc])* pub $field: f32, )*
        }

        impl Default for SearchParams {
            fn default() -> Self {
                SearchParams { $( $field: $def as f32 / $scale as f32, )* }
            }
        }

        /// (UCI name, default, min, max, scale, smallest SPSA step) of every parameter.
        pub const TUNABLES: &[(&str, i64, i64, i64, i64, f64)] = &[
            $( ($uci, $def, $min, $max, $scale, $c as f64), )*
        ];

        impl SearchParams {
            /// Current values in search units, in [`TUNABLES`] order.
            pub fn values(&self) -> Vec<f32> {
                vec![$( self.$field, )*]
            }

            /// Set a parameter from its UCI integer value (clamped to its
            /// range; OpenBench may send decimals, which are rounded).
            /// Returns false if `name` is not a search parameter.
            pub fn set(&mut self, name: &str, value: &str) -> bool {
                $(
                    if name.eq_ignore_ascii_case($uci) {
                        if let Ok(v) = value.trim().parse::<f64>() {
                            let v = (v.round() as i64).clamp($min, $max);
                            self.$field = v as f32 / $scale as f32;
                        } else {
                            println!("info string {}: bad value '{value}'", $uci);
                        }
                        return true;
                    }
                )*
                false
            }
        }
    };
}

tunables! {
    /// PUCT exploration constant (at small visit counts if CPuctFactor > 0).
    "CPuct" => cpuct: default 139, min 1, max 1000, scale 100, step 1;
    /// Power of the parent's visits in the exploration term (0.5 = √N).
    "CPuctExponent" => cpuct_exponent: default 50, min 10, max 120, scale 100, step 2;
    /// Logarithmic growth of cpuct with the parent's visits (0 = constant).
    "CPuctFactor" => cpuct_factor: default 282, min 0, max 2000, scale 100, step 5;
    "CPuctBase" => cpuct_base: default 18368, min 100, max 200000, scale 1, step 200;
    /// Gini scaling: more exploration where the policy is sharp.
    /// Base 1, factor 0 = off.
    "CPuctGiniBase" => cpuct_gini_base: default 100, min 0, max 1000, scale 100, step 3;
    "CPuctGiniFactor" => cpuct_gini_factor: default 0, min 0, max 1000, scale 100, step 5;
    "CPuctGiniMax" => cpuct_gini_max: default 300, min 0, max 2000, scale 100, step 5;
    /// FPU inside the tree: weight of the parent's Q (0 = unvisited
    /// children start from a draw) ...
    "FpuParentWeight" => fpu_parent_weight: default 100, min 0, max 200, scale 100, step 5;
    /// ... minus this times √(prior mass of the visited children).
    "FpuReduction" => fpu_reduction: default 43, min 0, max 400, scale 100, step 2;
    /// The same at the root.
    "FpuParentWeightRoot" => fpu_parent_weight_root: default 100, min 0, max 200, scale 100, step 5;
    "FpuReductionRoot" => fpu_reduction_root: default 87, min 0, max 600, scale 100, step 3;
    /// Softmax temperature of the policy priors inside the tree.
    "PolicyTemp" => policy_temp: default 162, min 25, max 800, scale 100, step 3;
    /// Softmax temperature of the policy priors at the root.
    "PolicyTempRoot" => policy_temp_root: default 212, min 25, max 2000, scale 100, step 3;
}

impl SearchParams {
    /// cpuct for a node with `visits` visits (without the gini scaling).
    pub fn cpuct_at(&self, visits: u32) -> f32 {
        self.cpuct + self.cpuct_factor * ((visits as f32 + self.cpuct_base) / self.cpuct_base).ln()
    }

    /// Exploration coefficient of a node: U = this · P / (1 + n).
    /// `gini` = 1 − Σ prior² of its children.
    pub fn explore_coef(&self, visits: u32, gini: f32) -> f32 {
        let gini_scale = (self.cpuct_gini_base - self.cpuct_gini_factor * (gini + 0.001).ln())
            .clamp(0.0, self.cpuct_gini_max);
        self.cpuct_at(visits) * gini_scale * (visits.max(1) as f32).powf(self.cpuct_exponent)
    }
}

/// `option name X type spin default D min A max B` for every parameter.
pub fn uci_options() -> Vec<String> {
    TUNABLES
        .iter()
        .map(|(name, def, min, max, _, _)| {
            format!("option name {name} type spin default {def} min {min} max {max}")
        })
        .collect()
}

/// The parameters with their current values in OpenBench's SPSA input
/// format, one per line; c_end = 7 % of the value, at least the minimum step.
pub fn spsa_lines(params: &SearchParams) -> Vec<String> {
    TUNABLES
        .iter()
        .zip(params.values())
        .map(|(&(name, _, min, max, scale, step), v)| {
            let value = (v as f64 * scale as f64).round() as i64;
            let c = (0.07 * value.abs() as f64 * 10.0).round() / 10.0;
            format!(
                "{name}, int, {value}, {min}, {max}, {}, {R_END}",
                c.max(step)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(values: &[(&str, &str)]) -> SearchParams {
        let mut p = SearchParams::default();
        for (k, v) in values {
            assert!(p.set(k, v), "{k}");
        }
        p
    }

    #[test]
    fn defaults() {
        let p = SearchParams::default();
        assert_eq!(p.cpuct, 2.15);
        assert_eq!(p.policy_temp, 1.61);
        assert_eq!((p.fpu_parent_weight, p.fpu_reduction), (1.0, 0.44));
        assert_eq!((p.fpu_parent_weight_root, p.fpu_reduction_root), (1.0, 1.0));
        // cpuct(N) · √N, independent of the policy's shape (gini off)
        for (n, gini) in [(1u32, 0.0f32), (400, 0.3), (10_000, 0.9)] {
            let want = p.cpuct_at(n) * (n as f32).sqrt();
            assert!((p.explore_coef(n, gini) - want).abs() < 1e-3 * want);
        }
        // the log term is ~0 at small N and grows with N
        assert!((p.cpuct_at(0) - 2.15).abs() < 1e-6);
        assert!(p.cpuct_at(1000) > p.cpuct_at(10) && p.cpuct_at(100_000) > 5.0);
    }

    #[test]
    fn original_search_is_reachable() {
        let p = with(&[
            ("CPuct", "150"),
            ("CPuctFactor", "0"),
            ("FpuParentWeight", "0"),
            ("FpuReduction", "0"),
            ("FpuParentWeightRoot", "0"),
            ("FpuReductionRoot", "0"),
            ("PolicyTemp", "100"),
            ("PolicyTempRoot", "100"),
        ]);
        assert!((p.explore_coef(400, 0.5) - 1.5 * 20.0).abs() < 1e-4);
        assert_eq!(p.policy_temp, 1.0);
    }

    #[test]
    fn gini_scaling_and_exponent() {
        // gini scaling: a sharp policy (low gini) explores more, capped at max
        let g = with(&[
            ("CPuctFactor", "0"),
            ("CPuctGiniBase", "68"),
            ("CPuctGiniFactor", "163"),
            ("CPuctGiniMax", "210"),
        ]);
        assert!(g.explore_coef(100, 0.1) > g.explore_coef(100, 0.8));
        assert!((g.explore_coef(100, 0.0) - 2.15 * 2.1 * 10.0).abs() < 1e-3);
        // exponent: N^0.84 instead of √N
        let e = with(&[("CPuctFactor", "0"), ("CPuctExponent", "84")]);
        assert!((e.explore_coef(1000, 0.5) - 2.15 * 1000f32.powf(0.84)).abs() < 1e-2);
    }

    #[test]
    fn set_parses_rounds_and_clamps() {
        let mut p = SearchParams::default();
        assert!(p.set("cpuct", "300"));
        assert_eq!(p.cpuct, 3.0);
        assert!(p.set("CPuct", "149.6")); // OpenBench sends decimals
        assert_eq!(p.cpuct, 1.5);
        assert!(p.set("FpuReduction", "100000"));
        assert_eq!(p.fpu_reduction, 4.0);
        assert!(p.set("PolicyTempRoot", "junk")); // known name, bad value: unchanged
        assert_eq!(p.policy_temp_root, 1.61);
        assert!(!p.set("Hash", "16"));
    }

    #[test]
    fn every_default_is_inside_its_range_and_listed() {
        for &(name, def, min, max, scale, c) in TUNABLES {
            assert!(min <= def && def <= max, "{name}");
            assert!(scale > 0 && c > 0.0 && c < (max - min) as f64, "{name}");
        }
        assert_eq!(uci_options().len(), TUNABLES.len());
        let lines = spsa_lines(&SearchParams::default());
        assert_eq!(lines.len(), TUNABLES.len());
        assert_eq!(lines[0], "CPuct, int, 215, 1, 1000, 15.1, 0.002");
        assert!(lines.contains(&"CPuctGiniFactor, int, 0, 0, 1000, 5, 0.002".to_string()));
    }
}
