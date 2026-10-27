//! Tunable search parameters, exposed as UCI spin options for SPSA.
//!
//! Every parameter is declared once in the `tunables!` table below:
//! UCI name, field, default, min, max, scale and the SPSA step `c_end`.
//! The UCI value is an integer; the search uses `value / scale` (so
//! `CPuct 215` means 2.15). Adding a parameter = one line in the table and
//! using `params.<field>` in the search.
//!
//! The non-standard UCI command `spsa` prints the table in OpenBench's SPSA
//! input format (`name, int, default, min, max, c_end, r_end`), ready to
//! paste into a tuning workload. After tuning, put the new defaults into
//! the table.

/// OpenBench SPSA learning rate at the end of the tune (its usual default).
const R_END: f64 = 0.002;

macro_rules! tunables {
    ($( $(#[$doc:meta])* $uci:literal => $field:ident : default $def:literal, min $min:literal, max $max:literal, scale $scale:literal, c_end $c:literal; )*) => {
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

        /// (UCI name, default, min, max, scale, c_end) of every parameter.
        pub const TUNABLES: &[(&str, i64, i64, i64, i64, f64)] = &[
            $( ($uci, $def, $min, $max, $scale, $c as f64), )*
        ];

        impl SearchParams {
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
    /// PUCT exploration constant at small visit counts.
    "CPuct" => cpuct: default 215, min 50, max 600, scale 100, c_end 15;
    /// Growth of cpuct with the parent's visits:
    /// cpuct(N) = CPuct + CPuctFactor · ln((N + CPuctBase) / CPuctBase).
    "CPuctFactor" => cpuct_factor: default 282, min 0, max 1000, scale 100, c_end 30;
    "CPuctBase" => cpuct_base: default 18368, min 500, max 100000, scale 1, c_end 2000;
    /// First-play urgency: an unvisited child is scored
    /// Q(parent) − FpuReduction · √(policy of the visited children).
    "FpuReduction" => fpu_reduction: default 44, min 0, max 200, scale 100, c_end 6;
    /// The same at the root.
    "FpuReductionRoot" => fpu_reduction_root: default 100, min 0, max 300, scale 100, c_end 10;
    /// Softmax temperature of the policy priors inside the tree.
    "PolicyTemp" => policy_temp: default 161, min 50, max 400, scale 100, c_end 12;
    /// Softmax temperature of the policy priors at the root.
    "PolicyTempRoot" => policy_temp_root: default 161, min 50, max 600, scale 100, c_end 15;
}

impl SearchParams {
    /// cpuct for a node with `visits` visits.
    pub fn cpuct_at(&self, visits: u32) -> f32 {
        self.cpuct + self.cpuct_factor * ((visits as f32 + self.cpuct_base) / self.cpuct_base).ln()
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

/// The parameters in OpenBench's SPSA input format, one per line.
pub fn spsa_lines() -> Vec<String> {
    TUNABLES
        .iter()
        .map(|(name, def, min, max, _, c)| {
            format!("{name}, int, {def}, {min}, {max}, {c}, {R_END}")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_scaling() {
        let p = SearchParams::default();
        assert_eq!(p.cpuct, 2.15);
        assert_eq!(p.cpuct_base, 18368.0);
        assert_eq!(p.policy_temp, 1.61);
        // the log term is ~0 at small N and grows with N
        assert!((p.cpuct_at(0) - 2.15).abs() < 1e-6);
        assert!(p.cpuct_at(1000) > p.cpuct_at(10) && p.cpuct_at(100_000) > 5.0);
    }

    #[test]
    fn set_parses_rounds_and_clamps() {
        let mut p = SearchParams::default();
        assert!(p.set("cpuct", "300"));
        assert_eq!(p.cpuct, 3.0);
        assert!(p.set("CPuct", "149.6")); // OpenBench sends decimals
        assert_eq!(p.cpuct, 1.5);
        assert!(p.set("FpuReduction", "100000"));
        assert_eq!(p.fpu_reduction, 2.0);
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
        assert_eq!(spsa_lines()[0], "CPuct, int, 215, 50, 600, 15, 0.002");
    }
}
