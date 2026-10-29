//! Time management: the time budget of one move.
//!
//! * `soft`: the target time for this move: remaining time / moves to go
//!   + a share of the increment, at most half the remaining time.
//! * `hard`: the ceiling: a multiple of `soft`, at most 3/4 of the
//!   remaining time.
//!
//! The alpha-beta search stops starting new work after `soft`. The MCTS
//! scales `soft` by how settled its search is (see mcts.rs) and never
//! passes `hard`; its constants are tunable (`TimeConfig`).

use std::time::Duration;

use peazero_core::Color;

use crate::engine::SearchLimits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    pub soft: Duration,
    pub hard: Duration,
    /// `go movetime`: use exactly this time (no scaling of `soft`).
    pub fixed: bool,
}

/// The constants of the budget formula.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeConfig {
    /// Moves to go when the GUI doesn't send `movestogo`.
    pub moves_to_go: u64,
    /// Share of the increment added to `soft`.
    pub inc_share: f64,
    /// `hard` = this × `soft` (at most 3/4 of the remaining time).
    pub hard_mult: f64,
}

impl Default for TimeConfig {
    fn default() -> Self {
        TimeConfig {
            moves_to_go: 30,
            inc_share: 0.75,
            hard_mult: 3.0,
        }
    }
}

/// `None` means no time limit (infinite / depth / nodes only).
pub fn budget(limits: &SearchLimits, stm: Color, overhead_ms: u64) -> Option<TimeBudget> {
    budget_with(limits, stm, overhead_ms, &TimeConfig::default())
}

/// [`budget`] with the given constants.
pub fn budget_with(
    limits: &SearchLimits,
    stm: Color,
    overhead_ms: u64,
    cfg: &TimeConfig,
) -> Option<TimeBudget> {
    if limits.infinite {
        return None;
    }
    if let Some(mt) = limits.movetime {
        let t = Duration::from_millis(mt.saturating_sub(overhead_ms).max(1));
        return Some(TimeBudget {
            soft: t,
            hard: t,
            fixed: true,
        });
    }
    let (time, inc) = match stm {
        Color::White => (limits.wtime?, limits.winc.unwrap_or(0)),
        Color::Black => (limits.btime?, limits.binc.unwrap_or(0)),
    };
    let available = time.saturating_sub(overhead_ms).max(1);
    let mtg = limits
        .movestogo
        .map_or(cfg.moves_to_go, u64::from)
        .clamp(1, 100);
    let soft = (available / mtg + (inc as f64 * cfg.inc_share) as u64)
        .min(available / 2)
        .max(1);
    let hard = ((soft as f64 * cfg.hard_mult) as u64)
        .min(available * 3 / 4)
        .max(soft);
    Some(TimeBudget {
        soft: Duration::from_millis(soft),
        hard: Duration::from_millis(hard),
        fixed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_are_sane() {
        let l = SearchLimits {
            wtime: Some(60_000),
            winc: Some(1000),
            ..Default::default()
        };
        let b = budget(&l, Color::White, 10).unwrap();
        assert!(b.soft <= b.hard && b.hard < Duration::from_millis(60_000));
        assert!(budget(&l, Color::Black, 10).is_none()); // no btime given

        let l = SearchLimits {
            movetime: Some(500),
            ..Default::default()
        };
        assert_eq!(
            budget(&l, Color::Black, 10).unwrap().hard,
            Duration::from_millis(490)
        );

        let l = SearchLimits {
            wtime: Some(5),
            ..Default::default()
        };
        let b = budget(&l, Color::White, 10).unwrap();
        assert!(b.hard >= Duration::from_millis(1));
        assert!(!b.fixed);

        let l = SearchLimits {
            infinite: true,
            wtime: Some(5),
            ..Default::default()
        };
        assert!(budget(&l, Color::White, 10).is_none());
    }

    #[test]
    fn config_changes_the_formula() {
        // 8+0.08: available 7990, soft = 7990/30 + 60 = 326, hard = 3 × soft
        let l = SearchLimits {
            wtime: Some(8000),
            winc: Some(80),
            ..Default::default()
        };
        let b = budget(&l, Color::White, 10).unwrap();
        assert_eq!(b.soft, Duration::from_millis(326));
        assert_eq!(b.hard, Duration::from_millis(978));
        let cfg = TimeConfig {
            moves_to_go: 20,
            inc_share: 1.0,
            hard_mult: 2.0,
        };
        let b = budget_with(&l, Color::White, 10, &cfg).unwrap();
        assert_eq!(b.soft, Duration::from_millis(7990 / 20 + 80));
        assert_eq!(b.hard, Duration::from_millis(2 * (7990 / 20 + 80)));
        // movetime is fixed and ignores the config
        let l = SearchLimits {
            movetime: Some(300),
            ..Default::default()
        };
        assert!(budget_with(&l, Color::White, 10, &cfg).unwrap().fixed);
    }
}
