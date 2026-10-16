//! Time management: a search stops starting new work after `soft` and never
//! exceeds `hard` (used by the alpha-beta search; MCTS will use the same).

use std::time::Duration;

use peazero_core::Color;

use crate::engine::SearchLimits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    pub soft: Duration,
    pub hard: Duration,
}

/// `None` means no time limit (infinite / depth / nodes only).
pub fn budget(limits: &SearchLimits, stm: Color, overhead_ms: u64) -> Option<TimeBudget> {
    if limits.infinite {
        return None;
    }
    if let Some(mt) = limits.movetime {
        let t = Duration::from_millis(mt.saturating_sub(overhead_ms).max(1));
        return Some(TimeBudget { soft: t, hard: t });
    }
    let (time, inc) = match stm {
        Color::White => (limits.wtime?, limits.winc.unwrap_or(0)),
        Color::Black => (limits.btime?, limits.binc.unwrap_or(0)),
    };
    let available = time.saturating_sub(overhead_ms).max(1);
    let mtg = limits.movestogo.unwrap_or(30).clamp(1, 50) as u64;
    let soft = (available / mtg + inc * 3 / 4).min(available / 2).max(1);
    let hard = (soft * 3).min(available * 3 / 4).max(soft);
    Some(TimeBudget {
        soft: Duration::from_millis(soft),
        hard: Duration::from_millis(hard),
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

        let l = SearchLimits {
            infinite: true,
            wtime: Some(5),
            ..Default::default()
        };
        assert!(budget(&l, Color::White, 10).is_none());
    }
}
