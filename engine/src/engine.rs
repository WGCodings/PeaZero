//! The interface between the UCI layer and whatever picks moves.
//! Sprint 1: `RandomEngine`. Later: the MuZero engine. UCI code never changes.

use std::sync::atomic::AtomicBool;

use peazero_core::{Game, Move};

/// Parsed `go` parameters. Times are in milliseconds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchLimits {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub movetime: Option<u64>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub mate: Option<u32>,
    pub infinite: bool,
    pub ponder: bool,
    /// Non-standard `go perft N`.
    pub perft: Option<u32>,
    /// `go searchmoves m1 m2 ...` (UCI strings, validated by the engine).
    pub searchmoves: Vec<String>,
}

impl SearchLimits {
    /// Parse the tokens after `go`. Unknown tokens are ignored.
    pub fn parse(tokens: &[&str]) -> SearchLimits {
        let mut l = SearchLimits::default();
        let mut i = 0;
        let num = |i: usize| tokens.get(i + 1).and_then(|t| t.parse::<i64>().ok());
        // Clocks can be negative in some GUIs when flagging; clamp to 0.
        let ms = |v: Option<i64>| v.map(|x| x.max(0) as u64);
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => l.wtime = ms(num(i)),
                "btime" => l.btime = ms(num(i)),
                "winc" => l.winc = ms(num(i)),
                "binc" => l.binc = ms(num(i)),
                "movestogo" => l.movestogo = num(i).map(|x| x.max(1) as u32),
                "movetime" => l.movetime = ms(num(i)),
                "depth" => l.depth = num(i).map(|x| x.max(1) as u32),
                "nodes" => l.nodes = num(i).map(|x| x.max(1) as u64),
                "mate" => l.mate = num(i).map(|x| x.max(1) as u32),
                "perft" => l.perft = num(i).map(|x| x.max(0) as u32),
                "infinite" => l.infinite = true,
                "ponder" => l.ponder = true,
                "searchmoves" => {
                    let mut j = i + 1;
                    while j < tokens.len()
                        && tokens[j].len() >= 4
                        && tokens[j].as_bytes()[1].is_ascii_digit()
                    {
                        l.searchmoves.push(tokens[j].to_string());
                        j += 1;
                    }
                    i = j;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        l
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    /// `Move::NULL` when there are no legal moves (printed as `0000`).
    pub best: Move,
    pub ponder: Option<Move>,
}

/// UCI-level settings every engine receives.
#[derive(Clone, Debug)]
pub struct CommonOptions {
    pub hash_mb: usize,
    pub threads: usize,
    pub move_overhead_ms: u64,
}

impl Default for CommonOptions {
    fn default() -> Self {
        CommonOptions {
            hash_mb: 16,
            threads: 1,
            move_overhead_ms: 10,
        }
    }
}

pub trait Engine: Send {
    /// Called on `ucinewgame`.
    fn new_game(&mut self);

    /// Extra `option name ...` lines (without the common Hash/Threads/MoveOverhead).
    fn uci_options(&self) -> Vec<String> {
        Vec::new()
    }

    /// Tunable parameters with their current values in OpenBench's SPSA
    /// input format (UCI command `spsa`).
    fn spsa_lines(&self) -> Vec<String> {
        Vec::new()
    }

    /// Handle an engine-specific option. Return false if unknown.
    fn set_option(&mut self, _name: &str, _value: &str) -> bool {
        false
    }

    /// Choose a move. Must return promptly once `stop` becomes true.
    /// May print `info ...` lines to stdout; must NOT print `bestmove`.
    fn search(
        &mut self,
        game: &Game,
        limits: &SearchLimits,
        common: &CommonOptions,
        stop: &AtomicBool,
    ) -> SearchResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_go() {
        let l = SearchLimits::parse(
            &"wtime 1000 btime -5 winc 10 binc 20 movestogo 7"
                .split(' ')
                .collect::<Vec<_>>(),
        );
        assert_eq!(l.wtime, Some(1000));
        assert_eq!(l.btime, Some(0));
        assert_eq!(l.winc, Some(10));
        assert_eq!(l.binc, Some(20));
        assert_eq!(l.movestogo, Some(7));
        let l = SearchLimits::parse(&["searchmoves", "e2e4", "d2d4", "infinite"]);
        assert_eq!(l.searchmoves, vec!["e2e4", "d2d4"]);
        assert!(l.infinite);
        let l = SearchLimits::parse(&["perft", "5"]);
        assert_eq!(l.perft, Some(5));
    }
}
