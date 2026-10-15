//! A game = current position + hash history, so repetitions and the final
//! result can be detected. Self-play and UCI both work on a `Game`.

use crate::moves::{Move, MoveList};
use crate::position::{FenError, Position};
use crate::types::Color;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    Stalemate,
    FiftyMoves,
    Repetition,
    InsufficientMaterial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    /// The given colour won by checkmate.
    Checkmate(Color),
    Draw(DrawReason),
}

impl GameResult {
    /// PGN-style result string.
    pub fn pgn(self) -> &'static str {
        match self {
            GameResult::Ongoing => "*",
            GameResult::Checkmate(Color::White) => "1-0",
            GameResult::Checkmate(Color::Black) => "0-1",
            GameResult::Draw(_) => "1/2-1/2",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pos: Position,
    /// Hashes of all positions before the current one.
    history: Vec<u64>,
    moves: Vec<Move>,
    start: Position,
}

impl Default for Game {
    fn default() -> Self {
        Game::new(Position::startpos())
    }
}

impl Game {
    pub fn new(pos: Position) -> Game {
        Game {
            pos,
            history: Vec::with_capacity(256),
            moves: Vec::with_capacity(256),
            start: pos,
        }
    }

    pub fn from_fen(fen: &str) -> Result<Game, FenError> {
        Ok(Game::new(Position::from_fen(fen)?))
    }

    #[inline]
    pub fn position(&self) -> &Position {
        &self.pos
    }

    pub fn start_position(&self) -> &Position {
        &self.start
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    /// Play a move assumed to be legal.
    pub fn push(&mut self, m: Move) {
        self.history.push(self.pos.hash());
        self.moves.push(m);
        self.pos.make_move(m);
    }

    /// Play a UCI move string; returns false (and does nothing) if illegal.
    pub fn push_uci(&mut self, s: &str) -> bool {
        match self.pos.parse_uci_move(s) {
            Some(m) => {
                self.push(m);
                true
            }
            None => false,
        }
    }

    /// How many earlier positions are identical to the current one
    /// (same pieces, side to move, castling and ep rights).
    pub fn repetitions(&self) -> usize {
        let key = self.pos.hash();
        let n = self.history.len();
        let lookback = (self.pos.halfmove_clock() as usize).min(n);
        // Same side to move: step back 2 plies at a time, starting 2 back.
        (2..=lookback)
            .step_by(2)
            .filter(|&back| self.history[n - back] == key)
            .count()
    }

    pub fn result(&self) -> GameResult {
        let mut list = MoveList::new();
        self.pos.generate_legal(&mut list);
        if list.is_empty() {
            return if self.pos.in_check() {
                GameResult::Checkmate(!self.pos.side_to_move())
            } else {
                GameResult::Draw(DrawReason::Stalemate)
            };
        }
        if self.pos.halfmove_clock() >= 100 {
            return GameResult::Draw(DrawReason::FiftyMoves);
        }
        if self.pos.insufficient_material() {
            return GameResult::Draw(DrawReason::InsufficientMaterial);
        }
        if self.repetitions() >= 2 {
            return GameResult::Draw(DrawReason::Repetition);
        }
        GameResult::Ongoing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefold_by_knight_shuffle() {
        let mut g = Game::default();
        for _ in 0..2 {
            for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
                assert_eq!(g.result(), GameResult::Ongoing);
                assert!(g.push_uci(m));
            }
        }
        assert_eq!(g.repetitions(), 2);
        assert_eq!(g.result(), GameResult::Draw(DrawReason::Repetition));
    }

    #[test]
    fn fools_mate() {
        let mut g = Game::default();
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            assert!(g.push_uci(m));
        }
        assert_eq!(g.result(), GameResult::Checkmate(Color::Black));
    }

    #[test]
    fn stalemate_and_fifty() {
        let g = Game::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(g.result(), GameResult::Draw(DrawReason::Stalemate));
        let g = Game::from_fen("7k/8/6K1/8/8/8/8/R7 w - - 100 80").unwrap();
        assert_eq!(g.result(), GameResult::Draw(DrawReason::FiftyMoves));
        // Mate on the 100th half-move still counts as mate.
        let g = Game::from_fen("R6k/8/6K1/8/8/8/8/8 b - - 100 80").unwrap();
        assert_eq!(g.result(), GameResult::Checkmate(Color::White));
    }

    #[test]
    fn illegal_uci_rejected() {
        let mut g = Game::default();
        assert!(!g.push_uci("e2e5"));
        assert!(!g.push_uci("e1g1"));
        assert_eq!(g.ply(), 0);
    }
}
