//! PeaZero core: board representation, legal/pseudolegal move generation,
//! perft and game-state bookkeeping (repetitions, results).
//!
//! No external dependencies, so OpenBench workers can build it anywhere.

pub mod attacks;
pub mod bitboard;
pub mod game;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod position;
pub mod rng;
pub mod types;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use game::{DrawReason, Game, GameResult};
pub use moves::{Move, MoveList};
pub use position::{FenError, Position, START_FEN};
pub use types::{Color, Piece, PieceType, Square};

/// Force initialisation of lazily-built lookup tables (slider attacks).
/// Call once at startup so the first search/perft isn't charged for it.
pub fn init() {
    attacks::init();
}
