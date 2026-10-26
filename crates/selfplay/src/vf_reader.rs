//! Read viriformat files (ours from `--viriformat`, or any engine's, e.g.
//! Pea's NNUE data): every position with the move played from it, its
//! score and the game result. Inverse of `viriformat.rs`.
//!
//! [`VfStream`] streams positions from any number of (large) files through a
//! shuffle buffer, cycling over the files forever, so training samples are
//! random without loading the files into memory.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use peazero_core::position::{BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
use peazero_core::rng::Rng;
use peazero_core::{Color, Move, PieceType, Position, Square};

use crate::viriformat::PACKED_BOARD_SIZE;

const UNMOVED_ROOK: u8 = 6;

/// One position of a viriformat game.
#[derive(Clone, Debug)]
pub struct VfPosition {
    pub pos: Position,
    /// The move played from this position.
    pub mv: Move,
    /// Score of the position, side to move, centipawns.
    pub score: i16,
    /// Game result for the side to move: 1 win, 0 draw, -1 loss.
    pub result: i8,
    /// Plies since the start of the game record.
    pub ply: u16,
}

/// Start position and white-relative result (0 black win, 1 draw, 2 white
/// win) of a marlinformat `PackedBoard`.
pub fn unpack_board(b: &[u8; PACKED_BOARD_SIZE]) -> Result<(Position, u8), String> {
    let occupancy = u64::from_le_bytes(b[0..8].try_into().unwrap());
    if occupancy.count_ones() > 32 {
        return Err("packed board with more than 32 pieces".into());
    }
    let mut pieces = [0u64; 12];
    let mut castling = 0u8;
    let mut bits = occupancy;
    let mut i = 0;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let nib = (b[8 + i / 2] >> ((i % 2) * 4)) & 0xF;
        i += 1;
        let black = nib & 8 != 0;
        let mut code = nib & 7;
        if code == UNMOVED_ROOK {
            code = PieceType::Rook.idx() as u8;
            let home = if black { 56 } else { 0 };
            let (oo, ooo) = if black {
                (BLACK_OO, BLACK_OOO)
            } else {
                (WHITE_OO, WHITE_OOO)
            };
            // standard chess: the castling rook stands in its corner
            if sq == home + 7 {
                castling |= oo;
            } else if sq == home {
                castling |= ooo;
            }
        }
        if code > 5 {
            return Err(format!("bad piece code {nib}"));
        }
        pieces[code as usize + 6 * black as usize] |= 1u64 << sq;
    }
    let stm = if b[24] & 0x80 != 0 {
        Color::Black
    } else {
        Color::White
    };
    let ep = match b[24] & 0x7F {
        s if s < 64 => Some(Square(s)),
        _ => None,
    };
    let halfmove = b[25] as u16;
    let pos = Position::from_compact(&pieces, stm, castling, ep, halfmove);
    Ok((pos, b[30]))
}

/// The legal move of `pos` that a viriformat 16-bit move stands for.
pub fn decode_move(pos: &Position, raw: u16) -> Option<Move> {
    let from = (raw & 63) as u8;
    let to = ((raw >> 6) & 63) as u8;
    let promo = (raw >> 12) & 3;
    let kind = raw >> 14;
    let legal = pos.legal_moves();
    legal.iter().copied().find(|m| {
        if m.from().0 != from {
            return false;
        }
        match kind {
            // castling written king-takes-rook: the rook's file gives the side
            2 => m.is_castle() && ((to > from) == (m.to().0 > from)),
            3 => {
                let want = [
                    PieceType::Knight,
                    PieceType::Bishop,
                    PieceType::Rook,
                    PieceType::Queen,
                ][promo as usize];
                m.to().0 == to && m.promo_piece() == Some(want)
            }
            _ => m.to().0 == to && !m.is_castle() && !m.is_promo(),
        }
    })
}

/// Decode one game from `bytes` starting at `*at`; advances `*at`. Ok(None)
/// at the end of the data. Games whose moves can't be replayed are an error.
pub fn read_game(bytes: &[u8], at: &mut usize) -> Result<Option<Vec<VfPosition>>, String> {
    if *at >= bytes.len() {
        return Ok(None);
    }
    if *at + PACKED_BOARD_SIZE > bytes.len() {
        return Err("truncated viriformat game header".into());
    }
    let (mut pos, wdl) = unpack_board(bytes[*at..*at + PACKED_BOARD_SIZE].try_into().unwrap())?;
    *at += PACKED_BOARD_SIZE;
    let white_result: i8 = match wdl {
        0 => -1,
        2 => 1,
        _ => 0,
    };
    let mut out = Vec::new();
    loop {
        if *at + 4 > bytes.len() {
            return Err("truncated viriformat game".into());
        }
        let raw = u16::from_le_bytes([bytes[*at], bytes[*at + 1]]);
        let score = i16::from_le_bytes([bytes[*at + 2], bytes[*at + 3]]);
        *at += 4;
        if raw == 0 && score == 0 {
            return Ok(Some(out));
        }
        let mv = decode_move(&pos, raw).ok_or_else(|| {
            format!(
                "viriformat move {raw:#06x} is not legal in {}",
                pos.to_fen()
            )
        })?;
        let white = pos.side_to_move() == Color::White;
        out.push(VfPosition {
            pos,
            mv,
            score: if white { score } else { score.saturating_neg() },
            result: if white { white_result } else { -white_result },
            ply: out.len().min(u16::MAX as usize) as u16,
        });
        pos.make_move(mv);
    }
}

/// Every game of a viriformat byte buffer.
pub fn read_all(bytes: &[u8]) -> Result<Vec<Vec<VfPosition>>, String> {
    let mut at = 0;
    let mut games = Vec::new();
    while let Some(g) = read_game(bytes, &mut at)? {
        games.push(g);
    }
    Ok(games)
}

/// Streams games from viriformat files, file after file, cycling forever.
struct GameSource {
    files: Vec<PathBuf>,
    file: usize,
    reader: Option<BufReader<File>>,
    buf: Vec<u8>,
    at: usize,
    /// passes over all files completed
    epochs: u64,
    bad_games: u64,
}

impl GameSource {
    fn new(files: Vec<PathBuf>) -> GameSource {
        GameSource {
            files,
            file: 0,
            reader: None,
            buf: Vec::new(),
            at: 0,
            epochs: 0,
            bad_games: 0,
        }
    }

    /// Top up the buffer with the next block of the current file; false at
    /// its end.
    fn fill(&mut self) -> Result<bool, String> {
        if self.reader.is_none() {
            let f = File::open(&self.files[self.file])
                .map_err(|e| format!("{}: {e}", self.files[self.file].display()))?;
            self.reader = Some(BufReader::with_capacity(1 << 20, f));
        }
        self.buf.drain(..self.at);
        self.at = 0;
        let mut block = vec![0u8; 8 << 20];
        let n = self
            .reader
            .as_mut()
            .unwrap()
            .read(&mut block)
            .map_err(|e| e.to_string())?;
        self.buf.extend_from_slice(&block[..n]);
        Ok(n > 0)
    }

    fn next_game(&mut self) -> Result<Vec<VfPosition>, String> {
        loop {
            // try to decode a complete game from what's buffered
            let mut at = self.at;
            match read_game(&self.buf, &mut at) {
                Ok(Some(g)) => {
                    self.at = at;
                    return Ok(g);
                }
                Ok(None) => {}
                Err(e) if e.starts_with("truncated") => {}
                Err(_) => {
                    // a game we can't replay (corrupt / unsupported): skip to its terminator
                    self.bad_games += 1;
                    let mut p = self.at + PACKED_BOARD_SIZE;
                    while p + 4 <= self.buf.len() && self.buf[p..p + 4] != [0; 4] {
                        p += 4;
                    }
                    if p + 4 <= self.buf.len() {
                        self.at = p + 4;
                        continue;
                    }
                }
            }
            if !self.fill()? {
                if self.at < self.buf.len() && self.buf.len() - self.at > 4 {
                    self.bad_games += 1; // truncated last game
                }
                self.buf.clear();
                self.at = 0;
                self.reader = None;
                self.file += 1;
                if self.file == self.files.len() {
                    self.file = 0;
                    self.epochs += 1;
                }
            }
        }
    }
}

/// Random positions from viriformat files through a shuffle buffer.
///
/// * `holdout_every` > 0 splits by game: every `holdout_every`-th game is
///   held out. `holdout = false` streams the others (training), `true`
///   only the held-out games (evaluation).
/// * `skip_plies`: positions within the first `skip_plies` plies of a game
///   record are skipped (book/opening duplicates).
pub struct VfStream {
    source: GameSource,
    buffer: Vec<VfPosition>,
    capacity: usize,
    rng: Rng,
    holdout_every: u64,
    holdout: bool,
    skip_plies: u16,
    games_seen: u64,
    pub positions_read: u64,
}

impl VfStream {
    pub fn new(
        files: Vec<PathBuf>,
        capacity: usize,
        seed: u64,
        holdout_every: u64,
        holdout: bool,
        skip_plies: u16,
    ) -> Result<VfStream, String> {
        if files.is_empty() {
            return Err("no viriformat files".into());
        }
        for f in &files {
            std::fs::metadata(f).map_err(|e| format!("{}: {e}", f.display()))?;
        }
        Ok(VfStream {
            source: GameSource::new(files),
            buffer: Vec::with_capacity(capacity),
            capacity: capacity.max(1),
            rng: Rng::new(seed ^ 0x7F4A_7C15),
            holdout_every,
            holdout,
            skip_plies,
            games_seen: 0,
            positions_read: 0,
        })
    }

    pub fn epochs(&self) -> u64 {
        self.source.epochs
    }

    pub fn bad_games(&self) -> u64 {
        self.source.bad_games
    }

    fn refill(&mut self) -> Result<(), String> {
        let mut guard = 0u64;
        while self.buffer.len() < self.capacity {
            let g = self.source.next_game()?;
            self.games_seen += 1;
            guard += 1;
            if self.holdout_every > 0
                && (self.games_seen.is_multiple_of(self.holdout_every) != self.holdout)
            {
                if guard > 50_000_000 {
                    return Err("no games pass the holdout filter".into());
                }
                continue;
            }
            for p in g {
                if p.ply >= self.skip_plies {
                    self.positions_read += 1;
                    self.buffer.push(p);
                }
            }
            if guard > 50_000_000 && self.buffer.is_empty() {
                return Err("no positions after --skip-plies".into());
            }
        }
        Ok(())
    }

    /// `n` random positions (each used once per pass through the buffer).
    pub fn take(&mut self, n: usize) -> Result<Vec<VfPosition>, String> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            if self.buffer.len() < self.capacity {
                self.refill()?;
            }
            let i = self.rng.below(self.buffer.len() as u64) as usize;
            out.push(self.buffer.swap_remove(i));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{GameRecord, PlyRecord, Termination};
    use crate::viriformat::write_game;
    use peazero_core::encoding::{legal_actions, move_to_action};
    use peazero_core::Game;

    fn random_game(seed: u64, plies: usize) -> GameRecord {
        let mut rng = Rng::new(seed);
        let mut g = Game::default();
        let mut recs = Vec::new();
        for _ in 0..plies {
            let la = legal_actions(g.position());
            if la.is_empty() || g.result() != peazero_core::GameResult::Ongoing {
                break;
            }
            let (a, m) = la[rng.below(la.len() as u64) as usize];
            let v = (rng.below(1000) as f32) / 1000.0;
            recs.push(
                PlyRecord::new(g.position(), g.repetitions(), a, &[(a, 1.0)]).with_value([
                    v,
                    0.1,
                    0.9 - v,
                ]),
            );
            g.push(m);
        }
        GameRecord {
            plies: recs,
            result: [1, 0, -1][(seed % 3) as usize],
            termination: Termination::MaxPlies,
        }
    }

    #[test]
    fn round_trip_with_the_writer() {
        peazero_core::init();
        let games: Vec<GameRecord> = (0..200).map(|s| random_game(s, 150)).collect();
        let mut bytes = Vec::new();
        for g in &games {
            write_game(g, &mut bytes).unwrap();
        }
        let back = read_all(&bytes).unwrap();
        assert_eq!(back.len(), games.len());
        let mut castles = 0;
        for (g, b) in games.iter().zip(&back) {
            assert_eq!(g.plies.len(), b.len());
            for (p, v) in g.plies.iter().zip(b) {
                let key = |f: String| f.split(' ').take(4).collect::<Vec<_>>().join(" ");
                assert_eq!(key(p.position().to_fen()), key(v.pos.to_fen()));
                assert_eq!(move_to_action(v.mv, v.pos.side_to_move()) as u16, p.action);
                castles += v.mv.is_castle() as usize;
                let cp = crate::viriformat::white_score(p);
                let stm_cp = if p.stm == Color::White { cp } else { -cp };
                assert_eq!(v.score, stm_cp);
                let white_res = g.result;
                let stm_res = if p.stm == Color::White {
                    white_res
                } else {
                    -white_res
                };
                assert_eq!(v.result, stm_res);
            }
        }
        let _ = castles;
    }

    #[test]
    fn stream_holdout_split_and_skip() {
        peazero_core::init();
        let dir = std::env::temp_dir().join(format!("pz_vf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.vf");
        let mut bytes = Vec::new();
        for s in 0..60 {
            write_game(&random_game(s, 40), &mut bytes).unwrap();
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut train = VfStream::new(vec![path.clone()], 500, 1, 5, false, 10).unwrap();
        let mut test = VfStream::new(vec![path.clone()], 500, 1, 5, true, 10).unwrap();
        let a = train.take(2000).unwrap();
        let b = test.take(200).unwrap();
        assert!(a.iter().all(|p| p.ply >= 10) && b.iter().all(|p| p.ply >= 10));
        // held-out and training positions never overlap
        let fens = |v: &[VfPosition]| {
            v.iter()
                .map(|p| p.pos.to_fen())
                .collect::<std::collections::HashSet<_>>()
        };
        let (fa, fb) = (fens(&a), fens(&b));
        assert!(fa.is_disjoint(&fb), "holdout leaks into training");
        assert!(train.epochs() > 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
