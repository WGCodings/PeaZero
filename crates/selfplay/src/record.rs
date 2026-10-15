//! Self-play data format v0. One chunk = one file (little-endian):
//!
//! ```text
//! header   : "PZSP" | u32 format version (1) | u32 ENCODING_VERSION | u32 game count
//! per game : u16 ply count | i8 result (+1 white wins, 0 draw, -1 black wins) | u8 termination
//! per ply  : 12 × u64 piece bitboards (white P N B R Q K, black P N B R Q K)
//!            u8 side to move (0 white, 1 black) | u8 castling | u8 ep square (64 = none)
//!            u8 halfmove clock | u8 repetitions | u16 action played
//!            u8 legal count L | L × (u16 action, u16 target probability × 65535)
//! ```
//!
//! The position *after* the last ply (the terminal position) is not stored;
//! its side to move is the opposite of the last ply's.

use peazero_core::encoding::ENCODING_VERSION;
use peazero_core::{Color, GameResult, Position, Square};

pub const MAGIC: &[u8; 4] = b"PZSP";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Termination {
    Checkmate = 0,
    Stalemate = 1,
    FiftyMoves = 2,
    Repetition = 3,
    InsufficientMaterial = 4,
    /// Reached the self-play ply limit; scored as a draw.
    MaxPlies = 5,
}

impl Termination {
    pub fn from_u8(v: u8) -> Option<Termination> {
        use Termination::*;
        Some(match v {
            0 => Checkmate,
            1 => Stalemate,
            2 => FiftyMoves,
            3 => Repetition,
            4 => InsufficientMaterial,
            5 => MaxPlies,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use Termination::*;
        match self {
            Checkmate => "checkmate",
            Stalemate => "stalemate",
            FiftyMoves => "fifty_moves",
            Repetition => "repetition",
            InsufficientMaterial => "insufficient_material",
            MaxPlies => "max_plies",
        }
    }

    /// (result from White's view, termination) for a finished game result.
    pub fn from_result(r: GameResult) -> Option<(i8, Termination)> {
        use peazero_core::DrawReason::*;
        Some(match r {
            GameResult::Ongoing => return None,
            GameResult::Checkmate(Color::White) => (1, Termination::Checkmate),
            GameResult::Checkmate(Color::Black) => (-1, Termination::Checkmate),
            GameResult::Draw(Stalemate) => (0, Termination::Stalemate),
            GameResult::Draw(FiftyMoves) => (0, Termination::FiftyMoves),
            GameResult::Draw(Repetition) => (0, Termination::Repetition),
            GameResult::Draw(InsufficientMaterial) => (0, Termination::InsufficientMaterial),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlyRecord {
    pub pieces: [u64; 12],
    pub stm: Color,
    pub castling: u8,
    pub ep: Option<Square>,
    pub halfmove: u8,
    pub repetitions: u8,
    pub action: u16,
    /// (action, probability × 65535)
    pub targets: Vec<(u16, u16)>,
}

impl PlyRecord {
    pub fn new(
        pos: &Position,
        repetitions: usize,
        action: usize,
        targets: &[(usize, f32)],
    ) -> PlyRecord {
        PlyRecord {
            pieces: pos.piece_bitboards(),
            stm: pos.side_to_move(),
            castling: pos.castling_rights(),
            ep: pos.ep_square(),
            halfmove: pos.halfmove_clock().min(255) as u8,
            repetitions: repetitions.min(255) as u8,
            action: action as u16,
            targets: targets
                .iter()
                .map(|&(a, p)| (a as u16, (p.clamp(0.0, 1.0) * 65535.0).round() as u16))
                .collect(),
        }
    }

    pub fn position(&self) -> Position {
        Position::from_compact(
            &self.pieces,
            self.stm,
            self.castling,
            self.ep,
            self.halfmove as u16,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    pub plies: Vec<PlyRecord>,
    /// +1 white won, 0 draw, -1 black won.
    pub result: i8,
    pub termination: Termination,
}

pub fn write_chunk(games: &[GameRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + games.iter().map(|g| g.plies.len() * 250).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    out.extend_from_slice(&(games.len() as u32).to_le_bytes());
    for g in games {
        out.extend_from_slice(&(g.plies.len() as u16).to_le_bytes());
        out.push(g.result as u8);
        out.push(g.termination as u8);
        for p in &g.plies {
            for bb in p.pieces {
                out.extend_from_slice(&bb.to_le_bytes());
            }
            out.push(p.stm as u8);
            out.push(p.castling);
            out.push(p.ep.map(|s| s.0).unwrap_or(64));
            out.push(p.halfmove);
            out.push(p.repetitions);
            out.extend_from_slice(&p.action.to_le_bytes());
            out.push(p.targets.len() as u8);
            for &(a, prob) in &p.targets {
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&prob.to_le_bytes());
            }
        }
    }
    out
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!("truncated chunk at byte {}", self.pos));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

pub fn read_chunk(bytes: &[u8]) -> Result<Vec<GameRecord>, String> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err("not a PeaZero self-play chunk (bad magic)".into());
    }
    let version = r.u32()?;
    let encoding = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported data format v{version}"));
    }
    if encoding != ENCODING_VERSION {
        return Err(format!(
            "data uses encoding v{encoding}, expected v{ENCODING_VERSION}"
        ));
    }
    let count = r.u32()? as usize;
    let mut games = Vec::with_capacity(count);
    for _ in 0..count {
        let n = r.u16()? as usize;
        let result = r.u8()? as i8;
        let termination = Termination::from_u8(r.u8()?).ok_or("bad termination code")?;
        let mut plies = Vec::with_capacity(n);
        for _ in 0..n {
            let mut pieces = [0u64; 12];
            for bb in &mut pieces {
                *bb = r.u64()?;
            }
            let stm = if r.u8()? == 0 {
                Color::White
            } else {
                Color::Black
            };
            let castling = r.u8()?;
            let ep = match r.u8()? {
                64 => None,
                s if s < 64 => Some(Square(s)),
                _ => return Err("bad ep square".into()),
            };
            let halfmove = r.u8()?;
            let repetitions = r.u8()?;
            let action = r.u16()?;
            let l = r.u8()? as usize;
            let mut targets = Vec::with_capacity(l);
            for _ in 0..l {
                targets.push((r.u16()?, r.u16()?));
            }
            plies.push(PlyRecord {
                pieces,
                stm,
                castling,
                ep,
                halfmove,
                repetitions,
                action,
                targets,
            });
        }
        games.push(GameRecord {
            plies,
            result,
            termination,
        });
    }
    if r.pos != bytes.len() {
        return Err(format!(
            "{} trailing bytes after chunk",
            bytes.len() - r.pos
        ));
    }
    Ok(games)
}
