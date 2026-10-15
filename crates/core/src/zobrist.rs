//! Zobrist keys, generated at compile time from a fixed seed.

use crate::rng::splitmix64;

pub struct Keys {
    pub piece_sq: [[u64; 64]; 12],
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub side: u64,
}

pub const KEYS: Keys = {
    let mut state = 0x5045_415A_4552_4F21u64; // "PEAZERO!"
    let mut piece_sq = [[0u64; 64]; 12];
    let mut p = 0;
    while p < 12 {
        let mut s = 0;
        while s < 64 {
            let (ns, v) = splitmix64(state);
            state = ns;
            piece_sq[p][s] = v;
            s += 1;
        }
        p += 1;
    }
    let mut castling = [0u64; 16];
    let mut i = 1; // castling[0] stays 0 so "no rights" doesn't perturb the hash
    while i < 16 {
        let (ns, v) = splitmix64(state);
        state = ns;
        castling[i] = v;
        i += 1;
    }
    let mut ep_file = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        let (ns, v) = splitmix64(state);
        state = ns;
        ep_file[f] = v;
        f += 1;
    }
    let (_, side) = splitmix64(state);
    Keys {
        piece_sq,
        castling,
        ep_file,
        side,
    }
};
