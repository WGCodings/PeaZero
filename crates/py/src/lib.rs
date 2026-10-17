//! `peazero_rs` — Python bindings for the trainer.
//!
//! ```python
//! import peazero_rs as pz
//! sp = pz.SelfPlay(games=1024, considered=16)
//! obs = sp.observations()                    # (N, 21, 8, 8) float32
//! acts = sp.expand(policy_logits)            # (N*m, 5, 8, 8) float32
//! sp.finish(root_value_logits, child_value_logits)
//! chunk, games, positions = sp.take_chunk()  # bytes in data format v0
//!
//! rb = pz.Replay(capacity=1_000_000)
//! rb.add_chunk(chunk)
//! batch = rb.sample(512, 5)                  # dict of numpy arrays
//! ```
//! Shapes are validated here and raise `ValueError` instead of panicking
//! (the release profile aborts on panic). The heavy calls release the GIL, so
//! self-play in a background Python thread runs alongside training.

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use peazero_core::encoding::{
    ACTION_PLANES, ACTION_SIZE, ENCODING_VERSION, NUM_ACTIONS, OBS_PLANES, OBS_SIZE,
};
use peazero_core::Game;
use peazero_selfplay::gumbel::GumbelConfig;
use peazero_selfplay::pool::PoolStats;
use peazero_selfplay::pool::{GamePool, PoolConfig};
use peazero_selfplay::record::{write_chunk, Termination};
use peazero_selfplay::replay::{Replay as RustReplay, MAX_LEGAL};
use peazero_selfplay::search::{SearchConfig, SearchPool};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn shaped<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    v.into_pyarray(py).reshape(shape.to_vec())
}

fn slice_of<'a>(
    a: &'a PyReadonlyArrayDyn<'_, f32>,
    expected: usize,
    what: &str,
) -> PyResult<&'a [f32]> {
    let s = a.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{what} must be a C-contiguous float32 array"))
    })?;
    if s.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {expected} values, got {} (shape {:?})",
            s.len(),
            a.shape()
        )));
    }
    Ok(s)
}

#[pyclass]
struct SelfPlay {
    pool: GamePool,
}

#[pymethods]
impl SelfPlay {
    /// `threads`: CPU threads for the per-game work (0 = one per logical CPU).
    /// The games played do not depend on it.
    #[new]
    #[pyo3(signature = (games=256, considered=16, gumbel_scale=1.0, c_visit=50.0, c_scale=0.1, max_plies=512, seed=0, threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        considered: usize,
        gumbel_scale: f32,
        c_visit: f32,
        c_scale: f32,
        max_plies: usize,
        seed: u64,
        threads: usize,
    ) -> PyResult<Self> {
        if games == 0 || considered == 0 {
            return Err(PyValueError::new_err("games and considered must be > 0"));
        }
        peazero_core::init();
        Ok(SelfPlay {
            pool: GamePool::new(PoolConfig {
                games,
                max_plies,
                gumbel: GumbelConfig {
                    considered,
                    gumbel_scale,
                    c_visit,
                    c_scale,
                },
                seed,
                threads,
            }),
        })
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    #[getter]
    fn num_games(&self) -> usize {
        self.pool.num_games()
    }

    #[getter]
    fn considered(&self) -> usize {
        self.pool.considered()
    }

    /// Root observations of all games: (N, 21, 8, 8) float32.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let n = self.pool.num_games();
        let pool = &self.pool;
        let v = py.detach(|| {
            let mut v = vec![0.0f32; n * OBS_SIZE];
            pool.observations(&mut v);
            v
        });
        shaped(py, v, &[n, OBS_PLANES, 8, 8])
    }

    /// Root policy logits (N, 4672) → action planes (N*m, 5, 8, 8) of the
    /// considered moves, game-major (game i owns rows i*m .. i*m+m).
    fn expand<'py>(
        &mut self,
        py: Python<'py>,
        policy: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let n = self.pool.num_games();
        let m = self.pool.considered();
        let p = slice_of(&policy, n * NUM_ACTIONS, "policy")?;
        let pool = &mut self.pool;
        let out = py.detach(|| {
            let mut out = vec![0.0f32; n * m * ACTION_SIZE];
            pool.expand(p, &mut out);
            out
        });
        shaped(py, out, &[n * m, ACTION_PLANES, 8, 8])
    }

    /// Root value logits (N, 3) and child value logits (N*m, 3) → play one
    /// move in every game. Returns the number of games that finished.
    fn finish(
        &mut self,
        py: Python<'_>,
        root_value: PyReadonlyArrayDyn<'_, f32>,
        child_value: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<usize> {
        let n = self.pool.num_games();
        let m = self.pool.considered();
        let r = slice_of(&root_value, n * 3, "root_value")?;
        let c = slice_of(&child_value, n * m * 3, "child_value")?;
        let pool = &mut self.pool;
        Ok(py.detach(|| pool.finish(r, c)))
    }

    /// Finished games as one data-format-v0 chunk: (bytes, games, positions).
    fn take_chunk<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyBytes>, usize, usize) {
        let games = self.pool.take_finished();
        let positions = games.iter().map(|g| g.plies.len()).sum();
        let bytes = py.detach(|| write_chunk(&games));
        (PyBytes::new(py, &bytes), games.len(), positions)
    }

    fn plies_in_progress(&self) -> usize {
        self.pool.plies_in_progress()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_dict(py, self.pool.stats())
    }
}

fn stats_dict<'py>(py: Python<'py>, s: &PoolStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("games", s.games_finished)?;
    d.set_item("plies", s.plies)?;
    d.set_item("white_wins", s.white_wins)?;
    d.set_item("black_wins", s.black_wins)?;
    d.set_item("draws", s.draws)?;
    let t = PyDict::new(py);
    for (i, &count) in s.terminations.iter().enumerate() {
        t.set_item(Termination::from_u8(i as u8).unwrap().name(), count)?;
    }
    d.set_item("terminations", t)?;
    Ok(d)
}

/// Self-play with Gumbel MuZero tree search on the real board (only the
/// `initial` network is needed). Lock-step rounds:
///
/// ```python
/// sp = pz.SearchSelfPlay(games=1024, simulations=64)
/// obs, legal = sp.requests()          # (N,21,8,8) float32, (N,218) int32
/// # policy (N,4672) → gather legal → (N,218); value (N,3)
/// sp.respond(legal_logits, value)     # one simulation for every game
/// ```
#[pyclass]
struct SearchSelfPlay {
    pool: SearchPool,
}

#[pymethods]
impl SearchSelfPlay {
    #[new]
    /// `noise_plies`: plies per game with the full Gumbel noise
    /// (`gumbel_scale`), then `late_gumbel_scale`; 0 = full noise all game.
    #[pyo3(signature = (games=1024, simulations=64, considered=16, gumbel_scale=1.0, c_visit=50.0, c_scale=0.1, max_plies=512, seed=0, threads=0, noise_plies=0, late_gumbel_scale=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        simulations: usize,
        considered: usize,
        gumbel_scale: f32,
        c_visit: f32,
        c_scale: f32,
        max_plies: usize,
        seed: u64,
        threads: usize,
        noise_plies: usize,
        late_gumbel_scale: f32,
    ) -> PyResult<Self> {
        if games == 0 || considered == 0 || simulations == 0 {
            return Err(PyValueError::new_err(
                "games, simulations and considered must be > 0",
            ));
        }
        peazero_core::init();
        Ok(SearchSelfPlay {
            pool: SearchPool::new(SearchConfig {
                games,
                simulations,
                max_plies,
                gumbel: GumbelConfig {
                    considered,
                    gumbel_scale,
                    c_visit,
                    c_scale,
                },
                seed,
                threads,
                noise_plies,
                late_gumbel_scale,
            }),
        })
    }

    #[getter]
    fn num_games(&self) -> usize {
        self.pool.num_games()
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    /// Positions to evaluate: (obs (N,21,8,8) float32, legal (N,218) int32
    /// policy indices of their legal moves, padded with 0).
    #[allow(clippy::type_complexity)]
    fn requests<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<i32>>)> {
        let n = self.pool.num_games();
        let pool = &self.pool;
        let (obs, legal) = py.detach(|| {
            let mut obs = vec![0.0f32; n * OBS_SIZE];
            let mut legal = vec![0i32; n * MAX_LEGAL];
            pool.requests(&mut obs, &mut legal);
            (obs, legal)
        });
        Ok((
            shaped(py, obs, &[n, OBS_PLANES, 8, 8])?,
            shaped(py, legal, &[n, MAX_LEGAL])?,
        ))
    }

    /// Legal-move logits (N,218) gathered at `legal`, value logits (N,3).
    /// Runs one simulation step for every game; returns games finished.
    fn respond(
        &mut self,
        py: Python<'_>,
        legal_logits: PyReadonlyArrayDyn<'_, f32>,
        value: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<usize> {
        let n = self.pool.num_games();
        let l = slice_of(&legal_logits, n * MAX_LEGAL, "legal_logits")?;
        let v = slice_of(&value, n * 3, "value")?;
        let pool = &mut self.pool;
        Ok(py.detach(|| pool.respond(l, v)))
    }

    /// Finished games as one data-format-v0 chunk: (bytes, games, positions).
    fn take_chunk<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyBytes>, usize, usize) {
        let games = self.pool.take_finished();
        let positions = games.iter().map(|g| g.plies.len()).sum();
        let bytes = py.detach(|| write_chunk(&games));
        (PyBytes::new(py, &bytes), games.len(), positions)
    }

    fn plies_in_progress(&self) -> usize {
        self.pool.plies_in_progress()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_dict(py, self.pool.stats())
    }
}

#[pyclass]
struct Replay {
    inner: RustReplay,
}

#[pymethods]
impl Replay {
    #[new]
    #[pyo3(signature = (capacity=1_000_000, seed=0))]
    fn new(capacity: usize, seed: u64) -> Self {
        peazero_core::init();
        Replay {
            inner: RustReplay::new(capacity, seed),
        }
    }

    /// Add a data-format-v0 chunk; returns the number of games added.
    fn add_chunk(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.detach(|| inner.add_chunk(data))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn positions(&self) -> usize {
        self.inner.positions()
    }

    #[getter]
    fn games(&self) -> usize {
        self.inner.games()
    }

    #[getter]
    fn total_added(&self) -> u64 {
        self.inner.total_added()
    }

    /// Sample a training batch. Keys: obs (B,21,8,8), actions (B,K,5,8,8),
    /// policy_index / policy_prob (B,K+1,218), policy_mask (B,K+1),
    /// value_target (B,K+1) int64 (0 win, 1 draw, 2 loss), value_mask (B,K+1),
    /// value_search (B,K+1,3) search W/D/L where recorded (data format v2),
    /// value_search_mask (B,K+1).
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        batch: usize,
        unroll: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.inner.positions() == 0 {
            return Err(PyValueError::new_err("replay buffer is empty"));
        }
        let inner = &mut self.inner;
        let b = py.detach(|| inner.sample(batch, unroll));
        let k1 = unroll + 1;
        let d = PyDict::new(py);
        d.set_item("obs", shaped(py, b.obs, &[batch, OBS_PLANES, 8, 8])?)?;
        d.set_item(
            "actions",
            shaped(py, b.actions, &[batch, unroll, ACTION_PLANES, 8, 8])?,
        )?;
        d.set_item(
            "policy_index",
            shaped(py, b.policy_index, &[batch, k1, MAX_LEGAL])?,
        )?;
        d.set_item(
            "policy_prob",
            shaped(py, b.policy_prob, &[batch, k1, MAX_LEGAL])?,
        )?;
        d.set_item("policy_mask", shaped(py, b.policy_mask, &[batch, k1])?)?;
        d.set_item("value_target", shaped(py, b.value_target, &[batch, k1])?)?;
        d.set_item("value_mask", shaped(py, b.value_mask, &[batch, k1])?)?;
        d.set_item("value_search", shaped(py, b.value_search, &[batch, k1, 3])?)?;
        d.set_item(
            "value_search_mask",
            shaped(py, b.value_search_mask, &[batch, k1])?,
        )?;
        Ok(d)
    }
}

/// Encode a FEN exactly like the engine does: (21, 8, 8) float32.
#[pyfunction]
#[pyo3(signature = (fen, repetitions=0))]
fn encode_fen<'py>(
    py: Python<'py>,
    fen: &str,
    repetitions: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    peazero_core::init();
    let game = Game::from_fen(fen).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut v = vec![0.0f32; OBS_SIZE];
    peazero_core::encoding::encode_observation(game.position(), repetitions, &mut v);
    shaped(py, v, &[OBS_PLANES, 8, 8])
}

#[pymodule]
fn peazero_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlay>()?;
    m.add_class::<SearchSelfPlay>()?;
    m.add_class::<Replay>()?;
    m.add_function(wrap_pyfunction!(encode_fen, m)?)?;
    m.add("ENCODING_VERSION", ENCODING_VERSION)?;
    m.add("MAX_LEGAL", MAX_LEGAL)?;
    Ok(())
}
