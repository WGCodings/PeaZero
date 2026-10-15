//! UCI protocol loop.
//!
//! The main thread reads stdin; searches run on a worker thread so `stop`,
//! `isready` and `quit` are handled while searching. Only the worker prints
//! `bestmove`, exactly once per `go`.

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use peazero_core::perft::divide;
use peazero_core::{Game, Position, START_FEN};

use crate::engine::{CommonOptions, Engine, SearchLimits};

const NAME: &str = "PeaZero";
const AUTHOR: &str = "WG";

pub struct UciLoop {
    game: Game,
    engine: Arc<Mutex<Box<dyn Engine>>>,
    common: CommonOptions,
    stop: Arc<AtomicBool>,
    search_thread: Option<JoinHandle<()>>,
}

pub fn print_divide(pos: &Position, depth: u32) {
    let start = Instant::now();
    let mut total = 0;
    for (m, n) in divide(pos, depth.max(1)) {
        println!("{m}: {n}");
        total += n;
    }
    let ms = start.elapsed().as_millis().max(1);
    println!();
    println!("Nodes searched: {total}");
    println!("Time: {ms} ms ({} nps)", total as u128 * 1000 / ms);
}

impl UciLoop {
    pub fn new(engine: Box<dyn Engine>) -> UciLoop {
        UciLoop {
            game: Game::default(),
            engine: Arc::new(Mutex::new(engine)),
            common: CommonOptions::default(),
            stop: Arc::new(AtomicBool::new(false)),
            search_thread: None,
        }
    }

    pub fn run(mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if !self.handle(line.trim()) {
                break;
            }
        }
        self.stop_search();
    }

    /// Handle one command; returns false on `quit`.
    pub fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else {
            return true;
        };
        match cmd {
            "uci" => self.cmd_uci(),
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.stop_search();
                self.game = Game::default();
                self.engine.lock().unwrap().new_game();
            }
            "position" => {
                self.stop_search();
                self.cmd_position(&tokens[1..]);
            }
            "go" => self.cmd_go(&tokens[1..]),
            "stop" => self.stop_search(),
            // We don't advertise Ponder; treat ponderhit as "finish now".
            "ponderhit" => self.stop.store(true, Ordering::SeqCst),
            "setoption" => {
                self.stop_search();
                self.cmd_setoption(&tokens[1..]);
            }
            "debug" | "register" => {}
            "quit" => {
                self.stop_search();
                return false;
            }
            // ---- non-standard helpers ----
            "d" => println!("{}", self.game.position()),
            "result" => println!(
                "result {} {:?}",
                self.game.result().pgn(),
                self.game.result()
            ),
            "bench" => {
                self.stop_search();
                crate::bench::run(&mut crate::policy::PolicyNet::embedded());
            }
            "perft" => {
                self.stop_search();
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
                print_divide(self.game.position(), depth);
            }
            _ => println!("info string unknown command: {line}"),
        }
        true
    }

    fn cmd_uci(&self) {
        println!("id name {NAME} {}", env!("CARGO_PKG_VERSION"));
        println!("id author {AUTHOR}");
        println!("option name Hash type spin default 16 min 1 max 65536");
        println!("option name Threads type spin default 1 min 1 max 1024");
        println!("option name MoveOverhead type spin default 10 min 0 max 10000");
        for o in self.engine.lock().unwrap().uci_options() {
            println!("{o}");
        }
        println!("uciok");
    }

    fn cmd_position(&mut self, tokens: &[&str]) {
        let moves_idx = tokens.iter().position(|&t| t == "moves");
        let setup = &tokens[..moves_idx.unwrap_or(tokens.len())];

        let game = match setup.first() {
            Some(&"startpos") => Game::from_fen(START_FEN),
            Some(&"fen") => Game::from_fen(&setup[1..].join(" ")),
            _ => {
                println!("info string position: expected 'startpos' or 'fen'");
                return;
            }
        };
        let mut game = match game {
            Ok(g) => g,
            Err(e) => {
                println!("info string {e}");
                return;
            }
        };
        if let Some(i) = moves_idx {
            for &m in &tokens[i + 1..] {
                if !game.push_uci(m) {
                    println!("info string illegal move '{m}', ignoring the rest");
                    break;
                }
            }
        }
        self.game = game;
    }

    fn cmd_go(&mut self, tokens: &[&str]) {
        self.stop_search();
        let limits = SearchLimits::parse(tokens);

        if let Some(depth) = limits.perft {
            print_divide(self.game.position(), depth);
            return;
        }

        self.stop.store(false, Ordering::SeqCst);
        let engine = Arc::clone(&self.engine);
        let stop = Arc::clone(&self.stop);
        let game = self.game.clone();
        let common = self.common.clone();

        self.search_thread = Some(thread::spawn(move || {
            let result = engine
                .lock()
                .unwrap()
                .search(&game, &limits, &common, &stop);
            // UCI: in infinite/ponder mode, bestmove only after stop/ponderhit.
            if limits.infinite || limits.ponder {
                while !stop.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            match result.ponder {
                Some(p) => println!("bestmove {} ponder {p}", result.best),
                None => println!("bestmove {}", result.best),
            }
        }));
    }

    fn cmd_setoption(&mut self, tokens: &[&str]) {
        // setoption name <name...> [value <value...>]
        let name_start = tokens
            .iter()
            .position(|&t| t == "name")
            .map(|i| i + 1)
            .unwrap_or(0);
        let value_idx = tokens.iter().position(|&t| t == "value");
        let name = tokens[name_start..value_idx.unwrap_or(tokens.len())].join(" ");
        let value = value_idx
            .map(|i| tokens[i + 1..].join(" "))
            .unwrap_or_default();

        let parse = |v: &str| v.parse::<u64>().ok();
        match name.to_ascii_lowercase().as_str() {
            "hash" => {
                if let Some(v) = parse(&value) {
                    self.common.hash_mb = v.clamp(1, 65536) as usize;
                }
            }
            "threads" => {
                if let Some(v) = parse(&value) {
                    self.common.threads = v.clamp(1, 1024) as usize;
                }
            }
            "moveoverhead" | "move overhead" => {
                if let Some(v) = parse(&value) {
                    self.common.move_overhead_ms = v.min(10000);
                }
            }
            _ => {
                if !self.engine.lock().unwrap().set_option(&name, &value) {
                    println!("info string unknown option '{name}'");
                }
            }
        }
    }

    fn stop_search(&mut self) {
        if let Some(h) = self.search_thread.take() {
            self.stop.store(true, Ordering::SeqCst);
            let _ = h.join();
        }
    }
}
