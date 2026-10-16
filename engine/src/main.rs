mod bench;
mod engine;
mod policy;
mod random;
mod search;
mod timeman;
mod uci;

fn main() {
    peazero_core::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        // OpenBench runs `<engine> bench` and parses "<nodes> nodes <nps> nps".
        Some("bench") => {
            let mut net = match args.get(1) {
                Some(path) => {
                    policy::PolicyNet::from_pzn(&std::fs::read(path).expect("read network"))
                        .expect("load network")
                }
                None => policy::PolicyNet::embedded(),
            };
            bench::run(&mut net);
        }
        Some("perft") => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                peazero_core::START_FEN.to_string()
            };
            match peazero_core::Position::from_fen(&fen) {
                Ok(pos) => uci::print_divide(&pos, depth),
                Err(e) => eprintln!("{e}"),
            }
        }
        // Baseline opponent for sanity matches: `peazero random`.
        Some("random") => uci::UciLoop::new(Box::new(random::RandomEngine::new())).run(),
        _ => uci::UciLoop::new(Box::new(policy::PolicyEngine::new())).run(),
    }
}
