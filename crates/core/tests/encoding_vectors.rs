//! The committed `tests/encoding/vectors.jsonl` (used by the Python trainer
//! tests) must match what the current Rust encoding produces.

#[allow(dead_code)]
#[path = "../examples/encoding_vectors.rs"]
mod vectors;

#[test]
fn committed_vectors_are_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/encoding/vectors.jsonl"
    );
    let committed = std::fs::read_to_string(path)
        .expect("read vectors.jsonl")
        .replace("\r\n", "\n");
    assert!(
        committed == vectors::generate(),
        "tests/encoding/vectors.jsonl is stale; regenerate with:\n  \
         cargo run --release -p peazero-core --example encoding_vectors > tests/encoding/vectors.jsonl"
    );
}
