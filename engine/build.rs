//! Embeds the network file into the binary.
//!
//! `EVALFILE` (env var, set by the Makefile / OpenBench) selects the `.pzn`
//! file; relative paths are resolved from the repository root. Without it,
//! `networks/default.pzn` is used.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=EVALFILE");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest.parent().unwrap().to_path_buf();

    let chosen = match std::env::var("EVALFILE") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => PathBuf::from("networks/default.pzn"),
    };
    let path = if chosen.is_absolute() {
        chosen
    } else {
        repo_root.join(chosen)
    };
    let path = path
        .canonicalize()
        .unwrap_or_else(|e| panic!("EVALFILE {} not found: {e}", path.display()));
    let path = strip_verbatim(&path);

    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:rustc-env=PEAZERO_EVALFILE={}", path.display());
}

/// `canonicalize` on Windows returns `\\?\C:\...`; keep paths plain.
fn strip_verbatim(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => p.to_path_buf(),
    }
}
