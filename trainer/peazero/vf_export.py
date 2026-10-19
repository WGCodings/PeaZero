#!/usr/bin/env python3
"""Keep a run's self-play games as one viriformat file (Viridithas / bullet
game format, see crates/selfplay/src/viriformat.rs).

    <run>/<run name>.vf        all exported games, appended one generation at a time
    <run>/<run name>.vf.json   which generations are in it (and its committed size)

With `train.py --viriformat`, every data/gen_XXXX folder is converted and
appended right before it is deleted (see --keep-data-gens), so nothing is
lost. Folders still on disk can be exported by hand while training is
stopped; they are then skipped (not duplicated) when they are deleted later:

    python -m peazero.vf_export --run ..\\runs\\fourth

Appends are crash-safe: the .json records the file size after each
complete generation, and anything beyond it (an interrupted append) is cut
off before the next append. Moving the .vf away (e.g. to train on it)
simply starts a new file; generations already exported are not repeated.
"""
import argparse
import glob
import json
import os
import re
import sys
import time


class VfExporter:
    def __init__(self, run):
        name = os.path.basename(os.path.normpath(os.path.abspath(run))) or "selfplay"
        self.path = os.path.join(run, f"{name}.vf")
        self.state_path = self.path + ".json"
        self.state = {"bytes": 0, "gens": [], "games": 0, "positions": 0}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state.update(json.load(f))
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if size < self.state["bytes"]:
            # The file was moved away or cut short by hand: continue with what is there.
            self.state["bytes"] = size
            self._save()
        self.exported = set(self.state["gens"])

    def _save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def summary(self):
        s = self.state
        return f"{self.path}: {s['games']:,} games, {s['positions']:,} positions, {s['bytes'] / 1e6:.1f} MB"

    def export_gen(self, gen, folder):
        """Append the games of data/gen_XXXX (`folder`) unless already done.
        Returns (games, positions) appended."""
        if gen in self.exported:
            return 0, 0
        import peazero_rs as pz
        if not hasattr(pz, "chunk_to_viriformat"):
            raise RuntimeError("peazero_rs is out of date (no viriformat export); run `pip install ./crates/py`")
        parts, games, positions = [], 0, 0
        for path in sorted(glob.glob(os.path.join(folder, "*.bin"))):
            with open(path, "rb") as f:
                data, g, p = pz.chunk_to_viriformat(f.read())
            parts.append(data)
            games += g
            positions += p
        mode = "r+b" if os.path.exists(self.path) else "wb"
        with open(self.path, mode) as f:
            f.seek(self.state["bytes"])
            f.truncate()  # drop the tail of an interrupted append, if any
            for data in parts:
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
            size = f.tell()
        self.state["bytes"] = size
        self.state["gens"] = sorted(self.exported | {gen})
        self.state["games"] += games
        self.state["positions"] += positions
        self._save()
        self.exported.add(gen)
        return games, positions


def data_gen_dirs(run):
    out = []
    for path in glob.glob(os.path.join(run, "data", "gen_*")):
        m = re.fullmatch(r"gen_(\d+)", os.path.basename(path))
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def trainer_running(run):
    try:
        with open(os.path.join(run, "status.json")) as f:
            st = json.load(f)
        return st.get("phase") != "stopped" and time.time() - st.get("updated", 0) < 90
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    if trainer_running(args.run):
        sys.exit("training is running for this run; stop it first (it exports by itself with --viriformat)")
    ex = VfExporter(args.run)
    todo = [(g, d) for g, d in data_gen_dirs(args.run) if g not in ex.exported]
    for g, d in todo:
        games, positions = ex.export_gen(g, d)
        print(f"gen {g}: {games} games, {positions:,} positions")
    print(f"exported {len(todo)} generations; {ex.summary()}")


if __name__ == "__main__":
    main()
