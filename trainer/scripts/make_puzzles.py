#!/usr/bin/env python3
"""Make the fixed puzzle set for the evaluator (trainer/data/puzzles.csv).

Streams the Lichess puzzle database (CC0, ~4M puzzles, ~250 MB download)
and keeps a fixed, seeded, rating-stratified sample of well-established
puzzles:

    pip install zstandard
    python trainer/scripts/make_puzzles.py                      # download + sample
    python trainer/scripts/make_puzzles.py --input lichess_db_puzzle.csv.zst   # from a local copy

Default: 250 puzzles per 200-point band from 400 to 3000 (~3,250 puzzles).
Only puzzles with enough plays, a settled rating and a decent popularity
are used, so the ratings mean something. Keep the file fixed for a run:
the curves are only comparable on the same set.
"""
import argparse
import csv
import io
import os
import random
import sys
import urllib.request

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "puzzles.csv")


def open_rows(source):
    """CSV rows from a local .csv / .csv.zst file or the URL."""
    if source.endswith(".csv"):
        f = open(source, newline="", encoding="utf-8")
        return csv.DictReader(f)
    try:
        import zstandard
    except ImportError:
        sys.exit("needs the zstandard package: pip install zstandard")
    raw = urllib.request.urlopen(source) if source.startswith("http") else open(source, "rb")
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", newline="")
    return csv.DictReader(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=URL, help="Lichess puzzle CSV (.csv / .csv.zst) path or URL")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--per-band", type=int, default=250)
    ap.add_argument("--band-width", type=int, default=200)
    ap.add_argument("--min-rating", type=int, default=400)
    ap.add_argument("--max-rating", type=int, default=3000)
    ap.add_argument("--min-plays", type=int, default=1000)
    ap.add_argument("--max-deviation", type=int, default=80)
    ap.add_argument("--min-popularity", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bands = {}  # band start -> reservoir
    seen = {}
    total = 0
    print(f"reading {args.input} ...")
    for row in open_rows(args.input):
        total += 1
        if total % 500_000 == 0:
            print(f"  {total:,} puzzles read")
        try:
            rating = int(row["Rating"])
            if (int(row["NbPlays"]) < args.min_plays or int(row["RatingDeviation"]) > args.max_deviation
                    or int(row["Popularity"]) < args.min_popularity):
                continue
        except (KeyError, ValueError):
            continue
        if not args.min_rating <= rating < args.max_rating:
            continue
        b = (rating - args.min_rating) // args.band_width
        res = bands.setdefault(b, [])
        seen[b] = seen.get(b, 0) + 1
        keep = {"PuzzleId": row["PuzzleId"], "FEN": row["FEN"], "Moves": row["Moves"],
                "Rating": rating, "Themes": row.get("Themes", "")}
        # reservoir sampling: a uniform sample of each band, independent of file order
        if len(res) < args.per_band:
            res.append(keep)
        else:
            j = rng.randrange(seen[b])
            if j < args.per_band:
                res[j] = keep

    rows = sorted((r for res in bands.values() for r in res), key=lambda r: (r["Rating"], r["PuzzleId"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PuzzleId", "FEN", "Moves", "Rating", "Themes"])
        w.writeheader()
        w.writerows(rows)
    print(f"read {total:,} puzzles; wrote {len(rows):,} to {args.out}")
    for b in sorted(bands):
        lo = args.min_rating + b * args.band_width
        print(f"  {lo}-{lo + args.band_width}: {len(bands[b])}")


if __name__ == "__main__":
    main()
