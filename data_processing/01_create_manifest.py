#!/usr/bin/env python3
"""Generate QIIME2 PairedEndFastqManifestPhred33V2 manifest from a FASTQ directory.

Usage:
    python3 01_create_manifest.py <fastq_dir> [-o manifest.csv] [--sample-map srr_map.csv]

--sample-map: two-column CSV (srr_id, sample_id) to rename SRR accessions.
"""

import re
import sys
import csv
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fastq_dir")
    p.add_argument("-o", "--output", default="manifest.csv")
    p.add_argument("--sample-map", default=None)
    return p.parse_args()


def load_sample_map(path):
    mapping = {}
    sep = "\t" if path.endswith(".tsv") else ","
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter=sep):
            if not row or row[0].startswith("#"):
                continue
            if row[0].lower() in ("srr_id", "run", "accession"):
                continue
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def collect_pairs(fastq_dir):
    fastq_dir = Path(fastq_dir).resolve()
    r1_pat = re.compile(r"^(.+?)(?:_R?1|_1)\.fastq(?:\.gz)?$", re.IGNORECASE)
    r2_pat = re.compile(r"^(.+?)(?:_R?2|_2)\.fastq(?:\.gz)?$", re.IGNORECASE)

    forward, reverse = {}, {}
    for f in sorted(fastq_dir.iterdir()):
        if not f.is_file():
            continue
        m = r1_pat.match(f.name)
        if m:
            forward[m.group(1)] = f
            continue
        m = r2_pat.match(f.name)
        if m:
            reverse[m.group(1)] = f

    pairs, errors = [], []
    for sid in sorted(set(forward) | set(reverse)):
        if sid not in forward:
            errors.append(f"Missing R1: {sid}")
        elif sid not in reverse:
            errors.append(f"Missing R2: {sid}")
        else:
            pairs.append((sid, forward[sid], reverse[sid]))

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    return pairs


def main():
    args = parse_args()

    sample_map = load_sample_map(args.sample_map) if args.sample_map else {}
    if sample_map:
        print(f"Loaded {len(sample_map)} name mappings")

    pairs = collect_pairs(args.fastq_dir)
    print(f"Found {len(pairs)} pairs in {args.fastq_dir}")

    rows = []
    for raw_sid, fwd, rev in pairs:
        sample_id = sample_map.get(raw_sid, raw_sid).replace(" ", "_")
        rows.append({
            "sample-id": sample_id,
            "forward-absolute-filepath": str(fwd.resolve()),
            "reverse-absolute-filepath": str(rev.resolve()),
        })

    out_path = Path(args.output)
    fields = ["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Manifest: {out_path} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
