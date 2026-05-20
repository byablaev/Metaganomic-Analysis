#!/usr/bin/env python3
"""Convert QIIME2 exports to BIOMIDAS CSV format.

Outputs:
    otu_table.csv  — samples x features, integer counts
    taxonomy.csv   — OTUID + Kingdom..Species columns
    metadata.csv   — aligned to samples present in otu_table
"""

import re
import sys
import argparse
from pathlib import Path
from io import StringIO

import pandas as pd


RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

# SILVA prefixes (d__/k__ both map to Kingdom)
_PREFIX_MAP = {
    "d__": "Kingdom", "k__": "Kingdom",
    "p__": "Phylum",  "c__": "Class",
    "o__": "Order",   "f__": "Family",
    "g__": "Genus",   "s__": "Species",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-table", required=True)
    p.add_argument("--taxonomy",      required=True)
    p.add_argument("--metadata",      required=True)
    p.add_argument("--output-dir",    default="./biomidas")
    p.add_argument("--min-prevalence", type=float, default=0.0,
                   help="Drop features in fewer than this fraction of samples (0-1)")
    p.add_argument("--no-integer", action="store_true",
                   help="Keep fractional counts instead of rounding")
    return p.parse_args()


def load_feature_table(path):
    # BIOM TSV has a leading comment line; drop it before parsing
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return pd.read_csv(StringIO("".join(lines)), sep="\t", index_col=0)


def _parse_taxon(taxon):
    result = {r: "" for r in RANKS}
    if not isinstance(taxon, str) or taxon.strip() in ("", "Unassigned"):
        return result
    parts = [p.strip() for p in re.split(r"[;,]", taxon) if p.strip()]
    for i, part in enumerate(parts):
        for prefix, rank in _PREFIX_MAP.items():
            if part.lower().startswith(prefix.lower()):
                value = part[len(prefix):].strip()
                if value and value != "__":
                    result[rank] = value
                break
        else:
            if i == 0:
                result["Kingdom"] = part  # unprefixed first token
    return result


def load_taxonomy(path):
    raw = pd.read_csv(path, sep="\t", index_col=0)
    if "Taxon" not in raw.columns:
        raise ValueError(f"No 'Taxon' column in {path} — found: {list(raw.columns)}")
    parsed = raw["Taxon"].apply(_parse_taxon).apply(pd.Series)
    parsed.index.name = "OTUID"
    return parsed


def load_metadata(path):
    with open(path, errors="replace") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    df = pd.read_csv(path, sep=sep, index_col=0, dtype=str)
    df.index.name = "SampleID"
    df.index = df.index.astype(str).str.strip()
    return df


def align_and_export(feature_table, taxonomy, metadata,
                     output_dir, min_prevalence, integer_counts):
    otu = feature_table.T.copy()
    otu.index.name = "SampleID"
    otu.index   = otu.index.astype(str).str.strip()
    otu.columns = otu.columns.astype(str)

    if integer_counts:
        otu = otu.round().astype(int)

    otu = otu.loc[:, otu.sum() > 0]

    if min_prevalence > 0:
        otu = otu.loc[:, (otu > 0).mean(axis=0) >= min_prevalence]

    common = sorted(set(otu.index) & set(metadata.index))
    if not common:
        # fallback: normalise case and separators
        norm = lambda s: s.lower().replace("-", "_").replace(" ", "_")
        otu.index      = [norm(i) for i in otu.index]
        metadata.index = [norm(i) for i in metadata.index]
        common = sorted(set(otu.index) & set(metadata.index))
        if common:
            print(f"WARNING: IDs normalised for matching ({len(common)} matched)")
        else:
            print("ERROR: no sample IDs match between OTU table and metadata",
                  file=sys.stderr)
            print(f"  OTU:  {list(otu.index[:5])}", file=sys.stderr)
            print(f"  Meta: {list(metadata.index[:5])}", file=sys.stderr)
            sys.exit(1)
    elif len(common) < len(metadata):
        print(f"WARNING: {len(common)}/{len(metadata)} metadata samples in OTU table")

    otu_out  = otu.loc[common]
    meta_out = metadata.loc[common]

    tax_out = taxonomy.reindex(otu_out.columns)
    for rank in RANKS:
        if rank not in tax_out.columns:
            tax_out[rank] = ""
    tax_out = tax_out[RANKS].fillna("")
    tax_out.index.name = "OTUID"

    output_dir.mkdir(parents=True, exist_ok=True)
    otu_out.to_csv(output_dir / "otu_table.csv")
    tax_out.to_csv(output_dir / "taxonomy.csv")
    meta_out.to_csv(output_dir / "metadata.csv")

    print(f"samples={len(otu_out)}  features={len(otu_out.columns)}  "
          f"reads={int(otu_out.values.sum()):,}  "
          f"sparsity={( otu_out == 0).values.mean():.1%}")
    print(f"output: {output_dir}")

    for rank in RANKS:
        n = (tax_out[rank] != "").sum()
        print(f"  {rank:<10} {n}/{len(tax_out)} ({n/max(len(tax_out),1):.0%})")


def main():
    args = parse_args()

    print("Loading feature table ...")
    ft = load_feature_table(args.feature_table)
    print(f"  {ft.shape[0]} features x {ft.shape[1]} samples")

    print("Loading taxonomy ...")
    tax = load_taxonomy(args.taxonomy)
    print(f"  {len(tax)} features classified")

    print("Loading metadata ...")
    meta = load_metadata(args.metadata)
    print(f"  {len(meta)} samples")

    align_and_export(
        feature_table=ft,
        taxonomy=tax,
        metadata=meta,
        output_dir=Path(args.output_dir),
        min_prevalence=args.min_prevalence,
        integer_counts=not args.no_integer,
    )


if __name__ == "__main__":
    main()
