"""
scripts/txt_to_tsv.py

Convert plain-text source and reference files (one sentence per line) into a
TSV that the pipeline can process with load_tsv().

The output has the same column schema as train.tsv / test.tsv. Annotation
columns (cs, informal_lex, dialect, phonetic, density) are left empty.

Usage:
    python scripts/txt_to_tsv.py --src source.txt --ref reference.txt --out data/test.tsv
"""

import argparse
import csv
from pathlib import Path

COLUMNS = ["source_es", "ref_informal", "cs", "informal_lex", "dialect", "phonetic", "density", "ref_batua"]


def convert(src_path: Path, ref_path: Path, output_path: Path):
    sources = [l.strip() for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    refs    = [l.strip() for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if len(sources) != len(refs):
        raise ValueError(
            f"Line count mismatch: {src_path.name} has {len(sources)} lines, "
            f"{ref_path.name} has {len(refs)} lines."
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for src, ref in zip(sources, refs):
            writer.writerow({"source_es": src, "ref_informal": ref})

    print(f"Written {len(sources)} sentence pairs → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert source+reference .txt files to pipeline-compatible .tsv")
    parser.add_argument("--src", type=Path, required=True, help="Source .txt file (informal Spanish, one sentence per line)")
    parser.add_argument("--ref", type=Path, required=True, help="Reference .txt file (informal Basque, one sentence per line)")
    parser.add_argument("--out", type=Path, default=Path("data/test.tsv"), help="Output .tsv path (default: data/test.tsv)")
    args = parser.parse_args()

    convert(args.src, args.ref, args.out)


if __name__ == "__main__":
    main()
