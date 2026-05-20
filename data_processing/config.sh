#!/usr/bin/env bash
# config.sh — edit before running 02_pipeline.sh

# All vars can be overridden by environment (Docker sets these automatically)
FASTQ_DIR="${FASTQ_DIR:-/mnt/e/metagenomes/fastq}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
METADATA_FILE="${METADATA_FILE:-./metadata_wang.csv}"
SILVA_CLASSIFIER="${SILVA_CLASSIFIER:-./silva-138-99-seqs-515-806.qza}"

PRIMER_F="GTGYCAGCMGCCGCGGTAA"    # 515F
PRIMER_R="GGACTACNVGGGTWTCTAAT"   # 806R

# Adjust after viewing output/qzv/01_demux_summary.qzv (truncate at Q25 drop)
TRUNC_LEN_F="${TRUNC_LEN_F:-250}"
TRUNC_LEN_R="${TRUNC_LEN_R:-200}"
TRIM_LEFT_F="${TRIM_LEFT_F:-0}"    # 0 = primers removed by cutadapt
TRIM_LEFT_R="${TRIM_LEFT_R:-0}"

MIN_FREQUENCY="${MIN_FREQUENCY:-10}"
MIN_SAMPLES="${MIN_SAMPLES:-2}"
N_THREADS="${N_THREADS:-8}"

QIIME2_ENV="qiime2-amplicon-2024.10"
