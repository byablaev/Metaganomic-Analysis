#!/usr/bin/env bash
# 16S amplicon pipeline: raw FASTQ -> OTU table + taxonomy (Wang et al.)
# Usage: conda activate qiime2-amplicon-2024.10 && bash 02_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
skip_if_exists() { [[ -f "$1" ]] && { log "  skip (exists): $1"; return 0; } || return 1; }

mkdir -p "${OUTPUT_DIR}"/{qza,qzv,exported,logs}

log "FASTQ: ${FASTQ_DIR}  |  OUT: ${OUTPUT_DIR}"

# --- manifest ----------------------------------------------------------------
MANIFEST="${OUTPUT_DIR}/manifest.csv"

python3 "${SCRIPT_DIR}/01_create_manifest.py" \
    "${FASTQ_DIR}" \
    --sample-map "${SCRIPT_DIR}/srr_to_sample_map.csv" \
    --output "${MANIFEST}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/00_manifest.log"

[[ -f "${MANIFEST}" ]] || die "Manifest not created"

# --- import ------------------------------------------------------------------
DEMUX_QZA="${OUTPUT_DIR}/qza/01_demux-paired.qza"

skip_if_exists "${DEMUX_QZA}" || \
qiime tools import \
    --type 'SampleData[PairedEndSequencesWithQuality]' \
    --input-path "${MANIFEST}" \
    --output-path "${DEMUX_QZA}" \
    --input-format PairedEndFastqManifestPhred33V2 \
    2>&1 | tee "${OUTPUT_DIR}/logs/01_import.log"

# view at https://view.qiime2.org to choose TRUNC_LEN_F/R
qiime demux summarize \
    --i-data "${DEMUX_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/01_demux_summary.qzv" \
    --quiet

# --- cutadapt ----------------------------------------------------------------
TRIMMED_QZA="${OUTPUT_DIR}/qza/02_trimmed-paired.qza"

skip_if_exists "${TRIMMED_QZA}" || \
qiime cutadapt trim-paired \
    --i-demultiplexed-sequences "${DEMUX_QZA}" \
    --p-front-f "${PRIMER_F}" \
    --p-front-r "${PRIMER_R}" \
    --p-discard-untrimmed \
    --p-overlap 12 \
    --p-cores "${N_THREADS}" \
    --o-trimmed-sequences "${TRIMMED_QZA}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/02_cutadapt.log"

qiime demux summarize \
    --i-data "${TRIMMED_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/02_trimmed_summary.qzv" \
    --quiet

# --- DADA2 -------------------------------------------------------------------
TABLE_QZA="${OUTPUT_DIR}/qza/03_table.qza"
REP_SEQS_QZA="${OUTPUT_DIR}/qza/03_rep-seqs.qza"
DADA2_STATS_QZA="${OUTPUT_DIR}/qza/03_dada2-stats.qza"

skip_if_exists "${TABLE_QZA}" || \
qiime dada2 denoise-paired \
    --i-demultiplexed-seqs "${TRIMMED_QZA}" \
    --p-trunc-len-f "${TRUNC_LEN_F}" \
    --p-trunc-len-r "${TRUNC_LEN_R}" \
    --p-trim-left-f "${TRIM_LEFT_F}" \
    --p-trim-left-r "${TRIM_LEFT_R}" \
    --p-n-threads "${N_THREADS}" \
    --p-chimera-method consensus \
    --o-table "${TABLE_QZA}" \
    --o-representative-sequences "${REP_SEQS_QZA}" \
    --o-denoising-stats "${DADA2_STATS_QZA}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/03_dada2.log"

qiime metadata tabulate \
    --m-input-file "${DADA2_STATS_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/03_dada2_stats.qzv" --quiet

qiime feature-table summarize \
    --i-table "${TABLE_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/03_table_summary.qzv" --quiet

# --- taxonomy ----------------------------------------------------------------
if [[ ! -f "${SILVA_CLASSIFIER}" ]]; then
    log "Downloading SILVA 138.1 classifier ..."
    wget -nv -O "${SILVA_CLASSIFIER}" \
        "https://data.qiime2.org/classifiers/sklearn-1.4.2/silva/silva-138-99-seqs-515-806.qza" \
        2>&1 | tee "${OUTPUT_DIR}/logs/04_classifier_download.log"
fi

TAXONOMY_QZA="${OUTPUT_DIR}/qza/04_taxonomy.qza"

skip_if_exists "${TAXONOMY_QZA}" || \
qiime feature-classifier classify-sklearn \
    --i-classifier "${SILVA_CLASSIFIER}" \
    --i-reads "${REP_SEQS_QZA}" \
    --p-n-jobs "${N_THREADS}" \
    --p-confidence 0.7 \
    --o-classification "${TAXONOMY_QZA}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/04_taxonomy.log"

qiime metadata tabulate \
    --m-input-file "${TAXONOMY_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/04_taxonomy.qzv" --quiet

# --- filter ------------------------------------------------------------------
TABLE_FILT_QZA="${OUTPUT_DIR}/qza/05_table-filtered.qza"
REP_SEQS_FILT_QZA="${OUTPUT_DIR}/qza/05_rep-seqs-filtered.qza"

qiime feature-table filter-features \
    --i-table "${TABLE_QZA}" \
    --p-min-frequency "${MIN_FREQUENCY}" \
    --p-min-samples "${MIN_SAMPLES}" \
    --o-filtered-table "${TABLE_FILT_QZA}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/05_filter.log"

qiime feature-table filter-seqs \
    --i-data "${REP_SEQS_QZA}" \
    --i-table "${TABLE_FILT_QZA}" \
    --o-filtered-data "${REP_SEQS_FILT_QZA}" --quiet

qiime feature-table summarize \
    --i-table "${TABLE_FILT_QZA}" \
    --o-visualization "${OUTPUT_DIR}/qzv/05_table_filtered_summary.qzv" --quiet

# --- export ------------------------------------------------------------------
EXPORT_DIR="${OUTPUT_DIR}/exported"

qiime tools export \
    --input-path "${TABLE_FILT_QZA}" \
    --output-path "${EXPORT_DIR}/feature-table" \
    2>&1 | tee "${OUTPUT_DIR}/logs/06_export.log"

biom convert \
    -i "${EXPORT_DIR}/feature-table/feature-table.biom" \
    -o "${EXPORT_DIR}/feature-table/feature-table.tsv" \
    --to-tsv 2>&1 | tee -a "${OUTPUT_DIR}/logs/06_export.log"

qiime tools export --input-path "${TAXONOMY_QZA}"        --output-path "${EXPORT_DIR}/taxonomy"
qiime tools export --input-path "${REP_SEQS_FILT_QZA}"  --output-path "${EXPORT_DIR}/rep-seqs"

# --- BIOMIDAS CSV ------------------------------------------------------------
python3 "${SCRIPT_DIR}/03_export_biomidas.py" \
    --feature-table  "${EXPORT_DIR}/feature-table/feature-table.tsv" \
    --taxonomy       "${EXPORT_DIR}/taxonomy/taxonomy.tsv" \
    --metadata       "${METADATA_FILE}" \
    --output-dir     "${EXPORT_DIR}/biomidas" \
    2>&1 | tee "${OUTPUT_DIR}/logs/07_biomidas_export.log"

log "Done. BIOMIDAS files: ${EXPORT_DIR}/biomidas/"
