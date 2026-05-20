#!/usr/bin/env bash
# config.sh — параметры пайплайна
# Все переменные можно переопределить через окружение (Docker делает это автоматически)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Пути — относительно папки со скриптами, менять не нужно
FASTQ_DIR="${FASTQ_DIR:-${SCRIPT_DIR}/fastq}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
METADATA_FILE="${METADATA_FILE:-${SCRIPT_DIR}/metadata.csv}"
SILVA_CLASSIFIER="${SILVA_CLASSIFIER:-${SCRIPT_DIR}/silva-138-99-seqs-515-806.qza}"

# --- Праймеры ----------------------------------------------------------------
# Замените на свои если протокол отличается от EMP V4 (515F/806R)
PRIMER_F="GTGYCAGCMGCCGCGGTAA"    # 515F  (EMP V4)
PRIMER_R="GGACTACNVGGGTWTCTAAT"   # 806R  (EMP V4)
#
# Другие распространённые варианты:
#   V3-V4:  PRIMER_F="CCTACGGGNGGCWGCAG"     PRIMER_R="GACTACHVGGGTATCTAATCC"
#   ITS1:   PRIMER_F="CTTGGTCATTTAGAGGAAGTAA" PRIMER_R="GCTGCGTTCTTCATCGATGC"

# --- DADA2: длины обрезки ----------------------------------------------------
# Установите ПОСЛЕ просмотра output/qzv/01_demux_summary.qzv на view.qiime2.org
# Правило: обрезать там где медиана качества падает ниже Q25
# Требование: TRUNC_LEN_F + TRUNC_LEN_R >= длина ампликона + 20 нп перекрытия
TRUNC_LEN_F="${TRUNC_LEN_F:-250}"
TRUNC_LEN_R="${TRUNC_LEN_R:-200}"
TRIM_LEFT_F="${TRIM_LEFT_F:-0}"    # 0 = праймеры уже удалены cutadapt
TRIM_LEFT_R="${TRIM_LEFT_R:-0}"

# --- Фильтрация --------------------------------------------------------------
MIN_FREQUENCY="${MIN_FREQUENCY:-10}"   # минимум ридов на ASV
MIN_SAMPLES="${MIN_SAMPLES:-2}"        # минимум образцов где ASV присутствует

# --- Ресурсы -----------------------------------------------------------------
N_THREADS="${N_THREADS:-8}"

QIIME2_ENV="qiime2-amplicon-2024.10"
