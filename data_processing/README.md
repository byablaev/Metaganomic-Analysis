# data_processing — Обработка сырых данных 16S рРНК

Универсальный конвейер обработки парных FASTQ-файлов ампликонного секвенирования 16S рРНК.  
Работает с **любым датасетом** (SRA, собственное секвенирование, публичные данные).

**Выход:** три CSV-файла для загрузки в [BIOMIDAS](https://github.com/byablaev/BiomidasMetaAnalysis):
`otu_table.csv`, `taxonomy.csv`, `metadata.csv`

---

## Требования

- **Docker Desktop** — https://www.docker.com/products/docker-desktop/
- **16 ГБ RAM** (требуется DADA2)
- **~30 ГБ свободного места** на диске

> QIIME2 не работает нативно на Windows. Docker решает эту проблему.

---

## Быстрый старт

### 1. Подготовить FASTQ-файлы

Скопировать парные FASTQ-файлы в папку `fastq/` рядом с этим README.  
Файлы должны называться по одному из шаблонов:
```
<имя_образца>_1.fastq.gz  +  <имя_образца>_2.fastq.gz
<имя_образца>_R1.fastq.gz +  <имя_образца>_R2.fastq.gz
```

### 2. Создать metadata.csv

Создать файл `metadata.csv` в этой же папке (не в `fastq/`).  
Первый столбец `SampleID` должен совпадать с именами файлов FASTQ (без `_1.fastq.gz`).

```
SampleID,group,timepoint
Sample1,Control,T0
Sample2,Control,T0
Sample3,Treatment,T1
```

Шаблон: `metadata_example.csv`  
Пример для датасета Wang et al.: `metadata_example_wang.csv`

### 3. Запустить

Дважды кликнуть **`run_pipeline.bat`** → выбрать `[1] Собрать образ + запустить`.

### 4. Скопировать результаты в BIOMIDAS

```
output/exported/biomidas/otu_table.csv  →  ../data/otu_table.csv
output/exported/biomidas/taxonomy.csv   →  ../data/taxonomy.csv
output/exported/biomidas/metadata.csv   →  ../data/metadata.csv
```

---

## Обработка датасета Wang et al. (SRA: PRJNA610525)

Датасет: 63 образца почвенного микробиома, парное секвенирование 2×250 bp,  
праймеры 515F/806R (регион V4), 3 группы лечения × 3 временные точки.

### Шаг 1 — Скачать FASTQ из SRA

```bash
# Установить SRA Toolkit если нет: https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
# Скачать все 63 образца (SRR11573995–SRR11574057)

for i in $(seq 11573995 11574057); do
    fastq-dump --split-files --gzip SRR${i} --outdir fastq/
done
```

На Windows скачать через браузер или использовать SRA Run Selector:
1. Открыть https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA610525
2. Выбрать все 63 образца → «Download» → «Accession List»
3. Использовать `prefetch` + `fasterq-dump` из SRA Toolkit

### Шаг 2 — Подготовить метаданные

Скопировать готовый файл метаданных:
```
metadata_example_wang.csv  →  metadata.csv
```

Файл содержит 63 образца с колонками `SampleID`, `treatment`, `phase`:
- `treatment`: Control / SingleMix / MultiMix
- `phase`: TP1 / TP2 / TP3

### Шаг 3 — Настроить маппинг SRR → имена образцов

FASTQ-файлы называются по SRR-номерам (`SRR11573995_1.fastq.gz`),  
а метаданные используют описательные имена (`PC1_1_A1`).  
Файл `srr_to_sample_map.csv` содержит маппинг — передать его в манифест:

```bash
python3 01_create_manifest.py fastq/ \
    --sample-map srr_to_sample_map.csv \
    --output output/manifest.csv
```

При запуске через `run_pipeline.bat` маппинг применяется автоматически.

**Обязательно проверьте маппинг** перед запуском:
1. Открыть https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA610525
2. Нажать «Metadata» → скачать `SraRunTable.txt`
3. Сравнить столбцы `Run` и `Sample Name` с `srr_to_sample_map.csv`

Если маппинг неверный — образцы окажутся перепутаны без предупреждения.

### Шаг 4 — Запустить пайплайн

Параметры для этого датасета уже настроены по умолчанию в `config.sh`:
- Праймеры: 515F/806R
- `TRUNC_LEN_F=250`, `TRUNC_LEN_R=200`

```
run_pipeline.bat → [1] Собрать образ + запустить
```

---

## Использование со своими данными

### FASTQ

Пайплайн принимает любые парные FASTQ с Illumina:
- Данные с вашего секвенатора
- Данные из SRA (скачать через `fastq-dump --split-files <SRR>`)
- Любые публичные датасеты

Единственное требование — парные риды (_1/_2 или _R1/_R2).

### Метаданные

Создайте `metadata.csv`. Минимальный формат:
```
SampleID,group
MySample1,Control
MySample2,Treatment
```
Столбцов может быть сколько угодно (`timepoint`, `age`, `site` и т.д.).  
`SampleID` должен **точно** совпадать с именем FASTQ-файла без суффикса `_1.fastq.gz`.

### Праймеры

Если ваш протокол использует другие праймеры — откройте `config.sh` и замените:
```bash
PRIMER_F="GTGYCAGCMGCCGCGGTAA"    # ← ваш форвард-праймер
PRIMER_R="GGACTACNVGGGTWTCTAAT"   # ← ваш реверс-праймер
```

Часто используемые праймеры:

| Регион | Форвард | Реверс |
|--------|---------|--------|
| V4 (EMP, **по умолчанию**) | `GTGYCAGCMGCCGCGGTAA` (515F) | `GGACTACNVGGGTWTCTAAT` (806R) |
| V3-V4 | `CCTACGGGNGGCWGCAG` (341F) | `GACTACHVGGGTATCTAATCC` (806R) |
| V1-V2 | `AGAGTTTGATCCTGGCTCAG` (8F) | `TGCTGCCTCCCGTAGGAGT` (338R) |
| ITS1 (грибы) | `CTTGGTCATTTAGAGGAAGTAA` | `GCTGCGTTCTTCATCGATGC` |

### Длины обрезки DADA2

Запустите пайплайн — он выполнит шаги 0–1 и создаст `output/qzv/01_demux_summary.qzv`.  
Откройте этот файл на https://view.qiime2.org.

На графике качества найдите позицию, где медианное качество (оранжевая линия) падает ниже Q25.  
Откройте `config.sh` и установите:
```bash
TRUNC_LEN_F=250   # ← позиция обрезки форвард-рида
TRUNC_LEN_R=200   # ← позиция обрезки реверс-рида
```
Требование: `TRUNC_LEN_F + TRUNC_LEN_R ≥ длина ампликона + 20` (для V4: ≥ 273).

Запустите снова — пайплайн продолжит с шага DADA2, пропустив уже выполненные.

---

## Файлы

| Файл | Назначение |
|------|-----------|
| `run_pipeline.bat` | Точка входа для Windows |
| `config.sh` | Все параметры пайплайна |
| `01_create_manifest.py` | Составляет манифест FASTQ для QIIME2 |
| `02_pipeline.sh` | Основной конвейер QIIME2 |
| `03_export_biomidas.py` | Конвертирует результаты в CSV для BIOMIDAS |
| `Dockerfile` | Docker-образ на базе QIIME2 2024.10 |
| `docker-compose.yml` | Альтернативный способ запуска |
| `environment.yml` | Conda-окружение (для Linux/WSL без Docker) |
| `metadata_example.csv` | Минимальный шаблон метаданных |
| `metadata_example_wang.csv` | Метаданные датасета Wang et al. |
| `srr_to_sample_map.csv` | Маппинг SRR → имена образцов Wang et al. |
| `fastq/` | Сюда кладутся FASTQ-файлы |
| `output/` | Сюда записываются все результаты |

---

## Структура вывода

```
output/
├── qzv/                          # QIIME2 визуализации (открывать на view.qiime2.org)
│   ├── 01_demux_summary.qzv      # качество ридов — смотреть для выбора TRUNC_LEN
│   ├── 03_dada2_stats.qzv        # статистика денойзинга
│   └── ...
├── logs/                         # логи каждого шага
└── exported/
    └── biomidas/
        ├── otu_table.csv         ← загружать в BIOMIDAS
        ├── taxonomy.csv          ← загружать в BIOMIDAS
        └── metadata.csv          ← загружать в BIOMIDAS
```

---

## Этапы пайплайна

| # | Инструмент | Что происходит |
|---|-----------|----------------|
| 0 | Python | Сканирует `fastq/`, создаёт манифест |
| 1 | QIIME2 | Импортирует парные риды |
| 2 | cutadapt | Удаляет праймеры |
| 3 | DADA2 | Денойзинг → ASV, удаление химер |
| 4 | sklearn + SILVA 138 | Таксономическая классификация |
| 5 | QIIME2 | Фильтрация редких ASV |
| 6 | biom | BIOM → TSV |
| 7 | Python | TSV → CSV с колонками Kingdom…Species |

Шаги пропускаются автоматически если уже выполнены (при повторном запуске).

---

## Диагностика ошибок

**"No sample IDs match"**  
`SampleID` в `metadata.csv` не совпадают с именами FASTQ-файлов.  
Имя файла `Sample1_1.fastq.gz` → SampleID должен быть `Sample1`.

**"Too few reads passed the filter"**  
Уменьшите `TRUNC_LEN_F` и `TRUNC_LEN_R` в `config.sh`.

**DADA2 работает несколько часов**  
Нормально для больших датасетов. Увеличьте `N_THREADS` и RAM в Docker (Settings → Resources).

---

## Запуск без Docker (Linux / WSL2)

```bash
conda env create -f environment.yml
conda activate qiime2-amplicon-2024.10
bash 02_pipeline.sh
```

---

## Цитирование

- **QIIME2**: Bolyen E. et al. (2019) *Nature Biotechnology* 37:852–857
- **DADA2**: Callahan B.J. et al. (2016) *Nature Methods* 13:581–583
- **SILVA**: Quast C. et al. (2013) *Nucleic Acids Research* 41:D590–D596
- **cutadapt**: Martin M. (2011) *EMBnet.journal* 17:10–12
