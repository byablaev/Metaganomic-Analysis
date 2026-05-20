# data_processing — Обработка сырых данных 16S рРНК

Эта папка содержит полный конвейер обработки сырых данных секвенирования для проекта [BiomidasMetaAnalysis](https://github.com/byablaev/BiomidasMetaAnalysis).

**Вход:** парные FASTQ-файлы (SRA: SRR11573995–SRR11574057, датасет Wang et al.)  
**Выход:** три CSV-файла, готовых к загрузке в BIOMIDAS (`otu_table.csv`, `taxonomy.csv`, `metadata.csv`)

---

## Место этой папки в проекте

```
BiomidasMetaAnalysis/
├── data_processing/       ← вы здесь (обработка сырых данных)
│   ├── run_pipeline.bat
│   ├── ...
├── data/                  ← сюда скопировать результаты
│   ├── otu_table.csv
│   ├── taxonomy.csv
│   └── metadata.csv
├── main.py                ← BIOMIDAS (запускается после обработки)
└── ...
```

Полный рабочий процесс:
1. Запустить пайплайн в `data_processing/` → получить три CSV
2. Скопировать CSV в папку `data/`
3. Запустить BIOMIDAS (`Запустить_BIOMIDAS.bat` в корне репо)

---

## Требования

- **Docker Desktop** — https://www.docker.com/products/docker-desktop/
- **16 ГБ RAM** (требуется для DADA2)
- **~30 ГБ свободного места** (SILVA-классификатор + промежуточные файлы)
- FASTQ-файлы в папке `E:\metagenomes\fastq`

> QIIME2 не работает нативно на Windows. Docker решает эту проблему — устанавливать Linux или WSL не нужно.

---

## Быстрый старт

### 1. Установить Docker Desktop и запустить его

### 2. Проверить FASTQ-файлы

Убедитесь, что в `E:\metagenomes\fastq` находятся файлы вида:
```
SRR11573995_1.fastq.gz   SRR11573995_2.fastq.gz
SRR11573996_1.fastq.gz   SRR11573996_2.fastq.gz
...
```
Всего должно быть 126 файлов (63 образца × 2).

### 3. Запустить пайплайн

Дважды кликнуть **`run_pipeline.bat`** → выбрать `[1] Собрать образ + запустить`.

При первом запуске Docker скачает QIIME2 (~5 ГБ) и SILVA-классификатор (~900 МБ) — это происходит один раз.  
Время выполнения: **30–90 минут** в зависимости от CPU.

### 4. Скопировать результаты в BIOMIDAS

```
data_processing/output/exported/biomidas/otu_table.csv  →  data/otu_table.csv
data_processing/output/exported/biomidas/taxonomy.csv   →  data/taxonomy.csv
data_processing/output/exported/biomidas/metadata.csv   →  data/metadata.csv
```

---

## Файлы

| Файл | Назначение |
|------|-----------|
| `run_pipeline.bat` | Точка входа для Windows — запускать этот файл |
| `config.sh` | Параметры пайплайна (праймеры, треды, пороги фильтрации) |
| `01_create_manifest.py` | Составляет список FASTQ-файлов для QIIME2 |
| `02_pipeline.sh` | Основной конвейер QIIME2 |
| `03_export_biomidas.py` | Конвертирует результаты QIIME2 в CSV для BIOMIDAS |
| `Dockerfile` | Docker-образ с QIIME2 и скриптами пайплайна |
| `metadata_wang.csv` | Метаданные 63 образцов Wang et al. |
| `srr_to_sample_map.csv` | Соответствие SRR-номеров описательным именам образцов |
| `environment.yml` | Conda-окружение (для запуска без Docker) |

---

## Этапы пайплайна

| # | Инструмент | Что происходит |
|---|-----------|----------------|
| 0 | Python | Сканирует папку FASTQ, создаёт манифест для QIIME2 |
| 1 | QIIME2 | Импортирует парные риды |
| 2 | cutadapt | Удаляет праймеры (515F / 806R) |
| 3 | DADA2 | Денойзинг → ASV, объединение парных ридов, удаление химер |
| 4 | sklearn + SILVA 138 | Таксономическая классификация каждого ASV |
| 5 | QIIME2 | Фильтрация: убирает ASV с < 10 ридов или < 2 образцов |
| 6 | biom | Конвертация таблицы BIOM → TSV |
| 7 | Python | Разбивает строку SILVA на столбцы Kingdom…Species, сохраняет CSV |

---

## Настройка параметров (`config.sh`)

### Длины обрезки DADA2

```bash
TRUNC_LEN_F=250   # форвард-рид
TRUNC_LEN_R=200   # реверс-рид
```

После первого запуска откройте `output/qzv/01_demux_summary.qzv` на  
https://view.qiime2.org и посмотрите графики качества.  
Обрезайте там, где медиана качества падает ниже Q25.  
Сумма `TRUNC_LEN_F + TRUNC_LEN_R` должна быть ≥ 273 (ампликон 515F/806R ≈ 253 нп + перекрытие 20 нп).

### Праймеры

```bash
PRIMER_F="GTGYCAGCMGCCGCGGTAA"    # 515F
PRIMER_R="GGACTACNVGGGTWTCTAAT"   # 806R
```

### Ресурсы

```bash
N_THREADS=8    # число потоков (рекомендуется: кол-во ядер CPU − 2)
```

---

## Проверка соответствия SRR → имя образца

FASTQ-файлы названы по SRA-номерам (`SRR11573995`), а метаданные используют описательные имена (`PC1_1_A1`).  
Файл `srr_to_sample_map.csv` содержит предполагаемое соответствие — его **необходимо проверить**:

1. Открыть https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA610525
2. Нажать «Metadata» → скачать `SraRunTable.txt`
3. Сравнить столбцы `Run` и `Sample Name` с `srr_to_sample_map.csv`
4. Исправить расхождения при необходимости

Если маппинг неверный — образцы будут перепутаны без предупреждения.

---

## Диагностика ошибок

**"No sample IDs match"**  
Имена образцов в манифесте не совпадают с метаданными. Проверьте `srr_to_sample_map.csv`.

**"Too few reads passed the filter"**  
Слишком агрессивная обрезка. Уменьшите `TRUNC_LEN_F` и `TRUNC_LEN_R` в `config.sh`.

**DADA2 работает несколько часов**  
Нормально для 63 образцов. Увеличьте `N_THREADS` и объём RAM в Docker (Settings → Resources).

Логи каждого шага — в папке `output/logs/`. Просмотр через пункт `[4] Логи` в `run_pipeline.bat`.

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
