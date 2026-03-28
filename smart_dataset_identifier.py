"""
SmartDatasetIdentifier — Умная идентификация микробиомных датасетов
═══════════════════════════════════════════════════════════════════
Автоматически понимает структуру любого датасета:
  • Формат файлов: QIIME2, BIOM, mothur, phyloseq, таблицы с любым разделителем
  • Ориентация OTU таблицы (образцы в строках или столбцах)
  • Маппинг колонок таксономии (k__, p__, Phylum, phylum, Fil, Отдел...)
  • Определение типа датасета (16S, WGS, ампликон, маркерный ген)
  • Интеллектуальный анализ метаданных (группировка, ковариаты, даты, парные образцы)
  • Движок сравнений: что с чем сравнивать, референсная группа, балансировка
"""

from __future__ import annotations

import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
class DatasetFingerprint:
    """
    Результат идентификации датасета.
    Содержит все обнаруженные характеристики и рекомендации.
    """
    def __init__(self):
        self.dataset_type: str = 'unknown'           # '16S', 'WGS', 'ITS', 'generic'
        self.otu_orientation: str = 'unknown'         # 'samples_as_rows' / 'samples_as_columns'
        self.n_samples: int = 0
        self.n_features: int = 0
        self.n_reads_total: float = 0
        self.sparsity: float = 0
        self.detected_taxonomy_map: Dict[str, str] = {}   # original → standard
        self.taxonomy_format: str = 'unknown'          # 'qiime2', 'standard', 'abbreviated', 'silva'
        self.metadata_profile: Dict = {}               # column → {type, role, n_unique, values}
        self.group_columns: List[str] = []             # рекомендуемые для группировки
        self.numeric_covariates: List[str] = []        # числовые ковариаты
        self.date_columns: List[str] = []              # даты
        self.id_columns: List[str] = []                # колонки-идентификаторы
        self.confidence: float = 0.0                   # 0–1, уверенность в идентификации
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.comparison_engine: Optional['ComparisonEngine'] = None
        # ID matching
        self.sample_match_ratio: float = 0.0
        self.otu_taxonomy_match_ratio: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
class SmartDatasetIdentifier:
    """
    Умный идентификатор: читает ЛЮБОЙ формат микробиомного датасета.
    """

    # ── Таксономические уровни: все возможные имена → стандарт ──
    TAXONOMY_SYNONYMS = {
        'Kingdom':  ['kingdom', 'k__', 'k', 'царство', 'domain', 'Superkingdom',
                     'Domain', 'd__', 'd', 'reich', 'domaine'],
        'Phylum':   ['phylum', 'p__', 'p', 'тип', 'филум', 'division', 'abteilung',
                     'fil', 'división', 'embranchement', 'فیلوم'],
        'Class':    ['class', 'c__', 'c', 'класс', 'klasse', 'clase', 'klasa',
                     'classe', 'cls'],
        'Order':    ['order', 'o__', 'o', 'отряд', 'порядок', 'orden', 'ordnung',
                     'ord'],
        'Family':   ['family', 'f__', 'f', 'семейство', 'familia', 'familie',
                     'fam', 'famille'],
        'Genus':    ['genus', 'g__', 'g', 'род', 'género', 'gattung', 'gen',
                     'genre'],
        'Species':  ['species', 's__', 's', 'вид', 'especie', 'art', 'sp', 'spp',
                     'espèce'],
    }

    # ── Паттерны OTU/ASV идентификаторов ──
    OTU_PATTERNS = [
        r'^OTU\d+', r'^ASV\d+', r'^Zotu\d+', r'^Feature\d+', r'^denovo\d+',
        r'^[0-9a-f]{32}$',                  # MD5 хэши (DADA2)
        r'^[0-9a-f]{40}$',                  # SHA1 хэши
        r'^M\d+', r'^SRR\d+',              # архивные ID
        r'^\d{6,}$',                        # числовые OTU ID (mothur)
        r'^[A-Z]{5,}$',                     # SILVA короткие коды
    ]

    # ── Слова, указывающие на группировочные переменные ──
    GROUP_KEYWORDS = [
        'group', 'treatment', 'condition', 'status', 'type', 'class',
        'cohort', 'arm', 'diet', 'disease', 'diagnosis', 'health',
        'patient', 'control', 'case', 'sex', 'gender', 'site', 'location',
        'country', 'habitat', 'source', 'collection', 'body_site', 'bodysite',
        'time_point', 'timepoint', 'stage', 'severity', 'ethnicity', 'race',
        'группа', 'лечение', 'статус', 'контроль', 'больной', 'здоровый',
        'пол', 'диета', 'диагноз'
    ]

    # ── Слова, указывающие на числовые ковариаты ──
    COVARIATE_KEYWORDS = [
        'age', 'bmi', 'weight', 'height', 'score', 'index', 'concentration',
        'level', 'count', 'value', 'measure', 'dose', 'duration', 'volume',
        'возраст', 'вес', 'рост', 'индекс', 'концентрация'
    ]

    # ── Слова, указывающие на ID ──
    ID_KEYWORDS = [
        'id', 'identifier', 'sample', 'name', 'label', 'code', 'barcode',
        'sampleid', 'sample_id', '#sampleid', 'субъект', 'пациент', 'id_образца'
    ]

    def __init__(self):
        self._otu_patterns = [re.compile(p, re.IGNORECASE) for p in self.OTU_PATTERNS]

    # ──────────────────────────────────────────────────────────────────
    def identify(self,
                 otu_df: pd.DataFrame,
                 metadata_df: Optional[pd.DataFrame] = None,
                 taxonomy_df: Optional[pd.DataFrame] = None) -> DatasetFingerprint:
        """
        Главный метод: принимает три DataFrame и возвращает DatasetFingerprint.
        Работает даже если taxonomy_df = None.
        """
        fp = DatasetFingerprint()

        # 1. OTU ориентация
        if metadata_df is not None:
            fp.otu_orientation, fp.sample_match_ratio = self._detect_orientation(
                otu_df, metadata_df)
        else:
            fp.otu_orientation = self._guess_orientation_no_meta(otu_df)
            fp.sample_match_ratio = 0.5

        # 2. Статистика OTU
        # Нормализуем ориентацию (samples×features)
        working_otu = self._normalize_orientation(otu_df, fp.otu_orientation)
        fp.n_samples  = working_otu.shape[0]
        fp.n_features = working_otu.shape[1]
        fp.n_reads_total = float(working_otu.values.sum())
        total_cells = fp.n_samples * fp.n_features
        fp.sparsity = float((working_otu == 0).values.sum() / total_cells) if total_cells > 0 else 0

        # 3. Тип датасета
        fp.dataset_type, fp.confidence = self._detect_dataset_type(working_otu, taxonomy_df)

        # 4. Таксономия
        if taxonomy_df is not None:
            fp.detected_taxonomy_map, fp.taxonomy_format = self._map_taxonomy_columns(taxonomy_df)
            otu_ids = set(working_otu.columns.astype(str))
            tax_ids = set(taxonomy_df.index.astype(str))
            fp.otu_taxonomy_match_ratio = len(otu_ids & tax_ids) / max(len(otu_ids), 1)

        # 5. Метаданные
        if metadata_df is not None:
            fp.metadata_profile = self._profile_metadata(metadata_df)
            fp.group_columns       = self._pick_group_columns(fp.metadata_profile)
            fp.numeric_covariates  = self._pick_numeric_columns(fp.metadata_profile)
            fp.date_columns        = self._pick_date_columns(fp.metadata_profile)
            fp.id_columns          = self._pick_id_columns(fp.metadata_profile)

            # 6. Движок сравнений
            fp.comparison_engine = ComparisonEngine(
                metadata_df   = metadata_df,
                group_columns = fp.group_columns,
                numeric_cols  = fp.numeric_covariates,
            )

        # 7. Предупреждения и рекомендации
        fp.warnings, fp.suggestions = self._generate_diagnostics(fp, working_otu, metadata_df)

        return fp

    # ──────────────────────────────────────────────────────────────────
    #  ОРИЕНТАЦИЯ OTU ТАБЛИЦЫ
    # ──────────────────────────────────────────────────────────────────

    def _detect_orientation(self, df: pd.DataFrame,
                             metadata: pd.DataFrame) -> Tuple[str, float]:
        """
        Надёжное определение ориентации с уверенностью.
        Возвращает (ориентация, доля совпадений ID образцов).
        """
        sample_ids = set(metadata.index.astype(str))
        
        # Пробуем 3 стратегии сопоставления
        def match_ratio(candidates):
            # Use len() — avoids ambiguous truth value on pandas Index
            if len(candidates) == 0 or len(sample_ids) == 0:
                return 0.0
            cand_set = set(str(c) for c in candidates)
            direct = len(cand_set & sample_ids)
            if direct / len(sample_ids) > 0.5:
                return direct / len(sample_ids)
            # Fuzzy: нижний регистр, замена пробелов и дефисов
            norm_meta = {s.lower().replace('-', '_').replace(' ', '_') for s in sample_ids}
            norm_cand = {str(c).lower().replace('-', '_').replace(' ', '_') for c in candidates}
            n_meta = len(norm_meta)
            return len(norm_meta & norm_cand) / n_meta if n_meta > 0 else 0.0

        row_ratio = match_ratio(df.index)
        col_ratio = match_ratio(df.columns)

        if col_ratio > row_ratio and col_ratio > 0.3:
            return 'samples_as_columns', col_ratio
        elif row_ratio > 0.3:
            return 'samples_as_rows', row_ratio
        
        # Fallback: морфология (OTU ID обычно длиннее и содержат паттерны)
        is_otu_index = self._looks_like_otu_ids(df.index)
        is_otu_cols  = self._looks_like_otu_ids(df.columns)

        if is_otu_index and not is_otu_cols:
            return 'samples_as_columns', 0.4
        elif is_otu_cols and not is_otu_index:
            return 'samples_as_rows', 0.4
        
        # Если больше строк чем столбцов → обычно OTU в строках
        if df.shape[0] > df.shape[1]:
            return 'samples_as_columns', 0.3
        return 'samples_as_rows', 0.3

    def _guess_orientation_no_meta(self, df: pd.DataFrame) -> str:
        """Угадывает ориентацию без метаданных."""
        is_otu_index = self._looks_like_otu_ids(df.index)
        is_otu_cols  = self._looks_like_otu_ids(df.columns)
        if is_otu_index and not is_otu_cols:
            return 'samples_as_columns'
        if is_otu_cols and not is_otu_index:
            return 'samples_as_rows'
        return 'samples_as_columns' if df.shape[0] > df.shape[1] else 'samples_as_rows'

    def _looks_like_otu_ids(self, index) -> bool:
        """Проверяет, похожи ли значения индекса на OTU/ASV идентификаторы."""
        sample = [str(v) for v in list(index)[:20]]
        matches = sum(
            any(p.match(s) for p in self._otu_patterns)
            for s in sample
        )
        # Также проверяем длину (OTU ID обычно ≥10 символов или содержат хэш-паттерны)
        avg_len = np.mean([len(s) for s in sample]) if sample else 0
        long_ids = avg_len > 10
        return (matches / max(len(sample), 1) > 0.3) or long_ids

    def _normalize_orientation(self, df: pd.DataFrame, orientation: str) -> pd.DataFrame:
        """Возвращает DataFrame в формате samples×features."""
        if orientation == 'samples_as_columns':
            return df.T
        return df

    # ──────────────────────────────────────────────────────────────────
    #  ТИП ДАТАСЕТА
    # ──────────────────────────────────────────────────────────────────

    def _detect_dataset_type(self, working_otu: pd.DataFrame,
                               taxonomy_df: Optional[pd.DataFrame]) -> Tuple[str, float]:
        """
        Определяет тип датасета по характеристикам OTU/таксономии.
        """
        confidence = 0.5
        dtype = 'generic'

        # Признаки 16S: высокая разреженность, OTU идентификаторы
        sparsity = float((working_otu == 0).values.sum() /
                         max(working_otu.shape[0] * working_otu.shape[1], 1))

        n_features = working_otu.shape[1]

        # Проверка таксономии
        if taxonomy_df is not None:
            tax_str = ' '.join(str(c) for c in taxonomy_df.columns).lower()
            if any(t in tax_str for t in ['phylum', 'class', 'genus', 'kingdom']):
                dtype = '16S_amplicon'
                confidence = 0.85
            elif 'its' in tax_str or 'fungus' in tax_str or 'fungi' in tax_str:
                dtype = 'ITS_fungal'
                confidence = 0.80
            elif 'cog' in tax_str or 'ko' in tax_str or 'pathway' in tax_str:
                dtype = 'WGS_functional'
                confidence = 0.80

        # Характеристики OTU
        if dtype == 'generic':
            if sparsity > 0.8 and n_features > 100:
                dtype = '16S_amplicon'
                confidence = 0.60
            elif sparsity < 0.5 and n_features < 500:
                dtype = 'WGS_species'
                confidence = 0.55

        # Проверка по именам колонок
        col_names = [str(c) for c in working_otu.columns[:20]]
        col_str = ' '.join(col_names).lower()
        if any(p in col_str for p in ['otu', 'asv', 'zotu', 'denovo']):
            dtype = '16S_amplicon' if dtype == 'generic' else dtype
            confidence = min(confidence + 0.15, 0.95)

        return dtype, confidence

    # ──────────────────────────────────────────────────────────────────
    #  ТАКСОНОМИЯ
    # ──────────────────────────────────────────────────────────────────

    def _map_taxonomy_columns(self, taxonomy_df: pd.DataFrame) -> Tuple[Dict[str, str], str]:
        """
        Автоматически маппирует любые имена колонок таксономии на стандартные.
        Поддерживает: QIIME2 (k__, p__...), SILVA, mothur, произвольные имена.
        """
        mapping = {}
        fmt = 'unknown'

        # Сначала пробуем разобрать сводный столбец "taxonomy" или "lineage"
        combined_col = None
        for col in taxonomy_df.columns:
            if str(col).lower() in ['taxonomy', 'lineage', 'classification',
                                     'taxon', 'lineage_string']:
                combined_col = col
                break

        if combined_col is not None:
            fmt = 'combined_string'
            mapping[combined_col] = 'TAXONOMY_STRING'
            return mapping, fmt

        # Иначе ищем соответствие по синонимам
        used_targets = set()
        for col in taxonomy_df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower().rstrip('_')
            matched = None
            for standard, synonyms in self.TAXONOMY_SYNONYMS.items():
                if standard in used_targets:
                    continue
                if any(col_lower == s.lower() or col_lower.startswith(s.lower())
                       for s in synonyms):
                    matched = standard
                    break
            if matched:
                mapping[col_str] = matched
                used_targets.add(matched)

        # Определяем формат
        cols_lower = [str(c).lower() for c in taxonomy_df.columns]
        if any('k__' in c or 'p__' in c for c in cols_lower):
            fmt = 'qiime2_prefix'
        elif any(c in ['kingdom', 'phylum', 'class'] for c in cols_lower):
            fmt = 'standard'
        elif len(mapping) > 3:
            fmt = 'abbreviated'
        else:
            fmt = 'unknown'

        return mapping, fmt

    def apply_taxonomy_mapping(self, taxonomy_df: pd.DataFrame,
                                mapping: Dict[str, str],
                                fmt: str) -> pd.DataFrame:
        """
        Применяет маппинг таксономии к DataFrame.
        Возвращает стандартизированный DataFrame.
        """
        df = taxonomy_df.copy()

        if fmt == 'combined_string':
            # Разбиваем строку вида "k__Bacteria;p__Firmicutes;..."
            tax_col = [c for c, t in mapping.items() if t == 'TAXONOMY_STRING']
            if tax_col:
                df = self._parse_taxonomy_string(df, tax_col[0])
            return df

        if fmt in ('qiime2_prefix', 'standard', 'abbreviated') and mapping:
            # Переименовываем колонки
            rename_map = {orig: std for orig, std in mapping.items() if std != 'TAXONOMY_STRING'}
            df = df.rename(columns=rename_map)

        # Чистим значения: убираем k__, p__... префиксы из ячеек
        prefix_pat = re.compile(r'^[a-z]__', re.IGNORECASE)
        for col in df.columns:
            if col in ('Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'):
                df[col] = df[col].astype(str).apply(
                    lambda v: prefix_pat.sub('', v).strip() if pd.notna(v) else v
                )
                # Заменяем пустые строки и "unclassified" на NaN
                df[col] = df[col].replace(['', 'unclassified', 'uncultured',
                                           'Unknown', 'unknown', 'nan',
                                           'Unclassified', 'NA'], np.nan)
        return df

    def _parse_taxonomy_string(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Разбирает строку таксономии вида 'k__Bacteria;p__Firmicutes;...'
        или 'Bacteria;Firmicutes;Bacilli;...'
        """
        levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        prefix_map = {'k': 'Kingdom', 'd': 'Kingdom', 'p': 'Phylum',
                      'c': 'Class', 'o': 'Order', 'f': 'Family',
                      'g': 'Genus', 's': 'Species'}

        def parse_row(val):
            if not isinstance(val, str) or not val.strip():
                return {l: np.nan for l in levels}
            parts = re.split(r'[;|,]', val.strip())
            result = {l: np.nan for l in levels}
            for part in parts:
                part = part.strip()
                m = re.match(r'^([a-z])__(.+)$', part, re.IGNORECASE)
                if m:
                    prefix, name = m.group(1).lower(), m.group(2).strip()
                    level = prefix_map.get(prefix)
                    if level and name and name not in ('', 'NA', 'unclassified'):
                        result[level] = name
            # Если нет префиксов — присваиваем по позиции
            if all(v is np.nan for v in result.values()):
                for i, part in enumerate(parts[:len(levels)]):
                    name = part.strip()
                    if name and name not in ('NA', 'unclassified', ''):
                        result[levels[i]] = name
            return result

        parsed = df[col].apply(parse_row).apply(pd.Series)
        df = df.drop(columns=[col])
        for lv in levels:
            if lv in parsed.columns:
                df[lv] = parsed[lv]
        return df

    # ──────────────────────────────────────────────────────────────────
    #  ПРОФИЛИРОВАНИЕ МЕТАДАННЫХ
    # ──────────────────────────────────────────────────────────────────

    def _profile_metadata(self, metadata: pd.DataFrame) -> Dict:
        """
        Детальный профиль каждой колонки метаданных.
        Определяет роль: group | numeric | date | id | text
        """
        profile = {}
        for col in metadata.columns:
            series = metadata[col].dropna()
            n_unique = series.nunique()
            n_total  = len(metadata)
            dtype    = str(series.dtype)

            # Пробуем числовое
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                is_numeric = numeric_series.notna().mean() > 0.9
            except Exception:
                is_numeric = False

            # Пробуем дату
            is_date = False
            if not is_numeric:
                try:
                    pd.to_datetime(series.head(10), errors='raise')
                    is_date = True
                except Exception:
                    pass

            col_lower = str(col).lower().strip('#').strip()

            # Определяем роль
            if any(kw in col_lower for kw in self.ID_KEYWORDS) and n_unique == n_total:
                role = 'id'
            elif is_date:
                role = 'date'
            elif is_numeric and n_unique > 15:
                role = 'numeric'
            elif is_numeric and n_unique <= 10:
                role = 'numeric_categorical'   # числа закодированы как категории
            elif n_unique == 1:
                role = 'constant'
            elif n_unique == n_total and n_total > 20:
                role = 'id'
            elif 2 <= n_unique <= 30:
                role = 'group'
            else:
                role = 'text'

            # Перепроверяем по ключевым словам
            if any(kw in col_lower for kw in self.GROUP_KEYWORDS) and role not in ('id', 'date'):
                role = 'group'
            elif any(kw in col_lower for kw in self.COVARIATE_KEYWORDS) and is_numeric:
                role = 'numeric'

            # Подсчёт классов для группировочных
            try:
                class_counts = series.dropna().value_counts().to_dict() if role in ('group', 'numeric_categorical') else {}
            except Exception:
                class_counts = {}

            profile[col] = {
                'role':         role,
                'n_unique':     n_unique,
                'n_total':      n_total,
                'is_numeric':   is_numeric,
                'is_date':      is_date,
                'dtype':        dtype,
                'class_counts': class_counts,
                'values_sample': list(series.unique()[:8]),
                'completeness': series.notna().mean(),
            }
        return profile

    def _pick_group_columns(self, profile: Dict) -> List[str]:
        """Выбирает колонки для группировки, отсортированные по релевантности."""
        groups = [(col, info) for col, info in profile.items()
                  if info['role'] in ('group', 'numeric_categorical')
                  and 2 <= info['n_unique'] <= 20]
        # Сортировка: колонки с меньшим числом групп и ключевыми словами выше
        def score(item):
            col, info = item
            kw_score = sum(1 for kw in self.GROUP_KEYWORDS if kw in str(col).lower())
            balance_score = 1 / max(info['n_unique'], 1)
            return kw_score * 2 + balance_score
        groups.sort(key=score, reverse=True)
        return [col for col, _ in groups]

    def _pick_numeric_columns(self, profile: Dict) -> List[str]:
        return [col for col, info in profile.items()
                if info['role'] == 'numeric' and info['n_unique'] > 3]

    def _pick_date_columns(self, profile: Dict) -> List[str]:
        return [col for col, info in profile.items() if info['role'] == 'date']

    def _pick_id_columns(self, profile: Dict) -> List[str]:
        return [col for col, info in profile.items() if info['role'] == 'id']

    # ──────────────────────────────────────────────────────────────────
    #  ДИАГНОСТИКА
    # ──────────────────────────────────────────────────────────────────

    def _generate_diagnostics(self, fp: DatasetFingerprint,
                               working_otu: pd.DataFrame,
                               metadata: Optional[pd.DataFrame]) -> Tuple[List, List]:
        warnings_list = []
        suggestions   = []

        if fp.n_samples < 10:
            warnings_list.append(f"Мало образцов ({fp.n_samples}). Для надёжного ML нужно >=30.")
        if fp.n_features > 10000:
            suggestions.append("Более 10,000 признаков — рекомендуется фильтрация по prevalence.")
        if fp.sparsity > 0.95:
            warnings_list.append(f"Очень высокая разреженность ({fp.sparsity:.0%}). Проверьте данные.")
        if fp.sample_match_ratio < 0.5:
            warnings_list.append(
                f"Только {fp.sample_match_ratio:.0%} ID образцов совпадает между OTU и метаданными. "
                "Проверьте имена образцов.")
        if fp.otu_taxonomy_match_ratio < 0.5 and fp.otu_taxonomy_match_ratio > 0:
            warnings_list.append(
                f"Только {fp.otu_taxonomy_match_ratio:.0%} OTU имеют таксономию. "
                "Возможно несоответствие ID.")
        if fp.otu_orientation != 'unknown' and fp.sample_match_ratio < 0.3:
            suggestions.append(
                "Низкое совпадение ID — попробуйте другую колонку индекса в 'Advanced Settings'.")
        if not fp.group_columns and metadata is not None:
            warnings_list.append("Не найдено подходящих переменных для группировки.")
        if fp.taxonomy_format == 'combined_string':
            suggestions.append("Таксономия в строке ('taxonomy string') — будет автоматически разобрана.")

        return warnings_list, suggestions

    # ──────────────────────────────────────────────────────────────────
    #  УТИЛИТЫ
    # ──────────────────────────────────────────────────────────────────

    def apply_to_data_loader(self, data_loader, fp: DatasetFingerprint) -> None:
        """
        Применяет результаты идентификации к DataLoader:
        - Транспонирует OTU если нужно
        - Переименовывает колонки таксономии
        """
        # Транспозиция
        if (fp.otu_orientation == 'samples_as_columns'
                and data_loader.otu_table is not None):
            data_loader.otu_table = data_loader.otu_table.T

        # Таксономия
        if (data_loader.taxonomy is not None
                and fp.detected_taxonomy_map
                and fp.taxonomy_format != 'unknown'):
            data_loader.taxonomy = self.apply_taxonomy_mapping(
                data_loader.taxonomy,
                fp.detected_taxonomy_map,
                fp.taxonomy_format)


# ═══════════════════════════════════════════════════════════════════════
class ComparisonEngine:
    """
    Движок сравнений: что с чем сравнивать.
    Анализирует метаданные и предлагает осмысленные сравнения.
    """

    def __init__(self, metadata_df: pd.DataFrame,
                 group_columns: List[str],
                 numeric_cols: List[str]):
        self.metadata    = metadata_df
        self.group_cols  = group_columns
        self.numeric_cols = numeric_cols
        self._comparison_cache: Dict = {}

    def get_groups_for_column(self, column: str) -> Dict:
        """
        Возвращает подробную информацию о группах для выбранной колонки.
        """
        if column not in self.metadata.columns:
            return {}
        series = self.metadata[column].dropna()
        vc = series.value_counts()

        groups = {}
        for grp, cnt in vc.items():
            samples = list(self.metadata[self.metadata[column] == grp].index)
            groups[str(grp)] = {
                'n':       cnt,
                'samples': samples,
                'pct':     cnt / len(self.metadata) * 100,
            }
        return groups

    def suggest_comparisons(self, column: str,
                             min_n_per_group: int = 3) -> List[Dict]:
        """
        Автоматически предлагает все осмысленные попарные сравнения.
        Оценивает каждое по балансу и размеру.
        """
        groups = self.get_groups_for_column(column)
        valid_groups = {g: info for g, info in groups.items()
                        if info['n'] >= min_n_per_group}
        group_names = sorted(valid_groups.keys())

        comparisons = []
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                n1 = valid_groups[g1]['n']
                n2 = valid_groups[g2]['n']
                balance = min(n1, n2) / max(n1, n2)    # 1.0 = идеальный баланс
                total   = n1 + n2
                comparisons.append({
                    'group1':     g1,
                    'group2':     g2,
                    'n1':         n1,
                    'n2':         n2,
                    'total':      total,
                    'balance':    round(balance, 2),
                    'label':      f'{g1} vs {g2}',
                    'recommended': balance >= 0.5 and total >= 10,
                })

        # Сортировка: сначала рекомендуемые, потом по балансу
        comparisons.sort(key=lambda x: (-int(x['recommended']), -x['balance']))
        return comparisons

    def get_all_vs_all_summary(self, column: str) -> Dict:
        """
        Сводка для многоклассового анализа (без попарных сравнений).
        """
        groups = self.get_groups_for_column(column)
        return {
            'mode': 'multi_class',
            'n_groups': len(groups),
            'groups': groups,
            'balanced': all(g['n'] >= 5 for g in groups.values()),
        }

    def filter_samples_for_comparison(self, column: str,
                                       group1: str, group2: str,
                                       metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Фильтрует метаданные для конкретного попарного сравнения.
        Возвращает отфильтрованный metadata и бинарный вектор меток.
        """
        mask = metadata[column].astype(str).isin([str(group1), str(group2)])
        filtered_meta = metadata[mask].copy()
        # Бинарная метка: group2 = 1, group1 = 0
        y = (filtered_meta[column].astype(str) == str(group2)).astype(int)
        return filtered_meta, y

    def get_reference_group_options(self, column: str) -> List[str]:
        """
        Предлагает варианты референсной группы (обычно 'Control', 'Healthy'...).
        """
        groups = list(self.get_groups_for_column(column).keys())
        control_keywords = ['control', 'ctrl', 'healthy', 'normal', 'vehicle',
                            'placebo', 'baseline', 'pre', 'untreated', 'wildtype',
                            'wt', 'контроль', 'здоровый', 'норма']
        # Сначала ставим те, что похожи на "контроль"
        ref_first = [g for g in groups
                     if any(kw in str(g).lower() for kw in control_keywords)]
        others = [g for g in groups if g not in ref_first]
        return ref_first + others


# ═══════════════════════════════════════════════════════════════════════
class DatasetCard:
    """
    Генерирует красивую HTML/Markdown карточку датасета для отображения в UI.
    """

    @staticmethod
    def render_streamlit(fp: DatasetFingerprint):
        """Отображает карточку в Streamlit."""
        try:
            import streamlit as st
        except ImportError:
            return

        # ── Заголовок ──
        type_icons = {
            '16S_amplicon': '16S Amplicon',
            'WGS_species': 'WGS / Metagenomics',
            'WGS_functional': 'WGS Functional',
            'ITS_fungal': 'ITS / Fungal',
            'generic': 'Generic Omics',
        }
        dtype_label = type_icons.get(fp.dataset_type, fp.dataset_type)
        conf_color  = '#27AE60' if fp.confidence > 0.7 else '#F39C12' if fp.confidence > 0.4 else '#E74C3C'

        st.markdown(
            f"<div style='background:#f8f9fa;border-radius:8px;padding:12px 16px;"
            f"border-left:4px solid {conf_color};margin-bottom:8px'>"
            f"<b>{dtype_label}</b> &nbsp;|&nbsp; "
            f"Confidence: <span style='color:{conf_color}'><b>{fp.confidence:.0%}</b></span>"
            f"</div>", unsafe_allow_html=True)

        # ── Метрики ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Samples",    f"{fp.n_samples:,}")
        m2.metric("Features",   f"{fp.n_features:,}")
        m3.metric("Reads",      f"{fp.n_reads_total:,.0f}" if fp.n_reads_total < 1e9
                                 else f"{fp.n_reads_total/1e9:.1f}B")
        m4.metric("Sparsity",   f"{fp.sparsity:.0%}")
        m5.metric("OTU↔Tax",   f"{fp.otu_taxonomy_match_ratio:.0%}")

        # ── Ориентация ──
        orient_icon = "↕️" if fp.otu_orientation == 'samples_as_columns' else "↔️"
        orient_text = {
            'samples_as_columns': 'OTUs→rows, Samples→columns (transposed automatically)',
            'samples_as_rows':    'Samples→rows, OTUs→columns (standard)',
            'unknown':            'Unknown orientation',
        }.get(fp.otu_orientation, fp.otu_orientation)
        st.caption(f"{orient_icon} OTU Orientation: **{orient_text}**  "
                   f"| Sample match: **{fp.sample_match_ratio:.0%}**  "
                   f"| Taxonomy format: **{fp.taxonomy_format}**")

        # ── Предупреждения ──
        for w in fp.warnings:
            st.warning(w)
        for s in fp.suggestions:
            st.info(s)

        # ── Метаданные профиль ──
        if fp.metadata_profile:
            with st.expander("Metadata Column Intelligence", expanded=False):
                rows = []
                for col, info in fp.metadata_profile.items():
                    role_icons = {
                        'group': 'Grouping',
                        'numeric': 'Numeric covariate',
                        'numeric_categorical': 'Numeric (categorical)',
                        'date': 'Date/Time',
                        'id': 'Identifier',
                        'constant': 'Constant (useless)',
                        'text': 'Text',
                    }
                    vals = ', '.join(str(v) for v in info['values_sample'][:5])
                    rows.append({
                        'Column':       col,
                        'Role':         role_icons.get(info['role'], info['role']),
                        'Unique':       info['n_unique'],
                        'Completeness': f"{info['completeness']:.0%}",
                        'Sample values': vals,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Taxonomy mapping ──
        if fp.detected_taxonomy_map:
            with st.expander("Taxonomy Column Mapping", expanded=False):
                map_rows = [{'Original': orig, '→ Standard': std}
                            for orig, std in fp.detected_taxonomy_map.items()]
                st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
class ComparisonUI:
    """
    Streamlit UI для выбора и настройки сравнений.
    """

    @staticmethod
    def render(fp: DatasetFingerprint, settings: Dict) -> Dict:
        """
        Отображает интерфейс выбора сравнения.
        Возвращает обновлённые settings с:
          - group_column, reference_group, comparison_groups,
            comparison_mode, selected_pair
        """
        try:
            import streamlit as st
        except ImportError:
            return settings

        if not fp.group_columns:
            st.warning("No grouping variables detected in metadata.")
            return settings

        engine = fp.comparison_engine
        if engine is None:
            return settings

        st.markdown("### Comparison Configuration")

        # ── Выбор переменной группировки ──
        col_options = fp.group_columns + [c for c in fp.metadata_profile
                                           if c not in fp.group_columns and
                                           fp.metadata_profile[c]['role'] in
                                           ('group', 'numeric_categorical')]

        selected_col = st.selectbox(
            "Grouping variable:",
            col_options,
            format_func=lambda c: (
                f"{c}  [{fp.metadata_profile.get(c, {}).get('n_unique', '?')} groups]"),
            key="cmp_group_col",
            help="The metadata variable that defines your experimental groups"
        )
        settings['group_column'] = selected_col

        if not selected_col:
            return settings

        groups = engine.get_groups_for_column(selected_col)
        n_groups = len(groups)

        # ── Режим сравнения ──
        if n_groups == 2:
            mode = 'binary'
            st.success(f"Binary comparison detected: **{list(groups.keys())[0]}** vs **{list(groups.keys())[1]}**")
        elif n_groups > 2:
            mode_opts = {
                'all_vs_all':   'All groups simultaneously (multi-class)',
                'pairwise':     'Pairwise (one-vs-one)',
                'one_vs_rest':  'One group vs all others',
            }
            mode_key = st.radio(
                "Comparison mode:",
                list(mode_opts.keys()),
                format_func=lambda k: mode_opts[k],
                horizontal=True,
                key="cmp_mode"
            )
            mode = mode_key
        else:
            st.warning(f"Only {n_groups} group(s) found — need at least 2.")
            return settings

        settings['comparison_mode'] = mode

        # ── Информация о группах ──
        with st.expander("Group sizes", expanded=True):
            gcols = st.columns(min(n_groups, 5))
            for i, (grp, info) in enumerate(groups.items()):
                with gcols[i % len(gcols)]:
                    st.metric(str(grp), f"n = {info['n']}",
                              delta=f"{info['pct']:.0f}%")

        # ── Пользовательский выбор групп ──
        st.markdown("**Include groups:**")
        all_group_names = list(groups.keys())
        selected_groups = st.multiselect(
            "Groups to include in analysis:",
            all_group_names,
            default=all_group_names,
            key="cmp_selected_groups",
            help="Uncheck groups to exclude them from analysis"
        )
        settings['comparison_groups'] = selected_groups

        # ── Референсная группа (для pairwise / one_vs_rest) ──
        if mode in ('pairwise', 'one_vs_rest', 'binary') and len(selected_groups) >= 2:
            ref_options = engine.get_reference_group_options(selected_col)
            ref_options = [r for r in ref_options if r in selected_groups]

            ref_group = st.selectbox(
                "Reference group (control/baseline):",
                ref_options if ref_options else selected_groups,
                key="cmp_ref_group",
                help="This group will be the reference/control for comparisons"
            )
            settings['reference_group'] = ref_group

        # ── Для pairwise: выбор конкретной пары ──
        if mode == 'pairwise' and len(selected_groups) >= 2:
            suggestions = engine.suggest_comparisons(selected_col)
            suggestions = [s for s in suggestions
                           if s['group1'] in selected_groups and s['group2'] in selected_groups]

            pair_labels = [f"{s['label']} (n={s['n1']}+{s['n2']}, "
                           f"balance={s['balance']:.0%}"
                           f"{' [recommended]' if s['recommended'] else ''})"
                           for s in suggestions]

            if pair_labels:
                st.markdown("**Comparison pairs** ([recommended] = balanced):")
                selected_pairs = st.multiselect(
                    "Select comparisons to run:",
                    pair_labels,
                    default=[pair_labels[0]] if pair_labels else [],
                    key="cmp_pairs"
                )
                settings['selected_pairs'] = [
                    suggestions[pair_labels.index(p)]
                    for p in selected_pairs if p in pair_labels
                ]

        # ── Preview таблица сравнений ──
        if mode != 'pairwise':
            suggestions = engine.suggest_comparisons(selected_col, min_n_per_group=2)
            if suggestions:
                with st.expander("All possible pairwise comparisons", expanded=False):
                    df_sugg = pd.DataFrame([{
                        'Comparison': s['label'],
                        'n group1': s['n1'],
                        'n group2': s['n2'],
                        'Balance': f"{s['balance']:.0%}",
                        'Total n': s['total'],
                        'Status': 'Recommended' if s['recommended'] else 'Small n',
                    } for s in suggestions])
                    st.dataframe(df_sugg, use_container_width=True, hide_index=True)

        return settings
