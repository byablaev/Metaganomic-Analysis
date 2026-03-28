"""
DataLoader — Universal Microbiome Data Loader with Smart Identification
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from typing import Optional

from smart_dataset_identifier import (
    SmartDatasetIdentifier,
    DatasetFingerprint,
)


class DataLoader:
    def __init__(self):
        self.otu_table:  Optional[pd.DataFrame] = None
        self.metadata:   Optional[pd.DataFrame] = None
        self.taxonomy:   Optional[pd.DataFrame] = None
        self.phylo_tree  = None   # skbio.TreeNode, если загружено дерево
        self.separator:  str = ','
        self.index_col:  int = 0
        self.fingerprint: Optional[DatasetFingerprint] = None
        self.identifier  = SmartDatasetIdentifier()
        self.orientation_detected: Optional[str] = None

    # ── LOADING ────────────────────────────────────────────────────────

    def load_uploaded_data(self, otu_file, metadata_file, taxonomy_file) -> bool:
        try:
            raw = otu_file.read().decode('utf-8', errors='replace'); otu_file.seek(0)
            sep = self._detect_separator(raw) if self.separator == ',' else self.separator
            self.otu_table = pd.read_csv(otu_file, sep=sep, index_col=self.index_col)
            raw = metadata_file.read().decode('utf-8', errors='replace'); metadata_file.seek(0)
            self.metadata = pd.read_csv(metadata_file, sep=self._detect_separator(raw), index_col=0)
            raw = taxonomy_file.read().decode('utf-8', errors='replace'); taxonomy_file.seek(0)
            self.taxonomy = pd.read_csv(taxonomy_file, sep=self._detect_separator(raw), index_col=0)
            return self._run_identification()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def load_default_data(self, file_paths: dict) -> bool:
        try:
            for path in file_paths.values():
                if not os.path.exists(path):
                    st.error(f"File not found: {path}"); return False
            def _read(p):
                with open(p, 'r', errors='replace') as f: content = f.read()
                return pd.read_csv(p, sep=self._detect_separator(content), index_col=0)
            self.otu_table = _read(file_paths['otu_table'])
            self.metadata  = _read(file_paths['metadata'])
            self.taxonomy  = _read(file_paths['taxonomy'])
            return self._run_identification()
        except Exception as e:
            st.error(f"Error loading default data: {e}"); return False

    # ── SMART IDENTIFICATION ───────────────────────────────────────────

    def _run_identification(self) -> bool:
        self._basic_clean()
        self.fingerprint = self.identifier.identify(
            otu_df=self.otu_table, metadata_df=self.metadata, taxonomy_df=self.taxonomy)

        orientation = self.fingerprint.otu_orientation
        self.orientation_detected = orientation
        if orientation == 'samples_as_columns':
            self.otu_table = self.otu_table.T
            st.info("🔄 Auto-detected: OTU table transposed (samples were columns → now rows)")
        else:
            st.info("✅ Auto-detected: OTU table in standard orientation (samples as rows)")

        if (self.fingerprint.detected_taxonomy_map
                and self.fingerprint.taxonomy_format != 'unknown'):
            self.taxonomy = self.identifier.apply_taxonomy_mapping(
                self.taxonomy, self.fingerprint.detected_taxonomy_map,
                self.fingerprint.taxonomy_format)

        self.otu_table.index = self.otu_table.index.astype(str)
        self.metadata.index  = self.metadata.index.astype(str)
        self.taxonomy.index  = self.taxonomy.index.astype(str)
        return self._validate_and_sync()

    def _basic_clean(self):
        self.otu_table = self.otu_table.dropna(how='all').dropna(how='all', axis=1).fillna(0)
        for col in self.otu_table.columns:
            self.otu_table[col] = pd.to_numeric(self.otu_table[col], errors='coerce').fillna(0)
        self.metadata = self.metadata.dropna(how='all')
        # Try basic taxonomy rename before full smart mapping
        standard = ['Kingdom','Phylum','Class','Order','Family','Genus','Species']
        rename = {}
        for col in self.taxonomy.columns:
            for lvl in standard:
                if lvl.lower() in col.lower() and col not in rename:
                    rename[col] = lvl; break
        if rename:
            self.taxonomy.rename(columns=rename, inplace=True)

    def _validate_and_sync(self) -> bool:
        if self.otu_table is None or self.metadata is None or self.taxonomy is None:
            st.error("Not all data loaded."); return False
        if self.otu_table.empty:
            st.error("OTU table is empty."); return False

        otu_s  = set(self.otu_table.index.astype(str))
        meta_s = set(self.metadata.index.astype(str))
        common = otu_s & meta_s

        if not common:
            # Fuzzy match attempt
            norm = lambda s: s.lower().replace('-','_').replace(' ','_')
            n_otu  = {norm(s): s for s in otu_s}
            n_meta = {norm(s): s for s in meta_s}
            fuzzy  = set(n_otu) & set(n_meta)
            if fuzzy:
                self.otu_table.index = pd.Index([norm(i) for i in self.otu_table.index])
                self.metadata.index  = pd.Index([norm(i) for i in self.metadata.index])
                common = set(self.otu_table.index) & set(self.metadata.index)

        if not common:
            st.error(f"❌ No matching sample IDs.\n"
                     f"OTU samples: {list(otu_s)[:5]}\n"
                     f"Metadata samples: {list(meta_s)[:5]}")
            return False

        if len(common) < len(meta_s) * 0.5:
            st.warning(f"⚠ Only {len(common)}/{len(meta_s)} samples matched.")

        common_list = sorted(common)
        self.otu_table = self.otu_table.loc[common_list]
        self.metadata  = self.metadata.loc[common_list]

        if not (set(self.otu_table.columns.astype(str)) & set(self.taxonomy.index.astype(str))):
            st.warning("⚠ OTU IDs don't match taxonomy IDs exactly. Taxonomy may be partial.")
        return True

    # ── HELPERS ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_separator(content: str) -> str:
        """
        Detect separator from the FIRST LINE only (header row).
        Avoids false positives from semicolons inside taxonomy strings
        like 'k__Bacteria;p__Firmicutes;c__Bacilli' which are data, not separators.
        """
        lines = [l for l in content.split('\n') if l.strip()]
        header = lines[0] if lines else ''
        counts = {'\t': header.count('\t'), ',': header.count(','),
                  ';': header.count(';'), '|': header.count('|')}
        # If comma count >= semicolon count and comma is present → use comma
        # (prevents semicolons inside QIIME2 taxonomy strings from winning)
        if counts[','] > 0 and counts[','] >= counts[';']:
            return ','
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ','

    def detect_separator(self, content): return self._detect_separator(content)

    # ── PUBLIC API ─────────────────────────────────────────────────────

    def get_filtered_otu_table(self, filter_zeros=True):
        if self.otu_table is None: return pd.DataFrame()
        if filter_zeros:
            return self.otu_table[self.otu_table.columns[self.otu_table.sum() > 0]]
        return self.otu_table

    def get_categorical_columns(self):
        if self.metadata is None: return []
        if self.fingerprint and self.fingerprint.group_columns:
            return [c for c in self.fingerprint.group_columns if c in self.metadata.columns]
        cols = []
        for col in self.metadata.columns:
            nu = self.metadata[col].nunique()
            if self.metadata[col].dtype in ['float64','int64']:
                if 2 <= nu <= 10: cols.append(col)
            elif 2 <= nu <= 50: cols.append(col)
        return cols

    def get_numeric_columns(self):
        if self.fingerprint:
            return [c for c in self.fingerprint.numeric_covariates
                    if self.metadata is not None and c in self.metadata.columns]
        if self.metadata is None: return []
        return [c for c in self.metadata.columns
                if pd.to_numeric(self.metadata[c], errors='coerce').notna().mean() > 0.9
                and self.metadata[c].nunique() > 5]

    def get_taxonomic_levels(self):
        if self.taxonomy is None: return []
        standard = ['Kingdom','Phylum','Class','Order','Family','Genus','Species']
        avail = [l for l in standard
                 if l in self.taxonomy.columns and self.taxonomy[l].notna().sum() > 0]
        return avail or [c for c in self.taxonomy.columns
                         if self.taxonomy[c].notna().sum() > 0]

    def get_data_summary(self):
        if self.otu_table is None: return {}
        s  = self.otu_table.sum(axis=1)
        fp = self.fingerprint
        return {
            'n_samples':             len(self.otu_table),
            'n_otus':                len(self.otu_table.columns),
            'total_reads':           float(self.otu_table.sum().sum()),
            'min_reads_per_sample':  float(s.min()),
            'max_reads_per_sample':  float(s.max()),
            'mean_reads_per_sample': float(s.mean()),
            'sparsity':              float((self.otu_table==0).sum().sum() /
                                          max(self.otu_table.size, 1)),
            'categorical_columns':   self.get_categorical_columns(),
            'numeric_columns':       self.get_numeric_columns(),
            'taxonomic_levels':      self.get_taxonomic_levels(),
            'orientation':           self.orientation_detected,
            'dataset_type':          fp.dataset_type if fp else 'unknown',
            'taxonomy_format':       fp.taxonomy_format if fp else 'unknown',
            'id_confidence':         fp.confidence if fp else 0.0,
        }

    def load_phylo_tree(self, tree_file) -> bool:
        """
        Загружает филогенетическое дерево в формате Newick.

        Поддерживает файлы .nwk / .newick / .tre / .tree из SILVA, Greengenes,
        QIIME2 и любых других источников.

        Возвращает True при успехе, False при ошибке.
        """
        try:
            import skbio
            content = tree_file.read().decode('utf-8', errors='replace')
            from io import StringIO
            self.phylo_tree = skbio.io.read(
                StringIO(content), format='newick', into=skbio.TreeNode)

            # Статистика
            tip_names = {tip.name for tip in self.phylo_tree.tips()}
            n_tips = len(tip_names)

            # Подсчёт совпадений с OTU-таблицей
            if self.otu_table is not None:
                otu_ids = set(self.otu_table.columns.astype(str))
                matched = len(tip_names & otu_ids)
            else:
                matched = 0

            return True, n_tips, matched

        except ImportError:
            return False, 0, 0
        except Exception as e:
            self.phylo_tree = None
            raise ValueError(f"Ошибка загрузки дерева: {e}")

    def export_processed_data(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name, df in [('otu_table', self.otu_table),
                         ('metadata', self.metadata),
                         ('taxonomy', self.taxonomy)]:
            if df is not None:
                df.to_csv(os.path.join(output_dir, f'processed_{name}.csv'))
