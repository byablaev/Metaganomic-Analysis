"""
Модуль анализа бета-разнообразия для микробиомных данных
Полный анализ: PCoA/NMDS, все метрики, PERMANOVA, ANOSIM, 
тепловые карты расстояний, попарные сравнения
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class BetaDiversityAnalyzer:
    """Комплексный анализатор бета-разнообразия"""

    METRIC_NAMES = {
        'bray_curtis': 'Bray-Curtis',
        'jaccard': 'Jaccard',
        'aitchison': 'Aitchison (CLR)',
        'weighted_unifrac': 'Weighted UniFrac (proxy)'  # updated in __init__
    }

    def __init__(self, data_loader, settings):
        self.data_loader = data_loader
        self.settings = settings
        self.distance_matrices = {}
        self.ordination_results = {}
        self.permanova_results = {}
        self.anosim_results = {}
        self.pairwise_results = {}
        self.METRIC_NAMES = dict(self.__class__.METRIC_NAMES)  # per-instance copy
        tree = getattr(data_loader, 'phylo_tree', None)
        if tree is not None:
            self.METRIC_NAMES['weighted_unifrac'] = 'Weighted UniFrac (phylogenetic)'
        self._create_color_palettes()

    def _create_color_palettes(self):
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            unique_groups = sorted(self.data_loader.metadata[gc].unique())
            n = len(unique_groups)
            colors = sns.color_palette("Set2", n) if n <= 8 else sns.color_palette("husl", n)
            self.group_colors = {}
            for i, g in enumerate(unique_groups):
                rgb = colors[i]
                self.group_colors[g] = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        else:
            self.group_colors = {}
        self.markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']

    # ─── Вычисление расстояний ───

    def calculate_distances(self):
        """Вычисление всех матриц расстояний"""
        from normalization import NormalizationProcessor

        auto_filter = self.settings.get('auto_filter_zeros', True)
        excluded = self.settings.get('excluded_samples', [])

        otu = self.data_loader.get_filtered_otu_table(auto_filter)

        if excluded:
            keep = [s for s in otu.index if s not in excluded]
            if len(keep) < 3:
                raise ValueError("Need at least 3 samples for beta diversity.")
            otu = otu.loc[keep]

        # Умное сопоставление образцов
        self._smart_match_samples(otu)

        self.active_samples = otu.index.tolist()
        self.excluded_samples = excluded

        normalizer = NormalizationProcessor()
        rel = normalizer.get_relative_abundance(otu)
        clr = normalizer.clr_transform(otu)

        # Bray-Curtis
        self.distance_matrices['bray_curtis'] = pd.DataFrame(
            pairwise_distances(rel.values, metric='braycurtis'),
            index=rel.index, columns=rel.index)

        # Jaccard
        binary = (rel > 0).astype(float)
        self.distance_matrices['jaccard'] = pd.DataFrame(
            pairwise_distances(binary.values, metric='jaccard'),
            index=binary.index, columns=binary.index)

        # Aitchison
        self.distance_matrices['aitchison'] = pd.DataFrame(
            pairwise_distances(clr.values, metric='euclidean'),
            index=clr.index, columns=clr.index)

        # Weighted UniFrac — real tree if available, else UPGMA proxy
        tree = getattr(self.data_loader, 'phylo_tree', None)
        wu = self._weighted_unifrac_real(otu, tree) if tree is not None else None
        if wu is not None:
            self.distance_matrices['weighted_unifrac'] = wu
        else:
            self.distance_matrices['weighted_unifrac'] = self._weighted_unifrac_proxy(rel)

        self.perform_ordination()

    def _weighted_unifrac_proxy(self, rel_abundance):
        """
        Weighted UniFrac distance matrix via UPGMA proxy tree.

        True Weighted UniFrac requires a newick phylogenetic tree. In its
        absence we build an approximate UPGMA tree from pairwise OTU-profile
        distances (log1p Euclidean), then apply the standard Weighted UniFrac
        formula per branch:

            WU(i,j) = Σ_b  branch_len(b) × |A_i(b) − A_j(b)|
                      ─────────────────────────────────────────
                                total_tree_length

        where A_k(b) = Σ relative abundances of all OTUs in clade b for
        sample k (i.e., the "abundance weight" on each side of every branch).

        Complexity: O(n_merges × n_samples²) — vectorised over sample pairs.

        Parameters
        ----------
        rel_abundance : pd.DataFrame  (samples × OTUs), relative abundances

        Returns
        -------
        pd.DataFrame  symmetric distance matrix (samples × samples)
        """
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage as hclust

        n_samples, n_otus = rel_abundance.shape

        if n_otus < 2:
            # Fallback: Bray-Curtis when only 1 OTU present
            return pd.DataFrame(
                pairwise_distances(rel_abundance.values, metric='braycurtis'),
                index=rel_abundance.index, columns=rel_abundance.index)

        # ── 1. Build UPGMA tree from OTU profiles ─────────────────────
        otu_profiles = np.log1p(rel_abundance.T.values)   # (n_otus, n_samples)
        dist_condensed = pdist(otu_profiles, metric='euclidean')
        Z = hclust(dist_condensed, method='average')       # UPGMA

        n_leaves = n_otus
        n_merges = len(Z)

        # ── 2. Pre-compute per-clade OTU membership lists ─────────────
        cluster_otus: dict[int, list] = {i: [i] for i in range(n_leaves)}
        for k in range(n_merges):
            c1, c2 = int(Z[k, 0]), int(Z[k, 1])
            cluster_otus[n_leaves + k] = (cluster_otus.get(c1, []) +
                                          cluster_otus.get(c2, []))

        # ── 3. Compute Weighted UniFrac pairwise distances ─────────────
        X = rel_abundance.values.astype(float)     # (n_samples, n_otus)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0              # avoid division by zero
        X_norm = X / row_sums                       # normalised relative abundance

        dist_matrix = np.zeros((n_samples, n_samples))
        total_length = float(Z[:, 2].sum())

        for k in range(n_merges):
            branch_len = float(Z[k, 2])
            clade_idx = cluster_otus[n_leaves + k]

            # Abundance weight for each sample over this clade
            clade_abund = X_norm[:, clade_idx].sum(axis=1)    # (n_samples,)

            # Vectorised pairwise |A_i - A_j|  →  (n_samples, n_samples)
            diff = np.abs(clade_abund[:, np.newaxis] - clade_abund[np.newaxis, :])
            dist_matrix += branch_len * diff

        # Normalise by total tree length (makes distances in [0, 1])
        if total_length > 0:
            dist_matrix /= total_length

        np.fill_diagonal(dist_matrix, 0.0)

        return pd.DataFrame(dist_matrix,
                            index=rel_abundance.index,
                            columns=rel_abundance.index)

    def _weighted_unifrac_real(self, otu, tree):
        """
        True Weighted UniFrac via scikit-bio (Lozupone et al. 2007).
        Requires a rooted Newick tree loaded into data_loader.phylo_tree.
        Returns None if scikit-bio is unavailable or OTU/tree overlap < 20%.
        """
        try:
            from skbio.diversity import beta_diversity
        except ImportError:
            return None

        tip_names = {tip.name for tip in tree.tips()}
        otu_ids = [str(c) for c in otu.columns if str(c) in tip_names]

        if len(otu_ids) < max(2, len(otu.columns) * 0.2):
            return None  # less than 20% overlap — fall back to proxy

        counts = otu[otu_ids].values.astype(int)
        counts = np.maximum(counts, 0)

        try:
            dm = beta_diversity('weighted_unifrac', counts, otu.index.tolist(),
                                tree=tree, otu_ids=otu_ids)
            return pd.DataFrame(dm.data, index=otu.index, columns=otu.index)
        except Exception:
            return None

    def _smart_match_samples(self, otu):
        """Умное сопоставление образцов OTU и метаданных"""
        if not hasattr(self.data_loader, 'metadata') or self.data_loader.metadata is None:
            return

        data_samples = list(otu.index)
        meta_samples = list(self.data_loader.metadata.index)
        direct = [s for s in data_samples if s in meta_samples]

        if len(direct) >= 3:
            return  # Прямое совпадение — всё ОК

        min_n = min(len(data_samples), len(meta_samples))
        if min_n < 3:
            raise ValueError("Insufficient samples for beta diversity analysis")

        new_names = [f"sample_{i}" for i in range(min_n)]
        rename_otu = dict(zip(data_samples[:min_n], new_names))
        otu.rename(index=rename_otu, inplace=True)

        temp_meta = self.data_loader.metadata.iloc[:min_n].copy()
        temp_meta.index = new_names
        self.original_metadata = self.data_loader.metadata
        self.data_loader.metadata = temp_meta

    # ─── Ординация ───

    def perform_ordination(self, method='pcoa'):
        """PCoA / NMDS ординация для всех метрик"""
        for name, dm in self.distance_matrices.items():
            try:
                # PCoA через MDS
                mds = MDS(n_components=3, dissimilarity='precomputed',
                         metric=True, random_state=42, max_iter=1000, normalized_stress='auto')
                coords_3d = mds.fit_transform(dm.values)

                eigenvals = np.var(coords_3d, axis=0)
                total = np.sum(eigenvals)
                explained = eigenvals / total if total > 0 else np.zeros(3)

                coords_df = pd.DataFrame(coords_3d, index=dm.index,
                                        columns=['PC1', 'PC2', 'PC3'])

                self.ordination_results[name] = {
                    'coords': coords_df,
                    'explained_variance': explained,
                    'stress': mds.stress_ if hasattr(mds, 'stress_') else None
                }

                # NMDS (2D)
                nmds = MDS(n_components=2, dissimilarity='precomputed',
                          metric=False, random_state=42, max_iter=1000, normalized_stress='auto')
                nmds_coords = nmds.fit_transform(dm.values)
                nmds_df = pd.DataFrame(nmds_coords, index=dm.index,
                                      columns=['NMDS1', 'NMDS2'])
                self.ordination_results[f'{name}_nmds'] = {
                    'coords': nmds_df,
                    'stress': nmds.stress_ if hasattr(nmds, 'stress_') else None
                }
            except Exception as e:
                print(f"Ordination error for {name}: {e}")

    # ─── Статистические тесты ───

    def run_permanova(self):
        """PERMANOVA для всех метрик и факторов"""
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return

        cat_cols = self.data_loader.get_categorical_columns()

        for metric_name, dm in self.distance_matrices.items():
            results = {}

            # Основной фактор
            results[gc] = self._permanova_test(dm, self.data_loader.metadata[gc])
            results[f'{gc}_dispersion'] = self._permdisp_test(dm, self.data_loader.metadata[gc])

            # Дополнительные факторы
            for col in cat_cols:
                if col != gc and self.data_loader.metadata[col].nunique() >= 2:
                    results[col] = self._permanova_test(dm, self.data_loader.metadata[col])
                    if self.data_loader.metadata[col].nunique() <= 5:
                        combined = (self.data_loader.metadata[gc].astype(str) + '_' +
                                   self.data_loader.metadata[col].astype(str))
                        results[f'{gc}×{col}'] = self._permanova_test(dm, combined)

            self.permanova_results[metric_name] = results

    def run_anosim(self):
        """ANOSIM тест для всех метрик"""
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return

        for metric_name, dm in self.distance_matrices.items():
            try:
                common = dm.index.intersection(self.data_loader.metadata.index)
                if len(common) < 3:
                    continue
                groups = self.data_loader.metadata.loc[common, gc]
                dm_sub = dm.loc[common, common]
                # Гарантируем симметричность матрицы
                dm_vals = dm_sub.values
                dm_vals = (dm_vals + dm_vals.T) / 2
                np.fill_diagonal(dm_vals, 0)

                n = len(common)
                n_perms = 999

                # Вычисление ANOSIM R статистики
                group_labels = groups.values
                unique_groups = np.unique(group_labels)

                within, between = [], []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = dm_vals[i, j]
                        if group_labels[i] == group_labels[j]:
                            within.append(d)
                        else:
                            between.append(d)

                if within and between:
                    r_obs = (np.mean(between) - np.mean(within)) / (n * (n - 1) / 4)

                    # Пермутационный тест
                    r_perms = []
                    for _ in range(n_perms):
                        perm_labels = np.random.permutation(group_labels)
                        w, b = [], []
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = dm_vals[i, j]
                                if perm_labels[i] == perm_labels[j]:
                                    w.append(d)
                                else:
                                    b.append(d)
                        if w and b:
                            r_perms.append((np.mean(b) - np.mean(w)) / (n * (n - 1) / 4))

                    p_value = (np.sum(np.array(r_perms) >= r_obs) + 1) / (n_perms + 1)

                    self.anosim_results[metric_name] = {
                        'R': r_obs,
                        'p_value': p_value,
                        'n_permutations': n_perms
                    }
            except Exception as e:
                print(f"ANOSIM error for {metric_name}: {e}")

    def run_pairwise_permanova(self):
        """Попарные PERMANOVA сравнения между группами"""
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return

        groups = sorted(self.data_loader.metadata[gc].unique())
        if len(groups) < 3:
            return  # Не нужно для 2 групп

        for metric_name, dm in self.distance_matrices.items():
            pairs = {}
            p_values = []
            pair_keys = []

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    g1, g2 = groups[i], groups[j]
                    mask = self.data_loader.metadata[gc].isin([g1, g2])
                    samples = self.data_loader.metadata[mask].index
                    common = dm.index.intersection(samples)

                    if len(common) >= 4:
                        sub_dm = dm.loc[common, common]
                        sub_groups = self.data_loader.metadata.loc[common, gc]
                        result = self._permanova_test(sub_dm, sub_groups)
                        key = f"{g1} vs {g2}"
                        pairs[key] = result
                        p_values.append(result['p_value'])
                        pair_keys.append(key)

            # Коррекция p-values
            if p_values:
                _, padj, _, _ = multipletests(p_values, method='fdr_bh')
                for k, key in enumerate(pair_keys):
                    pairs[key]['p_adjusted'] = padj[k]

            self.pairwise_results[metric_name] = pairs

    def _permanova_F(self, dist_sq, groups):
        """Вычисление pseudo-F статистики PERMANOVA (внутренний метод)."""
        n = len(groups)
        group_names = np.unique(groups)
        total_ss = dist_sq.sum() / (2 * n)
        within_ss = 0.0
        for g in group_names:
            idx = np.where(groups == g)[0]
            ni = len(idx)
            if ni > 1:
                within_ss += dist_sq[np.ix_(idx, idx)].sum() / (2 * ni)
        between_ss = total_ss - within_ss
        df_b = len(group_names) - 1
        df_w = n - 1 - df_b
        if df_w > 0 and within_ss > 0:
            F = (between_ss / df_b) / (within_ss / df_w)
            r2 = between_ss / total_ss if total_ss > 0 else 0.0
        else:
            F, r2 = 0.0, 0.0
        return F, r2, df_b, df_w

    def _permanova_test(self, dm, grouping_var, n_perms=999):
        """
        PERMANOVA (McArdle & Anderson 2001) — пермутационный тест.

        p-value вычисляется через пермутацию меток групп (n_perms итераций),
        а НЕ через параметрическое F-распределение. Только так гарантируется
        корректность теста при произвольной структуре данных.
        """
        common = dm.index.intersection(grouping_var.index)
        if len(common) < 3:
            return {'F_statistic': 0, 'p_value': 1.0, 'r_squared': 0,
                    'df_between': 0, 'df_within': 0, 'n_permutations': n_perms}

        dist_sq = dm.loc[common, common].values ** 2
        groups = grouping_var.loc[common].values

        F_obs, r2, df_b, df_w = self._permanova_F(dist_sq, groups)

        # Пермутационный p-value
        rng = np.random.default_rng(42)
        count_geq = 0
        for _ in range(n_perms):
            perm_groups = rng.permutation(groups)
            F_perm, _, _, _ = self._permanova_F(dist_sq, perm_groups)
            if F_perm >= F_obs:
                count_geq += 1

        p_value = (count_geq + 1) / (n_perms + 1)

        return {'F_statistic': F_obs, 'p_value': p_value, 'r_squared': r2,
                'df_between': df_b, 'df_within': df_w, 'n_permutations': n_perms}

    def _permdisp_test(self, dm, grouping_var, n_perms=999):
        """
        PERMDISP (Anderson 2006) — тест однородности дисперсий групп.

        Дисперсия группы = среднее расстояние от образца до центроида группы.
        p-value — пермутационный (не параметрический F от ANOVA), что корректно
        для данных с произвольным распределением расстояний.
        """
        common = dm.index.intersection(grouping_var.index)
        if len(common) < 3:
            return {'F_statistic': 0, 'p_value': 1.0}
        distances = dm.loc[common, common].values
        groups = grouping_var.loc[common].values
        group_names = np.unique(groups)
        if len(group_names) < 2:
            return {'F_statistic': 0, 'p_value': 1.0}

        def _disp_F(grps):
            """Pseudo-F на основе отклонений от центроидов."""
            all_devs = []
            group_devs = []
            for g in np.unique(grps):
                idx = np.where(grps == g)[0]
                if len(idx) < 2:
                    continue
                # Расстояние каждого образца до центроида группы
                sub = distances[np.ix_(idx, idx)]
                centroid_dist = sub.mean(axis=1)
                group_devs.append(centroid_dist)
                all_devs.extend(centroid_dist)
            if len(group_devs) < 2:
                return 0.0
            grand_mean = np.mean(all_devs)
            ss_between = sum(len(gd) * (np.mean(gd) - grand_mean)**2 for gd in group_devs)
            ss_within = sum(np.sum((gd - np.mean(gd))**2) for gd in group_devs)
            n_grp = len(group_devs)
            n_total = len(all_devs)
            df_b = n_grp - 1
            df_w = n_total - n_grp
            if df_w > 0 and ss_within > 0:
                return (ss_between / df_b) / (ss_within / df_w)
            return 0.0

        F_obs = _disp_F(groups)

        rng = np.random.default_rng(42)
        count_geq = 0
        for _ in range(n_perms):
            if _disp_F(rng.permutation(groups)) >= F_obs:
                count_geq += 1

        p_value = (count_geq + 1) / (n_perms + 1)
        return {'F_statistic': F_obs, 'p_value': p_value, 'n_permutations': n_perms}

    # ─── Plotly визуализации ───

    def plot_2d_plotly(self, metric='bray_curtis', group_by=None,
                       show_ellipses=True, ellipse_confidence=0.95,
                       ellipse_opacity=0.12, ellipse_style='dot',
                       show_labels=False, point_size=12):
        """
        Интерактивный 2D PCoA (Plotly)

        Parameters:
        -----------
        show_ellipses : bool — отображать доверительные эллипсы
        ellipse_confidence : float — уровень доверия (0.80..0.99)
        ellipse_opacity : float — прозрачность заливки эллипса
        ellipse_style : str — стиль линии: 'solid', 'dot', 'dash', 'dashdot'
        show_labels : bool — подписи образцов
        point_size : int — размер точек
        """
        if metric not in self.ordination_results:
            return None

        res = self.ordination_results[metric]
        coords = res['coords']
        ev = res.get('explained_variance', [0, 0, 0])

        fig = go.Figure()
        gc = group_by or self.settings.get('group_column')

        if gc and gc in self.data_loader.metadata.columns:
            groups = sorted(self.data_loader.metadata[gc].unique())
            for i, g in enumerate(groups):
                mask = self.data_loader.metadata[gc] == g
                common = coords.index.intersection(self.data_loader.metadata[mask].index)
                gc_coords = coords.loc[common]
                if len(gc_coords) == 0:
                    continue

                color = self.group_colors.get(g, '#333')
                marker_sym = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down',
                             'pentagon', 'hexagon', 'star', 'cross', 'x'][i % 10]

                fig.add_trace(go.Scatter(
                    x=gc_coords['PC1'], y=gc_coords['PC2'],
                    mode='markers+text' if show_labels else 'markers',
                    marker=dict(size=point_size, color=color, symbol=marker_sym,
                               line=dict(width=1.5, color='black')),
                    name=str(g),
                    text=[str(idx) for idx in gc_coords.index] if show_labels else
                         [f'Sample: {idx}<br>Group: {g}' for idx in gc_coords.index],
                    textposition='top center' if show_labels else None,
                    textfont=dict(size=8) if show_labels else None,
                    hovertemplate='Sample: %{customdata}<br>Group: ' + str(g) +
                                  '<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                    customdata=[str(idx) for idx in gc_coords.index]
                ))

                # Доверительный эллипс
                if show_ellipses and len(gc_coords) >= 3:
                    self._add_plotly_ellipse(
                        fig, gc_coords['PC1'].values, gc_coords['PC2'].values,
                        color, str(g),
                        confidence=ellipse_confidence,
                        fill_opacity=ellipse_opacity,
                        line_width=2.0,
                        line_dash=ellipse_style
                    )
        else:
            fig.add_trace(go.Scatter(
                x=coords['PC1'], y=coords['PC2'], mode='markers',
                marker=dict(size=12, color='#3498DB', line=dict(width=1.5, color='black')),
                name='Samples'
            ))

        title = f'PCoA — {self.METRIC_NAMES.get(metric, metric)}'

        # Добавляем PERMANOVA
        ann_text = self._get_permanova_annotation(metric, gc)
        annotations = []
        if ann_text:
            annotations.append(dict(
                text=ann_text, xref="paper", yref="paper", x=0.98, y=0.98,
                showarrow=False, font=dict(size=11),
                bgcolor="rgba(255,255,224,0.9)", bordercolor="gray", borderwidth=1, borderpad=6))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial Black'), x=0.5),
            xaxis_title=f'PC1 ({ev[0]:.1%} variance)',
            yaxis_title=f'PC2 ({ev[1]:.1%} variance)',
            height=600, template='plotly_white',
            annotations=annotations,
            legend=dict(font=dict(size=12))
        )
        return fig

    def plot_nmds_plotly(self, metric='bray_curtis', group_by=None,
                        show_ellipses=True, ellipse_confidence=0.95,
                        ellipse_opacity=0.12, ellipse_style='dot',
                        show_labels=False, point_size=12):
        """NMDS ординация (Plotly) с управляемыми эллипсами"""
        key = f'{metric}_nmds'
        if key not in self.ordination_results:
            return None

        res = self.ordination_results[key]
        coords = res['coords']

        fig = go.Figure()
        gc = group_by or self.settings.get('group_column')

        if gc and gc in self.data_loader.metadata.columns:
            groups = sorted(self.data_loader.metadata[gc].unique())
            for i, g in enumerate(groups):
                mask = self.data_loader.metadata[gc] == g
                common = coords.index.intersection(self.data_loader.metadata[mask].index)
                gc_coords = coords.loc[common]
                if len(gc_coords) == 0:
                    continue
                color = self.group_colors.get(g, '#333')
                fig.add_trace(go.Scatter(
                    x=gc_coords['NMDS1'], y=gc_coords['NMDS2'],
                    mode='markers+text' if show_labels else 'markers',
                    marker=dict(size=point_size, color=color, line=dict(width=1.5, color='black')),
                    name=str(g),
                    text=[str(idx) for idx in gc_coords.index] if show_labels else
                         [f'Sample: {idx}' for idx in gc_coords.index],
                    textposition='top center' if show_labels else None,
                    textfont=dict(size=8) if show_labels else None,
                    hovertemplate='Sample: %{customdata}<br>NMDS1: %{x:.3f}<br>NMDS2: %{y:.3f}<extra></extra>',
                    customdata=[str(idx) for idx in gc_coords.index]
                ))
                if show_ellipses and len(gc_coords) >= 3:
                    self._add_plotly_ellipse(
                        fig, gc_coords['NMDS1'].values, gc_coords['NMDS2'].values,
                        color, str(g),
                        confidence=ellipse_confidence,
                        fill_opacity=ellipse_opacity,
                        line_dash=ellipse_style
                    )

        stress = res.get('stress')
        stress_text = f'<br>Stress: {stress:.4f}' if stress is not None else ''

        fig.update_layout(
            title=dict(text=f'NMDS — {self.METRIC_NAMES.get(metric, metric)}{stress_text}',
                      font=dict(size=18, family='Arial Black'), x=0.5),
            xaxis_title='NMDS1', yaxis_title='NMDS2',
            height=600, template='plotly_white',
            legend=dict(font=dict(size=12))
        )
        return fig

    def plot_3d(self, metric='bray_curtis'):
        """3D PCoA график"""
        if metric not in self.ordination_results:
            return None
        res = self.ordination_results[metric]
        coords = res['coords']
        ev = res.get('explained_variance', [0, 0, 0])
        gc = self.settings.get('group_column')
        fig = go.Figure()

        if gc and gc in self.data_loader.metadata.columns:
            for i, g in enumerate(sorted(self.data_loader.metadata[gc].unique())):
                mask = self.data_loader.metadata[gc] == g
                common = coords.index.intersection(self.data_loader.metadata[mask].index)
                gc_coords = coords.loc[common]
                if len(gc_coords) == 0:
                    continue
                color = self.group_colors.get(g, '#333')
                fig.add_trace(go.Scatter3d(
                    x=gc_coords['PC1'], y=gc_coords['PC2'], z=gc_coords['PC3'],
                    mode='markers',
                    marker=dict(size=8, color=color, line=dict(width=1, color='black')),
                    name=str(g),
                    text=[f'Sample: {idx}<br>Group: {g}' for idx in gc_coords.index],
                    hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=coords['PC1'], y=coords['PC2'], z=coords['PC3'],
                mode='markers', marker=dict(size=8, color='#3498DB'),
                name='Samples'
            ))

        fig.update_layout(
            title=f'3D PCoA: {self.METRIC_NAMES.get(metric, metric)}',
            scene=dict(
                xaxis_title=f'PC1 ({ev[0]:.1%})',
                yaxis_title=f'PC2 ({ev[1]:.1%})',
                zaxis_title=f'PC3 ({ev[2]:.1%})'
            ),
            height=700
        )
        return fig

    def plot_distance_heatmap(self, metric='bray_curtis'):
        """Тепловая карта матрицы расстояний"""
        if metric not in self.distance_matrices:
            return None

        dm = self.distance_matrices[metric]
        gc = self.settings.get('group_column')

        # Сортируем по группам если есть
        if gc and gc in self.data_loader.metadata.columns:
            common = dm.index.intersection(self.data_loader.metadata.index)
            order = self.data_loader.metadata.loc[common].sort_values(gc).index
            dm = dm.loc[order, order]

        fig = go.Figure(data=go.Heatmap(
            z=dm.values,
            x=[str(x) for x in dm.columns],
            y=[str(y) for y in dm.index],
            colorscale='YlOrRd',
            colorbar=dict(title='Distance'),
            hovertemplate='%{x} ↔ %{y}<br>Distance: %{z:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=f'Distance Matrix — {self.METRIC_NAMES.get(metric, metric)}',
                      font=dict(size=16), x=0.5),
            height=600,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
        return fig

    def plot_4metric_grid_publication(self, show_ellipses=True, ellipse_confidence=0.95,
                                      ellipse_opacity=0.14, ellipse_style='dot',
                                      show_labels=False, point_size=11):
        """
        2×2 publication-quality grid: PCoA for all 4 distance metrics.

        Designed for scientific papers — clean white template, PERMANOVA
        annotations, consistent colour palette, SVG-exportable.
        """
        metrics = [m for m in ['bray_curtis', 'jaccard', 'aitchison', 'weighted_unifrac']
                   if m in self.ordination_results]
        if not metrics:
            return None

        # Always build 2×2 grid (pad with empty cells if fewer than 4 metrics)
        titles = [self.METRIC_NAMES.get(m, m) for m in metrics]
        while len(titles) < 4:
            titles.append('')

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=titles,
            horizontal_spacing=0.12,
            vertical_spacing=0.20
        )

        gc = self.settings.get('group_column')

        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1

            res = self.ordination_results[metric]
            coords = res['coords']
            ev = res.get('explained_variance', [0, 0, 0])

            groups_plotted = set()

            if gc and gc in self.data_loader.metadata.columns:
                for i, g in enumerate(sorted(self.data_loader.metadata[gc].unique())):
                    mask = self.data_loader.metadata[gc] == g
                    common = coords.index.intersection(self.data_loader.metadata[mask].index)
                    gc_coords = coords.loc[common]
                    if len(gc_coords) == 0:
                        continue
                    color = self.group_colors.get(g, '#333333')
                    show_leg = (idx == 0) and (str(g) not in groups_plotted)
                    groups_plotted.add(str(g))

                    fig.add_trace(go.Scatter(
                        x=gc_coords['PC1'], y=gc_coords['PC2'],
                        mode='markers+text' if show_labels else 'markers',
                        marker=dict(size=point_size, color=color,
                                    line=dict(width=1.2, color='black')),
                        name=str(g),
                        text=[str(s) for s in gc_coords.index] if show_labels else None,
                        textposition='top center' if show_labels else None,
                        textfont=dict(size=7) if show_labels else None,
                        showlegend=show_leg,
                        legendgroup=str(g),
                        hovertemplate=f'<b>{g}</b><br>Sample: %{{customdata}}<br>'
                                      f'PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
                        customdata=[str(s) for s in gc_coords.index]
                    ), row=row, col=col)

                    if show_ellipses and len(gc_coords) >= 3:
                        self._add_plotly_ellipse(
                            fig, gc_coords['PC1'].values, gc_coords['PC2'].values,
                            color, str(g),
                            confidence=ellipse_confidence,
                            fill_opacity=ellipse_opacity,
                            line_dash=ellipse_style,
                            row=row, col=col
                        )
            else:
                fig.add_trace(go.Scatter(
                    x=coords['PC1'], y=coords['PC2'], mode='markers',
                    marker=dict(size=point_size, color='#3498DB',
                                line=dict(width=1, color='black')),
                    name='Samples', showlegend=(idx == 0)
                ), row=row, col=col)

            # PERMANOVA annotation per panel - outside plot area at bottom
            perm_ann = self._get_permanova_annotation(metric, gc)
            if perm_ann:
                # Use paper-relative x/y for each subplot quadrant (bottom-left corner)
                # Subplot positions in paper coords: col1=0..0.45, col2=0.55..1; row1=0.55..1, row2=0..0.45
                x_paper = [0.01, 0.56, 0.01, 0.56][idx]
                y_paper = [0.51, 0.51, 0.01, 0.01][idx]
                fig.add_annotation(
                    text=perm_ann,
                    xref='paper', yref='paper',
                    x=x_paper, y=y_paper,
                    xanchor='left', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=9.5, color='#222'),
                    bgcolor='rgba(255,255,245,0.92)',
                    bordercolor='#bbb', borderwidth=1, borderpad=5
                )

            fig.update_xaxes(title_text=f'PC1 ({ev[0]:.1%})', row=row, col=col,
                             showgrid=True, gridcolor='rgba(200,200,200,0.4)',
                             zeroline=True, zerolinecolor='rgba(150,150,150,0.5)')
            fig.update_yaxes(title_text=f'PC2 ({ev[1]:.1%})', row=row, col=col,
                             showgrid=True, gridcolor='rgba(200,200,200,0.4)',
                             zeroline=True, zerolinecolor='rgba(150,150,150,0.5)')

        # Make subplot titles bold
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=13, family='Arial', color='#222')

        fig.update_layout(
            title=dict(
                text='PCoA — All Distance Metrics',
                font=dict(size=17, family='Arial Black', color='#111'),
                x=0.5
            ),
            height=820,
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(
                title=dict(text=gc or 'Group', font=dict(size=11)),
                font=dict(size=11),
                borderwidth=1,
                bordercolor='#ddd',
                bgcolor='rgba(255,255,255,0.95)'
            ),
            margin=dict(t=90, b=90, l=70, r=40)
        )
        return fig

    def plot_all_ordinations(self, show_ellipses=True, ellipse_confidence=0.95,
                             ellipse_opacity=0.10):
        """Обзорный график: PCoA для всех 4 метрик с эллипсами"""
        metrics = [m for m in ['bray_curtis', 'jaccard', 'aitchison', 'weighted_unifrac']
                   if m in self.ordination_results]
        if not metrics:
            return None

        n = len(metrics)
        fig = make_subplots(rows=1, cols=n,
                           subplot_titles=[self.METRIC_NAMES.get(m, m) for m in metrics],
                           horizontal_spacing=0.06)
        gc = self.settings.get('group_column')

        for idx, metric in enumerate(metrics):
            res = self.ordination_results[metric]
            coords = res['coords']
            c = idx + 1

            if gc and gc in self.data_loader.metadata.columns:
                for i, g in enumerate(sorted(self.data_loader.metadata[gc].unique())):
                    mask = self.data_loader.metadata[gc] == g
                    common = coords.index.intersection(self.data_loader.metadata[mask].index)
                    gc_coords = coords.loc[common]
                    if len(gc_coords) == 0:
                        continue
                    color = self.group_colors.get(g, '#333')
                    fig.add_trace(go.Scatter(
                        x=gc_coords['PC1'], y=gc_coords['PC2'], mode='markers',
                        marker=dict(size=8, color=color, line=dict(width=1, color='black')),
                        name=str(g), showlegend=(idx == 0), legendgroup=str(g)
                    ), row=1, col=c)

                    if show_ellipses and len(gc_coords) >= 3:
                        self._add_plotly_ellipse(
                            fig, gc_coords['PC1'].values, gc_coords['PC2'].values,
                            color, str(g), confidence=ellipse_confidence,
                            fill_opacity=ellipse_opacity, row=1, col=c
                        )

            ev = res.get('explained_variance', [0, 0])
            fig.update_xaxes(title_text=f'PC1 ({ev[0]:.1%})', row=1, col=c)
            fig.update_yaxes(title_text=f'PC2 ({ev[1]:.1%})', row=1, col=c)

        fig.update_layout(
            title=dict(text='PCoA Overview — All Distance Metrics',
                      font=dict(size=18, family='Arial Black'), x=0.5),
            height=500, template='plotly_white'
        )
        return fig

    @staticmethod
    def _hex_to_rgba(hex_color, alpha=0.15):
        """Конвертация HEX → rgba() для Plotly"""
        hex_color = str(hex_color).lstrip('#')
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        else:
            r, g, b = 100, 100, 100
        return f'rgba({r},{g},{b},{alpha})'

    def _add_plotly_ellipse(self, fig, x, y, color, name,
                            confidence=0.95, fill_opacity=0.12,
                            line_width=2.0, line_dash='dot',
                            row=None, col=None):
        """
        Доверительный эллипс для PCoA/NMDS

        Parameters:
        -----------
        confidence : float — уровень доверия (0.80, 0.90, 0.95, 0.99)
        fill_opacity : float — прозрачность заливки (0..1)
        line_width : float — толщина линии контура
        line_dash : str — 'solid', 'dot', 'dash', 'dashdot'
        """
        try:
            if len(x) < 3:
                return
            from scipy.stats import chi2 as chi2_dist

            mx, my = np.mean(x), np.mean(y)
            cov_matrix = np.cov(x, y)

            # Проверка вырожденности
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                return

            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 1e-12)

            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            chi2_val = chi2_dist.ppf(confidence, 2)
            w = 2 * np.sqrt(chi2_val * eigvals[1])  # major axis
            h = 2 * np.sqrt(chi2_val * eigvals[0])  # minor axis

            t = np.linspace(0, 2 * np.pi, 120)
            a_rad = np.radians(angle)
            ex = mx + w/2 * np.cos(t) * np.cos(a_rad) - h/2 * np.sin(t) * np.sin(a_rad)
            ey = my + w/2 * np.cos(t) * np.sin(a_rad) + h/2 * np.sin(t) * np.cos(a_rad)

            fill_rgba = self._hex_to_rgba(color, fill_opacity)

            ellipse_trace = go.Scatter(
                x=np.append(ex, ex[0]),  # замыкаем контур
                y=np.append(ey, ey[0]),
                mode='lines',
                line=dict(color=color, width=line_width, dash=line_dash),
                fill='toself',
                fillcolor=fill_rgba,
                name=f'{name} ({confidence:.0%} CI)',
                legendgroup=name,
                showlegend=False,
                hoverinfo='skip'
            )

            if row is not None and col is not None:
                fig.add_trace(ellipse_trace, row=row, col=col)
            else:
                fig.add_trace(ellipse_trace)

        except Exception as e:
            print(f"Ellipse error for {name}: {e}")

    def _get_permanova_annotation(self, metric, gc):
        if not gc or metric not in self.permanova_results:
            return None
        if gc not in self.permanova_results[metric]:
            return None
        r = self.permanova_results[metric][gc]
        return f"PERMANOVA: R²={r['r_squared']:.3f}, p={r['p_value']:.4f}"

    # ─── Matplotlib (для публикаций) ───

    def plot_2d(self, metric='bray_curtis', group_by=None):
        if metric not in self.ordination_results:
            return None
        fig, ax = plt.subplots(figsize=(12, 10))
        res = self.ordination_results[metric]
        coords = res['coords']
        ev = res.get('explained_variance', [0, 0, 0])
        gc = group_by or self.settings.get('group_column')

        if gc and gc in self.data_loader.metadata.columns:
            for i, g in enumerate(sorted(self.data_loader.metadata[gc].unique())):
                mask = self.data_loader.metadata[gc] == g
                common = coords.index.intersection(self.data_loader.metadata[mask].index)
                gc_coords = coords.loc[common]
                if len(gc_coords) == 0:
                    continue
                color = self.group_colors.get(g, '#333')
                marker = self.markers[i % len(self.markers)]
                ax.scatter(gc_coords['PC1'], gc_coords['PC2'], color=color, s=100,
                          alpha=0.7, edgecolors='black', linewidth=1, marker=marker, label=str(g))
                if len(gc_coords) >= 3:
                    self._add_mpl_ellipse(ax, gc_coords['PC1'].values, gc_coords['PC2'].values, color)

        ax.set_xlabel(f'PC1 ({ev[0]:.1%} variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({ev[1]:.1%} variance)', fontsize=14, fontweight='bold')
        ax.set_title(f'PCoA — {self.METRIC_NAMES.get(metric, metric)}', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        if gc:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=0.95)
        plt.tight_layout()
        return fig

    def _add_mpl_ellipse(self, ax, x, y, color, alpha=0.2):
        if len(x) < 3: return
        try:
            from scipy.stats import chi2
            mx, my = np.mean(x), np.mean(y)
            cov = np.cov(x, y)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-10)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            chi2_val = chi2.ppf(0.95, 2)
            w = 2 * np.sqrt(chi2_val * eigvals[0])
            h = 2 * np.sqrt(chi2_val * eigvals[1])
            e = Ellipse((mx, my), w, h, angle=angle, facecolor=color, alpha=alpha,
                       edgecolor=color, linewidth=2)
            ax.add_patch(e)
        except Exception:
            pass

    # ─── Сводки ───

    def get_permanova_summary(self):
        rows = []
        for metric in self.distance_matrices:
            if metric in self.permanova_results:
                for factor, s in self.permanova_results[metric].items():
                    if '_dispersion' not in factor:
                        row = {
                            'Distance Metric': self.METRIC_NAMES.get(metric, metric),
                            'Factor': factor,
                            'R²': f"{s['r_squared']:.3f}",
                            'F': f"{s['F_statistic']:.2f}",
                            'p-value': f"{s['p_value']:.4f}"
                        }
                        dk = f"{factor}_dispersion"
                        if dk in self.permanova_results[metric]:
                            row['PERMDISP p'] = f"{self.permanova_results[metric][dk]['p_value']:.4f}"
                        else:
                            row['PERMDISP p'] = 'N/A'
                        rows.append(row)
        return pd.DataFrame(rows)

    def get_anosim_summary(self):
        if not self.anosim_results:
            return pd.DataFrame()
        rows = []
        for metric, r in self.anosim_results.items():
            rows.append({
                'Distance Metric': self.METRIC_NAMES.get(metric, metric),
                'ANOSIM R': f"{r['R']:.3f}",
                'p-value': f"{r['p_value']:.4f}",
                'Permutations': r['n_permutations']
            })
        return pd.DataFrame(rows)

    def get_pairwise_summary(self, metric='bray_curtis'):
        if metric not in self.pairwise_results:
            return pd.DataFrame()
        rows = []
        for pair, r in self.pairwise_results[metric].items():
            row = {'Comparison': pair, 'R²': f"{r['r_squared']:.3f}",
                  'F': f"{r['F_statistic']:.2f}", 'p-value': f"{r['p_value']:.4f}"}
            if 'p_adjusted' in r:
                row['p-value (adj)'] = f"{r['p_adjusted']:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)
