"""
Модуль анализа альфа-разнообразия для микробиомных данных
Plotly для интерактивных графиков (zoom, download, screenshot)
+ Matplotlib для публикационных Grid-графиков
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AlphaDiversityAnalyzer:
    """Анализатор альфа-разнообразия (Plotly + Matplotlib)"""

    METRIC_LABELS = {
        'observed': 'Observed Richness',
        'shannon': 'Shannon Diversity',
        'simpson': 'Simpson Diversity',
        'pielou': 'Pielou Evenness',
        'chao1': 'Chao1 Estimator',
        'faith_pd': 'Faith PD (taxonomy proxy)'   # обновляется в __init__
    }

    def __init__(self, data_loader, settings):
        self.data_loader = data_loader
        self.settings = settings
        self.results = {}
        self.statistics = {}
        self.excluded_samples = []
        self.active_samples = []
        # Копируем метки чтобы не мутировать class-level dict
        self.METRIC_LABELS = dict(self.__class__.METRIC_LABELS)
        self._update_faith_pd_label()
        self._create_color_palettes()

    def _update_faith_pd_label(self):
        """Обновляет метку Faith PD в зависимости от наличия дерева."""
        has_tree = (hasattr(self.data_loader, 'phylo_tree')
                    and self.data_loader.phylo_tree is not None)
        self.METRIC_LABELS['faith_pd'] = (
            'Faith PD (phylogenetic)' if has_tree else 'Faith PD (taxonomy proxy)')
        self._faith_pd_uses_tree = has_tree

    def _create_color_palettes(self):
        group_column = self.settings.get('group_column')
        if group_column and group_column in self.data_loader.metadata.columns:
            unique_groups = sorted(self.data_loader.metadata[group_column].unique())
            n = len(unique_groups)
            colors = sns.color_palette("Set2", n) if n <= 8 else sns.color_palette("husl", n)
            self.group_colors = {g: colors[i] for i, g in enumerate(unique_groups)}
            self.group_colors_hex = {
                g: '#{:02x}{:02x}{:02x}'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
                for g, c in zip(unique_groups, colors)
            }
        else:
            self.group_colors = {}
            self.group_colors_hex = {}

    # ─── Метрики ───

    def calculate_metric(self, otu_data, metric='shannon'):
        if metric == 'observed':
            return (otu_data > 0).sum(axis=1)
        elif metric == 'shannon':
            return otu_data.apply(self._shannon, axis=1)
        elif metric == 'simpson':
            return otu_data.apply(self._simpson, axis=1)
        elif metric == 'pielou':
            sh = otu_data.apply(self._shannon, axis=1)
            obs = (otu_data > 0).sum(axis=1)
            return sh / np.log(obs.replace(0, 1))
        elif metric == 'chao1':
            return otu_data.apply(self._chao1, axis=1)
        elif metric == 'faith_pd':
            # Приоритет: настоящее дерево > таксономический прокси > NaN
            if self._faith_pd_uses_tree:
                tree = self.data_loader.phylo_tree
                return otu_data.apply(
                    lambda row: self._faith_pd_real(row, tree), axis=1)
            tax_df, rank_cols = self._get_taxonomy_info()
            if tax_df is not None and rank_cols:
                return otu_data.apply(
                    lambda row: self._faith_pd_taxonomy(row, tax_df, rank_cols), axis=1)
            import warnings
            warnings.warn(
                "Faith PD: не загружено ни дерево, ни таксономия. "
                "Загрузите Newick-дерево для истинного Faith PD.")
            return pd.Series(np.nan, index=otu_data.index)
        raise ValueError(f"Unknown metric: {metric}")

    def calculate_metrics(self):
        auto_filter = self.settings.get('auto_filter_zeros', True)
        excluded = self.settings.get('excluded_samples', [])
        otu = self.data_loader.get_filtered_otu_table(auto_filter)

        if excluded:
            keep = [s for s in otu.index if s not in excluded]
            if not keep:
                raise ValueError("All samples excluded.")
            otu = otu.loc[keep]
            meta = self.data_loader.metadata.loc[self.data_loader.metadata.index.isin(keep)]
        else:
            meta = self.data_loader.metadata

        for metric in self.METRIC_LABELS:
            vals = self.calculate_metric(otu, metric)
            df = pd.DataFrame({metric: vals})
            for col in meta.columns:
                df[col] = meta.reindex(df.index, fill_value=np.nan)[col]
            self.results[metric] = df

        self.excluded_samples = excluded
        self.active_samples = otu.index.tolist()
        self._calc_stats()

    def _shannon(self, c):
        c = c[c > 0]
        if len(c) == 0: return 0
        p = c / c.sum()
        return -np.sum(p * np.log(p))

    def _simpson(self, c):
        c = c[c > 0]
        if len(c) == 0: return 0
        n = c.sum()
        if n <= 1: return 0
        return 1 - np.sum((c * (c - 1)) / (n * (n - 1)))

    def _chao1(self, c):
        obs = (c > 0).sum()
        s1 = (c == 1).sum()
        s2 = (c == 2).sum()
        if s2 > 0 and s1 > 0:
            return obs + (s1**2) / (2 * s2)
        elif s1 > 0:
            return obs + s1 * (s1 - 1) / 2
        return obs

    # ─── Таксономия для Faith PD ───

    def _get_taxonomy_info(self):
        """
        Возвращает (taxonomy_df, rank_cols) из data_loader.
        rank_cols — упорядоченный список столбцов от Kingdom до Species.
        """
        if not hasattr(self.data_loader, 'taxonomy') or self.data_loader.taxonomy is None:
            return None, []
        tax = self.data_loader.taxonomy
        if tax.empty:
            return None, []

        # Ищем стандартные ранги (QIIME2 / SILVA / Greengenes)
        rank_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        rank_cols = []
        lower_cols = {c.lower().strip(): c for c in tax.columns}
        for rank in rank_order:
            if rank in lower_cols:
                rank_cols.append(lower_cols[rank])
            elif rank[0] in lower_cols:          # k, p, c, o, f, g, s
                rank_cols.append(lower_cols[rank[0]])

        if not rank_cols:
            # Попытка найти одну строку-таксономию (вида "k__Bacteria;p__...")
            for col in tax.columns:
                if tax[col].astype(str).str.contains(';').any():
                    rank_cols = [col]
                    break

        return tax, rank_cols

    def _faith_pd_taxonomy(self, sample_series, taxonomy_df, rank_cols):
        """
        Faith's Phylogenetic Diversity через таксономический прокси-дерево
        (Faith 1992, doi:10.1016/0006-3207(92)91201-3).

        Алгоритм:
        - Для каждого присутствующего OTU прослеживаем путь по таксономическому
          дереву (Kingdom → Species).
        - Каждый уникальный узел дерева считается «ветвью».
        - Длина ветви = 1/rank_depth (более высокие ранги вносят больший вклад,
          т.к. разделяют бо́льшую эволюционную историю).
        - PD = сумма длин всех уникальных ветвей, пройденных образцом.

        Без реального Newick-дерева это лучший возможный proxy.
        """
        present_otus = sample_series[sample_series > 0].index
        if len(present_otus) == 0:
            return 0.0

        # Длины ветвей по рангу: Kingdom самая длинная
        n_ranks = len(rank_cols)
        rank_weights = {i: 1.0 / (i + 1) for i in range(n_ranks)}

        # Если один столбец — разбиваем строку "k__X;p__Y;..."
        if len(rank_cols) == 1:
            return self._faith_pd_from_string(
                sample_series, taxonomy_df, rank_cols[0], rank_weights)

        unique_nodes: set = set()
        for otu in present_otus:
            if otu not in taxonomy_df.index:
                continue
            row = taxonomy_df.loc[otu]
            for rank_idx, col in enumerate(rank_cols):
                val = str(row[col]).strip()
                if val and val not in ('nan', 'NA', '', 'None'):
                    unique_nodes.add((rank_idx, val))

        return sum(rank_weights[ri] for ri, _ in unique_nodes)

    def _faith_pd_from_string(self, sample_series, taxonomy_df, tax_col, rank_weights):
        """Вспомогательный: Faith PD когда таксономия хранится одной строкой."""
        present_otus = sample_series[sample_series > 0].index
        unique_nodes: set = set()
        for otu in present_otus:
            if otu not in taxonomy_df.index:
                continue
            lineage = str(taxonomy_df.loc[otu, tax_col])
            for rank_idx, taxon in enumerate(lineage.split(';')):
                taxon = taxon.strip()
                # Убираем префиксы k__, p__, etc.
                if '__' in taxon:
                    taxon = taxon.split('__', 1)[-1].strip()
                if taxon and taxon not in ('', 'nan', 'NA', 'unidentified'):
                    unique_nodes.add((rank_idx, taxon))
        return sum(rank_weights.get(ri, 0.1) for ri, _ in unique_nodes)

    def _faith_pd_real(self, sample_series, tree):
        """
        Настоящий Faith's PD (Faith 1992) через scikit-bio.

        Использует реальное Newick-дерево (SILVA / Greengenes / QIIME2 и др.).
        Вычисляет сумму длин ветвей, соединяющих все присутствующие OTU/ASV
        с корнем дерева — строго по оригинальной формуле.

        Если OTU из таблицы отсутствуют в дереве — они пропускаются;
        если ни один OTU не совпал → возвращается NaN.
        """
        try:
            from skbio.diversity.alpha import faith_pd as skbio_faith_pd
        except ImportError:
            # scikit-bio не установлен — fallback на таксономический прокси
            tax_df, rank_cols = self._get_taxonomy_info()
            if tax_df is not None and rank_cols:
                return self._faith_pd_taxonomy(sample_series, tax_df, rank_cols)
            return np.nan

        present = sample_series[sample_series > 0]
        if len(present) == 0:
            return 0.0

        # Имена концов дерева
        tree_tips = {tip.name for tip in tree.tips()}

        # Оставляем только OTU, присутствующие в дереве
        otu_ids = [str(otu) for otu in present.index if str(otu) in tree_tips]
        if not otu_ids:
            return np.nan   # OTU не совпали с деревом — честнее вернуть NaN

        counts = present[otu_ids].values.astype(int)
        # Убеждаемся что нет отрицательных значений (после округления)
        counts = np.maximum(counts, 0)

        try:
            return float(skbio_faith_pd(counts, otu_ids, tree))
        except Exception:
            return np.nan

    # ─── Статистика ───

    def _calc_stats(self):
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return
        groups = self.data_loader.metadata[gc].unique()
        if len(groups) < 2:
            return

        results = {}
        for metric in self.METRIC_LABELS:
            if metric not in self.results:
                continue
            md = self.results[metric]
            if not isinstance(md, pd.DataFrame) or metric not in md.columns:
                continue
            if gc not in md.columns:
                continue

            gd = [md[md[gc] == g][metric].dropna() for g in groups]
            gd = [x for x in gd if len(x) > 0]
            if len(gd) < 2:
                continue

            if len(gd) == 2:
                stat, pv = stats.mannwhitneyu(gd[0], gd[1], alternative='two-sided')
                ps = np.sqrt((gd[0].var() + gd[1].var()) / 2)
                es = abs(gd[1].mean() - gd[0].mean()) / ps if ps > 0 else 0
            else:
                stat, pv = stats.kruskal(*gd)
                tn = sum(len(x) for x in gd)
                es = (stat - len(gd) + 1) / (tn - len(gd)) if tn > len(gd) else 0

            results[metric] = {'statistic': stat, 'p_value': pv, 'effect_size': es, 'n_groups': len(gd)}

        if results:
            pvs = [results[m]['p_value'] for m in results]
            _, padj, _, _ = multipletests(pvs, method='fdr_bh') if len(pvs) > 1 else (None, pvs, None, None)
            for i, m in enumerate(results):
                results[m]['p_adjusted'] = padj[i]

        self.statistics[gc] = results

    def _sig(self, p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return 'ns'

    # ─── Plotly: один график ───

    def plot_metric_plotly(self, metric, metric_label=None):
        """Интерактивный Plotly-график для одной метрики (полный тулбар)"""
        if metric not in self.results:
            return None
        label = metric_label or self.METRIC_LABELS.get(metric, metric)
        md = self.results[metric]
        if not isinstance(md, pd.DataFrame):
            return None

        clean = md.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        gc = self.settings.get('group_column')
        fig = go.Figure()

        if gc and gc in clean.columns:
            groups = sorted(clean[gc].dropna().unique())
            for group in groups:
                sub = clean[clean[gc] == group][metric]
                if len(sub) == 0:
                    continue
                color = self.group_colors_hex.get(group, '#3498DB')
                fig.add_trace(go.Box(
                    y=sub.values, name=str(group), marker_color=color,
                    boxmean='sd', jitter=0.3, pointpos=-1.5, boxpoints='all',
                    marker=dict(size=6, opacity=0.7, line=dict(width=1, color='black')),
                    line=dict(width=2), fillcolor=color, opacity=0.75
                ))

            # Статистика
            ann = self._stats_ann(metric, gc)
            if ann:
                fig.add_annotation(text=ann, xref="paper", yref="paper",
                                   x=0.98, y=0.98, showarrow=False, font=dict(size=12),
                                   bgcolor="rgba(255,228,181,0.9)", bordercolor="#DEB887",
                                   borderwidth=1, borderpad=6)
        else:
            fig.add_trace(go.Histogram(x=clean[metric].values, nbinsx=30,
                                       marker_color='#3498DB', opacity=0.75, name=label))

        fig.update_layout(
            title=dict(text=f'{label} Distribution', font=dict(size=18, family='Arial Black'), x=0.5),
            yaxis_title=label,
            xaxis_title=gc.title() if gc and gc in clean.columns else label,
            height=550, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=80, r=40, t=100, b=80)
        )
        return fig

    # ─── Plotly: сетка всех метрик ───

    def plot_all_metrics_plotly(self, selected_metrics):
        """Plotly Grid-график со всеми метриками"""
        if not selected_metrics:
            return None
        n = len(selected_metrics)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        titles = [self.METRIC_LABELS.get(m, m) for m in selected_metrics]

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                           vertical_spacing=0.12, horizontal_spacing=0.08)
        gc = self.settings.get('group_column')

        for idx, metric in enumerate(selected_metrics):
            r, c = idx // cols + 1, idx % cols + 1
            if metric not in self.results:
                continue
            md = self.results[metric]
            if not isinstance(md, pd.DataFrame):
                continue
            clean = md.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

            if gc and gc in clean.columns:
                groups = sorted(clean[gc].dropna().unique())
                for group in groups:
                    sub = clean[clean[gc] == group][metric]
                    if len(sub) == 0:
                        continue
                    color = self.group_colors_hex.get(group, '#3498DB')
                    fig.add_trace(
                        go.Box(y=sub.values, name=str(group), marker_color=color,
                               boxmean='sd', jitter=0.3, pointpos=-1.5, boxpoints='all',
                               marker=dict(size=4, opacity=0.6), line=dict(width=1.5),
                               fillcolor=color, opacity=0.7,
                               showlegend=(idx == 0), legendgroup=str(group)),
                        row=r, col=c
                    )
            else:
                fig.add_trace(
                    go.Histogram(x=clean[metric].values, nbinsx=15,
                                 marker_color='#3498DB', opacity=0.7, showlegend=False),
                    row=r, col=c
                )

        fig.update_layout(
            title=dict(text='Alpha Diversity — Comprehensive Overview',
                      font=dict(size=20, family='Arial Black'), x=0.5),
            height=max(450, rows * 380), template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        return fig

    def _stats_ann(self, metric, gc):
        if gc not in self.statistics or metric not in self.statistics[gc]:
            return None
        s = self.statistics[gc][metric]
        p = s['p_adjusted']
        e = s['effect_size']
        sig = self._sig(p)
        if s['n_groups'] == 2:
            return f"p(adj)={p:.4f} {sig}<br>|d|={abs(e):.3f}"
        return f"p(adj)={p:.4f} {sig}<br>η²={e:.3f}"

    # ─── Matplotlib (публикационный стиль) ───

    def plot_metric(self, metric, metric_label):
        if metric not in self.results:
            return None
        fig, ax = plt.subplots(figsize=(12, 7))
        md = self.results[metric]
        if not isinstance(md, pd.DataFrame):
            return None
        # subset=[metric]: only drop rows where the metric is NaN,
        # not rows where unrelated metadata columns (e.g. 'replicate') are NaN
        clean = md.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        gc = self.settings.get('group_column')

        if gc and gc in clean.columns:
            groups = sorted(clean[gc].dropna().unique())
            bd, bp, bc = [], [], []
            for i, g in enumerate(groups):
                sub = clean[clean[gc] == g]
                if len(sub) > 0:
                    bd.append(sub[metric].values)
                    bp.append(i)
                    bc.append(self.group_colors.get(g, sns.color_palette("Set2")[i % 8]))
            if bd:
                bx = ax.boxplot(bd, positions=bp, widths=0.6, patch_artist=True,
                               showfliers=False, notch=True,
                               medianprops={'color': 'black', 'linewidth': 2})
                for patch, col in zip(bx['boxes'], bc):
                    patch.set_facecolor(col); patch.set_alpha(0.8); patch.set_edgecolor('black')
                for data, pos, col in zip(bd, bp, bc):
                    jx = pos + np.random.normal(0, 0.08, len(data))
                    ax.scatter(jx, data, color=col, s=25, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)
                ax.set_xticks(bp)
                ax.set_xticklabels([str(g)[:15] for g in groups], rotation=45, ha='right', fontsize=11)
                ax.set_xlabel(gc.title(), fontweight='bold', fontsize=13)
                legend_el = [mpatches.Patch(facecolor=c, label=g, alpha=0.8, edgecolor='black') for g, c in zip(groups, bc)]
                ax.legend(handles=legend_el, bbox_to_anchor=(0.02, 0.98), loc='upper left',
                         title=gc.title(), title_fontsize=12, fontsize=11, framealpha=0.95)
        else:
            ax.hist(clean[metric].values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{metric_label} Distribution', fontweight='bold', fontsize=15, pad=15)
        ax.set_ylabel(metric_label, fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.15)
        return fig

    def plot_grid(self, selected_metrics, metric_labels):
        if not selected_metrics:
            return None
        n = len(selected_metrics)
        cols = min(3, n); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(max(15, cols*5), max(8, rows*4.5)))
        fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.15, wspace=0.25, hspace=0.35)
        fig.suptitle('Alpha Diversity Analysis', fontsize=16, fontweight='bold', y=0.96)
        if rows == 1 and cols == 1: axes = np.array([[axes]])
        elif rows == 1: axes = axes.reshape(1, -1)
        elif cols == 1: axes = axes.reshape(-1, 1)
        gc = self.settings.get('group_column')
        for i, m in enumerate(selected_metrics):
            ax = axes[i // cols, i % cols]
            if m not in self.results:
                ax.set_visible(False); continue
            md = self.results[m]
            if not isinstance(md, pd.DataFrame):
                ax.set_visible(False); continue
            # subset=[m]: drop only rows where the metric itself is NaN,
            # NOT rows where any unrelated metadata column (e.g. 'replicate') is NaN
            clean = md.replace([np.inf, -np.inf], np.nan).dropna(subset=[m])
            if gc and gc in clean.columns:
                groups = sorted(clean[gc].dropna().unique())
                bd, bp, bc = [], [], []
                for j, g in enumerate(groups):
                    sub = clean[clean[gc] == g]
                    if len(sub) > 0:
                        bd.append(sub[m].values); bp.append(j)
                        bc.append(self.group_colors.get(g, sns.color_palette("Set2")[j % 8]))
                if bd:
                    bx = ax.boxplot(bd, positions=bp, widths=0.6, patch_artist=True, showfliers=False,
                                   medianprops={'color': 'black', 'linewidth': 1.5})
                    for patch, col in zip(bx['boxes'], bc):
                        patch.set_facecolor(col); patch.set_alpha(0.7); patch.set_edgecolor('black')
                    for data, pos, col in zip(bd, bp, bc):
                        ax.scatter(pos + np.random.normal(0, 0.06, len(data)), data, color=col, s=12, alpha=0.6, edgecolors='black', linewidth=0.3)
                    ax.set_xticks(bp)
                    ax.set_xticklabels([str(g)[:10] for g in groups], rotation=45, ha='right', fontsize=9)
            else:
                ax.hist(clean[m].values, bins=15, color='lightblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'{chr(65+i)}. {metric_labels.get(m, m)}', fontweight='bold', fontsize=11, pad=8)
            ax.set_ylabel(metric_labels.get(m, m), fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for i in range(n, rows * cols):
            axes[i // cols, i % cols].set_visible(False)
        return fig

    # ─── Сводки ───

    def get_statistics_summary(self):
        rows = []
        gc = self.settings.get('group_column')
        for m in self.METRIC_LABELS:
            if m not in self.results: continue
            row = {'Metric': self.METRIC_LABELS[m]}
            md = self.results[m]
            if isinstance(md, pd.DataFrame) and m in md.columns:
                v = md[m].dropna()
                row['Mean ± SD'] = f"{v.mean():.3f} ± {v.std():.3f}"
                row['Median [IQR]'] = f"{v.median():.3f} [{v.quantile(0.25):.3f}-{v.quantile(0.75):.3f}]"
            if gc and gc in self.statistics and m in self.statistics[gc]:
                s = self.statistics[gc][m]
                p = s['p_adjusted']
                row['p-value (adj)'] = f"{p:.4f} {self._sig(p)}"
                row['Effect Size'] = (f"d = {s['effect_size']:.3f}" if s['n_groups'] == 2
                                     else f"η² = {s['effect_size']:.3f}")
            rows.append(row)
        return pd.DataFrame(rows)

    def get_group_summary_table(self):
        gc = self.settings.get('group_column')
        if not gc: return pd.DataFrame()
        rows = []
        for m in self.METRIC_LABELS:
            if m not in self.results: continue
            md = self.results[m]
            if not isinstance(md, pd.DataFrame) or gc not in md.columns: continue
            for g in sorted(md[gc].unique()):
                v = md[md[gc] == g][m].dropna()
                if len(v) > 0:
                    rows.append({'Metric': self.METRIC_LABELS[m], 'Group': g, 'N': len(v),
                                'Mean': round(v.mean(), 4), 'SD': round(v.std(), 4),
                                'Median': round(v.median(), 4), 'Min': round(v.min(), 4),
                                'Max': round(v.max(), 4)})
        return pd.DataFrame(rows)
