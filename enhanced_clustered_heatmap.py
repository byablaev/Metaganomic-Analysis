"""
Кластеризованные тепловые карты для микробиомных данных
Иерархическая кластеризация, K-means, DBSCAN
Профессиональная визуализация с дендрограммами и аннотациями групп

Layout (domain-based axes, всё НЕ перекрывается):
┌─────────────────────────────────────────────────────────┐
│                    Group Legend (top)                    │
├────────────┬──────────────────────────┬──────────────────┤
│            │   Column Dendrogram      │                  │
│            ├──────────────────────────┤                  │
│            │   Column Group Bar       │                  │
├────────────┼──────────────────────────┤    Taxa Labels   │
│   Row      │                          │    (right side   │
│   Dendro-  │       HEATMAP            │     of yaxis)    │
│   gram     │                          │                  │
├────────────┼──────────────────────────┼──────────────────┤
│            │   Column Labels (bottom) │                  │
└────────────┴──────────────────────────┴──────────────────┘
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class EnhancedClusteredHeatmapAnalyzer:
    """Кластеризованные тепловые карты с универсальной группировкой"""

    def __init__(self, data_loader, settings):
        self.data_loader = data_loader
        self.settings = settings
        self.processed_data = None
        self.group_means = None
        self.cluster_results = {}
        self.grouped_samples = {}
        self._create_group_colors()

    def _create_group_colors(self):
        """Палитра цветов для групп"""
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            unique_groups = sorted(self.data_loader.metadata[gc].unique().astype(str))
            n = len(unique_groups)
            palette = [
                '#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
                '#E91E63', '#00BCD4', '#8BC34A', '#FF9800',
            ]
            if n <= len(palette):
                colors = palette[:n]
            else:
                sc = sns.color_palette("husl", n)
                colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sc]
            self.group_colors = dict(zip(unique_groups, colors))
        else:
            self.group_colors = {}

    # ─── Обработка данных ───

    def process_data(self):
        """Обработка данных для кластеризованной тепловой карты"""
        try:
            from normalization import NormalizationProcessor
            normalizer = NormalizationProcessor()

            tax_level = self.settings.get('taxonomic_level', 'Genus')
            auto_filter = self.settings.get('auto_filter_zeros', True)
            gc = self.settings.get('group_column')

            otu = self.data_loader.get_filtered_otu_table(auto_filter)
            if otu.empty:
                self._empty(); return

            if (hasattr(self.data_loader, 'taxonomy') and
                self.data_loader.taxonomy is not None and
                tax_level in self.data_loader.taxonomy.columns):
                otu_t = otu.T
                merged = otu_t.merge(self.data_loader.taxonomy[[tax_level]],
                                    left_index=True, right_index=True, how='left')
                merged[tax_level] = merged[tax_level].fillna('Unclassified')
                sample_cols = list(otu.index)
                agg = merged.groupby(tax_level)[sample_cols].sum()
            else:
                agg = otu.T

            non_zero = agg.sum(axis=1) > 0
            agg = agg[non_zero]
            if agg.empty:
                self._empty(); return

            rel = normalizer.get_relative_abundance(agg.T)

            if gc and gc in self.data_loader.metadata.columns:
                common_samples = list(set(rel.index) & set(self.data_loader.metadata.index))
                if len(common_samples) < 3:
                    common_samples = self._smart_match(rel, self.data_loader.metadata)

                if len(common_samples) >= 3:
                    rel_sub = rel.loc[common_samples]
                    meta_sub = self.data_loader.metadata.loc[common_samples]

                    self.grouped_samples = {}
                    for group in sorted(meta_sub[gc].unique()):
                        g_samples = meta_sub[meta_sub[gc] == group].index.tolist()
                        if g_samples:
                            self.grouped_samples[str(group)] = g_samples

                    self.group_means = pd.DataFrame()
                    for gname, gsamples in self.grouped_samples.items():
                        avail = [s for s in gsamples if s in rel_sub.index]
                        if avail:
                            self.group_means[gname] = rel_sub.loc[avail].mean(axis=0)

                    self.processed_data = rel_sub
                else:
                    self.group_means = rel.T
                    self.processed_data = rel
                    self.grouped_samples = {'All': list(rel.index)}
            else:
                self.group_means = rel.T
                self.processed_data = rel
                self.grouped_samples = {'All': list(rel.index)}

            if self.group_means.empty:
                self._empty()

        except Exception as e:
            print(f"Error in process_data: {e}")
            import traceback; traceback.print_exc()
            self._empty()

    def _smart_match(self, rel, metadata):
        data_samples = list(rel.index)
        meta_samples = list(metadata.index)
        n = min(len(data_samples), len(meta_samples))
        if n < 3: return []
        new_names = meta_samples[:n]
        rename = dict(zip(data_samples[:n], new_names))
        rel.rename(index=rename, inplace=True)
        return new_names

    def _empty(self):
        self.group_means = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        self.grouped_samples = {}

    # ═══════════════════════════════════════════════════
    #  ОСНОВНОЙ МЕТОД: Кластеризованная тепловая карта
    # ═══════════════════════════════════════════════════

    def create_clustered_heatmap(self, top_n=50, clustering_algorithm='hierarchical',
                                  linkage_method='average', distance_metric='euclidean',
                                  show_row_dendrogram=True, show_col_dendrogram=True,
                                  colorscale='YlOrRd', show_values=False,
                                  hide_zeros=False, font_scale=1.0):
        try:
            if self.group_means is None or self.group_means.empty:
                return None

            data = self.group_means.copy()
            if len(data) > top_n:
                variance = data.var(axis=1)
                data = data.loc[variance.nlargest(top_n).index]

            if data.empty or data.shape[0] < 2:
                return None

            data = data.replace([np.inf, -np.inf], 0).fillna(0)

            # Кластеризация
            if clustering_algorithm == 'hierarchical':
                ordered, row_link, col_link = self._hierarchical(data, linkage_method, distance_metric)
            elif clustering_algorithm == 'kmeans':
                ordered, row_link, col_link = self._kmeans(data)
            elif clustering_algorithm == 'dbscan':
                ordered, row_link, col_link = self._dbscan(data)
            else:
                ordered, row_link, col_link = data, None, None

            if ordered is None or ordered.empty:
                return None

            z = ordered.values
            x_labels = [str(c) for c in ordered.columns]

            tax_level = self.settings.get('taxonomic_level', 'Genus')
            use_italic = tax_level in ['Genus', 'Species']
            y_labels_raw = [str(r) for r in ordered.index]
            y_labels_display = [f'<i>{t}</i>' if use_italic else t for t in y_labels_raw]

            n_taxa = len(y_labels_raw)
            n_cols = len(x_labels)

            if hide_zeros:
                z_display = z.copy().astype(float)
                z_display[z_display == 0] = np.nan
            else:
                z_display = z

            has_row_dendro = (show_row_dendrogram and row_link is not None
                             and clustering_algorithm == 'hierarchical')
            has_col_dendro = (show_col_dendrogram and col_link is not None
                             and clustering_algorithm == 'hierarchical')
            has_groups = len(self.grouped_samples) > 1 and len(self.group_colors) > 0

            # ═══ DOMAIN LAYOUT CALCULATION ═══
            # Всё вычисляется заранее, чтобы ничего не перекрывалось

            GAP = 0.015  # зазор между областями

            # ── X-домены (горизонтально) ──
            x_cur = 0.0
            if has_row_dendro:
                row_dendro_x0 = x_cur
                row_dendro_x1 = x_cur + 0.08
                x_cur = row_dendro_x1 + GAP
            else:
                row_dendro_x0 = row_dendro_x1 = None

            heat_x0 = x_cur
            heat_x1 = 0.87  # место для colorbar и подписей справа

            # ── Y-домены (вертикально, снизу вверх) ──
            heat_y0 = 0.0
            y_cur_top = 1.0

            if has_col_dendro:
                col_dendro_y1 = y_cur_top
                col_dendro_y0 = y_cur_top - 0.10
                y_cur_top = col_dendro_y0 - GAP
            else:
                col_dendro_y0 = col_dendro_y1 = None

            # Цветовая полоса групп сверху (тонкая)
            if has_groups:
                group_bar_y1 = y_cur_top
                group_bar_y0 = y_cur_top - 0.025
                y_cur_top = group_bar_y0 - GAP * 0.5
            else:
                group_bar_y0 = group_bar_y1 = None

            heat_y1 = y_cur_top

            # ═══ BUILD FIGURE ═══
            fig = go.Figure()
            axes_layout = {}

            # ── 1. HEATMAP (xaxis / yaxis) ──
            y_font = max(8, int(10 * font_scale))
            x_font = max(9, int(11 * font_scale))
            text_vals = np.round(z, 2).astype(str) if show_values else None

            fig.add_trace(go.Heatmap(
                z=z_display,
                x=x_labels,
                y=y_labels_display,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text='Abundance (%)', font=dict(size=int(11 * font_scale))),
                    len=0.4, thickness=13,
                    tickfont=dict(size=int(10 * font_scale)),
                    x=1.0, xanchor='left', yanchor='middle', y=0.5
                ),
                text=text_vals,
                texttemplate='%{text}' if show_values else None,
                textfont=dict(size=max(7, int(8 * font_scale))),
                hovertemplate='<b>%{y}</b><br>Group: %{x}<br>Abundance: %{z:.3f}%<extra></extra>',
                showscale=True,
                xgap=1, ygap=1,
                xaxis='x', yaxis='y'
            ))

            axes_layout['xaxis'] = dict(
                domain=[heat_x0, heat_x1],
                tickangle=-45 if n_cols > 5 else 0,
                tickfont=dict(size=x_font, family='Arial'),
                side='bottom',
                showgrid=False, zeroline=False,
            )
            axes_layout['yaxis'] = dict(
                domain=[heat_y0, heat_y1],
                tickfont=dict(size=y_font, family='Arial'),
                autorange='reversed',
                dtick=1,
                side='right',  # Подписи таксонов СПРАВА
                showgrid=False, zeroline=False,
            )

            # ── 2. ROW DENDROGRAM (xaxis2 / yaxis2) ──
            if has_row_dendro:
                d = dendrogram(row_link, no_plot=True, labels=y_labels_raw, orientation='left')
                for seg_i in range(len(d['icoord'])):
                    x_v = d['dcoord'][seg_i]
                    y_v = [(yi - 5) / 10 for yi in d['icoord'][seg_i]]
                    fig.add_trace(go.Scatter(
                        x=x_v, y=y_v, mode='lines',
                        line=dict(color='#444', width=1.0),
                        showlegend=False, hoverinfo='skip',
                        xaxis='x2', yaxis='y2'
                    ))

                axes_layout['xaxis2'] = dict(
                    domain=[row_dendro_x0, row_dendro_x1],
                    showticklabels=False, showgrid=False, zeroline=False,
                    autorange='reversed',
                    anchor='y2',
                )
                axes_layout['yaxis2'] = dict(
                    domain=[heat_y0, heat_y1],   # точно совпадает с heatmap
                    showticklabels=False, showgrid=False, zeroline=False,
                    # ВАЖНО: при явном range нельзя одновременно ставить autorange —
                    # Plotly молча игнорирует range и использует autorange,
                    # что сдвигает дендрограмму. Задаём range явно, autorange=False.
                    range=[n_taxa - 0.5, -0.5],  # reversed вручную
                    autorange=False,
                    anchor='x2',
                )

            # ── 3. COLUMN DENDROGRAM (xaxis3 / yaxis3) ──
            if has_col_dendro:
                d = dendrogram(col_link, no_plot=True, labels=x_labels)
                for seg_i in range(len(d['icoord'])):
                    x_v = [(xi - 5) / 10 for xi in d['icoord'][seg_i]]
                    y_v = d['dcoord'][seg_i]
                    fig.add_trace(go.Scatter(
                        x=x_v, y=y_v, mode='lines',
                        line=dict(color='#444', width=1.0),
                        showlegend=False, hoverinfo='skip',
                        xaxis='x3', yaxis='y3'
                    ))

                axes_layout['xaxis3'] = dict(
                    domain=[heat_x0, heat_x1],  # Совпадает с heatmap по X
                    showticklabels=False, showgrid=False, zeroline=False,
                    range=[-0.5, n_cols - 0.5],
                    anchor='y3',
                )
                axes_layout['yaxis3'] = dict(
                    domain=[col_dendro_y0, col_dendro_y1],
                    showticklabels=False, showgrid=False, zeroline=False,
                    anchor='x3',
                )

            # ── 4. GROUP COLOR BAR (shapes сверху heatmap) ──
            shapes = []
            if has_groups and group_bar_y0 is not None:
                for i, label in enumerate(x_labels):
                    # Определяем группу для каждого столбца
                    color = '#CCCCCC'
                    for gname in self.grouped_samples:
                        if label == gname or label in [str(s) for s in self.grouped_samples[gname]]:
                            color = self.group_colors.get(gname, '#CCCCCC')
                            break

                    # Координаты в paper space
                    x_span = heat_x1 - heat_x0
                    col_w = x_span / n_cols
                    sx0 = heat_x0 + i * col_w
                    sx1 = heat_x0 + (i + 1) * col_w

                    shapes.append(dict(
                        type='rect',
                        xref='paper', yref='paper',
                        x0=sx0, y0=group_bar_y0,
                        x1=sx1, y1=group_bar_y1,
                        fillcolor=color,
                        line=dict(width=0.5, color='white'),
                        layer='above'
                    ))

            # Добавляем разделители групп на heatmap (белые линии)
            if has_groups and n_cols > 1:
                prev_group = None
                for i, label in enumerate(x_labels):
                    cur_group = None
                    for gname in self.grouped_samples:
                        if label == gname or label in [str(s) for s in self.grouped_samples[gname]]:
                            cur_group = gname
                            break
                    if prev_group is not None and cur_group != prev_group:
                        # Белая линия-разделитель на heatmap
                        x_span = heat_x1 - heat_x0
                        col_w = x_span / n_cols
                        line_x = heat_x0 + i * col_w

                        shapes.append(dict(
                            type='line',
                            xref='paper', yref='paper',
                            x0=line_x, y0=heat_y0,
                            x1=line_x, y1=heat_y1,
                            line=dict(color='white', width=3),
                            layer='above'
                        ))
                    prev_group = cur_group

            # ── 5. GROUP LEGEND ──
            if has_groups:
                for gname, color in self.group_colors.items():
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=14, color=color, symbol='square'),
                        name=gname,
                        showlegend=True,
                        hoverinfo='skip'
                    ))

            # ═══ GLOBAL LAYOUT ═══
            # Адаптивная высота строки: минимум 18px, максимум 28px
            row_h = max(18, min(28, int(20 * font_scale)))
            fig_height = max(600, n_taxa * row_h + 280)

            max_lbl = max(len(t) for t in y_labels_raw) if y_labels_raw else 10
            # Адаптивный правый отступ под подписи таксонов
            right_margin = max(160, int(max_lbl * y_font * 0.52) + 80)

            # Build study context subtitle
            gc = self.settings.get('group_column')
            tax_level = self.settings.get('taxonomic_level', '')
            title_text = f'Clustered Heatmap: Top {n_taxa} Taxa'
            if tax_level:
                title_text += f' ({tax_level} level)'

            subtitle = ''
            if gc and gc in self.data_loader.metadata.columns:
                groups = sorted(self.data_loader.metadata[gc].unique().astype(str))
                n_samples = len(self.data_loader.metadata)
                subtitle = f'<br><span style="font-size:12px;color:#666">Grouped by: {gc} ({", ".join(groups)}) · {n_samples} samples</span>'

            fig.update_layout(
                **axes_layout,
                shapes=shapes,
                title=dict(
                    text=title_text + subtitle,
                    font=dict(size=int(16 * font_scale), family='Arial'),
                    x=0.5, y=0.99
                ),
                height=fig_height,
                margin=dict(
                    l=25,
                    r=right_margin,
                    t=130 if (has_col_dendro or has_groups) else 80,
                    b=max(80, n_cols * 4 + 60)
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=has_groups,
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.01,
                    xanchor='left', x=0.0,
                    font=dict(size=int(11 * font_scale)),
                    itemsizing='constant',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#DDD', borderwidth=1
                ) if has_groups else None,
            )

            return fig

        except Exception as e:
            print(f"Error creating clustered heatmap: {e}")
            import traceback; traceback.print_exc()
            return None

    # ─── Алгоритмы кластеризации ───

    def _hierarchical(self, data, method='average', metric='euclidean'):
        """Иерархическая кластеризация"""
        try:
            # scipy pdist не принимает 'manhattan' — правильное название 'cityblock'
            _METRIC_MAP = {'manhattan': 'cityblock', 'cosine': 'cosine',
                           'correlation': 'correlation', 'euclidean': 'euclidean'}
            metric = _METRIC_MAP.get(metric, metric)

            row_link = None
            col_link = None

            if len(data) > 1:
                row_data = data.values.copy()
                row_std = np.std(row_data, axis=1)
                valid_rows = row_std > 0

                if valid_rows.sum() > 1:
                    if metric == 'correlation':
                        # pdist должен получать только строки с ненулевой дисперсией;
                        # сохраняем исходные индексы чтобы row_order относился
                        # к полному набору строк, а не к подмножеству
                        safe_idx = np.where(valid_rows)[0]
                        safe_data = row_data[safe_idx]
                        try:
                            row_dist = pdist(safe_data, metric=metric)
                        except Exception:
                            row_dist = pdist(safe_data, metric='euclidean')
                        safe_link = linkage(
                            np.nan_to_num(row_dist, nan=0.0, posinf=0.0, neginf=0.0),
                            method=method)
                        safe_order = dendrogram(safe_link, no_plot=True)['leaves']
                        # Расставляем нестатичные строки по их отсортированным позициям;
                        # константные строки добавляем в конец
                        constant_idx = np.where(~valid_rows)[0]
                        row_order = list(safe_idx[safe_order]) + list(constant_idx)
                        row_link = safe_link  # для отрисовки дендрограммы
                    else:
                        row_dist = pdist(row_data, metric=metric)
                        row_dist = np.nan_to_num(row_dist, nan=0.0, posinf=0.0, neginf=0.0)
                        row_link = linkage(row_dist, method=method)
                        row_order = dendrogram(row_link, no_plot=True)['leaves']
                else:
                    row_order = list(range(len(data)))
            else:
                row_order = [0]

            if len(data.columns) > 1:
                col_data = data.T.values.copy()
                col_std = np.std(col_data, axis=1)
                valid_cols = col_std > 0
                if valid_cols.sum() > 1:
                    try:
                        col_dist = pdist(col_data, metric=metric)
                    except Exception:
                        col_dist = pdist(col_data, metric='euclidean')
                    col_dist = np.nan_to_num(col_dist, nan=0.0, posinf=0.0, neginf=0.0)
                    col_link = linkage(col_dist, method=method)
                    col_order = dendrogram(col_link, no_plot=True)['leaves']
                else:
                    col_order = list(range(len(data.columns)))
            else:
                col_order = [0]

            return data.iloc[row_order, col_order], row_link, col_link

        except Exception as e:
            print(f"Hierarchical clustering error: {e}")
            return data, None, None

    def _kmeans(self, data, max_k=8):
        """K-means кластеризация"""
        try:
            if len(data) > 3:
                k_row = min(max_k, len(data) // 2)
                if k_row >= 2:
                    km = KMeans(n_clusters=k_row, random_state=42, n_init=10)
                    labels = km.fit_predict(data.values)
                    row_order = np.argsort(labels).tolist()
                else:
                    row_order = list(range(len(data)))
            else:
                row_order = list(range(len(data)))

            if len(data.columns) > 3:
                k_col = min(max_k, len(data.columns) // 2)
                if k_col >= 2:
                    km_c = KMeans(n_clusters=k_col, random_state=42, n_init=10)
                    labels_c = km_c.fit_predict(data.T.values)
                    col_order = np.argsort(labels_c).tolist()
                else:
                    col_order = list(range(len(data.columns)))
            else:
                col_order = list(range(len(data.columns)))

            return data.iloc[row_order, col_order], None, None

        except Exception as e:
            print(f"K-means error: {e}")
            return data, None, None

    def _dbscan(self, data, eps=0.5, min_samples=2):
        """DBSCAN кластеризация"""
        try:
            scaler = StandardScaler()
            if len(data) > min_samples:
                scaled = scaler.fit_transform(data.values)
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
                row_order = np.argsort(labels).tolist()
            else:
                row_order = list(range(len(data)))

            if len(data.columns) > min_samples:
                scaled_c = scaler.fit_transform(data.T.values)
                labels_c = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_c)
                col_order = np.argsort(labels_c).tolist()
            else:
                col_order = list(range(len(data.columns)))

            return data.iloc[row_order, col_order], None, None

        except Exception as e:
            print(f"DBSCAN error: {e}")
            return data, None, None

    # ─── Совместимость ───

    def _add_col_dendrogram_aligned(self, fig, link, labels, row=1, col=1, n_items=4):
        pass

    def _add_row_dendrogram_aligned(self, fig, link, labels, row=1, col=1, n_items=30):
        pass

    def _add_col_dendrogram(self, fig, link, labels, row=1, col=1):
        pass

    def _add_row_dendrogram(self, fig, link, labels, row=1, col=1):
        pass

    # ─── Статистика ───

    def get_comprehensive_statistics(self):
        if self.group_means is None or self.group_means.empty:
            return {}
        return {
            'total_taxa': len(self.group_means),
            'total_groups': len(self.group_means.columns),
            'abundant_taxa': int((self.group_means.mean(axis=1) >= 1.0).sum()),
            'max_abundance': float(self.group_means.max().max()),
            'mean_abundance': float(self.group_means.mean().mean()),
        }
