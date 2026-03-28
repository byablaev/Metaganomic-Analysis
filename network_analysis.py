"""
Модуль сетевого анализа микробиома — ОПТИМИЗИРОВАННАЯ ВЕРСИЯ

Ключевые оптимизации:
1. Векторизация O(n²) вычислений (numpy вместо Python-циклов)
2. Пакетная отрисовка рёбер (2 trace вместо N)
3. Кэширование метрик центральности
4. Настраиваемый max_taxa для контроля нагрузки
5. Progress callback для отображения прогресса в UI
6. Удаление изолированных узлов
7. Ограничение тяжёлых вычислений (diameter, eigenvector)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalyzer:
    """Оптимизированный анализатор микробиомных сетей"""

    def __init__(self, data_loader, settings):
        self.data_loader = data_loader
        self.settings = settings
        self.correlation_matrix = None
        self.network_data = None
        self.graph = None
        self.p_values = None
        self.communities = None
        self._tax_level_for_labels = settings.get('taxonomic_level', 'Genus')
        self._centrality_cache = {}
        self._build_time = 0
        self._n_taxa_used = 0

    # ═══════════════════════════════════════════
    #  КОРРЕЛЯЦИИ — ВЕКТОРИЗОВАННЫЕ
    # ═══════════════════════════════════════════

    def calculate_correlations(self, method='spearman', min_prevalence=0.1,
                               max_taxa=150, progress_cb=None):
        """
        Вычисление корреляций (векторизовано).

        Parameters
        ----------
        method : str — 'pearson', 'spearman', 'kendall', 'sparcc', 'proportionality'
        min_prevalence : float — мин. распространённость (0–1)
        max_taxa : int — максимум таксонов (главный контроль скорости)
        progress_cb : callable(text, pct) — для прогресс-бара
        """
        t0 = time.time()
        if progress_cb:
            progress_cb("Filtering taxa...", 5)

        from normalization import NormalizationProcessor

        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu_data = self.data_loader.get_filtered_otu_table(auto_filter)
        otu_t = otu_data.T  # taxa × samples

        # Фильтрация по prevalence
        prevalence = (otu_t > 0).mean(axis=1)
        filtered = otu_t[prevalence >= min_prevalence]

        if len(filtered) < 2:
            raise ValueError("Too few taxa after filtering (increase prevalence or data)")

        # Агрегация по таксономическому уровню
        tax_level = self._tax_level_for_labels
        if tax_level in self.data_loader.taxonomy.columns:
            tax_col = self.data_loader.taxonomy[tax_level].fillna('Unclassified')
            common_idx = list(set(filtered.index) & set(tax_col.index))
            if common_idx and len(common_idx) > 10:
                filtered = filtered.loc[common_idx]
                labels = tax_col[common_idx]
                filtered.index = [
                    f"{labels[idx]}" if labels[idx] != 'Unclassified'
                    else idx for idx in filtered.index
                ]
                filtered = filtered.groupby(filtered.index).sum()

        # Ограничение max_taxa — берём самые обильные
        if len(filtered) > max_taxa:
            total_ab = filtered.sum(axis=1).sort_values(ascending=False)
            filtered = filtered.loc[total_ab.head(max_taxa).index]

        self._n_taxa_used = len(filtered)
        n_taxa = self._n_taxa_used

        if progress_cb:
            progress_cb(f"Computing {method} correlations ({n_taxa} taxa)...", 15)

        normalizer = NormalizationProcessor()

        if method in ('sparcc', 'proportionality'):
            corr_matrix, p_matrix = self._compositional_correlation_fast(
                filtered, method=method, progress_cb=progress_cb)
        else:
            # CLR нормализация
            if self.settings.get('normalization', 'clr') == 'clr':
                try:
                    norm_data = normalizer._clr_transform(filtered.T).T
                except Exception:
                    norm_data = filtered
            else:
                norm_data = filtered

            # Pandas vectorized correlation
            corr_matrix = norm_data.T.corr(method=method if method in ('pearson', 'spearman') else 'spearman')

            if progress_cb:
                progress_cb("Computing p-values (vectorized)...", 35)

            p_matrix = self._pvalues_from_corr(norm_data.T, corr_matrix, method)

        self.correlation_matrix = corr_matrix
        self.p_values = p_matrix
        self._build_time = time.time() - t0

        if progress_cb:
            progress_cb(f"Correlations done ({self._build_time:.1f}s, {n_taxa} taxa)", 50)

    def _pvalues_from_corr(self, data_samples, corr_df, method='spearman'):
        """Векторизованные p-values из корреляционной матрицы.
        Используем t-статистику: t = r * sqrt(n-2) / sqrt(1-r²)"""
        n = len(data_samples)
        r = corr_df.values.copy()
        np.fill_diagonal(r, 0)

        r_sq = np.clip(r ** 2, 0, 0.9999)
        t_stat = np.abs(r) * np.sqrt(n - 2) / np.sqrt(1 - r_sq)
        p = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)
        p = np.clip(p, 0, 1)
        np.fill_diagonal(p, 1.0)

        return pd.DataFrame(p, index=corr_df.index, columns=corr_df.columns)

    def _compositional_correlation_fast(self, data, method='sparcc',
                                         n_iter=10, n_boot=50, progress_cb=None):
        """Оптимизированный SparCC / Proportionality через numpy."""
        data_np = data.T.values + 1  # samples × taxa
        n_samples, n_taxa = data_np.shape
        taxa_names = list(data.index)
        log_data = np.log(data_np)

        if method == 'sparcc':
            if progress_cb:
                progress_cb(f"SparCC ({n_iter} iter, {n_taxa} taxa)...", 20)

            # SparCC (Friedman & Alm 2012): оценка дисперсий через log-ratio матрицу T_ij
            # T_ij = Var(log xi - log xj) = Var_i + Var_j - 2*Cov_ij
            var_est = np.var(log_data, axis=0)
            corr_est = np.zeros((n_taxa, n_taxa))

            for _ in range(n_iter):
                cov_mat = np.cov(log_data.T)
                # T_ij — матрица log-ratio дисперсий
                T_mat = np.maximum(
                    var_est[:, None] + var_est[None, :] - 2 * cov_mat, 0)
                cov_ij = (var_est[:, None] + var_est[None, :] - T_mat) / 2
                denom = np.sqrt(np.outer(var_est, var_est))
                denom = np.maximum(denom, 1e-10)
                corr_est = cov_ij / denom

                # Обновление дисперсий по оригинальной формуле Friedman & Alm:
                # T_i = sum_j T_ij; Var_i = (T_i - max_j T_ij) / (n - 2)
                T_row = T_mat.sum(axis=1)  # Σ_j T_ij для каждого i
                T_max = T_mat.max(axis=1)  # max_j T_ij
                var_est = np.maximum(0.001,
                    (T_row - T_max) / max(1, n_taxa - 2))

            np.fill_diagonal(corr_est, 1.0)
            corr_est = np.clip(corr_est, -1, 1)
            corr_matrix = pd.DataFrame(corr_est, index=taxa_names, columns=taxa_names)

        else:  # proportionality
            if progress_cb:
                progress_cb(f"Proportionality ({n_taxa} taxa)...", 20)

            var_each = np.var(log_data, axis=0)
            cov_mat = np.cov(log_data.T)
            var_lr = np.maximum(
                var_each[:, None] + var_each[None, :] - 2 * cov_mat, 0)
            denom = np.maximum(var_each[:, None] + var_each[None, :], 1e-10)
            rho = 1 - var_lr / denom
            np.fill_diagonal(rho, 1.0)
            corr_matrix = pd.DataFrame(rho, index=taxa_names, columns=taxa_names)

        # Bootstrap p-values (permutation-based null distribution)
        # p = доля bootstrap-итераций где |r_boot| >= |r_obs|
        if progress_cb:
            progress_cb(f"Bootstrap p-values ({n_boot} iter)...", 35)

        obs = np.abs(corr_matrix.values)
        count_geq = np.zeros((n_taxa, n_taxa))
        rng = np.random.default_rng(42)

        for b in range(n_boot):
            idx = rng.choice(n_samples, n_samples, replace=True)
            boot_r = np.abs(np.corrcoef(np.log(data_np[idx] + 1).T))
            count_geq += (boot_r >= obs).astype(float)

        p_vals = count_geq / n_boot
        p_vals = np.clip(p_vals, 0, 1)
        np.fill_diagonal(p_vals, 1.0)

        return corr_matrix, pd.DataFrame(p_vals, index=taxa_names, columns=taxa_names)

    # ═══════════════════════════════════════════
    #  ПОСТРОЕНИЕ СЕТИ — ВЕКТОРИЗОВАННОЕ
    # ═══════════════════════════════════════════

    def build_network(self, correlation_threshold=0.6, p_value_threshold=0.05,
                      progress_cb=None):
        if self.correlation_matrix is None:
            raise ValueError("Calculate correlations first")

        if progress_cb:
            progress_cb("Building network graph...", 55)

        self.graph = nx.Graph()
        self._centrality_cache = {}
        taxa = list(self.correlation_matrix.index)

        # Taxonomy info (быстрая загрузка)
        tax_df = self.data_loader.taxonomy
        tax_info = {}
        for taxon in taxa:
            if taxon in tax_df.index:
                row = tax_df.loc[taxon]
                k = str(row.get('Kingdom', 'Unknown')) if isinstance(row, pd.Series) else 'Unknown'
                p = str(row.get('Phylum', 'Unknown')) if isinstance(row, pd.Series) else 'Unknown'
                k = k if k != 'nan' else 'Unknown'
                p = p if p != 'nan' else 'Unknown'
            else:
                k, p = 'Unknown', 'Unknown'
            tax_info[taxon] = {'kingdom': k, 'phylum': p}

        # Добавляем узлы
        otu_cols = set(self.data_loader.otu_table.columns)
        for taxon in taxa:
            ab = float(self.data_loader.otu_table[taxon].mean()) if taxon in otu_cols else 1.0
            self.graph.add_node(taxon, abundance=ab,
                               kingdom=tax_info[taxon]['kingdom'],
                               phylum=tax_info[taxon]['phylum'])

        # ВЕКТОРИЗОВАННОЕ добавление рёбер
        corr = self.correlation_matrix.values
        pvals = self.p_values.values
        n = len(taxa)

        mask = (np.triu(np.ones((n, n), dtype=bool), k=1)
                & (np.abs(corr) >= correlation_threshold)
                & (pvals <= p_value_threshold))

        rows, cols = np.where(mask)
        edges = []
        for k in range(len(rows)):
            i, j = int(rows[k]), int(cols[k])
            cv, pv = float(corr[i, j]), float(pvals[i, j])
            self.graph.add_edge(taxa[i], taxa[j],
                               weight=abs(cv), correlation=cv, p_value=pv)
            edges.append({'source': taxa[i], 'target': taxa[j],
                         'correlation': cv, 'p_value': pv})

        # Убираем изолированные узлы
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

        self.network_data = {'nodes': list(self.graph.nodes()),
                            'edges': edges, 'tax_info': tax_info}

        if progress_cb:
            progress_cb(f"Network: {len(self.graph.nodes())} nodes, {len(edges)} edges", 65)

        return len(self.graph.nodes()), len(edges)

    # ═══════════════════════════════════════════
    #  КЭШИРОВАННЫЕ ЦЕНТРАЛЬНОСТИ
    # ═══════════════════════════════════════════

    def _get_centrality(self, kind='degree'):
        if kind not in self._centrality_cache:
            if self.graph is None or len(self.graph.nodes()) == 0:
                return {}
            if kind == 'degree':
                self._centrality_cache[kind] = nx.degree_centrality(self.graph)
            elif kind == 'betweenness':
                self._centrality_cache[kind] = nx.betweenness_centrality(self.graph)
            elif kind == 'closeness':
                self._centrality_cache[kind] = nx.closeness_centrality(self.graph)
            elif kind == 'eigenvector':
                try:
                    self._centrality_cache[kind] = nx.eigenvector_centrality(
                        self.graph, max_iter=200, tol=1e-4)
                except Exception:
                    self._centrality_cache[kind] = {n: 0 for n in self.graph.nodes()}
        return self._centrality_cache.get(kind, {})

    # ═══════════════════════════════════════════
    #  ДЕТЕКЦИЯ СООБЩЕСТВ
    # ═══════════════════════════════════════════

    def detect_communities(self, method='louvain', resolution=1.0, progress_cb=None):
        if self.graph is None or len(self.graph.nodes()) == 0:
            return {}

        if progress_cb:
            progress_cb("Detecting communities...", 70)

        try:
            if method == 'louvain':
                self.communities = nx.community.louvain_communities(
                    self.graph, resolution=resolution, seed=42)
            elif method == 'label_propagation':
                self.communities = list(
                    nx.community.label_propagation_communities(self.graph))
            elif method == 'greedy_modularity':
                self.communities = list(
                    nx.community.greedy_modularity_communities(self.graph))
            else:
                self.communities = nx.community.louvain_communities(
                    self.graph, resolution=resolution, seed=42)
        except Exception:
            self.communities = list(nx.connected_components(self.graph))

        cmap = {}
        for i, comm in enumerate(self.communities):
            for node in comm:
                cmap[node] = i
        nx.set_node_attributes(self.graph, cmap, 'community')

        if progress_cb:
            progress_cb(f"{len(self.communities)} communities", 80)

        return {
            'n_communities': len(self.communities),
            'sizes': [len(c) for c in self.communities],
            'modularity': self._calc_modularity(),
            'community_map': cmap
        }

    def _calc_modularity(self):
        if self.communities is None:
            return 0
        try:
            return nx.community.modularity(self.graph, self.communities)
        except Exception:
            return 0

    # ═══════════════════════════════════════════
    #  KEYSTONE SPECIES
    # ═══════════════════════════════════════════

    def identify_keystone_species(self, top_n=10):
        if self.graph is None or len(self.graph.nodes()) == 0:
            return None

        dc = self._get_centrality('degree')
        bc = self._get_centrality('betweenness')
        cc = self._get_centrality('closeness')
        ec = self._get_centrality('eigenvector')

        rows = []
        for node in self.graph.nodes():
            deg = self.graph.degree(node)
            score = (0.3 * dc.get(node, 0) + 0.3 * bc.get(node, 0)
                     + 0.2 * cc.get(node, 0) + 0.2 * ec.get(node, 0))
            pos_e = sum(1 for _, _, d in self.graph.edges(node, data=True)
                        if d.get('correlation', 0) > 0)
            rows.append({
                'Taxon': node, 'Degree': deg,
                'Degree_Centrality': round(dc.get(node, 0), 4),
                'Betweenness': round(bc.get(node, 0), 4),
                'Closeness': round(cc.get(node, 0), 4),
                'Eigenvector': round(ec.get(node, 0), 4),
                'Abundance': round(self.graph.nodes[node].get('abundance', 0), 4),
                'Keystone_Score': round(score, 4),
                'Positive_Edges': pos_e,
                'Negative_Edges': deg - pos_e,
                'Community': self.graph.nodes[node].get('community', -1)
            })

        return pd.DataFrame(rows).sort_values('Keystone_Score', ascending=False)

    # ═══════════════════════════════════════════
    #  РОБАСТНОСТЬ — ОПТИМИЗИРОВАННАЯ
    # ═══════════════════════════════════════════

    def analyze_robustness(self, n_steps=12):
        if self.graph is None or len(self.graph.nodes()) < 5:
            return None

        n_nodes = len(self.graph.nodes())
        fracs = np.linspace(0, 0.85, n_steps)

        sorted_nodes = [n for n, _ in sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True)]

        t_res, r_res = [], []

        for frac in fracs:
            nr = int(frac * n_nodes)

            # Targeted
            G = self.graph.copy()
            G.remove_nodes_from(sorted_nodes[:nr])
            if len(G) > 0:
                lc = max(len(c) for c in nx.connected_components(G))
                t_res.append({'frac': frac, 'lc': lc / n_nodes})

            # Random (3x average)
            vals = []
            for _ in range(3):
                G2 = self.graph.copy()
                rm = np.random.choice(list(G2.nodes()),
                                      min(nr, len(G2) - 1), replace=False)
                G2.remove_nodes_from(rm)
                if len(G2) > 0:
                    vals.append(max(len(c) for c in nx.connected_components(G2)) / n_nodes)
                else:
                    vals.append(0)
            r_res.append({'frac': frac, 'lc': np.mean(vals), 'std': np.std(vals)})

        td, rd = pd.DataFrame(t_res), pd.DataFrame(r_res)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=td['frac'], y=td['lc'], mode='lines+markers',
            name='Targeted Attack', line=dict(color='#E74C3C', width=2),
            marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=rd['frac'], y=rd['lc'], mode='lines+markers',
            name='Random Failure', line=dict(color='#3498DB', width=2),
            marker=dict(size=5),
            error_y=dict(type='data', array=rd['std'].values, visible=True)))
        fig.update_layout(
            title=dict(text='Network Robustness', x=0.5, font=dict(size=15)),
            xaxis_title='Fraction Removed',
            yaxis_title='Largest Component (relative)',
            yaxis=dict(range=[0, 1.05]),
            height=420, template='plotly_white',
            legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5),
            margin=dict(t=50, b=80))
        return fig

    # ═══════════════════════════════════════════
    #  ВИЗУАЛИЗАЦИЯ — ПАКЕТНЫЕ РЁБРА
    # ═══════════════════════════════════════════

    def create_network_plot(self, layout='spring', node_size_factor=1.0,
                           color_by='kingdom', show_labels=True):
        if self.graph is None or len(self.graph.nodes()) == 0:
            return None

        # --- Layout ---
        nn = len(self.graph.nodes())
        if layout == 'spring':
            k = max(1.5, 3.0 / np.sqrt(max(nn, 1)))
            pos = nx.spring_layout(self.graph, k=k, iterations=30, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(self.graph)
            except Exception:
                pos = nx.spring_layout(self.graph, seed=42)
        elif layout == 'spectral':
            try:
                pos = nx.spectral_layout(self.graph)
            except Exception:
                pos = nx.spring_layout(self.graph, seed=42)
        elif layout == 'shell':
            if self.communities:
                pos = nx.shell_layout(self.graph,
                                      nlist=[list(c) for c in self.communities])
            else:
                pos = nx.shell_layout(self.graph)
        elif layout == 'community':
            pos = self._community_layout() if self.communities else nx.spring_layout(self.graph, seed=42)
        else:
            pos = nx.spring_layout(self.graph, seed=42)

        fig = go.Figure()

        # --- Рёбра: 2 trace вместо N (главная оптимизация) ---
        pos_ex, pos_ey, neg_ex, neg_ey = [], [], [], []
        for u, v, d in self.graph.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            if d.get('correlation', 0) >= 0:
                pos_ex.extend([x0, x1, None]); pos_ey.extend([y0, y1, None])
            else:
                neg_ex.extend([x0, x1, None]); neg_ey.extend([y0, y1, None])

        if pos_ex:
            fig.add_trace(go.Scatter(
                x=pos_ex, y=pos_ey, mode='lines',
                line=dict(width=0.8, color='rgba(39,174,96,0.35)'),
                hoverinfo='skip', showlegend=True, name='Positive'))
        if neg_ex:
            fig.add_trace(go.Scatter(
                x=neg_ex, y=neg_ey, mode='lines',
                line=dict(width=0.8, color='rgba(231,76,60,0.35)'),
                hoverinfo='skip', showlegend=True, name='Negative'))

        # --- Узлы ---
        nx_arr, ny_arr, texts, colors, sizes = [], [], [], [], []
        nodes_list = list(self.graph.nodes())

        kc = {'Bacteria': '#3498DB', 'Archaea': '#E74C3C', 'Fungi': '#F39C12',
              'Eukaryota': '#9B59B6', 'Unknown': '#95A5A6'}
        phylum_cmap = {}
        pc = px.colors.qualitative.D3 + px.colors.qualitative.Plotly
        comm_colors = px.colors.qualitative.D3 + px.colors.qualitative.Plotly

        cent_vals = None
        if color_by in ('degree', 'betweenness'):
            cent_vals = self._get_centrality(color_by)

        for node in nodes_list:
            x, y = pos[node]
            nx_arr.append(x); ny_arr.append(y)
            deg = self.graph.degree(node)
            sizes.append(max(6, 7 + deg * 2.5 * node_size_factor))
            nd = self.graph.nodes[node]

            if cent_vals is not None:
                cv = cent_vals.get(node, 0)
                colors.append(cv)
                texts.append(f"<b>{node}</b><br>{color_by}: {cv:.3f}<br>Deg: {deg}")
            elif color_by == 'community':
                cm = nd.get('community', 0)
                colors.append(comm_colors[cm % len(comm_colors)])
                texts.append(f"<b>{node}</b><br>C{cm}<br>Deg: {deg}")
            elif color_by == 'phylum':
                ph = nd.get('phylum', 'Unknown')
                if ph not in phylum_cmap:
                    phylum_cmap[ph] = pc[len(phylum_cmap) % len(pc)]
                colors.append(phylum_cmap[ph])
                texts.append(f"<b>{node}</b><br>{ph}<br>Deg: {deg}")
            else:
                colors.append(kc.get(nd.get('kingdom', 'Unknown'), '#95A5A6'))
                texts.append(f"<b>{node}</b><br>K:{nd.get('kingdom','?')}"
                            f"<br>P:{nd.get('phylum','?')}<br>Deg:{deg}")

        # Labels: только top-20 по degree
        labels = None
        if show_labels and nodes_list:
            deg_dict = dict(self.graph.degree())
            top20 = set(sorted(deg_dict, key=deg_dict.get, reverse=True)[:20])
            labels = [str(n)[:15] if n in top20 else '' for n in nodes_list]

        marker_kw = dict(size=sizes, line=dict(width=0.8, color='white'))
        if cent_vals is not None:
            marker_kw.update(color=colors, colorscale='Viridis',
                            colorbar=dict(title=color_by.title(), thickness=12))
        else:
            marker_kw['color'] = colors

        fig.add_trace(go.Scatter(
            x=nx_arr, y=ny_arr,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text', hovertext=texts,
            text=labels, textposition='top center', textfont=dict(size=8),
            marker=marker_kw, showlegend=False))

        n_pos = sum(1 for _, _, d in self.graph.edges(data=True) if d.get('correlation', 0) > 0)
        n_neg = len(self.graph.edges()) - n_pos

        fig.update_layout(
            title=dict(
                text=f'Co-occurrence Network<br>'
                     f'<span style="font-size:11px;color:#666">'
                     f'{nn} nodes, {len(self.graph.edges())} edges '
                     f'(+{n_pos}/-{n_neg}) | {layout}</span>',
                x=0.5),
            showlegend=True, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white', height=700,
            margin=dict(b=30, l=5, r=5, t=70),
            legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5))
        return fig

    def _community_layout(self):
        if not self.communities:
            return nx.spring_layout(self.graph, seed=42)
        pos = {}
        nc = len(self.communities)
        for i, comm in enumerate(self.communities):
            angle = 2 * np.pi * i / nc
            cx, cy = 3 * np.cos(angle), 3 * np.sin(angle)
            sg = self.graph.subgraph(comm)
            sp = nx.spring_layout(sg, seed=42, k=1.5, iterations=20)
            for node, (x, y) in sp.items():
                pos[node] = (x + cx, y + cy)
        return pos

    # ═══════════════════════════════════════════
    #  ВИЗУАЛИЗАЦИЯ СООБЩЕСТВ
    # ═══════════════════════════════════════════

    def create_community_summary_plot(self):
        if self.communities is None or self.graph is None:
            return None

        rows = []
        for i, comm in enumerate(self.communities):
            sg = self.graph.subgraph(comm)
            internal = len(sg.edges())
            external = sum(1 for n in comm for nb in self.graph.neighbors(n)
                          if nb not in comm)
            hub = max(comm, key=lambda n: self.graph.degree(n)) if comm else 'N/A'
            rows.append({'Community': i, 'Size': len(comm),
                        'Internal': internal, 'External': external, 'Hub': hub})

        df = pd.DataFrame(rows)
        clrs = px.colors.qualitative.D3[:len(df)]

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=['Sizes', 'Internal vs External'],
            horizontal_spacing=0.12)
        fig.add_trace(go.Bar(
            x=[f'C{i}' for i in df['Community']], y=df['Size'],
            marker_color=clrs, text=df['Size'], textposition='auto',
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[f'C{i}' for i in df['Community']], y=df['Internal'],
            name='Internal', marker_color='#3498DB'), row=1, col=2)
        fig.add_trace(go.Bar(
            x=[f'C{i}' for i in df['Community']], y=df['External'],
            name='External', marker_color='#E74C3C'), row=1, col=2)

        mod = self._calc_modularity()
        fig.update_layout(
            title=dict(text=f'Communities (Modularity: {mod:.3f})', x=0.5, font=dict(size=15)),
            height=380, template='plotly_white', barmode='group',
            margin=dict(t=60, b=40))
        return fig

    # ═══════════════════════════════════════════
    #  DEGREE DISTRIBUTION
    # ═══════════════════════════════════════════

    def create_degree_distribution_plot(self):
        if self.graph is None or len(self.graph.nodes()) == 0:
            return None

        degrees = [d for _, d in self.graph.degree()]
        if not degrees:
            return None
        counts = Counter(degrees)

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=['Degree Distribution', 'Log-Log (Scale-Free Test)'],
            horizontal_spacing=0.12)

        fig.add_trace(go.Histogram(
            x=degrees, nbinsx=max(5, max(degrees) // 2),
            marker_color='#3498DB', opacity=0.8, showlegend=False), row=1, col=1)

        k_v = sorted(counts.keys())
        p_v = [counts[k] / len(degrees) for k in k_v]
        kl = [np.log10(k) for k in k_v if k > 0]
        pl = [np.log10(p) for k, p in zip(k_v, p_v) if k > 0]

        fig.add_trace(go.Scatter(x=kl, y=pl, mode='markers',
            marker=dict(size=7, color='#E74C3C'), showlegend=False), row=1, col=2)

        if len(kl) > 2:
            sl, ic, rv, _, _ = stats.linregress(kl, pl)
            xf = np.linspace(min(kl), max(kl), 30)
            fig.add_trace(go.Scatter(
                x=xf.tolist(), y=(sl * xf + ic).tolist(),
                mode='lines', line=dict(color='gray', dash='dash'),
                showlegend=False), row=1, col=2)

        fig.update_xaxes(title_text='Degree', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_xaxes(title_text='log(k)', row=1, col=2)
        fig.update_yaxes(title_text='log(P(k))', row=1, col=2)
        fig.update_layout(
            title=dict(text=f'Degree Distribution (mean: {np.mean(degrees):.1f})',
                      x=0.5, font=dict(size=15)),
            height=380, template='plotly_white', margin=dict(t=60, b=40))
        return fig

    # ═══════════════════════════════════════════
    #  МЕТРИКИ
    # ═══════════════════════════════════════════

    def calculate_network_metrics(self):
        if self.graph is None:
            raise ValueError("Build network first")

        nn = len(self.graph.nodes())
        ne = len(self.graph.edges())

        metrics = {
            'nodes': nn, 'edges': ne,
            'density': nx.density(self.graph),
            'average_degree': np.mean([d for _, d in self.graph.degree()]) if nn else 0,
            'clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'positive_edges': sum(1 for _, _, d in self.graph.edges(data=True)
                                 if d.get('correlation', 0) > 0),
            'negative_edges': sum(1 for _, _, d in self.graph.edges(data=True)
                                 if d.get('correlation', 0) < 0),
        }

        # Diameter — только если < 300 узлов (иначе слишком долго)
        if nn < 300:
            try:
                if nx.is_connected(self.graph):
                    metrics['diameter'] = nx.diameter(self.graph)
                    metrics['average_path_length'] = nx.average_shortest_path_length(self.graph)
                else:
                    lcc = max(nx.connected_components(self.graph), key=len)
                    if len(lcc) > 2:
                        sg = self.graph.subgraph(lcc)
                        metrics['diameter'] = nx.diameter(sg)
                        metrics['average_path_length'] = nx.average_shortest_path_length(sg)
                        metrics['largest_component_size'] = len(lcc)
            except Exception:
                pass

        # Small-world
        try:
            metrics['transitivity'] = nx.transitivity(self.graph)
            k = metrics['average_degree']
            if k > 1 and 'average_path_length' in metrics:
                eC = k / nn
                eL = np.log(nn) / np.log(k)
                aL = metrics['average_path_length']
                if eC > 0 and eL > 0 and aL > 0:
                    sigma = (metrics['clustering_coefficient'] / eC) / (aL / eL)
                    metrics['small_world_sigma'] = round(sigma, 3)
                    metrics['is_small_world'] = sigma > 1
        except Exception:
            pass

        dc = self._get_centrality('degree')
        bc = self._get_centrality('betweenness')
        metrics['top_degree_nodes'] = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_betweenness_nodes'] = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]

        return metrics

    def export_network_data(self):
        if self.network_data is None:
            return None, None

        dc = self._get_centrality('degree')
        nodes = [{'Node': n, 'Kingdom': self.graph.nodes[n].get('kingdom', '?'),
                  'Phylum': self.graph.nodes[n].get('phylum', '?'),
                  'Abundance': round(self.graph.nodes[n].get('abundance', 0), 4),
                  'Degree': self.graph.degree(n),
                  'Centrality': round(dc.get(n, 0), 4),
                  'Community': self.graph.nodes[n].get('community', -1)}
                 for n in self.graph.nodes()]

        edges = [{'Source': u, 'Target': v,
                  'Correlation': round(d['correlation'], 4),
                  'Weight': round(d['weight'], 4),
                  'P_value': round(d['p_value'], 6)}
                 for u, v, d in self.graph.edges(data=True)]

        return pd.DataFrame(nodes), pd.DataFrame(edges)

    def create_correlation_heatmap(self, top_n=40):
        if self.correlation_matrix is None:
            return None
        mc = self.correlation_matrix.abs().mean().sort_values(ascending=False)
        top = mc.head(top_n).index
        sub = self.correlation_matrix.loc[top, top]

        fig = go.Figure(go.Heatmap(
            z=sub.values,
            x=[str(t)[:22] for t in sub.columns],
            y=[str(t)[:22] for t in sub.index],
            colorscale='RdBu_r', zmid=0, colorbar=dict(title='r'),
            hovertemplate='%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>'))
        fig.update_layout(
            title=dict(text=f'Correlation Matrix (Top {top_n})', x=0.5, font=dict(size=15)),
            height=max(450, top_n * 13 + 100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7)),
            margin=dict(l=130, b=130, r=40, t=50))
        return fig
