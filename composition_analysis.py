"""
Модуль анализа таксономического состава микробиома
Профессиональные фильтры таксонов, процентное отображение,
Sunburst, кладограмма, дифференциальный анализ,
многоуровневый обзор иерархии таксономии, Core Microbiome,
Rank-Abundance, Indicator Species, Sankey, Treemap
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class CompositionAnalyzer:
    """Анализатор таксономического состава с профессиональными фильтрами"""

    def __init__(self, data_loader, settings):
        self.data_loader = data_loader
        self.settings = settings
        self.taxonomy_results = {}
        self.all_taxa_info = pd.DataFrame()
        self._create_color_palettes()

    def _create_color_palettes(self):
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            unique_groups = sorted(self.data_loader.metadata[gc].unique())
            n = len(unique_groups)
            colors = sns.color_palette("Set2", n) if n <= 8 else sns.color_palette("husl", n)
            self.group_colors = {
                g: '#{:02x}{:02x}{:02x}'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
                for g, c in zip(unique_groups, colors)
            }
        else:
            self.group_colors = {}

    def process_data(self):
        """Обработка таксономических данных"""
        tax_level = self.settings['taxonomic_level']
        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu = self.data_loader.get_filtered_otu_table(auto_filter)
        sample_cols = list(otu.index)
        otu_t = otu.T

        if tax_level in self.data_loader.taxonomy.columns:
            non_null = self.data_loader.taxonomy[tax_level].notna().sum()
            if non_null > 10:
                merged = otu_t.merge(self.data_loader.taxonomy[[tax_level]],
                                    left_index=True, right_index=True)
                merged[tax_level] = merged[tax_level].fillna('Unclassified')
                level_abundance = merged.groupby(tax_level)[sample_cols].sum()
            else:
                available = self.data_loader.get_taxonomic_levels()
                fb = available[0] if available else 'Kingdom'
                merged = otu_t.merge(self.data_loader.taxonomy[[fb]],
                                    left_index=True, right_index=True)
                merged[tax_level] = merged[fb].fillna('Unclassified')
                level_abundance = merged.groupby(tax_level)[sample_cols].sum()
        else:
            available = self.data_loader.get_taxonomic_levels()
            fb = available[0] if available else 'Kingdom'
            merged = otu_t.merge(self.data_loader.taxonomy[[fb]],
                                left_index=True, right_index=True)
            merged[tax_level] = merged[fb].fillna('Unclassified')
            level_abundance = merged.groupby(tax_level)[sample_cols].sum()

        totals = level_abundance.sum(axis=0)
        rel = level_abundance.div(totals, axis=1) * 100
        rel = rel.fillna(0)
        non_zero = rel.sum(axis=1) > 0
        rel = rel[non_zero]
        level_abundance = level_abundance[non_zero]

        self.taxonomy_results[tax_level] = {
            'relative_abundance': rel,
            'absolute_abundance': level_abundance
        }
        self._build_taxa_info(rel, tax_level)

    def _build_taxa_info(self, rel, tax_level):
        info = pd.DataFrame({
            'Taxon': rel.index,
            'Mean Abundance (%)': rel.mean(axis=1).values,
            'Max Abundance (%)': rel.max(axis=1).values,
            'Min Abundance (%)': rel.min(axis=1).values,
            'SD': rel.std(axis=1).values,
            'Prevalence (%)': ((rel > 0).sum(axis=1) / rel.shape[1] * 100).values,
            'Total Abundance (%)': rel.sum(axis=1).values
        })
        info = info.sort_values('Mean Abundance (%)', ascending=False).reset_index(drop=True)
        info['Rank'] = range(1, len(info) + 1)
        self.all_taxa_info = info

    def get_all_taxa_list(self):
        return self.all_taxa_info.copy()

    def get_taxa_by_filter(self, min_abundance=0.0, max_abundance=100.0,
                           min_prevalence=0.0, exclude_taxa=None, include_only=None):
        if self.all_taxa_info.empty:
            return []
        filtered = self.all_taxa_info.copy()
        if include_only:
            filtered = filtered[filtered['Taxon'].isin(include_only)]
        else:
            filtered = filtered[
                (filtered['Mean Abundance (%)'] >= min_abundance) &
                (filtered['Mean Abundance (%)'] <= max_abundance) &
                (filtered['Prevalence (%)'] >= min_prevalence)
            ]
            if exclude_taxa:
                filtered = filtered[~filtered['Taxon'].isin(exclude_taxa)]
        return filtered['Taxon'].tolist()

    # ═══════════════════════════════════════════
    #  STACKED BAR (ОРИГИНАЛЬНЫЙ)
    # ═══════════════════════════════════════════

    def create_plot(self, top_n=15, exclude_taxa=None, include_only=None,
                    min_abundance=0.0, min_prevalence=0.0, show_individual_samples=False,
                    font_scale=1.0):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        if rel.empty:
            return None

        if include_only:
            taxa = [t for t in include_only if t in rel.index]
        else:
            mean_ab = rel.mean(axis=1).sort_values(ascending=False)
            taxa = mean_ab.index.tolist()
            if min_abundance > 0:
                taxa = [t for t in taxa if mean_ab[t] >= min_abundance]
            if min_prevalence > 0:
                prev = (rel > 0).sum(axis=1) / rel.shape[1] * 100
                taxa = [t for t in taxa if prev.get(t, 0) >= min_prevalence]
            if exclude_taxa:
                taxa = [t for t in taxa if t not in exclude_taxa]
            taxa = taxa[:top_n]

        if not taxa:
            return None

        plot_data = rel.loc[taxa].T
        gc = self.settings.get('group_column')
        sample_info = []
        group_positions = {}
        pos = 0

        if gc and gc in self.data_loader.metadata.columns:
            grouped = {}
            for s in plot_data.index:
                if s in self.data_loader.metadata.index:
                    g = self.data_loader.metadata.loc[s, gc]
                    grouped.setdefault(g, []).append(s)
            for i, (g, samples) in enumerate(sorted(grouped.items())):
                group_positions[g] = {'start': pos, 'samples': samples}
                for j, s in enumerate(samples):
                    sample_info.append({'sample': s, 'group': g, 'position': pos + j})
                pos += len(samples) + 1
        else:
            for i, s in enumerate(plot_data.index):
                sample_info.append({'sample': s, 'group': 'All', 'position': i})

        sample_df = pd.DataFrame(sample_info)
        if sample_df.empty:
            return None

    # ═══════════════════════════════════════════
    #  STACKED BAR — Individual Samples
    # ═══════════════════════════════════════════

    def create_plot(self, top_n=15, exclude_taxa=None, include_only=None,
                    min_abundance=0.0, min_prevalence=0.0, show_individual_samples=False,
                    font_scale=1.0):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        if rel.empty:
            return None

        if include_only:
            taxa = [t for t in include_only if t in rel.index]
        else:
            mean_ab = rel.mean(axis=1).sort_values(ascending=False)
            taxa = mean_ab.index.tolist()
            if min_abundance > 0:
                taxa = [t for t in taxa if mean_ab[t] >= min_abundance]
            if min_prevalence > 0:
                prev = (rel > 0).sum(axis=1) / rel.shape[1] * 100
                taxa = [t for t in taxa if prev.get(t, 0) >= min_prevalence]
            if exclude_taxa:
                taxa = [t for t in taxa if t not in exclude_taxa]
            taxa = taxa[:top_n]

        if not taxa:
            return None

        plot_data = rel.loc[taxa].T
        gc = self.settings.get('group_column')
        sample_info = []
        group_positions = {}
        pos = 0

        if gc and gc in self.data_loader.metadata.columns:
            grouped = {}
            for s in plot_data.index:
                if s in self.data_loader.metadata.index:
                    g = self.data_loader.metadata.loc[s, gc]
                    grouped.setdefault(g, []).append(s)
            for i, (g, samples) in enumerate(sorted(grouped.items())):
                group_positions[g] = {'start': pos, 'samples': samples}
                for j, s in enumerate(samples):
                    sample_info.append({'sample': s, 'group': g, 'position': pos + j})
                pos += len(samples) + 1
        else:
            for i, s in enumerate(plot_data.index):
                sample_info.append({'sample': s, 'group': 'All', 'position': i})

        sample_df = pd.DataFrame(sample_info)
        if sample_df.empty:
            return None

        n_taxa = len(taxa)
        colors = self._get_taxa_colors(n_taxa)

        fig = go.Figure()
        for idx, taxon in enumerate(taxa):
            values = []
            for _, row in sample_df.iterrows():
                if row['sample'] in plot_data.index and taxon in plot_data.columns:
                    values.append(plot_data.loc[row['sample'], taxon])
                else:
                    values.append(0)
            display_name = f'<i>{taxon}</i>' if tax_level in ['Genus', 'Species'] else str(taxon)
            fig.add_trace(go.Bar(
                x=sample_df['position'].tolist(), y=values,
                name=display_name, marker_color=colors[idx % len(colors)],
                hovertemplate=(f'<b>{taxon}</b><br>Sample: %{{customdata}}<br>'
                              f'Abundance: %{{y:.2f}}%<extra></extra>'),
                customdata=sample_df['sample'].tolist()
            ))

        if gc and group_positions:
            x_pos = [info['start'] + (len(info['samples']) - 1) / 2
                    for g, info in sorted(group_positions.items())]
            x_labels = [str(g) for g in sorted(group_positions.keys())]
        else:
            x_pos = list(range(len(sample_df)))
            x_labels = sample_df['sample'].tolist()

        shapes = []
        if gc and len(group_positions) > 1:
            sorted_g = sorted(group_positions.items())
            for i in range(len(sorted_g) - 1):
                line_pos = sorted_g[i][1]['start'] + len(sorted_g[i][1]['samples']) - 0.5
                shapes.append(dict(type="line", x0=line_pos, x1=line_pos,
                                  y0=0, y1=100, line=dict(color="gray", width=1.5, dash="dash")))

        # Dynamic font sizes based on taxa count + font_scale
        legend_size = max(10, int((14 - n_taxa * 0.01) * font_scale))
        title_size = int(18 * font_scale)
        axis_size = int(14 * font_scale)
        tick_size = int(12 * font_scale)
        max_name_len = max(len(str(t)) for t in taxa)
        r_margin = max(220, min(500, max_name_len * 9 + 80))

        title = f'Taxonomic Composition: {tax_level} (Top {len(taxa)})'
        if gc:
            title += f' — grouped by {gc}'

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=title_size, family='Arial')),
            xaxis=dict(
                title=dict(text=gc.title() if gc else 'Samples', font=dict(size=axis_size)),
                tickvals=x_pos, ticktext=x_labels,
                tickangle=-30 if not gc else 0,
                tickfont=dict(size=max(12, tick_size), family='Arial', color='#111')
            ),
            yaxis=dict(
                title=dict(text='Relative Abundance (%)', font=dict(size=axis_size)),
                range=[0, 100],
                tickfont=dict(size=tick_size)
            ),
            barmode='stack', height=720,
            legend=dict(
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                font=dict(size=legend_size, family='Arial'),
                tracegroupgap=1, itemwidth=30,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ddd', borderwidth=1
            ),
            shapes=shapes,
            margin=dict(l=80, r=r_margin, t=100, b=120),
            plot_bgcolor='white', paper_bgcolor='white'
        )
        return fig

    # ═══════════════════════════════════════════
    #  STACKED BAR — Group Mean (Publication)
    # ═══════════════════════════════════════════

    def create_grouped_mean_plot(self, top_n=50, exclude_taxa=None,
                                  min_abundance=0.0, min_prevalence=0.0,
                                  font_scale=1.0, show_error_bars=False,
                                  bar_width=0.7):
        """
        Publication-quality stacked bar: ONE wide bar per group (mean ± SD).
        Ideal for many taxa (50-300) — much more readable than per-sample bars.

        Parameters
        ----------
        top_n : int — number of top taxa to display
        show_error_bars : bool — overlay SD whiskers on each segment
        bar_width : float — width of bars (0.3-0.95)
        font_scale : float — scale all fonts
        """
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        if rel.empty:
            return None

        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            # Fallback: single "All" bar
            gc = None

        # ── Filter taxa ──
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        taxa = mean_ab.index.tolist()
        if min_abundance > 0:
            taxa = [t for t in taxa if mean_ab[t] >= min_abundance]
        if min_prevalence > 0:
            prev = (rel > 0).sum(axis=1) / rel.shape[1] * 100
            taxa = [t for t in taxa if prev.get(t, 0) >= min_prevalence]
        if exclude_taxa:
            taxa = [t for t in taxa if t not in exclude_taxa]
        taxa = taxa[:top_n]

        if not taxa:
            return None

        n_taxa = len(taxa)
        colors = self._get_taxa_colors(n_taxa)

        # ── Compute group means ──
        if gc:
            groups = sorted(self.data_loader.metadata[gc].unique())
        else:
            groups = ['All']

        group_means = {}
        group_sds = {}
        for g in groups:
            if gc:
                g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
                common = list(set(g_samples) & set(rel.columns))
            else:
                common = list(rel.columns)
            if common:
                group_means[g] = rel.loc[taxa, common].mean(axis=1)
                group_sds[g] = rel.loc[taxa, common].std(axis=1)
            else:
                group_means[g] = pd.Series(0, index=taxa)
                group_sds[g] = pd.Series(0, index=taxa)

        x_labels = [str(g) for g in groups]

        # Dynamic font sizes
        legend_size = max(11, int((15 - n_taxa * 0.005) * font_scale))
        title_size = int(20 * font_scale)
        axis_size = int(15 * font_scale)
        tick_size = int(14 * font_scale)
        max_name_len = max(len(str(t)) for t in taxa)
        r_margin = max(240, min(520, max_name_len * 9 + 80))

        # ── Build figure ──
        fig = go.Figure()

        for idx, taxon in enumerate(taxa):
            y_vals = [group_means[g][taxon] if taxon in group_means[g] else 0 for g in groups]
            sd_vals = [group_sds[g][taxon] if taxon in group_sds[g] else 0 for g in groups] if show_error_bars else None
            display_name = f'<i>{taxon}</i>' if tax_level in ['Genus', 'Species'] else str(taxon)

            trace_kwargs = dict(
                x=x_labels, y=y_vals,
                name=display_name,
                marker=dict(color=colors[idx % len(colors)],
                            line=dict(color='rgba(255,255,255,0.25)', width=0.3)),
                width=bar_width,
                hovertemplate=(f'<b>{taxon}</b><br>Group: %{{x}}<br>'
                               f'Mean: %{{y:.3f}}%<extra></extra>')
            )
            if show_error_bars and sd_vals:
                trace_kwargs['error_y'] = dict(
                    type='data', array=sd_vals, visible=True,
                    thickness=1.5, width=4, color='rgba(0,0,0,0.5)'
                )
            fig.add_trace(go.Bar(**trace_kwargs))

        # ── Group-size annotation below x-axis ──
        annotations = []
        for g in groups:
            if gc:
                n = (self.data_loader.metadata[gc] == g).sum()
                annotations.append(dict(
                    x=str(g), y=-8, text=f'n={n}',
                    xref='x', yref='y',
                    showarrow=False,
                    font=dict(size=max(11, int(12 * font_scale)), color='#555'),
                    xanchor='center'
                ))

        title = f'Taxonomic Composition: {tax_level} — Group Mean (Top {n_taxa})'
        if gc:
            title += f'<br><span style="font-size:{int(13*font_scale)}px;color:#555">Grouped by: {gc}</span>'

        fig.update_layout(
            title=dict(text=title, x=0.5,
                       font=dict(size=title_size, family='Arial', color='#111')),
            xaxis=dict(
                title=dict(text=gc.title() if gc else 'Group',
                           font=dict(size=axis_size, family='Arial')),
                tickfont=dict(size=tick_size, family='Arial',
                              color='#111'),
                tickangle=0
            ),
            yaxis=dict(
                title=dict(text='Mean Relative Abundance (%)',
                           font=dict(size=axis_size, family='Arial')),
                range=[0, 103],
                tickfont=dict(size=tick_size - 1)
            ),
            barmode='stack',
            height=700,
            legend=dict(
                orientation='v', yanchor='top', y=1, xanchor='left', x=1.01,
                font=dict(size=legend_size, family='Arial'),
                tracegroupgap=0, itemwidth=30,
                bgcolor='rgba(255,255,255,0.97)',
                bordercolor='#ccc', borderwidth=1
            ),
            annotations=annotations,
            margin=dict(l=80, r=r_margin, t=110, b=100),
            plot_bgcolor='white', paper_bgcolor='white',
            bargap=0.35
        )
        return fig

    def _get_taxa_colors(self, n_taxa):
        """
        Return a list of n_taxa visually distinct colors.
        Uses a hand-curated high-contrast palette for large n.
        """
        # Base palette — 26 Alphabet + D3 + Plotly combined
        base = (list(px.colors.qualitative.Alphabet) +
                list(px.colors.qualitative.D3) +
                list(px.colors.qualitative.Plotly) +
                list(px.colors.qualitative.Set1) +
                list(px.colors.qualitative.Pastel))
        # Deduplicate while preserving order
        seen_c = set()
        palette = []
        for c in base:
            if c not in seen_c:
                palette.append(c)
                seen_c.add(c)

        if n_taxa <= len(palette):
            return palette[:n_taxa]

        # For very large n: cycle with lightness variation
        import colorsys
        colors = list(palette)
        while len(colors) < n_taxa:
            for c in palette:
                c_hex = c.lstrip('#')
                try:
                    r, g, b = int(c_hex[0:2], 16)/255, int(c_hex[2:4], 16)/255, int(c_hex[4:6], 16)/255
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    v2 = max(0.3, v - 0.18)
                    r2, g2, b2 = colorsys.hsv_to_rgb(h, min(1, s + 0.1), v2)
                    colors.append('#{:02x}{:02x}{:02x}'.format(int(r2*255), int(g2*255), int(b2*255)))
                except Exception:
                    colors.append(c)
                if len(colors) >= n_taxa:
                    break
        return colors[:n_taxa]



    def create_percentage_table(self, top_n=50, exclude_taxa=None):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return pd.DataFrame()
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        taxa = mean_ab.head(top_n).index.tolist()
        if exclude_taxa:
            taxa = [t for t in taxa if t not in exclude_taxa]
        if gc and gc in self.data_loader.metadata.columns:
            groups = sorted(self.data_loader.metadata[gc].unique())
            rows = []
            for taxon in taxa:
                row = {'Taxon': taxon}
                for g in groups:
                    g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
                    common = list(set(g_samples) & set(rel.columns))
                    if common:
                        vals = rel.loc[taxon, common]
                        row[f'{g} Mean%'] = round(vals.mean(), 3)
                        row[f'{g} SD'] = round(vals.std(), 3)
                    else:
                        row[f'{g} Mean%'] = 0
                        row[f'{g} SD'] = 0
                row['Overall Mean%'] = round(mean_ab[taxon], 3)
                row['Prevalence%'] = round((rel.loc[taxon] > 0).sum() / len(rel.columns) * 100, 1)
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            rows = []
            for taxon in taxa:
                rows.append({
                    'Taxon': taxon, 'Mean%': round(mean_ab[taxon], 3),
                    'SD': round(rel.loc[taxon].std(), 3),
                    'Max%': round(rel.loc[taxon].max(), 3),
                    'Prevalence%': round((rel.loc[taxon] > 0).sum() / len(rel.columns) * 100, 1)
                })
            return pd.DataFrame(rows)

    def create_full_taxonomy_table(self):
        """Full taxonomy table (QIIME2-style): Feature ID + all levels + lineage + abundance stats."""
        tax = self.data_loader.taxonomy
        otu = self.data_loader.get_filtered_otu_table(self.settings.get('auto_filter_zeros', True))
        if tax is None or tax.empty or otu is None or otu.empty:
            return pd.DataFrame()

        standard_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        prefix_map = {'Kingdom': 'k__', 'Phylum': 'p__', 'Class': 'c__',
                      'Order': 'o__', 'Family': 'f__', 'Genus': 'g__', 'Species': 's__'}
        available_levels = [l for l in standard_levels if l in tax.columns and tax[l].notna().sum() > 0]
        if not available_levels:
            return pd.DataFrame()

        tax_sub = tax[available_levels].copy()

        def make_lineage(row):
            parts = []
            for lvl in available_levels:
                val = row[lvl]
                p = prefix_map.get(lvl, '')
                if pd.notna(val) and str(val).strip() not in ('', 'nan', 'Unclassified', 'unclassified'):
                    parts.append(f"{p}{val}")
                else:
                    parts.append(f"{p}__")
            return '; '.join(parts)

        tax_sub['Full Lineage'] = tax_sub.apply(make_lineage, axis=1)

        # OTU table: rows=samples, cols=OTUs → transpose to rows=OTUs
        otu_t = otu.T
        common = otu_t.index.intersection(tax_sub.index)
        if len(common) == 0:
            return pd.DataFrame()
        otu_sub = otu_t.loc[common]

        # Per-sample totals for relative abundance
        sample_totals = otu.sum(axis=1)
        rel = otu_sub.div(sample_totals, axis=1) * 100
        rel = rel.fillna(0)

        result = tax_sub.loc[common].copy()
        result.index.name = 'Feature ID'
        result['Mean Abundance (%)'] = rel.mean(axis=1).round(4)
        result['Max Abundance (%)']  = rel.max(axis=1).round(4)
        result['Prevalence (%)']     = ((rel > 0).sum(axis=1) / rel.shape[1] * 100).round(1)
        result['Total Reads']        = otu_sub.sum(axis=1).astype(int)

        result = result.sort_values('Mean Abundance (%)', ascending=False).reset_index()
        cols = (['Feature ID'] + available_levels +
                ['Full Lineage', 'Mean Abundance (%)', 'Max Abundance (%)',
                 'Prevalence (%)', 'Total Reads'])
        return result[[c for c in cols if c in result.columns]]

    def _compute_level_rel(self, level, otu, otu_t, sample_cols):
        """Compute relative abundance DataFrame for one taxonomic level."""
        tax = self.data_loader.taxonomy
        if level not in tax.columns or tax[level].notna().sum() == 0:
            return None
        merged = otu_t.merge(tax[[level]], left_index=True, right_index=True)
        merged[level] = merged[level].fillna('Unclassified')
        level_ab = merged.groupby(level)[sample_cols].sum()
        totals = level_ab.sum(axis=0)
        rel = level_ab.div(totals, axis=1) * 100
        rel = rel.fillna(0)
        rel = rel[rel.sum(axis=1) > 0]
        return rel

    def generate_taxonomy_html_report(self, min_abundance=0.0):
        """
        Self-contained HTML taxa summary.
        One Plotly chart per taxonomic level, rendered by plotly.io.to_html().
        Tab buttons show/hide level divs — no custom Plotly JS API calls.
        Works fully offline (plotly.min.js embedded inline).
        """
        import plotly.io as pio
        import plotly.graph_objects as go
        import json, datetime, os

        # ── Inline Plotly.js (self-contained, offline) ──────────────────────
        try:
            import plotly as _px
            _js_path = os.path.join(os.path.dirname(_px.__file__),
                                    'package_data', 'plotly.min.js')
            with open(_js_path, 'r', encoding='utf-8') as _f:
                _plotly_tag = '<script>' + _f.read() + '</script>'
        except Exception:
            _plotly_tag = ('<script src="https://cdn.plot.ly/'
                           'plotly-2.27.0.min.js"></script>')

        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu    = self.data_loader.get_filtered_otu_table(auto_filter)
        gc     = self.settings.get('group_column')
        levels = self.data_loader.get_taxonomic_levels()

        if otu is None or otu.empty or not levels:
            return '<p>No data available.</p>'

        sample_cols = list(otu.index)
        otu_t       = otu.T

        level_rel = {}
        for level in levels:
            rel = self._compute_level_rel(level, otu, otu_t, sample_cols)
            if rel is not None and not rel.empty:
                level_rel[level] = rel

        if not level_rel:
            return '<p>Could not compute relative abundance.</p>'

        # ── Build one Plotly figure per level ────────────────────────────────
        level_ru = {
            'Kingdom': 'царства', 'Phylum': 'типа', 'Class': 'класса',
            'Order': 'порядка', 'Family': 'семейства',
            'Genus': 'рода', 'Species': 'вида',
        }
        n_samples  = len(sample_cols)
        n_features = len(otu.columns)
        now        = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        group_info = self.settings.get('group_column') or 'none'
        min_ab_str = (f'>= {min_abundance:.2f}%' if min_abundance > 0
                      else 'all taxa included')

        ldata      = {}   # compact JSON for TSV download
        tabs_html  = ''
        sections   = []

        for i, (level, rel) in enumerate(level_rel.items()):
            mean_ab = rel.mean(axis=1).sort_values(ascending=False)
            taxa = (mean_ab[mean_ab >= min_abundance].index.tolist()
                    if min_abundance > 0 else mean_ab.index.tolist())
            samples = [s for s in sample_cols if s in rel.columns]
            colors  = self._get_taxa_colors(len(taxa))
            italic  = level in ('Genus', 'Species')

            # Compact data for TSV
            ldata[level] = {
                'taxa': taxa, 'samples': samples,
                'matrix': {t: rel.loc[t, samples].round(6).tolist() for t in taxa}
            }

            # Plotly figure — one trace per taxon, reversed so max is at bottom
            fig = go.Figure()
            for j, taxon in enumerate(reversed(taxa)):
                color = colors[len(taxa) - 1 - j]
                name  = f'<i>{taxon}</i>' if italic else taxon
                fig.add_trace(go.Bar(
                    x=samples,
                    y=rel.loc[taxon, samples].round(4).tolist(),
                    name=name,
                    marker_color=color,
                    marker_line_width=0,
                    hovertemplate=(
                        '<b>' + taxon + '</b><br>'
                        '\u041e\u0431\u0440\u0430\u0437\u0435\u0446: %{x}<br>'
                        '\u041e\u0442\u043d. \u0447\u0438\u0441\u043b\u0435\u043d\u043d\u043e\u0441\u0442\u044c: %{y:.3f}%'
                        '<extra></extra>'
                    ),
                ))

            ht  = max(500, min(820, len(samples) * 18 + 180))
            tsz = max(7, 11 - len(samples) // 20)
            fig.update_layout(
                barmode='stack',
                xaxis=dict(
                    title='\u041e\u0431\u0440\u0430\u0437\u0435\u0446',
                    tickangle=-55, tickfont=dict(size=tsz, family='Arial'),
                    showgrid=False, linecolor='#555', linewidth=1,
                    ticks='outside', ticklen=3,
                ),
                yaxis=dict(
                    title=('\u041e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d\u0430\u044f '
                           '\u0447\u0438\u0441\u043b\u0435\u043d\u043d\u043e\u0441\u0442\u044c (%)'),
                    range=[0, 101], gridcolor='#e8e8e8', gridwidth=0.5,
                    zeroline=True, zerolinecolor='#555', zerolinewidth=1,
                    linecolor='#555', linewidth=1, ticks='outside', ticklen=3,
                ),
                legend=dict(
                    title=dict(text=level, font=dict(size=10, color='#222')),
                    font=dict(size=9, family='Arial'),
                    bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    x=1.01, y=1, xanchor='left', yanchor='top',
                ),
                height=ht, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=70, r=240, t=16, b=120),
                bargap=0.06,
                font=dict(family='Arial, sans-serif', size=10, color='#222'),
            )

            chart_div = pio.to_html(
                fig, full_html=False, include_plotlyjs=False,
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'svg', 'filename': f'taxa_{level}',
                        'width': 1400, 'height': ht, 'scale': 2,
                    },
                    'responsive': True,
                },
            )

            lvru    = level_ru.get(level, level)
            caption = (f'\u0420\u0438\u0441. \u041e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d\u0430\u044f '
                       f'\u0447\u0438\u0441\u043b\u0435\u043d\u043d\u043e\u0441\u0442\u044c \u0432\u0441\u0435\u0445 '
                       f'{len(taxa)} \u0442\u0430\u043a\u0441\u043e\u043d\u043e\u0432 '
                       f'\u043d\u0430 \u0443\u0440\u043e\u0432\u043d\u0435 {lvru} '
                       f'\u0432 {len(samples)} \u043e\u0431\u0440\u0430\u0437\u0446\u0430\u0445.')

            display = 'block' if i == 0 else 'none'
            sections.append(
                f'<div class="level-section" id="section-{level}" style="display:{display}">'
                f'{chart_div}'
                f'<p class="fig-caption">{caption}</p>'
                f'<div class="dt-toolbar">'
                f'  <span class="dt-info">{len(taxa)} taxa &times; {len(samples)} samples</span>'
                f'  <div class="dt-actions">'
                f'    <span class="toggle-link" onclick="toggleTbl(\'{level}\')">Show / hide data table</span>'
                f'    <button class="dl-btn" onclick="dlTSV(\'{level}\')">Download TSV</button>'
                f'  </div>'
                f'</div>'
                f'<div class="dt-outer" id="dto-{level}"><div id="dtc-{level}"></div></div>'
                f'</div>'
            )

            active = ' active' if i == 0 else ''
            tabs_html += (f'<button class="level-tab{active}" id="tab-{level}" '
                          f'onclick="showLevel(\'{level}\')">'
                          f'{level}<span class="tab-cnt"> ({len(taxa)})</span></button>')

        sections_html = '\n'.join(sections)
        ldata_json    = json.dumps(ldata, ensure_ascii=False)

        # ── Feature table (capped at 500 rows) ──────────────────────────────
        full_tax  = self.create_full_taxonomy_table()
        _MAX_ROWS = 500
        if not full_tax.empty:
            _tot  = len(full_tax)
            _note = (f'<p class="tbl-note">Top {_MAX_ROWS} of {_tot:,} features '
                     f'(by mean abundance).</p>' if _tot > _MAX_ROWS else '')
            ftable_html = _note + full_tax.head(_MAX_ROWS).to_html(
                index=False, classes='tax-table', border=0,
                float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
        else:
            ftable_html = '<p>Feature table not available.</p>'

        meta_rows = (
            f'<tr><td>Samples</td><td>{n_samples}</td></tr>'
            f'<tr><td>Features (OTUs/ASVs)</td><td>{n_features:,}</td></tr>'
            f'<tr><td>Taxonomic levels</td><td>{", ".join(level_rel.keys())}</td></tr>'
            f'<tr><td>Grouping variable</td><td>{group_info}</td></tr>'
            f'<tr><td>Abundance filter</td><td>{min_ab_str}</td></tr>'
            f'<tr><td>Report generated</td><td>{now}</td></tr>'
            f'<tr><td>Developer</td><td>Ablaev Aziz Yakubovich</td></tr>'
        )

        # ── Assemble HTML ────────────────────────────────────────────────────
        html = (
"<!DOCTYPE html>\n<html lang='ru'>\n<head>\n"
"<meta charset='UTF-8'>\n"
"<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
"<title>Taxa Summary \u2014 BIOMIDAS</title>\n"
+ _plotly_tag + "\n"
+ """<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,Helvetica,sans-serif;background:#fff;color:#111;font-size:13px;line-height:1.5}
.page-header{border-bottom:2px solid #111;padding:16px 40px 10px}
.page-header h1{font-size:15px;font-weight:bold}
.page-header .sub{font-size:11px;color:#666;margin-top:3px}
.level-nav{background:#f5f5f5;border-bottom:1px solid #ccc;padding:0 40px;display:flex;flex-wrap:wrap}
.level-tab{padding:9px 20px;border:none;border-bottom:3px solid transparent;background:none;
  cursor:pointer;font-size:12px;font-family:Arial,sans-serif;color:#555;font-weight:500}
.level-tab:hover{color:#111;border-bottom-color:#aaa}
.level-tab.active{color:#111;border-bottom-color:#111;font-weight:bold}
.tab-cnt{font-weight:normal;color:#999;font-size:11px}
.container{max-width:1700px;margin:0 auto;padding:20px 40px 40px}
.section{margin-bottom:28px}
.section-title{font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:.5px;
  color:#111;border-bottom:1px solid #ccc;padding-bottom:4px;margin-bottom:12px}
.meta-table{border-collapse:collapse;font-size:12px}
.meta-table td{padding:3px 16px 3px 0;vertical-align:top}
.meta-table td:first-child{color:#666;white-space:nowrap;min-width:190px}
.meta-table td:last-child{font-weight:500}
.fig-caption{font-size:11px;color:#555;font-style:italic;margin-top:6px}
.dt-toolbar{display:flex;justify-content:space-between;align-items:center;
  margin-top:12px;border-top:1px solid #ccc;padding-top:8px}
.dt-info{font-size:11px;color:#555}
.dt-actions{display:flex;gap:10px;align-items:center}
.toggle-link{font-size:11px;color:#444;cursor:pointer;text-decoration:underline}
.dl-btn{padding:4px 14px;background:#333;color:#fff;border:none;border-radius:2px;
  font-size:11px;cursor:pointer;font-family:Arial,sans-serif}
.dl-btn:hover{background:#111}
.dt-outer{display:none;overflow:auto;max-height:400px;border:1px solid #ccc;margin-top:6px}
.raw-table{border-collapse:collapse;font-size:11px;white-space:nowrap}
.raw-table thead th{position:sticky;top:0;background:#222;color:#fff;
  padding:6px 10px;text-align:right;font-size:10px;font-weight:bold;z-index:5}
.raw-table thead th:first-child{text-align:left;min-width:200px;position:sticky;left:0;z-index:10}
.raw-table tbody td{padding:4px 10px;border-bottom:1px solid #eee;text-align:right}
.raw-table tbody td:first-child{text-align:left;background:#fff;position:sticky;left:0}
.raw-table tbody tr:nth-child(even) td{background:#f8f8f8}
.raw-table tbody tr:nth-child(even) td:first-child{background:#f8f8f8}
.raw-table tbody tr:hover td{background:#fffde7!important}
.search-row{display:flex;gap:10px;margin-bottom:10px;align-items:center}
.search-box{width:360px;padding:5px 10px;border:1px solid #aaa;border-radius:2px;
  font-size:12px;font-family:Arial,sans-serif;outline:none}
.search-box:focus{border-color:#333}
.row-count{font-size:11px;color:#888}
.tbl-note{font-size:11px;color:#666;margin-bottom:6px}
.ftbl-wrapper{overflow:auto;max-height:500px;border:1px solid #ccc}
.tax-table{width:100%;border-collapse:collapse;font-size:11.5px;min-width:800px}
.tax-table thead th{position:sticky;top:0;z-index:5;background:#222;color:#fff;
  padding:7px 12px;text-align:left;font-size:11px;font-weight:bold;cursor:pointer;
  user-select:none;white-space:nowrap}
.tax-table thead th:hover{background:#444}
.tax-table tbody tr:nth-child(even) td{background:#f7f7f7}
.tax-table tbody tr:hover td{background:#fffde7!important}
.tax-table td{padding:4px 12px;border-bottom:1px solid #e8e8e8;color:#111}
.tax-table td:first-child{font-family:'Courier New',monospace;font-size:10.5px;color:#666}
.page-footer{border-top:1px solid #ccc;padding:10px 40px;font-size:10px;color:#999}
</style>
</head>
<body>
"""
+ f"""<div class="page-header">
  <h1>Taxa Summary &mdash; Microbiome Analysis</h1>
  <div class="sub">Generated {now} &bull; BIOMIDAS analysis platform</div>
</div>
<div class="level-nav">{tabs_html}</div>
<div class="container">
  <div class="section">
    <div class="section-title">Dataset characteristics</div>
    <table class="meta-table"><tbody>{meta_rows}</tbody></table>
  </div>
  <div class="section">
    <div class="section-title">Taxonomic bar plots</div>
    {sections_html}
  </div>
  <div class="section">
    <div class="section-title">Full feature table</div>
    <div class="search-row">
      <input type="text" class="search-box" id="tblSearch"
             placeholder="Search by taxon name or lineage..." oninput="filterTable()">
      <span class="row-count" id="rowCount"></span>
    </div>
    <div class="ftbl-wrapper">{ftable_html}</div>
  </div>
</div>
<div class="page-footer">
  Taxa Summary &bull; BIOMIDAS &bull; Developer: Ablaev Aziz Yakubovich &bull; {now}
</div>
<script>
var LDATA = {ldata_json};

function showLevel(level) {{
  var secs = document.querySelectorAll('.level-section');
  for (var i = 0; i < secs.length; i++) secs[i].style.display = 'none';
  var tabs = document.querySelectorAll('.level-tab');
  for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
  var sec = document.getElementById('section-' + level);
  if (sec) {{
    sec.style.display = 'block';
    // Resize any Plotly chart inside now that it is visible
    var pd = sec.querySelector('.plotly-graph-div');
    if (pd && window.Plotly) Plotly.Plots.resize(pd);
  }}
  var tab = document.getElementById('tab-' + level);
  if (tab) tab.classList.add('active');
}}

function toggleTbl(level) {{
  var el = document.getElementById('dto-' + level);
  if (!el) return;
  if (el.style.display === 'block') {{
    el.style.display = 'none';
  }} else {{
    buildRawTable(level);
    el.style.display = 'block';
  }}
}}

function buildRawTable(level) {{
  var d = LDATA[level];
  var h = '<table class="raw-table"><thead><tr><th>Taxon</th>';
  for (var i = 0; i < d.samples.length; i++) h += '<th>' + d.samples[i] + '</th>';
  h += '</tr></thead><tbody>';
  for (var j = 0; j < d.taxa.length; j++) {{
    var t = d.taxa[j];
    h += '<tr><td>' + t + '</td>';
    var vals = d.matrix[t];
    for (var k = 0; k < vals.length; k++) h += '<td>' + vals[k].toFixed(4) + '</td>';
    h += '</tr>';
  }}
  h += '</tbody></table>';
  document.getElementById('dtc-' + level).innerHTML = h;
}}

function dlTSV(level) {{
  var d = LDATA[level];
  var tsv = 'Taxon\\t' + d.samples.join('\\t') + '\\n';
  for (var j = 0; j < d.taxa.length; j++) {{
    var t = d.taxa[j];
    tsv += t + '\\t' + d.matrix[t].map(function(v){{return v.toFixed(6);}}).join('\\t') + '\\n';
  }}
  var blob = new Blob([tsv], {{type: 'text/tab-separated-values'}});
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href = url; a.download = 'taxa_' + level + '.tsv'; a.click();
  URL.revokeObjectURL(url);
}}

function filterTable() {{
  var q  = document.getElementById('tblSearch').value.toLowerCase().trim();
  var tb = document.querySelector('.tax-table tbody');
  if (!tb) return;
  var rows = tb.querySelectorAll('tr'), n = 0;
  for (var i = 0; i < rows.length; i++) {{
    var show = !q || rows[i].textContent.toLowerCase().indexOf(q) >= 0;
    rows[i].style.display = show ? '' : 'none';
    if (show) n++;
  }}
  document.getElementById('rowCount').textContent =
    q ? n + '\\u00a0/\\u00a0' + rows.length + ' features' : rows.length + ' features';
}}

window.addEventListener('load', function() {{
  var tb = document.querySelector('.tax-table tbody');
  if (tb) document.getElementById('rowCount').textContent =
    tb.querySelectorAll('tr').length + ' features';
  // Sort feature table by clicking header
  var tbl = document.querySelector('.tax-table');
  if (!tbl) return;
  var sc = -1, sa = true;
  var ths = tbl.querySelectorAll('thead th');
  for (var x = 0; x < ths.length; x++) {{
    (function(idx) {{
      ths[idx].addEventListener('click', function() {{
        sa = sc === idx ? !sa : true; sc = idx;
        var rows = Array.from(tbl.querySelector('tbody').querySelectorAll('tr'));
        rows.sort(function(a, b) {{
          var av = (a.cells[idx] ? a.cells[idx].textContent.trim() : '');
          var bv = (b.cells[idx] ? b.cells[idx].textContent.trim() : '');
          var an = parseFloat(av), bn = parseFloat(bv);
          if (!isNaN(an) && !isNaN(bn)) return sa ? an-bn : bn-an;
          return sa ? av.localeCompare(bv) : bv.localeCompare(av);
        }});
        var tbody = tbl.querySelector('tbody');
        rows.forEach(function(r) {{ tbody.appendChild(r); }});
        for (var h2 = 0; h2 < ths.length; h2++) {{
          ths[h2].textContent = ths[h2].textContent.replace(/ [\u2191\u2193]$/, '');
        }}
        ths[idx].textContent += sa ? ' \u2191' : ' \u2193';
      }});
    }})(x);
  }}
}});
</script>
</body>
</html>"""
        )
        return html

    def create_group_comparison_plot(self, top_n=20, exclude_taxa=None, font_scale=1.0):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return None
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        taxa = mean_ab.head(top_n).index.tolist()
        if exclude_taxa:
            taxa = [t for t in taxa if t not in exclude_taxa]
        groups = sorted(self.data_loader.metadata[gc].unique())
        use_italic = tax_level in ['Genus', 'Species']
        display_taxa = [f'<i>{t}</i>' if use_italic else t for t in taxa]

        fig = go.Figure()
        for g in groups:
            g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
            common = list(set(g_samples) & set(rel.columns))
            means = [rel.loc[t, common].mean() if common else 0 for t in taxa]
            sds = [rel.loc[t, common].std() if common and len(common) > 1 else 0 for t in taxa]
            color = self.group_colors.get(g, '#888')
            fig.add_trace(go.Bar(
                y=display_taxa, x=means, name=str(g), orientation='h',
                marker_color=color,
                error_x=dict(type='data', array=sds, visible=True, thickness=1.5),
                hovertemplate='<b>%{y}</b><br>Mean: %{x:.2f}% ± %{error_x.array:.2f}%<extra>' + str(g) + '</extra>'
            ))

        bar_height = max(30, int(30 * font_scale))
        fig_height = max(500, len(taxa) * bar_height + 150)
        fig.update_layout(
            title=dict(text=f'Group Comparison: {tax_level} Level (Top {len(taxa)})',
                      font=dict(size=int(16 * font_scale)), x=0.5),
            xaxis_title=dict(text='Mean Relative Abundance (%)', font=dict(size=int(13 * font_scale))),
            yaxis=dict(autorange='reversed', tickfont=dict(size=max(10, int(12 * font_scale))), dtick=1),
            barmode='group', height=fig_height, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                       font=dict(size=int(12 * font_scale))),
            margin=dict(l=max(180, max(len(str(t)) for t in taxa) * 7), r=40, t=80, b=60)
        )
        return fig

    def get_data_table(self, top_n=15):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return pd.DataFrame()
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        top_taxa = mean_ab.head(top_n).index.tolist()
        table = rel.loc[top_taxa].T
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            table.insert(0, gc, self.data_loader.metadata[gc])
        return {'by_sample': table, 'by_taxa': rel.loc[top_taxa]}

    def get_summary_statistics(self):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return {}
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        return {
            'total_taxa': len(rel),
            'mean_abundance': rel.mean().mean(),
            'abundant_taxa': len(rel[rel.mean(axis=1) >= 1.0]),
            'rare_taxa': len(rel[rel.mean(axis=1) < 0.1]),
        }

    # ═══════════════════════════════════════════
    #  НОВЫЕ ВИЗУАЛИЗАЦИИ
    # ═══════════════════════════════════════════

    def _build_hierarchy_data(self, top_n=100):
        """Строит иерархические данные для Sunburst/Treemap/Icicle.
        Использует remainder branchvalues — значения только на листьях."""
        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu = self.data_loader.get_filtered_otu_table(auto_filter)
        tax = self.data_loader.taxonomy.copy()
        levels = self.data_loader.get_taxonomic_levels()
        if not levels:
            return None
        otu_means = otu.mean(axis=0)
        common_otus = list(set(otu_means.index) & set(tax.index))
        if not common_otus:
            return None
        tax_sub = tax.loc[common_otus].copy()
        tax_sub['abundance'] = otu_means[common_otus].values
        available_levels = [lv for lv in levels if lv in tax_sub.columns]
        if not available_levels:
            return None
        for lv in available_levels:
            tax_sub[lv] = tax_sub[lv].fillna(f'Unclassified_{lv}')
        agg = tax_sub.groupby(available_levels)['abundance'].sum().reset_index()
        agg = agg.sort_values('abundance', ascending=False).head(top_n)

        ids, labels, parents, values = [], [], [], []
        seen = {}
        # Собираем все узлы
        for _, row in agg.iterrows():
            path_parts = []
            for lv in available_levels:
                path_parts.append(str(row[lv]))
                current_id = '/'.join(path_parts)
                parent_id = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
                if current_id not in seen:
                    seen[current_id] = len(ids)
                    ids.append(current_id)
                    labels.append(str(row[lv]))
                    parents.append(parent_id)
                    values.append(0)

            # Накапливаем abundance только на листьях
            leaf_id = '/'.join([str(row[lv]) for lv in available_levels])
            values[seen[leaf_id]] += row['abundance']

        if not ids:
            return None

        return {'ids': ids, 'labels': labels, 'parents': parents,
                'values': values, 'levels': available_levels}

    def create_sunburst(self, top_n=100):
        hierarchy = self._build_hierarchy_data(top_n=top_n)
        if not hierarchy:
            return None
        level_colors = {
            'Kingdom': '#636EFA', 'Phylum': '#EF553B', 'Class': '#00CC96',
            'Order': '#AB63FA', 'Family': '#FFA15A', 'Genus': '#19D3F3',
            'Species': '#FF6692'
        }
        colors = []
        for i, node_id in enumerate(hierarchy['ids']):
            depth = node_id.count('/')
            if depth < len(hierarchy['levels']):
                lv = hierarchy['levels'][depth]
                colors.append(level_colors.get(lv, '#B6E880'))
            else:
                colors.append('#B6E880')

        fig = go.Figure(go.Sunburst(
            ids=hierarchy['ids'], labels=hierarchy['labels'],
            parents=hierarchy['parents'], values=hierarchy['values'],
            branchvalues='remainder',
            marker=dict(colors=colors, line=dict(width=1, color='white')),
            hovertemplate='<b>%{label}</b><br>Abundance: %{value:.2f}<br>Path: %{id}<extra></extra>',
            maxdepth=5, insidetextorientation='radial'
        ))
        levels_str = ' -> '.join(hierarchy['levels'])
        fig.update_layout(
            title=dict(text=f'Taxonomic Hierarchy (Sunburst)<br>'
                     f'<span style="font-size:12px;color:#666">{levels_str}</span>',
                     x=0.5, font=dict(size=16)),
            height=750, margin=dict(t=80, l=10, r=10, b=10)
        )
        return fig

    def create_treemap(self, top_n=100):
        hierarchy = self._build_hierarchy_data(top_n=top_n)
        if not hierarchy:
            return None
        fig = go.Figure(go.Treemap(
            ids=hierarchy['ids'], labels=hierarchy['labels'],
            parents=hierarchy['parents'], values=hierarchy['values'],
            branchvalues='remainder',
            marker=dict(colorscale='Viridis', line=dict(width=2, color='white')),
            hovertemplate='<b>%{label}</b><br>Abundance: %{value:.2f}<br>Path: %{id}<extra></extra>',
            maxdepth=4, textinfo='label+percent parent'
        ))
        fig.update_layout(
            title=dict(text=f'Taxonomic Hierarchy (Treemap)',
                     x=0.5, font=dict(size=16)),
            height=650, margin=dict(t=60, l=10, r=10, b=10)
        )
        return fig

    def create_icicle(self, top_n=100):
        hierarchy = self._build_hierarchy_data(top_n=top_n)
        if not hierarchy:
            return None
        fig = go.Figure(go.Icicle(
            ids=hierarchy['ids'], labels=hierarchy['labels'],
            parents=hierarchy['parents'], values=hierarchy['values'],
            branchvalues='remainder',
            marker=dict(colorscale='RdYlBu_r', line=dict(width=1, color='white')),
            hovertemplate='<b>%{label}</b><br>Abundance: %{value:.2f}<extra></extra>',
            maxdepth=4, tiling=dict(orientation='v')
        ))
        fig.update_layout(
            title=dict(text='Taxonomic Hierarchy (Icicle Chart)', x=0.5, font=dict(size=16)),
            height=650, margin=dict(t=60, l=10, r=10, b=10)
        )
        return fig

    def create_multi_level_overview(self, top_n_per_level=10):
        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu = self.data_loader.get_filtered_otu_table(auto_filter)
        tax = self.data_loader.taxonomy.copy()
        levels = self.data_loader.get_taxonomic_levels()
        available_levels = [lv for lv in levels if lv in tax.columns and tax[lv].notna().sum() > 5]
        if not available_levels:
            return None
        n_levels = len(available_levels)
        n_cols = min(3, n_levels)
        n_rows = int(np.ceil(n_levels / n_cols))

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'{lv} Level' for lv in available_levels],
            specs=[[{'type': 'domain'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.12 if n_rows > 1 else 0.05,
            horizontal_spacing=0.06
        )

        level_colors = [
            px.colors.qualitative.Plotly, px.colors.qualitative.D3,
            px.colors.qualitative.Set2, px.colors.qualitative.Set3,
            px.colors.qualitative.Pastel, px.colors.qualitative.Bold,
            px.colors.qualitative.Safe,
        ]
        sample_cols = list(otu.index)
        otu_t = otu.T
        for i, lv in enumerate(available_levels):
            row = i // n_cols + 1
            col = i % n_cols + 1
            merged = otu_t.merge(tax[[lv]], left_index=True, right_index=True)
            merged[lv] = merged[lv].fillna('Unclassified')
            agg = merged.groupby(lv)[sample_cols].sum()
            total = agg.sum(axis=1).sort_values(ascending=False)
            top_taxa = total.head(top_n_per_level)
            other_val = total.iloc[top_n_per_level:].sum() if len(total) > top_n_per_level else 0
            lbl = top_taxa.index.tolist()
            vals = top_taxa.values.tolist()
            if other_val > 0:
                lbl.append('Other')
                vals.append(other_val)
            # Truncate long labels
            short_lbl = [l[:20] + '..' if len(str(l)) > 22 else str(l) for l in lbl]
            fig.add_trace(go.Pie(
                labels=short_lbl, values=vals,
                textinfo='percent', textposition='inside',
                marker=dict(colors=level_colors[i % len(level_colors)][:len(lbl)]),
                hole=0.35,
                hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
                showlegend=False, textfont=dict(size=10)
            ), row=row, col=col)

        fig.update_layout(
            title=dict(text=f'Multi-Level Taxonomic Overview (Top {top_n_per_level} per level)',
                      x=0.5, font=dict(size=16)),
            height=max(400, 350 * n_rows),
            margin=dict(t=80, l=20, r=20, b=30)
        )
        # Adjust subplot title font
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=13, color='#2C3E50')

        return fig

    def create_differential_abundance(self, top_n=25, method='kruskal',
                                       p_threshold=0.05, correction='bonferroni'):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None, None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return None, None
        groups = sorted(self.data_loader.metadata[gc].unique())
        if len(groups) < 2:
            return None, None
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        taxa = mean_ab.head(top_n).index.tolist()

        results = []
        for taxon in taxa:
            group_values = {}
            for g in groups:
                g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
                common = list(set(g_samples) & set(rel.columns))
                group_values[g] = rel.loc[taxon, common].values if common else np.array([0])
            valid_groups = [v for v in group_values.values() if len(v) > 0]
            if len(valid_groups) < 2:
                continue
            try:
                if len(groups) == 2:
                    stat_val, p_val = stats.mannwhitneyu(valid_groups[0], valid_groups[1], alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    stat_val, p_val = stats.kruskal(*valid_groups)
                    test_name = 'Kruskal-Wallis'
            except Exception:
                stat_val, p_val = 0, 1.0
                test_name = 'N/A'

            row = {'Taxon': taxon, 'Test': test_name, 'Statistic': round(stat_val, 4),
                   'P-value': p_val, 'Overall_Mean': round(mean_ab[taxon], 4)}
            for g in groups:
                g_vals = group_values.get(g, np.array([0]))
                row[f'{g}_Mean'] = round(np.mean(g_vals), 4)
                row[f'{g}_SD'] = round(np.std(g_vals), 4)
            if len(groups) == 2:
                m1, m2 = np.mean(group_values[groups[0]]), np.mean(group_values[groups[1]])
                if m1 > 0 and m2 > 0:
                    row['Log2FC'] = round(np.log2(m2 / m1), 4)
                elif m2 > 0:
                    row['Log2FC'] = float('inf')
                elif m1 > 0:
                    row['Log2FC'] = float('-inf')
                else:
                    row['Log2FC'] = 0
            results.append(row)

        if not results:
            return None, None
        df = pd.DataFrame(results)
        n_tests = len(df)
        if correction == 'bonferroni' and n_tests > 0:
            df['P-adjusted'] = np.minimum(df['P-value'] * n_tests, 1.0)
        elif correction == 'fdr' and n_tests > 0:
            ranked = df['P-value'].rank()
            df['P-adjusted'] = np.minimum(df['P-value'] * n_tests / ranked, 1.0)
        else:
            df['P-adjusted'] = df['P-value']
        df['Significant'] = df['P-adjusted'] < p_threshold
        df = df.sort_values('P-adjusted')

        if len(groups) == 2 and 'Log2FC' in df.columns:
            fig = self._create_volcano_plot(df, groups, p_threshold)
        else:
            fig = self._create_differential_dot_plot(df, groups, p_threshold)
        return fig, df

    def _create_volcano_plot(self, df, groups, p_threshold):
        plot_df = df[df['Log2FC'].apply(lambda x: np.isfinite(x))].copy()
        plot_df['-log10(P-adj)'] = -np.log10(plot_df['P-adjusted'].clip(lower=1e-300))
        fig = go.Figure()
        ns = plot_df[~plot_df['Significant']]
        if len(ns) > 0:
            fig.add_trace(go.Scatter(
                x=ns['Log2FC'], y=ns['-log10(P-adj)'], mode='markers', name='Not significant',
                marker=dict(color='#BDC3C7', size=8, opacity=0.6),
                text=ns['Taxon'],
                hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(P): %{y:.2f}<extra></extra>'
            ))
        for sign, color, label in [(1, '#E74C3C', groups[1]), (-1, '#3498DB', groups[0])]:
            sig = plot_df[plot_df['Significant'] & ((plot_df['Log2FC'] * sign) > 0)]
            if len(sig) > 0:
                # Show labels only for top 8 most significant
                sig_sorted = sig.sort_values('-log10(P-adj)', ascending=False)
                labels = []
                for i, t in enumerate(sig_sorted['Taxon']):
                    short_name = t[:20] + '..' if len(str(t)) > 22 else str(t)
                    labels.append(short_name if i < 8 else '')
                fig.add_trace(go.Scatter(
                    x=sig_sorted['Log2FC'], y=sig_sorted['-log10(P-adj)'],
                    mode='markers+text', name=f'Enriched in {label}',
                    marker=dict(size=10, color=color),
                    text=labels,
                    textposition='top right' if sign > 0 else 'top left',
                    textfont=dict(size=8),
                    hovertemplate='<b>%{hovertext}</b><br>Log2FC: %{x:.2f}<br>-log10(P): %{y:.2f}<extra></extra>',
                    hovertext=sig_sorted['Taxon']
                ))
        fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="gray",
                     annotation_text=f"P = {p_threshold}")
        sig_count = plot_df['Significant'].sum()
        fig.update_layout(
            title=dict(text=f'Volcano Plot: {groups[0]} vs {groups[1]}<br>'
                     f'<span style="font-size:12px;color:#666">{sig_count} significant taxa</span>',
                     x=0.5, font=dict(size=16)),
            xaxis_title='Log2 Fold Change', yaxis_title='-log10(P-adjusted)',
            height=650, template='plotly_white',
            legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5),
            margin=dict(t=80, l=60, r=60, b=100)
        )
        return fig

    def _create_differential_dot_plot(self, df, groups, p_threshold):
        sig_df = df.head(25).copy()
        fig = go.Figure()
        for g in groups:
            col_mean = f'{g}_Mean'
            col_sd = f'{g}_SD'
            if col_mean in sig_df.columns:
                color = self.group_colors.get(g, '#888')
                short_names = [t[:28] + '..' if len(str(t)) > 30 else str(t) for t in sig_df['Taxon']]
                fig.add_trace(go.Scatter(
                    y=short_names, x=sig_df[col_mean], mode='markers', name=str(g),
                    marker=dict(size=np.where(sig_df['Significant'], 12, 8), color=color,
                               symbol=np.where(sig_df['Significant'], 'diamond', 'circle'),
                               line=dict(width=1, color='white')),
                    error_x=dict(type='data', array=sig_df[col_sd].values, visible=True),
                    hovertemplate='<b>%{y}</b><br>Mean: %{x:.3f}%<extra>' + str(g) + '</extra>'
                ))
        sig_count = sig_df['Significant'].sum()
        n_taxa = len(sig_df)
        fig.update_layout(
            title=dict(text=f'Differential Abundance<br>'
                     f'<span style="font-size:12px;color:#666">{sig_count} significant</span>',
                     x=0.5, font=dict(size=16)),
            xaxis_title='Mean Relative Abundance (%)',
            yaxis=dict(autorange='reversed', tickfont=dict(size=10), dtick=1),
            height=max(450, n_taxa * 24 + 150), template='plotly_white',
            legend=dict(orientation='h', yanchor='top', y=-0.05, xanchor='center', x=0.5,
                       font=dict(size=11)),
            margin=dict(l=220, r=40, t=80, b=80)
        )
        return fig

    def create_taxonomy_sankey(self, top_n=30):
        auto_filter = self.settings.get('auto_filter_zeros', True)
        otu = self.data_loader.get_filtered_otu_table(auto_filter)
        tax = self.data_loader.taxonomy.copy()
        levels = self.data_loader.get_taxonomic_levels()
        available_levels = [lv for lv in levels if lv in tax.columns and tax[lv].notna().sum() > 5]
        if len(available_levels) < 2:
            return None
        available_levels = available_levels[:5]
        otu_t = otu.T
        common_otus = list(set(otu_t.index) & set(tax.index))
        tax_sub = tax.loc[common_otus, available_levels].copy()
        for lv in available_levels:
            tax_sub[lv] = tax_sub[lv].fillna('Unclassified')
        means = otu.mean(axis=0)
        tax_sub['abundance'] = means[common_otus].values
        top_otus = tax_sub.nlargest(top_n * 10, 'abundance')

        all_labels, label_map = [], {}
        for lv in available_levels:
            for val in top_otus[lv].unique():
                label_key = f'{lv}:{val}'
                if label_key not in label_map:
                    label_map[label_key] = len(all_labels)
                    short_val = str(val)[:25] + '..' if len(str(val)) > 27 else str(val)
                    all_labels.append(f'{short_val} ({lv[0]})')

        sources, targets, values = [], [], []
        for i in range(len(available_levels) - 1):
            lv1, lv2 = available_levels[i], available_levels[i + 1]
            flow = top_otus.groupby([lv1, lv2])['abundance'].sum().reset_index()
            flow = flow.nlargest(top_n, 'abundance')
            for _, row in flow.iterrows():
                src_key, tgt_key = f'{lv1}:{row[lv1]}', f'{lv2}:{row[lv2]}'
                if src_key in label_map and tgt_key in label_map:
                    sources.append(label_map[src_key])
                    targets.append(label_map[tgt_key])
                    values.append(row['abundance'])
        if not sources:
            return None

        lc = {'K': 'rgba(99,110,250,0.7)', 'P': 'rgba(239,85,59,0.7)',
              'C': 'rgba(0,204,150,0.7)', 'O': 'rgba(171,99,250,0.7)',
              'F': 'rgba(255,161,90,0.7)', 'G': 'rgba(25,211,243,0.7)',
              'S': 'rgba(255,102,146,0.7)'}
        node_colors = []
        for label in all_labels:
            matched = False
            for letter, color in lc.items():
                if f'({letter})' in label:
                    node_colors.append(color)
                    matched = True
                    break
            if not matched:
                node_colors.append('rgba(180,180,180,0.7)')

        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                     label=all_labels, color=node_colors),
            link=dict(source=sources, target=targets, value=values, color='rgba(200,200,200,0.3)')
        ))
        fig.update_layout(
            title=dict(text=f'Taxonomic Flow (Sankey)<br>'
                     f'<span style="font-size:12px;color:#666">{" → ".join(available_levels)}</span>',
                     x=0.5, font=dict(size=16)),
            height=700, margin=dict(t=80, l=10, r=10, b=10)
        )
        return fig

    def create_taxa_abundance_heatmap_by_group(self, top_n=30, font_scale=1.0):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return None
        groups = sorted(self.data_loader.metadata[gc].unique())
        mean_ab = rel.mean(axis=1).sort_values(ascending=False)
        taxa = mean_ab.head(top_n).index.tolist()

        matrix = pd.DataFrame(index=taxa, columns=groups, dtype=float)
        for g in groups:
            g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
            common = list(set(g_samples) & set(rel.columns))
            for taxon in taxa:
                matrix.loc[taxon, g] = rel.loc[taxon, common].mean() if common else 0

        z_matrix = matrix.copy()
        for taxon in taxa:
            row = matrix.loc[taxon].values.astype(float)
            z_matrix.loc[taxon] = (row - row.mean()) / row.std() if row.std() > 0 else 0

        # Truncate long taxon names
        short_taxa = [str(t)[:35] + '..' if len(str(t)) > 37 else str(t) for t in taxa]

        # Show text annotation only if groups <= 8 to avoid crowding
        show_text = len(groups) <= 8
        fig = go.Figure(go.Heatmap(
            z=z_matrix.values.astype(float),
            x=[str(g) for g in groups], y=short_taxa,
            colorscale='RdBu_r', zmid=0, colorbar=dict(title='Z-score'),
            text=matrix.values.round(2).astype(str) if show_text else None,
            texttemplate='%{text}%' if show_text else None,
            textfont=dict(size=max(7, int(9 * font_scale))) if show_text else None,
            hovertemplate='<b>%{y}</b><br>Group: %{x}<br>Z: %{z:.2f}<extra></extra>'
        ))
        max_name_len = max(len(str(t)) for t in short_taxa) if short_taxa else 10
        fig.update_layout(
            title=dict(text=f'Taxa Abundance by Group (Z-score)<br>'
                     f'<span style="font-size:12px;color:#666">{tax_level}</span>',
                     x=0.5, font=dict(size=int(16 * font_scale))),
            height=max(450, len(taxa) * 22 + 150),
            yaxis=dict(tickfont=dict(size=max(8, int(10 * font_scale))), autorange='reversed', dtick=1),
            xaxis=dict(tickfont=dict(size=max(9, int(11 * font_scale)))),
            margin=dict(l=min(250, max(120, max_name_len * 6)), r=80, t=80, b=60)
        )
        return fig

    def create_core_microbiome_plot(self, prevalence_thresholds=None):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        if prevalence_thresholds is None:
            prevalence_thresholds = [50, 60, 70, 80, 90, 100]
        prevalence = ((rel > 0).sum(axis=1) / rel.shape[1] * 100)
        mean_ab = rel.mean(axis=1)

        counts_data = []
        for thresh in prevalence_thresholds:
            n = (prevalence >= thresh).sum()
            counts_data.append({'Threshold': thresh, 'N_Taxa': n})
        counts = pd.DataFrame(counts_data)

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=['Core Microbiome Taxa', 'Taxa Count by Prevalence'],
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.12,
            specs=[[{'type': 'scatter'}, {'type': 'bar'}]])

        core_90 = set(prevalence[prevalence >= 90].index)
        core_70 = set(prevalence[prevalence >= 70].index) - core_90
        others = set(prevalence.index) - core_90 - core_70
        for taxa_set, color, name in [
            (core_90, '#E74C3C', 'Core (>=90%)'),
            (core_70, '#F39C12', 'Core (70-90%)'),
            (others, '#BDC3C7', 'Non-core (<70%)')
        ]:
            mask = prevalence.index.isin(taxa_set)
            if mask.any():
                texts = [t if prevalence[t] >= 90 else '' for t in prevalence[mask].index]
                fig.add_trace(go.Scatter(
                    x=prevalence[mask], y=mean_ab[mask], mode='markers', name=name,
                    marker=dict(size=8, color=color, opacity=0.8),
                    text=texts, textposition='top right', textfont=dict(size=8),
                    hovertemplate='<b>%{hovertext}</b><br>Prev: %{x:.1f}%<br>Ab: %{y:.3f}%<extra></extra>',
                    hovertext=[str(t) for t in prevalence[mask].index]
                ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=counts['Threshold'], y=counts['N_Taxa'], marker_color='#3498DB',
            showlegend=False, text=counts['N_Taxa'], textposition='auto'
        ), row=1, col=2)

        fig.update_xaxes(title_text='Prevalence (%)', row=1, col=1)
        fig.update_yaxes(title_text='Mean Abundance (%)', row=1, col=1)
        fig.update_xaxes(title_text='Threshold (%)', row=1, col=2)
        fig.update_yaxes(title_text='N Core Taxa', row=1, col=2)
        fig.update_layout(
            title=dict(text=f'Core Microbiome ({tax_level})', x=0.5, font=dict(size=16)),
            height=550, template='plotly_white',
            legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.3,
                       font=dict(size=11)),
            margin=dict(t=80, l=60, r=40, b=100)
        )
        return fig

    def create_rank_abundance_curve(self, top_n=50):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        fig = go.Figure()
        if gc and gc in self.data_loader.metadata.columns:
            groups = sorted(self.data_loader.metadata[gc].unique())
            for g in groups:
                g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
                common = list(set(g_samples) & set(rel.columns))
                if common:
                    means = rel[common].mean(axis=1).sort_values(ascending=False).head(top_n)
                    color = self.group_colors.get(g, '#888')
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(means) + 1)),
                        y=np.log10(means.values + 0.001),
                        mode='lines+markers', name=str(g),
                        marker=dict(size=5, color=color), line=dict(color=color),
                        text=means.index.tolist(), customdata=means.values,
                        hovertemplate='<b>Rank %{x}</b><br>%{text}<br>Ab: %{customdata:.3f}%<extra></extra>'
                    ))
        else:
            means = rel.mean(axis=1).sort_values(ascending=False).head(top_n)
            fig.add_trace(go.Scatter(
                x=list(range(1, len(means) + 1)), y=np.log10(means.values + 0.001),
                mode='lines+markers', name='All', marker=dict(size=5, color='#3498DB'),
                text=means.index.tolist(), customdata=means.values,
                hovertemplate='<b>Rank %{x}</b><br>%{text}<br>Ab: %{customdata:.3f}%<extra></extra>'
            ))
        fig.update_layout(
            title=dict(text=f'Rank-Abundance Curve ({tax_level})', x=0.5, font=dict(size=16)),
            xaxis_title='Species Rank', yaxis_title='log10(Relative Abundance %)',
            height=500, template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    def create_group_indicator_species(self, top_n=10):
        tax_level = self.settings['taxonomic_level']
        if tax_level not in self.taxonomy_results:
            return None
        rel = self.taxonomy_results[tax_level]['relative_abundance']
        gc = self.settings.get('group_column')
        if not gc or gc not in self.data_loader.metadata.columns:
            return None
        groups = sorted(self.data_loader.metadata[gc].unique())
        if len(groups) < 2:
            return None

        all_results = []
        for g in groups:
            g_samples = self.data_loader.metadata[self.data_loader.metadata[gc] == g].index
            common_g = list(set(g_samples) & set(rel.columns))
            other_samples = list(set(rel.columns) - set(common_g))
            if not common_g or not other_samples:
                continue
            for taxon in rel.index:
                mean_in = rel.loc[taxon, common_g].mean() if common_g else 0
                mean_out = rel.loc[taxon, other_samples].mean() if other_samples else 0
                total_mean = mean_in + mean_out
                specificity = mean_in / total_mean if total_mean > 0 else 0
                fidelity = (rel.loc[taxon, common_g] > 0).mean() if common_g else 0
                indval = specificity * fidelity
                all_results.append({
                    'Taxon': taxon, 'Group': g,
                    'Specificity': round(specificity, 4), 'Fidelity': round(fidelity, 4),
                    'IndVal': round(indval, 4),
                    'Mean_in_group': round(mean_in, 4), 'Mean_outside': round(mean_out, 4)
                })
        if not all_results:
            return None
        df = pd.DataFrame(all_results)

        # Для каждой группы берём top_n и собираем единый список таксонов
        fig = go.Figure()
        all_taxa_ordered = []
        group_data = {}
        for g in groups:
            g_df = df[df['Group'] == g].nlargest(top_n, 'IndVal')
            group_data[g] = g_df
            for t in g_df['Taxon'].tolist():
                if t not in all_taxa_ordered:
                    all_taxa_ordered.append(t)

        for g in groups:
            g_df = group_data[g]
            color = self.group_colors.get(g, '#888')
            # Truncate long taxon names for readability
            short_names = [t[:30] + '..' if len(str(t)) > 32 else str(t) for t in g_df['Taxon']]
            fig.add_trace(go.Bar(
                y=short_names, x=g_df['IndVal'], name=str(g), orientation='h',
                marker_color=color, width=0.7 / len(groups),
                hovertemplate='<b>%{y}</b><br>IndVal: %{x:.3f}<extra>' + str(g) + '</extra>',
            ))

        n_taxa = min(len(all_taxa_ordered), top_n * len(groups))
        fig.update_layout(
            title=dict(text=f'Indicator Species ({tax_level})<br>'
                     f'<span style="font-size:12px;color:#666">IndVal = Specificity x Fidelity</span>',
                     x=0.5, font=dict(size=16)),
            xaxis_title='Indicator Value',
            yaxis=dict(autorange='reversed', tickfont=dict(size=10), dtick=1),
            barmode='group', height=max(450, n_taxa * 22 + 150),
            template='plotly_white',
            legend=dict(orientation='h', yanchor='top', y=-0.05, xanchor='center', x=0.5,
                       font=dict(size=11)),
            margin=dict(l=220, r=40, t=100, b=80),
            bargap=0.15, bargroupgap=0.05
        )
        return fig
