"""
Heatmap analysis module for microbiome data.

Provides HeatmapAnalyzer — a class that aggregates OTU counts by taxonomic
level, normalises the table, and renders an interactive Plotly heatmap with
adaptive layout and configurable colour transforms.

Colour-transform rationale
--------------------------
Raw relative-abundance data from real 16S datasets is typically highly skewed:
a handful of dominant taxa (e.g. "uncultured", "Unclassified") can account for
>20 % each while most taxa in the top-100 are below 0.5 %.  With a global
linear colour scale the bottom 90 % of cells appear white, making the plot
unreadable.  The module therefore offers four transforms:

    none        -- raw values; usable only when top_n <= 20 and deliberate
    log         -- log10(x + 1); improves moderate skew
    row_scale   -- min-max [0, 1] per taxon; every row uses the full range
    row_zscore  -- z-score per taxon; the only option that renders all taxa
                   clearly regardless of how large top_n is; hover always
                   shows the original % value

The 'auto' mode measures the skew of the actual matrix being displayed and
selects row_zscore whenever > 50 % of non-zero cells would fall below 5 % of
the global colour range with raw values.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from scipy.spatial.distance import pdist      # noqa: F401  (kept for future use)
import warnings
warnings.filterwarnings('ignore')

# Recognised colour-transform identifiers
_TRANSFORMS = ('auto', 'none', 'log', 'row_scale', 'row_zscore')

# Human-readable labels used in UI and plot titles
_TRANSFORM_LABELS = {
    'none':       'raw values',
    'log':        'log\u2081\u2080',
    'row_scale':  'row 0\u20131 scale',
    'row_zscore': 'row z-score',
}


class HeatmapAnalyzer:
    """
    Interactive heatmap with automatic colour-scale selection.

    Workflow
    --------
    1. Instantiate with a DataLoader and a settings dict.
    2. Call process_data() to aggregate + normalise.
    3. Call create_heatmap() to obtain a Plotly Figure.
    """

    def __init__(self, data_loader, settings: dict):
        self.data_loader  = data_loader
        self.settings     = settings
        self.heatmap_data = None          # taxa x samples, normalised
        self._create_color_palettes()

    # ── Colour palettes ───────────────────────────────────────────────────

    def _create_color_palettes(self):
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            unique_groups = sorted(self.data_loader.metadata[gc].unique())
            n      = len(unique_groups)
            colors = sns.color_palette("Set2", n) if n <= 8 else sns.color_palette("husl", n)
            self.group_colors = {
                g: f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                for g, c in zip(unique_groups, colors)
            }
        else:
            self.group_colors = {}

    # ── Data processing ───────────────────────────────────────────────────

    def process_data(self):
        """
        Aggregate OTU counts by taxonomic level and normalise.

        After this call, self.heatmap_data is a DataFrame (taxa x samples)
        with all-zero taxa removed.
        """
        from normalization import NormalizationProcessor

        tax_level   = self.settings.get('taxonomic_level', 'Genus')
        norm_method = self.settings.get('normalization', 'relative')
        auto_filter = self.settings.get('auto_filter_zeros', True)

        otu = self.data_loader.get_filtered_otu_table(auto_filter)

        # Aggregate by taxonomic level when taxonomy is available
        if (hasattr(self.data_loader, 'taxonomy')
                and self.data_loader.taxonomy is not None
                and tax_level in self.data_loader.taxonomy.columns):
            otu_t  = otu.T
            merged = otu_t.merge(
                self.data_loader.taxonomy[[tax_level]],
                left_index=True, right_index=True, how='left'
            )
            merged[tax_level] = merged[tax_level].fillna('Unclassified')
            sample_cols       = list(otu.index)
            agg               = merged.groupby(tax_level)[sample_cols].sum()
        else:
            agg = otu.T

        normalizer = NormalizationProcessor()
        if norm_method != 'none':
            data_for_norm    = agg.T                              # samples x taxa
            normalized       = normalizer.normalize(data_for_norm, method=norm_method)
            self.heatmap_data = normalized.T                      # back to taxa x samples
        else:
            self.heatmap_data = agg

        # Remove rows that are entirely zero
        self.heatmap_data = self.heatmap_data[self.heatmap_data.sum(axis=1) > 0]

    # ── Colour-transform helpers ──────────────────────────────────────────

    @staticmethod
    def _detect_skew(z_raw: np.ndarray) -> tuple:
        """
        Return (skew_ratio, white_fraction) for the raw matrix.

        skew_ratio    = max / median of non-zero values
        white_fraction = proportion of cells below 5 % of global max
        """
        nz = z_raw[z_raw > 0]
        if len(nz) == 0:
            return 1.0, 0.0
        skew        = float(z_raw.max()) / float(np.median(nz))
        white_frac  = float((z_raw < z_raw.max() * 0.05).sum()) / z_raw.size
        return skew, white_frac

    @staticmethod
    def _auto_color_transform(z_raw: np.ndarray) -> str:
        """
        Choose the best colour transform for z_raw.

        Returns 'row_zscore' when the data are too skewed for a global scale
        (> 50 % cells would appear white with raw values), otherwise 'none'.
        """
        _, white_frac = HeatmapAnalyzer._detect_skew(z_raw)
        return 'row_zscore' if white_frac > 0.50 else 'none'

    @staticmethod
    def _apply_color_transform(z_raw: np.ndarray, transform: str) -> tuple:
        """
        Apply a colour transform to the raw abundance matrix.

        Parameters
        ----------
        z_raw     : 2-D float array (taxa x samples), raw values
        transform : one of 'none', 'log', 'row_scale', 'row_zscore'

        Returns
        -------
        z_disp        : transformed matrix for the colour scale
        zmin          : explicit colour-scale minimum
        zmax          : explicit colour-scale maximum
        cb_title      : colorbar axis title
        cs_override   : Plotly colorscale name to override user choice, or None
        """
        if transform == 'row_zscore':
            means        = z_raw.mean(axis=1, keepdims=True)
            stds         = z_raw.std(axis=1, keepdims=True)
            stds[stds == 0] = 1.0
            z_disp       = (z_raw - means) / stds
            absmax       = float(np.nanpercentile(np.abs(z_disp), 98))
            absmax       = max(absmax, 0.01)
            return z_disp, -absmax, absmax, 'Z-score (per taxon)', 'RdBu_r'

        elif transform == 'row_scale':
            mins         = z_raw.min(axis=1, keepdims=True)
            maxs         = z_raw.max(axis=1, keepdims=True)
            rng          = maxs - mins
            rng[rng == 0] = 1.0
            z_disp       = (z_raw - mins) / rng
            return z_disp, 0.0, 1.0, 'Scaled (0\u21921 per taxon)', None

        elif transform == 'log':
            z_disp = np.log10(z_raw + 1)
            return z_disp, 0.0, float(z_disp.max()), 'log\u2081\u2080(% + 1)', None

        else:  # 'none'
            nz   = z_raw[z_raw > 0]
            zmax = float(np.percentile(nz, 99)) if len(nz) else 1.0
            zmax = max(zmax, float(z_raw.max()))
            return z_raw.copy(), 0.0, zmax, 'Abundance (%)', None

    # ── Main visualisation ────────────────────────────────────────────────

    def create_heatmap(
        self,
        top_n:            int   = 50,
        hide_zeros:       bool  = False,
        zero_threshold:   float = 0.0,
        colorscale:       str   = 'YlOrRd',
        show_values:      bool  = False,
        color_transform:  str   = 'auto',
    ):
        """
        Build and return a Plotly Figure for the heatmap.

        Parameters
        ----------
        top_n           : number of taxa to display (ranked by mean abundance)
        hide_zeros      : replace values <= zero_threshold with grey
        zero_threshold  : threshold below which values are treated as zero
        colorscale      : Plotly colorscale name; overridden to 'RdBu_r' for
                          row_zscore regardless of this parameter
        show_values     : annotate each cell with its raw % value
        color_transform : 'auto' | 'none' | 'log' | 'row_scale' | 'row_zscore'
            auto        -- detects skew and picks the optimal transform
            none        -- raw relative abundance; readable only for top_n <= 20
            log         -- log10(x + 1); suitable for moderate skew
            row_scale   -- min-max [0, 1] per taxon
            row_zscore  -- z-score per taxon; 0 % white cells for any top_n;
                           hover always shows original % values

        Returns
        -------
        plotly.graph_objects.Figure or None if data is unavailable
        """
        if self.heatmap_data is None or self.heatmap_data.empty:
            return None

        # ── Select top taxa by mean abundance ─────────────────────────
        mean_vals = self.heatmap_data.mean(axis=1).sort_values(ascending=False)
        top_taxa  = mean_vals.head(top_n).index.tolist()
        data      = self.heatmap_data.loc[top_taxa]

        # ── Sort samples by group ──────────────────────────────────────
        gc = self.settings.get('group_column')
        if gc and gc in self.data_loader.metadata.columns:
            common = [s for s in data.columns if s in self.data_loader.metadata.index]
            if common:
                order = self.data_loader.metadata.loc[common].sort_values(gc).index
                data  = data[order]

        z_raw    = data.values.copy().astype(float)
        x_labels = [str(c) for c in data.columns]
        y_labels = [str(r) for r in data.index]

        # ── Resolve colour transform ───────────────────────────────────
        if color_transform == 'auto':
            color_transform = self._auto_color_transform(z_raw)

        z_disp, zmin, zmax, cb_title, cs_override = self._apply_color_transform(
            z_raw, color_transform
        )
        final_colorscale = cs_override if cs_override else colorscale

        # ── Hover text always shows raw % ──────────────────────────────
        customdata = [
            [f'{z_raw[i, j]:.4f}%' for j in range(z_raw.shape[1])]
            for i in range(z_raw.shape[0])
        ]
        hover_tmpl = '<b>%{y}</b><br>%{x}<br>Abundance: %{customdata}<extra></extra>'

        # ── Build heatmap trace(s) ─────────────────────────────────────
        # Colorbar is positioned horizontally below the plot so it never
        # overlaps the taxon labels on the right axis.
        colorbar_cfg = dict(
            orientation='h',
            x=0.5, xanchor='center',
            y=-0.08, yanchor='top',
            len=0.5,
            thickness=14,
            title=dict(text=cb_title, side='top', font=dict(size=11)),
            tickfont=dict(size=10),
        )

        if hide_zeros:
            mask     = z_raw <= zero_threshold
            z_masked = z_disp.copy()
            z_masked[mask] = np.nan

            fig = go.Figure(go.Heatmap(
                z=z_masked, x=x_labels, y=y_labels,
                colorscale=final_colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=colorbar_cfg,
                customdata=customdata,
                hovertemplate=hover_tmpl,
                text=np.round(z_raw, 2).astype(str) if show_values else None,
                texttemplate='%{text}' if show_values else None,
            ))
            # Grey overlay for absent/zero cells — must share the same axes
            fig.add_trace(go.Heatmap(
                z=np.where(mask, 0.0, np.nan),
                x=x_labels, y=y_labels,
                colorscale=[[0, 'rgb(222,222,222)'], [1, 'rgb(222,222,222)']],
                showscale=False,
                xaxis='x', yaxis='y',
                hovertemplate='<b>%{y}</b><br>%{x}<br>Absent / below threshold<extra></extra>',
            ))
        else:
            fig = go.Figure(go.Heatmap(
                z=z_disp, x=x_labels, y=y_labels,
                colorscale=final_colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=colorbar_cfg,
                customdata=customdata,
                hovertemplate=hover_tmpl,
                text=np.round(z_raw, 2).astype(str) if show_values else None,
                texttemplate='%{text}' if show_values else None,
            ))

        # ── Group separator lines (data-space coordinates) ────────────
        shapes = []
        if gc and gc in self.data_loader.metadata.columns:
            common = [s for s in data.columns if s in self.data_loader.metadata.index]
            if common:
                sorted_meta = self.data_loader.metadata.loc[common].sort_values(gc)
                prev_g      = None
                for i, (_, row_m) in enumerate(sorted_meta.iterrows()):
                    g = row_m[gc]
                    if prev_g is not None and g != prev_g:
                        shapes.append(dict(
                            type='line', xref='x', yref='y',
                            x0=x_labels[i - 1], x1=x_labels[i - 1],
                            y0=y_labels[0],     y1=y_labels[-1],
                            line=dict(color='white', width=3),
                        ))
                    prev_g = g

        # ── Title ─────────────────────────────────────────────────────
        norm_name = self.settings.get('normalization', 'none')
        tax_level = self.settings.get('taxonomic_level', '')
        n_zeros   = int((z_raw <= zero_threshold).sum())
        zero_pct  = n_zeros / z_raw.size * 100

        title_parts = [f'Top {len(top_taxa)} Taxa']
        if tax_level:
            title_parts.append(f'at {tax_level} level')
        title_parts.append(f'({norm_name})')
        if color_transform != 'none':
            title_parts.append(f'· {_TRANSFORM_LABELS.get(color_transform, color_transform)}')
        title = ' '.join(title_parts)

        subtitle = ''
        if gc and gc in self.data_loader.metadata.columns:
            groups  = sorted(self.data_loader.metadata[gc].unique().astype(str))
            n_samp  = len(self.data_loader.metadata)
            subtitle = (
                f'<br><span style="font-size:12px;color:#666">'
                f'Grouped by: {gc} ({", ".join(groups)}) \u00b7 {n_samp} samples'
                f'</span>'
            )
        if hide_zeros:
            title += f' \u00b7 zeros hidden ({zero_pct:.1f}%)'

        # ── Adaptive sizing ───────────────────────────────────────────
        n_rows = len(top_taxa)
        n_cols = len(x_labels)

        row_px  = max(18, min(30, int(700 / max(n_rows, 1))))
        fig_h   = max(500, n_rows * row_px + 220)

        y_font  = max(7, min(12, row_px - 6))
        x_font  = max(7, min(11, int(600 / max(n_cols, 1)) + 6))

        # Right margin: space for taxon labels only (colorbar is now horizontal below)
        max_y_chars   = max((len(t) for t in y_labels), default=10)
        right_margin  = max(160, int(max_y_chars * y_font * 0.52) + 25)

        # Bottom margin: sample labels (-45°) + horizontal colorbar (≈60px) + padding
        max_x_chars   = max((len(s) for s in x_labels), default=6)
        label_height  = max(60, int(max_x_chars * x_font * 0.52) + 20)
        bottom_margin = label_height + 75   # 75px reserved for colorbar + gap

        fig.update_layout(
            title=dict(text=title + subtitle, font=dict(size=16), x=0.5),
            height=fig_h,
            xaxis=dict(
                type='category',          # prevent Plotly from auto-casting sample names as dates
                tickangle=-45,
                tickfont=dict(size=x_font),
                showgrid=False,
                automargin=True,
            ),
            yaxis=dict(
                type='category',          # prevent Plotly from auto-casting taxon names
                tickfont=dict(size=y_font),
                autorange='reversed',
                side='right',
                showgrid=False,
                dtick=1,                  # show every taxon label
                automargin=True,
            ),
            shapes=shapes,
            margin=dict(l=30, r=right_margin, t=80, b=bottom_margin),
            plot_bgcolor='white',
        )
        return fig

    # ── Statistics ────────────────────────────────────────────────────────

    def get_extended_statistics(self) -> pd.DataFrame:
        """
        Return a DataFrame with per-taxon descriptive statistics.

        Columns: Taxon, Mean, SD, Min, Max, Zeros, Non-zeros, Prevalence%
        Sorted by mean abundance descending.
        """
        if self.heatmap_data is None or self.heatmap_data.empty:
            return pd.DataFrame()

        rows = []
        for taxon in self.heatmap_data.index:
            vals = self.heatmap_data.loc[taxon]
            rows.append({
                'Taxon':       taxon,
                'Mean':        round(float(vals.mean()),  4),
                'SD':          round(float(vals.std()),   4),
                'Min':         round(float(vals.min()),   4),
                'Max':         round(float(vals.max()),   4),
                'Zeros':       int((vals == 0).sum()),
                'Non-zeros':   int((vals  > 0).sum()),
                'Prevalence%': round((vals > 0).sum() / len(vals) * 100, 1),
            })

        return pd.DataFrame(rows).sort_values('Mean', ascending=False)
