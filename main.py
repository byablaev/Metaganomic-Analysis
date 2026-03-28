import streamlit as st
import warnings, os, sys, logging, time
from datetime import datetime
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.getLogger().setLevel(logging.ERROR)
for _l in ['streamlit', 'matplotlib', 'plotly']:
    logging.getLogger(_l).setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="BioMidas16s",
    page_icon=None, layout="wide", initial_sidebar_state="expanded"
)

from data_loader import DataLoader
from normalization import NormalizationProcessor
from smart_dataset_identifier import DatasetCard, ComparisonUI
from alpha_diversity import AlphaDiversityAnalyzer
from beta_diversity import BetaDiversityAnalyzer
from composition_analysis import CompositionAnalyzer
from heatmap_analysis import HeatmapAnalyzer
from network_analysis import NetworkAnalyzer
from enhanced_clustered_heatmap import EnhancedClusteredHeatmapAnalyzer
from microbiome_ml_analyzer import MicrobiomeMLAnalyzer

def _phylo_tree_sidebar_section():
    loader = st.session_state.get('data_loader')
    if loader is None:
        return

    has_tree = (hasattr(loader, 'phylo_tree') and loader.phylo_tree is not None)

    st.subheader("Phylogenetic Tree")
    if has_tree:
        tip_count = sum(1 for _ in loader.phylo_tree.tips())
        st.success(f"Tree loaded ({tip_count} tips)\nFaith PD: **phylogenetic**")
        if st.button("Remove tree", key="remove_tree_btn"):
            loader.phylo_tree = None
            st.rerun()
    else:
        st.info("Faith PD: **taxonomy proxy**")

    with st.expander("Load / replace tree", expanded=not has_tree):
        st.caption("Newick format — SILVA, Greengenes, QIIME2 rooted trees")
        tree_f = st.file_uploader(
            "Newick file (.nwk / .newick / .tre)",
            type=['nwk','newick','tre','tree','txt'],
            key="tree_sidebar_upload")
        if tree_f is not None:
            if st.button("Apply tree", key="apply_tree_btn", type="primary"):
                try:
                    ok, n_tips, matched = loader.load_phylo_tree(tree_f)
                    if ok:
                        st.success(
                            f"{n_tips} tips loaded, {matched} OTUs matched.")
                        st.rerun()
                except Exception as e:
                    st.error(str(e))


def main():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'settings' not in st.session_state:
        st.session_state.settings = {}

    st.title("BioMidas16s")
    st.markdown("Professional metagenomic data analysis toolkit | Developer: Ablaev Aziz Yakubovich")

    with st.sidebar:
        st.header("Data Management")
        data_section()
        if st.session_state.data_loaded:
            st.markdown("---")
            st.header("Global Settings")
            global_settings_section()
            st.markdown("---")
            _phylo_tree_sidebar_section()
        st.sidebar.markdown("---")
        st.sidebar.caption("Developer: Ablaev Aziz Yakubovich")

    if st.session_state.data_loaded:
        analysis_tabs()
    else:
        show_welcome_screen()

def show_welcome_screen():
    st.markdown("""
    <div style='background:#f8f9fa;border:1px solid #dee2e6;
                border-radius:6px;padding:20px;margin-bottom:16px'>
    <h2 style='margin:0;color:#212529'>BioMidas16s</h2>
    <p style='color:#6c757d;margin:8px 0 0'>Upload three data files to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**OTU / ASV Table**")
        st.markdown("- Samples x OTUs (any orientation)\n- Any separator: comma, tab, semicolon\n- Integer counts or relative abundance\n- QIIME2, mothur, phyloseq, custom")
    with c2:
        st.markdown("**Sample Metadata**")
        st.markdown("- Sample IDs must match OTU table\n- Grouping variables (treatment, diagnosis)\n- Numeric covariates (age, BMI)\n- Any number of columns")
    with c3:
        st.markdown("**Taxonomy Table**")
        st.markdown("- OTU IDs must match OTU table\n- Any column names auto-mapped\n- Supports: k__, Phylum, SILVA, lineage strings\n- Optional")

    st.info("Smart Identification automatically detects orientation, column roles, and comparison groups.")

def data_section():
    st.subheader("Data Source")
    st.markdown("### Upload Your Files")
    st.caption("Any format accepted — orientation and column names detected automatically")
    otu_file      = st.file_uploader("OTU/ASV Table",   type=['csv','txt','tsv','biom'], key="otu_upload")
    metadata_file = st.file_uploader("Sample Metadata", type=['csv','txt','tsv'],        key="metadata_upload")
    taxonomy_file = st.file_uploader("Taxonomy Table",  type=['csv','txt','tsv'],        key="taxonomy_upload")

    with st.expander("Phylogenetic Tree — optional (true Faith PD)", expanded=False):
        st.caption(
            "Without tree: Faith PD is calculated via taxonomy proxy.\n"
            "With tree: true Faith PD (Faith 1992) via scikit-bio. "
            "Accepted: SILVA, Greengenes, QIIME2 rooted trees.")
        tree_file_upload = st.file_uploader(
            "Newick tree (.nwk / .newick / .tre / .tree)",
            type=['nwk','newick','tre','tree','txt'],
            key="tree_upload")
        st.session_state['tree_file_for_load'] = tree_file_upload

    with st.expander("Accepted formats & examples", expanded=False):
        t1, t2, t3 = st.tabs(["OTU Table", "Metadata", "Taxonomy"])
        with t1:
            st.markdown("**Standard (OTUs as rows, samples as columns):**")
            st.code(",Sample1,Sample2,Sample3\nOTU_001,150,230,0\nOTU_002,0,45,310")
            st.markdown("**Transposed (samples as rows) — also accepted automatically:**")
            st.code(",OTU_001,OTU_002,OTU_003\nSample1,150,0,88\nSample2,230,45,12")
            st.markdown("**QIIME2 / DADA2 feature table** — MD5 ASV IDs also supported")
        with t2:
            st.markdown("**Standard:**")
            st.code(",Group,Treatment,Age\nSample1,Control,Placebo,32\nSample2,Disease,DrugA,45")
            st.markdown("First column = Sample IDs (must match OTU table). "
                        "Column names are analysed automatically — no strict naming needed.")
        with t3:
            st.markdown("**Standard columns:**")
            st.code(",Kingdom,Phylum,Class,Order,Family,Genus,Species\nOTU_001,Bacteria,Firmicutes,...")
            st.markdown("**QIIME2 prefix format:**")
            st.code(",k__,p__,c__,o__,f__,g__,s__\nOTU_001,k__Bacteria,p__Firmicutes,...")
            st.markdown("**Lineage string:**")
            st.code(",taxonomy\nOTU_001,k__Bacteria;p__Firmicutes;c__Bacilli;g__Lactobacillus")
        st.info("Separator (comma, tab, semicolon, pipe) is detected automatically.")

    with st.expander("Advanced Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sep = st.selectbox("Force separator (leave comma for auto-detect):",
                                [",", "\t", ";", "|"], key="separator")
        with c2:
            index_col = st.number_input("Index column number:", min_value=0, value=0,
                                         key="index_col")

    all_uploaded = otu_file and metadata_file and taxonomy_file
    if all_uploaded:
        if st.button("Load & Identify Dataset", type="primary"):
            with st.spinner("Loading files and running smart identification..."):
                try:
                    loader = DataLoader()
                    loader.separator  = sep
                    loader.index_col  = index_col
                    success = loader.load_uploaded_data(otu_file, metadata_file, taxonomy_file)
                    if success:
                        tree_file = st.session_state.get('tree_file_for_load')
                        if tree_file is not None:
                            try:
                                ok, n_tips, matched = loader.load_phylo_tree(tree_file)
                                if ok:
                                    st.success(
                                        f"Tree loaded: {n_tips} tips, "
                                        f"{matched} matched OTUs. "
                                        f"Faith PD will use true phylogenetic calculation.")
                            except Exception as e:
                                st.warning(f"Tree load failed: {e}. Using taxonomy proxy.")

                        st.session_state.data_loader   = loader
                        st.session_state.data_loaded   = True
                        st.session_state.ml_analyzer   = None
                        st.success("Data loaded and identified successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        missing = [n for f, n in [(otu_file, "OTU Table"),
                                   (metadata_file, "Metadata"),
                                   (taxonomy_file, "Taxonomy")] if not f]
        if missing:
            st.warning(f"Still need: {', '.join(missing)}")

def show_data_summary(loader):
    fp = loader.fingerprint
    if fp:
        DatasetCard.render_streamlit(fp)
    else:
        summary = loader.get_data_summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples",    f"{summary['n_samples']:,}")
        c2.metric("OTUs/ASVs",  f"{summary['n_otus']:,}")
        c3.metric("Total Reads", f"{summary['total_reads']:,.0f}")

def global_settings_section():
    loader   = st.session_state.data_loader
    settings = st.session_state.settings
    fp       = loader.fingerprint

    if fp:
        st.markdown(f"**Dataset:** `{fp.dataset_type}` &nbsp; "
                    f"**Confidence:** {fp.confidence:.0%}",
                    unsafe_allow_html=True)
        n_s  = fp.n_samples
        n_f  = fp.n_features
        spar = fp.sparsity
        st.caption(f"{n_s} samples · {n_f} features · {spar:.0%} sparse")
        st.markdown("---")

    st.subheader("Grouping & Comparison")
    cat_cols = loader.get_categorical_columns()

    if cat_cols:

        if fp and fp.metadata_profile:
            def _fmt(c):
                info = fp.metadata_profile.get(c, {})
                nu   = info.get('n_unique', '?')
                vals = info.get('values_sample', [])
                preview = ', '.join(str(v) for v in vals[:3])
                return f"{c}  [{nu} groups: {preview}…]"
            fmt_fn = _fmt
        else:
            def fmt_fn(c):
                nu = loader.metadata[c].nunique() if loader.metadata is not None else '?'
                return f"{c}  [{nu} unique values]"

        gc = st.selectbox("Grouping variable:", cat_cols,
                           format_func=fmt_fn,
                           key="global_group_column",
                           help="Primary variable defining experimental groups")
        settings['group_column'] = gc

        if gc and loader.metadata is not None:
            vc = loader.metadata[gc].value_counts()
            grp_cols = st.columns(min(len(vc), 4))
            for i, (grp, cnt) in enumerate(vc.items()):
                grp_cols[i % len(grp_cols)].metric(str(grp), f"n={cnt}")

        if fp and fp.comparison_engine and gc:
            engine = fp.comparison_engine
            suggestions = engine.suggest_comparisons(gc, min_n_per_group=3)
            rec  = [s for s in suggestions if s['recommended']]
            all_ = suggestions

            with st.expander("Comparison suggestions", expanded=len(rec) > 0):
                if rec:
                    st.markdown("**Recommended comparisons:**")
                    for s in rec:
                        b = s['balance']
                        bal_color = "" if b >= 0.7 else "" if b >= 0.4 else ""
                        st.markdown(
                            f"- **{s['label']}** &nbsp; "
                            f"n={s['n1']}+{s['n2']} &nbsp; "
                            f"balance={b:.0%}")
                if len(all_) > len(rec):
                    st.markdown("**All possible pairs:**")
                    import pandas as _pd
                    df_s = _pd.DataFrame([{
                        'Comparison': s['label'],
                        'n₁': s['n1'], 'n₂': s['n2'],
                        'Balance': f"{s['balance']:.0%}",
                        'Recommended': 'Yes' if s['recommended'] else 'No'
                    } for s in all_])
                    st.dataframe(df_s, use_container_width=True, hide_index=True)

    else:
        st.warning("No grouping variables found in metadata.")

    levels = loader.get_taxonomic_levels()
    if levels and 'taxonomic_level' not in settings:
        idx = levels.index('Genus') if 'Genus' in levels else 0
        settings['taxonomic_level'] = levels[idx]

    st.markdown("---")
    st.subheader("Filtering")
    c1, c2 = st.columns(2)
    with c1:
        ma = st.number_input("Min abundance (%):", 0.0, 10.0, 0.01, 0.01,
                              key="global_min_abundance")
        settings['min_abundance'] = ma
    with c2:
        af = st.checkbox("Auto-filter zeros", True, key="global_auto_filter")
        settings['auto_filter_zeros'] = af

    st.subheader("Normalization")
    normalizer = NormalizationProcessor()
    nm = st.selectbox("Method:", list(normalizer.normalization_methods.keys()),
                       format_func=lambda x: normalizer.normalization_methods[x],
                       index=1, key="global_normalization")
    settings['normalization'] = nm

def _taxonomic_level_selector(key_suffix: str) -> str:
    loader = st.session_state.data_loader
    levels = loader.get_taxonomic_levels()
    if levels:
        current = st.session_state.settings.get('taxonomic_level', 'Genus')
        idx = levels.index(current) if current in levels else (levels.index('Genus') if 'Genus' in levels else 0)
        tl = st.selectbox("Taxonomic level:", levels, index=idx, key=f"tax_level_{key_suffix}",
                          help="Kingdom > Phylum > Class > Order > Family > Genus > Species")
        st.session_state.settings['taxonomic_level'] = tl
        return tl
    return st.session_state.settings.get('taxonomic_level', 'Genus')

def analysis_tabs():
    tabs = st.tabs(["Dataset Intelligence",
                    "Overview", "Alpha Diversity", "Beta Diversity",
                    "Composition", "Heatmap", "Clustered Heatmap",
                    "Machine Learning", "Network Analysis", "Report", "Export Results"])
    with tabs[0]:  dataset_intelligence_tab()
    with tabs[1]:  overview_tab()
    with tabs[2]:  alpha_diversity_tab()
    with tabs[3]:  beta_diversity_tab()
    with tabs[4]:  composition_analysis_tab()
    with tabs[5]:  heatmap_analysis_tab()
    with tabs[6]:  enhanced_clustered_heatmap_tab()
    with tabs[7]:  machine_learning_tab()
    with tabs[8]:  network_analysis_tab()
    with tabs[9]:  report_generation_tab()
    with tabs[10]: export_results_tab()

def dataset_intelligence_tab():
    import pandas as _pd
    loader   = st.session_state.data_loader
    settings = st.session_state.settings
    fp       = loader.fingerprint

    st.header("Dataset Intelligence")
    st.markdown(
        "Automatic analysis of your dataset — format, orientation, "
        "metadata roles, and comparison recommendations."
    )

    if not fp:
        st.warning("Fingerprint not available. Reload your data.")
        return

    st.subheader("Dataset Fingerprint")
    DatasetCard.render_streamlit(fp)
    st.markdown("---")

    st.subheader("Metadata Column Roles")
    if fp.metadata_profile:
        role_icons = {
            'group': 'Grouping', 'numeric': 'Numeric',
            'numeric_categorical': 'Numeric (categorical)',
            'date': 'Date/Time', 'id': 'Identifier',
            'constant': 'Constant', 'text': 'Text',
        }
        rows = []
        for col, info in fp.metadata_profile.items():
            vals = info.get('values_sample', [])
            rows.append({
                'Column':        col,
                'Detected Role': role_icons.get(info['role'], info['role']),
                'Unique Values': info['n_unique'],
                'Completeness':  f"{info['completeness']:.0%}",
                'Sample Values': ', '.join(str(v) for v in vals[:5]),
                'Use for':       ('Grouping' if info['role'] in ('group','numeric_categorical')
                                  else 'Covariate' if info['role']=='numeric'
                                  else 'Skip' if info['role'] in ('constant','id') else '—'),
            })
        st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Comparison Builder")
    st.markdown(
        "Select which groups to compare. Recommendations are based on "
        "group sizes and statistical balance."
    )

    cat_cols = loader.get_categorical_columns()
    if not cat_cols:
        st.warning("No grouping variables detected in metadata.")
        return

    def _col_label(c):
        info = fp.metadata_profile.get(c, {})
        nu   = info.get('n_unique', '?')
        vals = info.get('values_sample', [])
        prev = ' · '.join(str(v) for v in vals[:4])
        return f"{c}  [{nu} groups: {prev}]"

    selected_col = st.selectbox(
        "Grouping variable:", cat_cols,
        format_func=_col_label, key="di_group_col",
        help="The metadata column that defines experimental groups")
    settings['group_column'] = selected_col

    engine = fp.comparison_engine
    if engine is None or not selected_col:
        return

    groups   = engine.get_groups_for_column(selected_col)
    n_groups = len(groups)

    st.markdown(f"**{n_groups} groups found in `{selected_col}`:**")
    g_cols = st.columns(min(n_groups, 6))
    for i, (grp, info) in enumerate(groups.items()):
        g_cols[i % len(g_cols)].metric(str(grp), f"n = {info['n']}", delta=f"{info['pct']:.0f}%")

    st.markdown("---")

    if n_groups == 2:
        grp_names = list(groups.keys())
        st.success(f"**Binary comparison:** `{grp_names[0]}` vs `{grp_names[1]}`")
        settings['comparison_mode']   = 'binary'
        settings['comparison_groups'] = grp_names
        ref_opts = engine.get_reference_group_options(selected_col)
        ref = st.selectbox("Reference group (baseline/control):", ref_opts,
                            key="di_ref_binary")
        settings['reference_group'] = ref
        other = [g for g in grp_names if g != ref][0]
        st.info(f"**Encoding:** `{ref}` = 0 (reference) · `{other}` = 1 (case)")

    elif n_groups > 2:
        mode_opts = {
            'all_vs_all':  ' All groups simultaneously (multi-class)',
            'pairwise':    'Pairwise (select specific pairs)',
            'one_vs_rest': ' One group vs all others',
        }
        mode = st.radio("Comparison mode:", list(mode_opts.keys()),
                         format_func=lambda k: mode_opts[k],
                         horizontal=True, key="di_mode")
        settings['comparison_mode'] = mode

        all_grp = list(groups.keys())
        sel_grps = st.multiselect("Include groups:", all_grp, default=all_grp,
                                   key="di_sel_groups")
        settings['comparison_groups'] = sel_grps

        if mode in ('pairwise','one_vs_rest') and len(sel_grps) >= 2:
            ref_opts = engine.get_reference_group_options(selected_col)
            ref_opts = [r for r in ref_opts if r in sel_grps]
            ref = st.selectbox("Reference group:", ref_opts or sel_grps,
                                key="di_ref")
            settings['reference_group'] = ref

    st.markdown("---")
    st.subheader("Pairwise Comparisons")
    sel_for_table = settings.get('comparison_groups', list(groups.keys()))
    suggestions   = engine.suggest_comparisons(selected_col, min_n_per_group=2)
    suggestions   = [s for s in suggestions
                     if s['group1'] in sel_for_table and s['group2'] in sel_for_table]

    if suggestions:
        df_sug = _pd.DataFrame([{
            'Comparison': s['label'],
            'n₁': s['n1'], 'n₂': s['n2'],
            'Total n': s['total'],
            'Balance': f"{s['balance']:.0%}",
            'Recommendation': 'Recommended' if s['recommended'] else 'Small/unbalanced',
            'ML-ready': 'Yes' if s['total'] >= 10 and s['balance'] >= 0.3 else 'No',
        } for s in suggestions])
        st.dataframe(df_sug, use_container_width=True, hide_index=True)

        if settings.get('comparison_mode') == 'pairwise' and len(suggestions) > 1:
            pair_labels = [s['label'] for s in suggestions]
            chosen = st.multiselect("Select pairs for ML / analysis:",
                                     pair_labels, default=[pair_labels[0]] if pair_labels else [],
                                     key="di_pairs")
            settings['selected_pairs'] = [
                suggestions[pair_labels.index(p)] for p in chosen if p in pair_labels
            ]

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply settings to all analyses", type="primary", key="di_apply"):
            st.session_state.settings = settings
            st.success("Settings applied to all tabs!")
    with col2:
        if st.button("Re-run identification", key="di_rerun"):
            loader._run_identification()
            st.rerun()

    if fp.warnings or fp.suggestions:
        st.markdown("---")
        st.subheader("Diagnostics")
        for w in fp.warnings:  st.warning(w)
        for s in fp.suggestions: st.info(s)

def overview_tab():
    st.header("Data Overview")
    loader = st.session_state.data_loader
    settings = st.session_state.settings
    af = settings.get('auto_filter_zeros', True)
    otu = loader.get_filtered_otu_table(af)
    meta = loader.metadata

    rps = otu.sum(axis=1)          # reads per sample
    sparsity = (otu == 0).sum().sum() / otu.size * 100
    gc = settings.get('group_column')

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Samples", f"{len(otu):,}")
    k2.metric("Active OTUs", f"{len(otu.columns):,}")
    k3.metric("Total Reads", f"{otu.sum().sum():,.0f}")
    k4.metric("Mean Reads/Sample", f"{rps.mean():,.0f}")
    k5.metric("Median Reads/Sample", f"{rps.median():,.0f}")
    k6.metric("Sparsity", f"{sparsity:.1f}%", help="% of zero values in the OTU table")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Reads per Sample Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rps.values, nbinsx=30,
            marker_color='#3498DB', opacity=0.80,
            name='Samples'
        ))

        fig_hist.add_trace(go.Scatter(
            x=rps.values, y=[-1] * len(rps),
            mode='markers', marker=dict(symbol='line-ns', size=8,
                                        color='#E74C3C', line=dict(width=1)),
            name='Individual', showlegend=False
        ))
        for val, label, color in [(rps.mean(), 'Mean', '#E74C3C'),
                                   (rps.median(), 'Median', '#27AE60')]:
            fig_hist.add_vline(x=val, line_dash='dash', line_color=color,
                               annotation_text=f'{label}: {val:,.0f}',
                               annotation_position='top right',
                               annotation_font_size=11)
        fig_hist.update_layout(
            xaxis_title="Reads per Sample", yaxis_title="Number of Samples",
            height=320, template='plotly_white',
            showlegend=False, margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("Reads per Sample — Box/Violin")
        if gc and gc in meta.columns:
            fig_box = go.Figure()
            groups_sorted = sorted(meta[gc].unique())
            colors_palette = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
                              '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
            for i, g in enumerate(groups_sorted):
                g_samples = meta[meta[gc] == g].index
                vals = rps[rps.index.isin(g_samples)].values
                color = colors_palette[i % len(colors_palette)]
                fig_box.add_trace(go.Violin(
                    y=vals, name=str(g), box_visible=True, meanline_visible=True,
                    fillcolor=color, opacity=0.65, line_color='black',
                    points='all', pointpos=0, jitter=0.25,
                    marker=dict(size=4, color=color, opacity=0.7)
                ))
            fig_box.update_layout(
                yaxis_title="Reads", height=320, template='plotly_white',
                violingap=0.15, showlegend=False,
                margin=dict(t=20, b=40)
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:

            fig_box = go.Figure(go.Box(
                y=rps.values, name='All Samples', marker_color='#3498DB',
                boxmean=True, boxpoints='all', jitter=0.3
            ))
            fig_box.update_layout(yaxis_title="Reads", height=320,
                                  template='plotly_white', margin=dict(t=20, b=40))
            st.plotly_chart(fig_box, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("OTU Prevalence Distribution")
        prevalence = (otu > 0).sum(axis=0) / len(otu) * 100
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Histogram(
            x=prevalence.values, nbinsx=40,
            marker_color='#2ECC71', opacity=0.80
        ))
        for p_thresh, label in [(10, '10%'), (50, '50%'), (90, '90%')]:
            cnt = (prevalence >= p_thresh).sum()
            fig_prev.add_vline(x=p_thresh, line_dash='dot', line_color='#95A5A6',
                               annotation_text=f'≥{label}: {cnt} OTUs',
                               annotation_font_size=10,
                               annotation_position='top left' if p_thresh > 20 else 'top right')
        fig_prev.update_layout(
            xaxis_title="Prevalence (% of samples)", yaxis_title="Number of OTUs",
            height=300, template='plotly_white', margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_prev, use_container_width=True)

    with col_d:
        st.subheader("Cumulative Reads (Sample Rarefaction Preview)")
        sorted_rps = rps.sort_values()
        cumulative = sorted_rps.cumsum()
        fig_rare = go.Figure()
        fig_rare.add_trace(go.Scatter(
            x=list(range(1, len(sorted_rps) + 1)),
            y=sorted_rps.values,
            mode='lines+markers', name='Reads',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(52,152,219,0.10)'
        ))
        fig_rare.update_layout(
            xaxis_title="Samples (sorted by depth)",
            yaxis_title="Reads per Sample",
            height=300, template='plotly_white',
            margin=dict(t=20, b=40), showlegend=False
        )
        st.plotly_chart(fig_rare, use_container_width=True)

    st.markdown("---")
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Sample Read Statistics")
        stat_data = {
            'Statistic': ['Total Reads', 'Mean ± SD', 'Median', 'Min', 'Max',
                          'CV (%)', 'IQR', 'Samples < 1k reads', 'Samples < 10k reads'],
            'Value': [
                f"{rps.sum():,.0f}",
                f"{rps.mean():,.0f} ± {rps.std():,.0f}",
                f"{rps.median():,.0f}",
                f"{rps.min():,.0f}",
                f"{rps.max():,.0f}",
                f"{rps.std() / rps.mean() * 100:.1f}" if rps.mean() > 0 else 'N/A',
                f"{(rps.quantile(0.75) - rps.quantile(0.25)):,.0f}",
                f"{(rps < 1000).sum()}",
                f"{(rps < 10000).sum()}"
            ]
        }
        st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

    with col_f:
        st.subheader("OTU Table Statistics")
        otu_means = otu.mean(axis=0)
        otu_data = {
            'Statistic': ['Total OTUs (filtered)', 'Core OTUs (≥50% samples)',
                          'Rare OTUs (<10% samples)', 'Singleton OTUs',
                          'Mean abundance per OTU', 'Max abundance (single OTU)',
                          'Overall sparsity', 'Mean non-zeros per sample'],
            'Value': [
                f"{len(otu.columns):,}",
                f"{((otu > 0).sum(axis=0) / len(otu) >= 0.5).sum():,}",
                f"{((otu > 0).sum(axis=0) / len(otu) < 0.1).sum():,}",
                f"{((otu > 0).sum(axis=0) == 1).sum():,}",
                f"{otu_means.mean():.2f}",
                f"{otu.max().max():,.0f}",
                f"{sparsity:.1f}%",
                f"{(otu > 0).sum(axis=1).mean():.1f}"
            ]
        }
        st.dataframe(pd.DataFrame(otu_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Metadata Variables")
    col_g, col_h = st.columns(2)

    with col_g:
        meta_rows = []
        for col in meta.columns:
            n_unique = meta[col].nunique()
            n_missing = meta[col].isna().sum()
            dtype_str = str(meta[col].dtype)
            if meta[col].dtype in ['object', 'category']:
                top_val = meta[col].value_counts().idxmax() if n_unique > 0 else 'N/A'
                val_info = f"Top: {top_val} ({meta[col].value_counts().iloc[0]})"
            else:
                val_info = f"{meta[col].min():.2g} – {meta[col].max():.2g}"
            meta_rows.append({
                'Variable': col, 'Type': dtype_str,
                'Unique': n_unique, 'Missing': n_missing, 'Range / Top': val_info
            })
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

    with col_h:
        if gc and gc in meta.columns:
            st.subheader(f"Group Sizes — {gc}")
            counts = meta[gc].value_counts().sort_index()
            fig_grp = go.Figure(go.Bar(
                x=[str(g) for g in counts.index],
                y=counts.values,
                marker_color=['#3498DB', '#E74C3C', '#2ECC71',
                              '#F39C12', '#9B59B6', '#1ABC9C'][:len(counts)],
                text=counts.values, textposition='auto'
            ))
            fig_grp.update_layout(
                xaxis_title=gc, yaxis_title="Sample Count",
                height=280, template='plotly_white',
                margin=dict(t=10, b=40), showlegend=False
            )
            st.plotly_chart(fig_grp, use_container_width=True)
        else:
            st.info("Select a grouping variable in the sidebar to see group sizes.")

    if hasattr(loader, 'taxonomy') and loader.taxonomy is not None and not loader.taxonomy.empty:
        st.markdown("---")
        st.subheader("Taxonomy Coverage")
        tax_levels = loader.get_taxonomic_levels()
        if tax_levels:
            tcols = st.columns(len(tax_levels))
            for i, lvl in enumerate(tax_levels):
                n_taxa = loader.taxonomy[lvl].nunique() if lvl in loader.taxonomy.columns else 0
                tcols[i].metric(lvl, f"{n_taxa:,}")

def alpha_diversity_tab():
    st.header("Alpha Diversity Analysis")
    loader = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Settings")

        available_samples = sorted(loader.metadata.index.tolist())
        excluded = st.multiselect("Exclude samples:", available_samples, default=[], key="alpha_excluded")
        active_count = len(available_samples) - len(excluded)
        st.info(f"Active: {active_count}/{len(available_samples)}")
        if active_count < 2:
            st.error("Need ≥2 samples"); return

        metrics = st.multiselect("Diversity metrics:",
            ['observed', 'shannon', 'simpson', 'pielou', 'chao1', 'faith_pd'],
            default=['observed', 'shannon', 'simpson', 'pielou'], key="alpha_metrics")

        display_mode = st.radio("Display mode:",
            ["Grid View (Plotly)", "Individual Plots", "Grid View (Matplotlib)"],
            index=0, key="alpha_display_mode")

        if display_mode == "Individual Plots" and metrics:
            current_metric = st.selectbox("Navigate metrics:",
                metrics, key="alpha_nav_metric",
                format_func=lambda x: AlphaDiversityAnalyzer.METRIC_LABELS.get(x, x))

        if st.button("Run Analysis", key="alpha_run", type="primary"):
            if not metrics:
                st.error("Select at least one metric"); return
            with st.spinner("Calculating..."):
                try:
                    settings['excluded_samples'] = excluded
                    analyzer = AlphaDiversityAnalyzer(loader, settings)
                    analyzer.calculate_metrics()
                    st.session_state['alpha_analyzer'] = analyzer
                    st.session_state['alpha_metrics_list'] = metrics
                    st.success(f"Done! {active_count} samples analyzed")
                except Exception as e:
                    st.error(f"Failed: {e}"); return

    with c2:
        if 'alpha_analyzer' not in st.session_state:
            st.info("Configure settings and click 'Run Analysis'")
            return

        analyzer = st.session_state['alpha_analyzer']
        stored_metrics = st.session_state.get('alpha_metrics_list', metrics)

        if display_mode == "Grid View (Plotly)":
            fig = analyzer.plot_all_metrics_plotly(stored_metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    'toImageButtonOptions': {'format': 'svg', 'filename': 'alpha_diversity_grid'},
                    'displayModeBar': True
                })

        elif display_mode == "Individual Plots":

            metric_to_show = current_metric if 'current_metric' in dir() else stored_metrics[0]

            if metric_to_show not in stored_metrics:
                metric_to_show = stored_metrics[0]

            fig = analyzer.plot_metric_plotly(metric_to_show)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    'toImageButtonOptions': {'format': 'svg', 'filename': f'alpha_{metric_to_show}'},
                    'displayModeBar': True,
                    'displaylogo': False
                })

            st.markdown("---")
            nav_cols = st.columns(len(stored_metrics))
            for i, m in enumerate(stored_metrics):
                label = AlphaDiversityAnalyzer.METRIC_LABELS.get(m, m)
                is_current = (m == metric_to_show)
                nav_cols[i].markdown(
                    f"{'**[ ' + label + ' ]**' if is_current else label}",
                    help=f"Select '{label}' in the dropdown above"
                )

        elif display_mode == "Grid View (Matplotlib)":
            labels = AlphaDiversityAnalyzer.METRIC_LABELS
            fig = analyzer.plot_grid(stored_metrics, labels)
            if fig:
                st.pyplot(fig, use_container_width=True)

                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                            facecolor='white')
                buf.seek(0)
                st.download_button(
                    label="Download PNG",
                    data=buf,
                    file_name="alpha_diversity_grid.png",
                    mime="image/png",
                    key="alpha_mpl_png_dl"
                )

        st.markdown("---")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            if st.checkbox("Statistical summary", key="alpha_stats"):
                df = analyzer.get_statistics_summary()
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
        with stat_col2:
            if st.checkbox("Group means table", key="alpha_groups"):
                gdf = analyzer.get_group_summary_table()
                if not gdf.empty:
                    st.dataframe(gdf, use_container_width=True)

def beta_diversity_tab():
    st.header("Beta Diversity Analysis")
    loader = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Settings")
        available_samples = sorted(loader.metadata.index.tolist())
        excluded = st.multiselect("Exclude samples:", available_samples, default=[], key="beta_excluded")
        active_count = len(available_samples) - len(excluded)
        st.info(f"Active: {active_count}/{len(available_samples)}")
        if active_count < 3:
            st.error("Need ≥3 samples"); return

        distance_metric = st.selectbox("Distance metric:", ['bray_curtis', 'jaccard', 'aitchison', 'weighted_unifrac'],
            format_func=lambda x: BetaDiversityAnalyzer.METRIC_NAMES[x], key="beta_metric")

        ordination = st.selectbox(
            "Ordination:",
            ['PCoA (2D)', 'NMDS (2D)', '3D PCoA',
             'All Metrics Overview', '4-Metric Grid (Publication)'],
            key="beta_ordination"
        )

        run_anosim = st.checkbox("Run ANOSIM test", True, key="beta_anosim")
        run_pairwise = st.checkbox("Pairwise PERMANOVA", False, key="beta_pairwise",
                                   help="Pairwise comparisons between all group pairs")

        st.markdown("#### Confidence Ellipses")
        show_ellipses = st.checkbox("Show confidence ellipses", True, key="beta_show_ell",
                                    help="Confidence ellipses around groups")
        if show_ellipses:
            ellipse_confidence = st.select_slider("Confidence level:",
                options=[0.80, 0.85, 0.90, 0.95, 0.99], value=0.95, key="beta_ell_conf")
            ellipse_opacity = st.slider("Ellipse fill opacity:", 0.0, 0.4, 0.12, 0.02, key="beta_ell_opa")
            ellipse_style = st.selectbox("Ellipse line style:",
                ['dot', 'solid', 'dash', 'dashdot'], key="beta_ell_style")
        else:
            ellipse_confidence, ellipse_opacity, ellipse_style = 0.95, 0.12, 'dot'

        st.markdown("#### Display")
        show_labels = st.checkbox("Show sample labels", False, key="beta_labels")
        point_size = st.slider("Point size:", 6, 24, 12, 2, key="beta_ptsize")

        if st.button("Run Full Analysis", key="beta_run", type="primary"):
            with st.spinner("Running comprehensive beta diversity analysis..."):
                try:
                    settings['excluded_samples'] = excluded
                    analyzer = BetaDiversityAnalyzer(loader, settings)
                    analyzer.calculate_distances()
                    analyzer.run_permanova()
                    if run_anosim:
                        analyzer.run_anosim()
                    if run_pairwise:
                        analyzer.run_pairwise_permanova()
                    st.session_state['beta_analyzer'] = analyzer
                    st.success(f"Full analysis complete! {active_count} samples, {len(analyzer.distance_matrices)} distance metrics")
                except Exception as e:
                    st.error(f"Failed: {e}"); return

    with c2:
        if 'beta_analyzer' not in st.session_state:
            st.info("Configure settings and click 'Run Full Analysis'")
            return

        analyzer = st.session_state['beta_analyzer']
        gc = settings.get('group_column')

        ell_params = dict(show_ellipses=show_ellipses, ellipse_confidence=ellipse_confidence,
                          ellipse_opacity=ellipse_opacity, ellipse_style=ellipse_style,
                          show_labels=show_labels, point_size=point_size)

        if ordination == 'PCoA (2D)':
            fig = analyzer.plot_2d_plotly(distance_metric, gc, **ell_params)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

        elif ordination == 'NMDS (2D)':
            fig = analyzer.plot_nmds_plotly(distance_metric, gc, **ell_params)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

        elif ordination == '3D PCoA':
            fig = analyzer.plot_3d(distance_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        elif ordination == 'All Metrics Overview':
            fig = analyzer.plot_all_ordinations(
                show_ellipses=show_ellipses,
                ellipse_confidence=ellipse_confidence,
                ellipse_opacity=ellipse_opacity
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        elif ordination == '4-Metric Grid (Publication)':
            fig = analyzer.plot_4metric_grid_publication(
                show_ellipses=show_ellipses,
                ellipse_confidence=ellipse_confidence,
                ellipse_opacity=ellipse_opacity,
                ellipse_style=ellipse_style,
                show_labels=show_labels,
                point_size=point_size
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'beta_diversity_4metric_grid',
                        'height': 820,
                        'width': 1300,
                        'scale': 3
                    }
                })

        st.markdown("---")
        if st.checkbox("Show distance matrix heatmap", key="beta_dm_heatmap"):
            fig_dm = analyzer.plot_distance_heatmap(distance_metric)
            if fig_dm:
                st.plotly_chart(fig_dm, use_container_width=True)

        st.subheader("PERMANOVA Results")
        perm_df = analyzer.get_permanova_summary()
        if not perm_df.empty:
            st.dataframe(perm_df, use_container_width=True)
        else:
            st.info("No PERMANOVA results available")

        if analyzer.anosim_results:
            st.subheader("ANOSIM Results")
            anosim_df = analyzer.get_anosim_summary()
            if not anosim_df.empty:
                st.dataframe(anosim_df, use_container_width=True)

        if analyzer.pairwise_results:
            st.subheader("Pairwise PERMANOVA")
            pw_df = analyzer.get_pairwise_summary(distance_metric)
            if not pw_df.empty:
                st.dataframe(pw_df, use_container_width=True)

def composition_analysis_tab():
    st.header("Taxonomic Composition")
    loader = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Settings")

        _taxonomic_level_selector("comp")
        settings = st.session_state.settings.copy()

        top_n = st.slider("Number of taxa:", 5, 300, 20, 5, key="comp_top_n")

        st.markdown("#### Taxon Filters")
        min_abundance = st.number_input("Min mean abundance (%):", 0.0, 50.0, 0.0, 0.1, key="comp_min_ab")
        min_prevalence = st.number_input("Min prevalence (%):", 0.0, 100.0, 0.0, 5.0, key="comp_min_prev")

        if 'comp_analyzer' in st.session_state:
            ana = st.session_state['comp_analyzer']
            all_taxa = ana.get_all_taxa_list()
            if not all_taxa.empty:
                taxa_names = all_taxa['Taxon'].tolist()
                exclude_taxa = st.multiselect("Exclude taxa:", taxa_names, default=[], key="comp_exclude",
                                              help="Select taxa to exclude from analysis")
            else:
                exclude_taxa = []
        else:
            exclude_taxa = []

        view_mode = st.radio("View:", [
            "Stacked Bar", "Group Mean Bars (Publication)",
            "Group Comparison", "Percentage Table",
            "Sunburst", "Treemap", "Icicle",
            "Multi-Level Overview", "Sankey Flow",
            "Differential Abundance", "Core Microbiome",
            "Rank-Abundance", "Indicator Species",
            "Group Heatmap"
        ], key="comp_view")

        st.markdown("#### Display")
        font_scale = st.slider("Font scale:", 0.8, 2.0, 1.0, 0.1, key="comp_font_scale",
                              help="Scale taxon names for better readability")

        if view_mode == "Group Mean Bars (Publication)":
            show_err = st.checkbox("Show SD error bars", False, key="comp_gmb_err")
            bar_width_val = st.slider("Bar width:", 0.3, 0.95, 0.65, 0.05, key="comp_gmb_bw")
        else:
            show_err = False
            bar_width_val = 0.65

        if view_mode == "Differential Abundance":
            st.markdown("#### Differential Settings")
            da_correction = st.selectbox("P-value correction:", ['bonferroni', 'fdr', 'none'],
                                         key="comp_da_corr")
            da_p_thresh = st.slider("P-value threshold:", 0.001, 0.1, 0.05, 0.005,
                                    format="%.3f", key="comp_da_p")

        if st.button("Run Analysis", key="comp_run", type="primary"):
            with st.spinner("Processing..."):
                analyzer = CompositionAnalyzer(loader, settings)
                analyzer.process_data()
                st.session_state['comp_analyzer'] = analyzer
                st.success("Done!")

    with c2:
        if 'comp_analyzer' not in st.session_state:
            st.info("Click 'Run Analysis'")
            return

        analyzer = st.session_state['comp_analyzer']

        if view_mode == "Stacked Bar":
            fig = analyzer.create_plot(top_n=top_n, exclude_taxa=exclude_taxa or None,
                                       min_abundance=min_abundance, min_prevalence=min_prevalence,
                                       font_scale=font_scale)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {'format': 'png', 'filename': 'taxonomy_stacked',
                                             'scale': 2}
                })

        elif view_mode == "Group Mean Bars (Publication)":
            st.caption(
                "One bar per group = **mean relative abundance** across all samples. "
                "Ideal for publications with many taxa. Use camera button to export PNG."
            )
            fig = analyzer.create_grouped_mean_plot(
                top_n=top_n, exclude_taxa=exclude_taxa or None,
                min_abundance=min_abundance, min_prevalence=min_prevalence,
                font_scale=font_scale, show_error_bars=show_err, bar_width=bar_width_val
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {
                        'format': 'png', 'filename': 'taxonomy_group_mean',
                        'height': 700, 'width': 1200, 'scale': 3
                    }
                })
            else:
                st.warning("Group Mean Bars requires a grouping variable — set it in the sidebar.")

        elif view_mode == "Group Comparison":
            fig = analyzer.create_group_comparison_plot(top_n=top_n, exclude_taxa=exclude_taxa or None,
                                                        font_scale=font_scale)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Group comparison requires a grouping variable")

        elif view_mode == "Percentage Table":
            pct = analyzer.create_percentage_table(top_n=top_n, exclude_taxa=exclude_taxa or None)
            if not pct.empty:
                st.dataframe(pct, use_container_width=True, height=600)
                csv = pct.to_csv(index=False)
                st.download_button("Download CSV", csv, "taxonomy_percentage.csv", "text/csv")

        elif view_mode == "Sunburst":
            fig = analyzer.create_sunburst(top_n=min(top_n * 5, 200))
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Insufficient taxonomy data for Sunburst")

        elif view_mode == "Treemap":
            fig = analyzer.create_treemap(top_n=min(top_n * 5, 200))
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Insufficient taxonomy data for Treemap")

        elif view_mode == "Icicle":
            fig = analyzer.create_icicle(top_n=min(top_n * 5, 200))
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Insufficient taxonomy data for Icicle")

        elif view_mode == "Multi-Level Overview":
            fig = analyzer.create_multi_level_overview(top_n_per_level=min(top_n, 12))
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Insufficient taxonomy levels")

        elif view_mode == "Sankey Flow":
            fig = analyzer.create_taxonomy_sankey(top_n=top_n)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Need at least 2 taxonomy levels for Sankey")

        elif view_mode == "Differential Abundance":
            fig, da_table = analyzer.create_differential_abundance(
                top_n=top_n,
                p_threshold=da_p_thresh,
                correction=da_correction
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                if da_table is not None and not da_table.empty:
                    st.subheader("Differential Abundance Results")
                    sig_count = da_table['Significant'].sum()
                    st.info(f"{sig_count} significant taxa out of {len(da_table)} tested "
                           f"(P-adjusted < {da_p_thresh}, correction: {da_correction})")
                    st.dataframe(da_table, use_container_width=True, height=400)
                    csv = da_table.to_csv(index=False)
                    st.download_button("Download Results", csv, "differential_abundance.csv", "text/csv")
            else:
                st.warning("Differential analysis requires a grouping variable with ≥2 groups")

        elif view_mode == "Core Microbiome":
            fig = analyzer.create_core_microbiome_plot()
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Insufficient data for Core Microbiome analysis")

        elif view_mode == "Rank-Abundance":
            fig = analyzer.create_rank_abundance_curve(top_n=top_n)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

        elif view_mode == "Indicator Species":
            fig = analyzer.create_group_indicator_species(top_n=min(top_n, 15))
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Indicator Species requires a grouping variable")

        elif view_mode == "Group Heatmap":
            fig = analyzer.create_taxa_abundance_heatmap_by_group(top_n=top_n, font_scale=font_scale)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Group Heatmap requires a grouping variable")

        summary = analyzer.get_summary_statistics()
        if summary:
            st.markdown("---")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: st.metric("Total Taxa", summary.get('total_taxa', 0))
            with sc2: st.metric("Abundant Taxa (≥1%)", summary.get('abundant_taxa', 0))
            with sc3: st.metric("Rare Taxa (<0.1%)", summary.get('rare_taxa', 0))
            with sc4: st.metric("Mean Abundance", f"{summary.get('mean_abundance', 0):.2f}%")

        if st.checkbox("Show all taxa info (for filtering)", key="comp_taxa_info"):
            all_info = analyzer.get_all_taxa_list()
            if not all_info.empty:
                st.dataframe(all_info, use_container_width=True, height=400)

def heatmap_analysis_tab():
    st.header("Heatmap Analysis")
    loader   = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    gc = settings.get('group_column')
    info_parts = []
    if gc and gc in loader.metadata.columns:
        groups = sorted(loader.metadata[gc].unique().astype(str))
        info_parts.append(f"**Grouping:** {gc} ({', '.join(groups)})")
    info_parts.append(f"**Samples:** {len(loader.metadata)}")
    info_parts.append(f"**Normalization:** {settings.get('normalization', 'relative')}")
    st.markdown(" · ".join(info_parts))

    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Settings")

        _taxonomic_level_selector("heat")
        settings = st.session_state.settings.copy()

        top_n = st.slider("Number of taxa:", 10, 300, 50, 10, key="heat_top_n")

        st.markdown("#### Colour Transform")
        color_transform = st.selectbox(
            "Display mode:",
            options=['auto', 'row_zscore', 'row_scale', 'log', 'none'],
            index=0,
            key="heat_color_transform",
            format_func=lambda x: {
                'auto':       'Auto (recommended)',
                'row_zscore': 'Row Z-score  —  best for any top_n',
                'row_scale':  'Row 0-1 scale  —  per-taxon range',
                'log':        'Log10  —  moderate skew',
                'none':       'Raw values  —  top 20 taxa only',
            }[x],
            help=(
                "Auto: detects data skew and applies Row Z-score when needed.\n\n"
                "Row Z-score: normalises each taxon independently so 0 % of "
                "cells appear white regardless of top_n.  Hover always shows "
                "the original % value.\n\n"
                "Row 0-1 scale: similar but uses a 0-1 range instead of z units.\n\n"
                "Log10: log10(x + 1); useful only when one or two taxa dominate.\n\n"
                "Raw values: not recommended above top_n = 20 on typical soil "
                "or gut datasets where a few taxa dominate by an order of magnitude."
            )
        )

        colorscale = st.selectbox(
            "Colour scale:",
            ['YlOrRd', 'Viridis', 'Blues', 'Plasma', 'Cividis'],
            key="heat_colorscale",
            help="Note: Row Z-score always uses the RdBu_r diverging scale "
                 "regardless of this setting."
        )

        st.markdown("#### Zero Values")
        hide_zeros = st.checkbox("Hide zero values", False, key="heat_hide_zeros",
                                 help="Replace values at or below the threshold with grey")
        zero_threshold = 0.0
        if hide_zeros:
            zero_threshold = st.number_input(
                "Zero threshold:", 0.0, 1.0, 0.0, 0.01,
                key="heat_zero_thresh",
                help="Values <= threshold are treated as absent"
            )

        show_values = st.checkbox("Show values in cells", False, key="heat_show_vals")

        if st.button("Run Analysis", key="heat_run", type="primary"):
            with st.spinner("Creating heatmap..."):
                analyzer = HeatmapAnalyzer(loader, settings)
                analyzer.process_data()
                st.session_state['heat_analyzer'] = analyzer
                st.success("Done!")

    with c2:
        if 'heat_analyzer' not in st.session_state:
            st.info("Click 'Run Analysis'"); return

        analyzer = st.session_state['heat_analyzer']

        if analyzer.heatmap_data is not None and not analyzer.heatmap_data.empty:
            d    = analyzer.heatmap_data
            top  = d.mean(axis=1).sort_values(ascending=False).head(top_n)
            z_c  = d.loc[top.index].values
            nz   = z_c[z_c > 0]
            if len(nz):
                white_frac  = (z_c < z_c.max() * 0.05).sum() / z_c.size * 100
                skew_ratio  = z_c.max() / float(np.median(nz))
                resolved    = (
                    analyzer._auto_color_transform(z_c)
                    if color_transform == 'auto'
                    else color_transform
                )
                if color_transform == 'auto' and resolved == 'row_zscore':
                    st.info(
                        f"Auto-selected Row Z-score "
                        f"(skew ratio = {skew_ratio:.0f}\u00d7, "
                        f"{white_frac:.0f}% of cells would appear white with raw values)."
                    )
                elif color_transform == 'none' and white_frac > 50:
                    st.warning(
                        f"{white_frac:.0f}% of cells appear white with raw values "
                        f"(skew ratio = {skew_ratio:.0f}\u00d7). "
                        f"Switch to Auto or Row Z-score for a readable heatmap."
                    )

        fig = analyzer.create_heatmap(
            top_n=top_n,
            hide_zeros=hide_zeros,
            zero_threshold=zero_threshold,
            colorscale=colorscale,
            show_values=show_values,
            color_transform=color_transform,
        )
        if fig:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {'format': 'svg', 'filename': 'heatmap'},
                }
            )

        if st.checkbox("Extended statistics", key="heat_ext_stats"):
            stats_df = analyzer.get_extended_statistics()
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)

def enhanced_clustered_heatmap_tab():
    st.header("Enhanced Clustered Heatmap")

    loader = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    gc = settings.get('group_column')
    info_parts = []
    if gc and gc in loader.metadata.columns:
        groups = sorted(loader.metadata[gc].unique().astype(str))
        n_per_group = loader.metadata[gc].value_counts().to_dict()
        group_desc = ", ".join([f"{g} (n={n_per_group.get(g, n_per_group.get(int(g) if g.isdigit() else g, '?'))})" for g in groups])
        info_parts.append(f"**Grouping:** {gc}: {group_desc}")
    info_parts.append(f"**Total samples:** {len(loader.metadata)}")
    st.markdown(" · ".join(info_parts))

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Analysis Parameters")

        _taxonomic_level_selector("clust")
        settings = st.session_state.settings.copy()

        top_n = st.slider("Number of top taxa", 10, 300, 50, key="clust_top_n")
        algorithm = st.selectbox("Clustering algorithm", ['hierarchical', 'kmeans', 'dbscan'], key="clust_algo")
        linkage_method = st.selectbox("Linkage method", ['average', 'complete', 'single', 'ward'], key="clust_link")
        distance_metric = st.selectbox("Distance metric", ['euclidean', 'correlation', 'manhattan', 'cosine'],
                                       key="clust_dist")

    with c2:
        st.subheader("Visualization")
        colorscale = st.selectbox("Color scale", ['YlOrRd', 'Viridis', 'Blues', 'RdYlBu_r', 'Plasma'], key="clust_color")
        show_row_dendro = st.checkbox("Row dendrogram", True, key="clust_row_d")
        show_col_dendro = st.checkbox("Column dendrogram", True, key="clust_col_d")
        show_vals = st.checkbox("Show values", False, key="clust_vals")
        hide_zeros = st.checkbox("Hide zeros", False, key="clust_hide_zeros")
        font_scale = st.slider("Font scale:", 0.8, 2.0, 1.0, 0.1, key="clust_font_scale",
                              help="Scale taxa labels for readability")

    if st.button("Generate Clustered Heatmap", type="primary", key="clust_run"):
        with st.spinner("Processing..."):
            try:
                ana = EnhancedClusteredHeatmapAnalyzer(st.session_state.data_loader, st.session_state.settings)
                ana.process_data()
                st.session_state['clust_analyzer'] = ana

                fig = ana.create_clustered_heatmap(
                    top_n=top_n, clustering_algorithm=algorithm,
                    linkage_method=linkage_method, distance_metric=distance_metric,
                    show_row_dendrogram=show_row_dendro, show_col_dendrogram=show_col_dendro,
                    colorscale=colorscale, show_values=show_vals, hide_zeros=hide_zeros,
                    font_scale=font_scale
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

                    stats = ana.get_comprehensive_statistics()
                    if stats:
                        st.subheader("Summary")
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        with sc1: st.metric("Total Taxa", stats.get('total_taxa', 'N/A'))
                        with sc2: st.metric("Groups", stats.get('total_groups', 'N/A'))
                        with sc3: st.metric("Abundant Taxa", stats.get('abundant_taxa', 'N/A'))
                        with sc4: st.metric("Max Abundance", f"{stats.get('max_abundance', 0):.2f}%")
                else:
                    st.error("Could not generate heatmap. Check data.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

def network_analysis_tab():
    st.header("Network Analysis")
    loader = st.session_state.data_loader
    settings = st.session_state.settings.copy()

    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Settings")

        st.markdown("#### Correlation")
        corr_method = st.selectbox("Correlation method:",
            ['spearman', 'pearson', 'sparcc', 'proportionality'],
            key="net_corr",
            help="SparCC/Proportionality — compositional-aware (slower)")
        corr_thresh = st.slider("Correlation threshold:", 0.3, 0.9, 0.6, 0.05, key="net_thresh")
        p_thresh = st.slider("P-value threshold:", 0.001, 0.05, 0.01, 0.001,
                            format="%.3f", key="net_p")
        min_prev = st.slider("Min prevalence (%):", 5, 50, 20, key="net_prev")

        st.markdown("#### Performance")
        max_taxa = st.slider("Max taxa (speed control):", 30, 300, 100, 10,
                            key="net_max_taxa",
                            help="Lower = faster. 50-100 is usually enough for good network")

        est_time = "< 5s"
        if max_taxa > 150:
            est_time = "10-30s"
        if corr_method in ('sparcc', 'proportionality') and max_taxa > 100:
            est_time = "15-60s"
        if max_taxa > 200 and corr_method == 'sparcc':
            est_time = "30-120s"
        st.caption(f"Est. time: **{est_time}**")

        st.markdown("#### Visualization")
        layout = st.selectbox("Layout:", ['spring', 'circular', 'kamada_kawai',
                                          'spectral', 'shell', 'community'],
                             key="net_layout")
        color_by = st.selectbox("Color by:", ['kingdom', 'phylum', 'community',
                                              'degree', 'betweenness'], key="net_color")
        show_labels = st.checkbox("Show labels", True, key="net_labels")
        node_factor = st.slider("Node size:", 0.5, 3.0, 1.0, 0.25, key="net_node_size")

        st.markdown("#### Community Detection")
        comm_method = st.selectbox("Algorithm:", ['louvain', 'label_propagation',
                                                   'greedy_modularity'], key="net_comm_method")
        comm_resolution = 1.0
        if comm_method == 'louvain':
            comm_resolution = st.slider("Resolution:", 0.5, 2.0, 1.0, 0.1,
                                       key="net_comm_res",
                                       help="Higher = more communities")

        if st.button("Build Network", key="net_run", type="primary"):
            progress = st.progress(0, text="Starting...")

            def update_progress(text, pct):
                progress.progress(min(pct, 100), text=text)

            try:
                import time as _time
                t0 = _time.time()

                ana = NetworkAnalyzer(loader, settings)

                ana.calculate_correlations(
                    method=corr_method,
                    min_prevalence=min_prev / 100,
                    max_taxa=max_taxa,
                    progress_cb=update_progress
                )

                nodes, edges = ana.build_network(
                    correlation_threshold=corr_thresh,
                    p_value_threshold=p_thresh,
                    progress_cb=update_progress
                )

                if nodes == 0:
                    progress.empty()
                    st.warning(
                        f"No edges found with threshold |r| >= {corr_thresh} and p <= {p_thresh}. "
                        f"Try lowering the correlation threshold or increasing max taxa.")
                    return

                comm_info = ana.detect_communities(
                    method=comm_method, resolution=comm_resolution,
                    progress_cb=update_progress
                )

                elapsed = _time.time() - t0
                update_progress("Done!", 100)

                st.session_state['net_analyzer'] = ana
                st.success(
                    f"Network built in **{elapsed:.1f}s**: "
                    f"{nodes} nodes, {edges} edges, "
                    f"{comm_info.get('n_communities', 0)} communities "
                    f"(from {ana._n_taxa_used} taxa)")

            except Exception as e:
                progress.empty()
                st.error(f"Error: {e}")

    with c2:
        if 'net_analyzer' not in st.session_state:
            st.info("Click 'Build Network' to start")
            return
        ana = st.session_state['net_analyzer']

        net_tabs = st.tabs(["Network Graph", "Communities", "Keystone Species",
                           "Degree Distribution", "Robustness", "Correlation Matrix",
                           "Metrics & Export"])

        with net_tabs[0]:
            fig = ana.create_network_plot(layout=layout, node_size_factor=node_factor,
                                          color_by=color_by, show_labels=show_labels)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

        with net_tabs[1]:
            comm_fig = ana.create_community_summary_plot()
            if comm_fig:
                st.plotly_chart(comm_fig, use_container_width=True)
            st.markdown("---")
            st.subheader("Community-Colored Network")
            fig_comm = ana.create_network_plot(layout='community', node_size_factor=node_factor,
                                               color_by='community', show_labels=show_labels)
            if fig_comm:
                st.plotly_chart(fig_comm, use_container_width=True)

        with net_tabs[2]:
            st.subheader("Keystone Species")
            ks = ana.identify_keystone_species(top_n=20)
            if ks is not None and not ks.empty:
                st.dataframe(ks.head(20), use_container_width=True, height=500)
                st.download_button("Download", ks.to_csv(index=False),
                                  "keystone_species.csv", "text/csv", key="dl_ks")

        with net_tabs[3]:
            deg_fig = ana.create_degree_distribution_plot()
            if deg_fig:
                st.plotly_chart(deg_fig, use_container_width=True)

        with net_tabs[4]:
            st.subheader("Robustness Analysis")
            rob = ana.analyze_robustness(n_steps=12)
            if rob:
                st.plotly_chart(rob, use_container_width=True)
            else:
                st.info("Need at least 5 nodes")

        with net_tabs[5]:
            cfig = ana.create_correlation_heatmap(top_n=min(40, ana._n_taxa_used))
            if cfig:
                st.plotly_chart(cfig, use_container_width=True)

        with net_tabs[6]:
            metrics = ana.calculate_network_metrics()
            st.subheader("Network Metrics")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Nodes", metrics['nodes'])
                st.metric("Edges", metrics['edges'])
            with mc2:
                st.metric("Density", f"{metrics['density']:.3f}")
                st.metric("Avg Degree", f"{metrics['average_degree']:.2f}")
            with mc3:
                st.metric("Clustering", f"{metrics['clustering_coefficient']:.3f}")
                st.metric("Components", metrics['connected_components'])
            with mc4:
                st.metric("+ Edges", metrics.get('positive_edges', 0))
                st.metric("- Edges", metrics.get('negative_edges', 0))
                if 'diameter' in metrics:
                    st.metric("Diameter", metrics['diameter'])
                if 'small_world_sigma' in metrics:
                    st.metric("Small-World σ", f"{metrics['small_world_sigma']:.2f}")

            if metrics.get('is_small_world'):
                st.success("Small-world network (σ > 1)")

            if 'top_degree_nodes' in metrics:
                st.subheader("Hub Nodes (Degree)")
                st.dataframe(pd.DataFrame([
                    {'Taxon': n, 'Centrality': f"{c:.3f}"}
                    for n, c in metrics['top_degree_nodes']
                ]), use_container_width=True)

            if 'top_betweenness_nodes' in metrics:
                st.subheader("Hub Nodes (Betweenness)")
                st.dataframe(pd.DataFrame([
                    {'Taxon': n, 'Centrality': f"{c:.3f}"}
                    for n, c in metrics['top_betweenness_nodes']
                ]), use_container_width=True)

            st.subheader("Export")
            nd, ed = ana.export_network_data()
            if nd is not None:
                e1, e2 = st.columns(2)
                with e1:
                    st.download_button("Nodes CSV", nd.to_csv(index=False),
                                      "network_nodes.csv", "text/csv", key="dl_nn")
                with e2:
                    if ed is not None:
                        st.download_button("Edges CSV", ed.to_csv(index=False),
                                          "network_edges.csv", "text/csv", key="dl_ne")

def machine_learning_tab():
    st.header("Machine Learning Analysis")
    st.markdown(
        "**Professional ML pipeline** implementing 2025 best practices: "
        "SHAP explainability · SMOTE class balancing · Outlier QC · UMAP/t-SNE · "
        "Permutation test · Learning curves · sPLS-DA · LEfSe · Normalization benchmark"
    )

    loader   = st.session_state.data_loader
    settings = st.session_state.settings

    if not hasattr(loader, 'metadata') or loader.metadata.empty:
        st.warning("ML requires metadata with a categorical target variable."); return

    if st.session_state.get('ml_analyzer') is None:
        st.session_state.ml_analyzer = MicrobiomeMLAnalyzer(loader, settings)
    ml = st.session_state.ml_analyzer

    tabs = st.tabs([
        "1. Data & QC",
        "2. Model Training",
        "3. Results & Metrics",
        "4. SHAP Explainability",
        "5. Biomarker Analysis",
        "6. Dimensionality",
        "7. Advanced Methods",
        "8. Scientific Report",
    ])

    with tabs[0]:
        st.subheader("Data Preparation, Quality Control & Class Balancing")

        col1, col2 = st.columns([1, 1])

        with col1:
            cat_vars = [v for v in loader.metadata.columns
                        if 2 <= loader.metadata[v].nunique() <= 10]
            if not cat_vars:
                st.error("No suitable categorical variables (2–10 unique classes)."); return
            target = st.selectbox("Target variable (what to predict):", cat_vars,
                                   key="ml_target")

            vc = loader.metadata[target].value_counts()
            st.markdown("**Class distribution:**")
            for cls, cnt in vc.items():
                st.write(f"  `{cls}`: {cnt} samples")

            norm_opts = {'clr': 'CLR (recommended for compositional data)',
                         'log': 'Log10(x+1)', 'relative': 'Relative abundance (%)',
                         'sqrt': 'Square root', 'none': 'No normalization'}
            ml_norm = st.selectbox("Normalization:", list(norm_opts.keys()),
                                    format_func=lambda x: norm_opts[x],
                                    index=0, key="ml_norm")
            st.info("**CLR** (Centered Log-Ratio) is the scientifically recommended "
                    "transformation for compositional microbiome data.")

        with col2:
            feat_sel = st.checkbox("Apply feature selection", True, key="ml_feat_sel")
            if feat_sel:
                fs_method = st.selectbox(
                    "Feature selection method:",
                    ['mutual_info', 'combined', 'lasso', 'rfe'],
                    format_func=lambda x: {
                        'mutual_info': 'Mutual Information (fast)',
                        'combined': 'MI + RF Combined (recommended)',
                        'lasso': 'ElasticNet LASSO (sparse)',
                        'rfe': 'RFECV (thorough, slow)'
                    }[x], key="ml_fs_method")
                max_feat = st.slider("Max features to keep:", 10, 300, 100, 10,
                                      key="ml_max_feat")
            else:
                fs_method = 'mutual_info'
                max_feat = 1000

            min_spc = st.slider("Min samples per class:", 2, 10, 3, key="ml_minspc")

        settings.update({'ml_normalization': ml_norm})

        if st.button("Prepare Data", type="primary", key="ml_prep"):
            with st.spinner("Preparing and selecting features..."):
                ok = ml.prepare_data_for_ml(
                    target_variable=target, feature_selection=feat_sel,
                    min_samples_per_class=min_spc, max_features=max_feat,
                    feature_selection_method=fs_method)
                if ok:
                    st.session_state.ml_data_prepared = True
                    d = ml.processed_data
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Samples",  len(d['sample_names']))
                    k2.metric("Features", len(d['feature_names']))
                    k3.metric("Classes",  len(d['class_names']))
                    k4.metric("CV Strategy", d.get('cv_strategy', 'N/A'))

        if getattr(st.session_state, 'ml_data_prepared', False):
            st.markdown("---")
            st.markdown("### Class Imbalance Check & SMOTE")
            imb = ml.detect_class_imbalance()
            if imb:
                sev = imb.get('severity', 'none')
                ratio = imb.get('ratio', 1)
                counts = imb.get('counts', {})
                color = {'none': 'green', 'mild': 'orange', 'severe': 'red'}.get(sev, 'gray')
                st.markdown(
                    f"Imbalance ratio: **{ratio:.1f}:1** &nbsp; "
                    f"Severity: <span style='color:{color};font-weight:bold'>{sev.upper()}</span> &nbsp; "
                    f"Counts: {dict(counts)}",
                    unsafe_allow_html=True)
                if imb.get('needs_resampling'):
                    smote_strat = st.selectbox(
                        "SMOTE strategy:",
                        ['smote', 'adasyn', 'borderline', 'svmsmote'],
                        format_func=lambda x: {
                            'smote': 'SMOTE (standard, recommended)',
                            'adasyn': 'ADASYN (adaptive density)',
                            'borderline': 'Borderline SMOTE',
                            'svmsmote': 'SVM SMOTE'
                        }[x], key="ml_smote_strat")
                    if st.button("Apply SMOTE Oversampling", key="ml_smote"):
                        with st.spinner("Applying SMOTE..."):
                            ok2 = ml.apply_smote(strategy=smote_strat)
                            if ok2:
                                st.success("SMOTE applied successfully!")
                else:
                    st.success("Classes are balanced — no SMOTE needed.")

            st.markdown("---")
            st.markdown("### Outlier Detection (Sample QC)")
            st.caption("Isolation Forest detects anomalous samples that may be "
                       "contaminated, mis-labelled, or biologically extreme.")
            contam = st.slider("Expected outlier fraction:", 0.02, 0.20, 0.05, 0.01,
                                key="ml_contam",
                                help="Isolation Forest contamination parameter")
            if st.button("Run Outlier Detection", key="ml_outlier"):
                with st.spinner("Running Isolation Forest..."):
                    outlier_res = ml.detect_outliers(contamination=contam)
                    st.session_state['ml_outlier_res'] = outlier_res
                    iso_out = outlier_res.get('iso_outliers', [])
                    st.info(f"Isolation Forest: {len(iso_out)} potential outliers detected")
                    fig_out = outlier_res.get('figure')
                    if fig_out:
                        st.plotly_chart(fig_out, use_container_width=True)
                    names = outlier_res.get('sample_outlier_names', [])
                    if names:
                        st.markdown(f"**Outlier samples:** {', '.join(names)}")
                        if st.button("Remove outliers from training data", key="ml_rm_out"):
                            ml.remove_outliers(iso_out)
                            st.success(f"Removed {len(iso_out)} outliers.")

            st.markdown("---")
            st.markdown("### Auto-Normalization Benchmark")
            st.caption("Tests CLR, log, relative abundance, sqrt — picks best for your data.")
            if st.button("Benchmark Normalizations (RF quick test)", key="ml_norm_bench"):
                with st.spinner("Benchmarking 4 normalizations with RF..."):
                    bench = ml.benchmark_normalizations(target_variable=target)
                    if bench:
                        fig_bench = bench.get('_figure')
                        best_nm = bench.get('_best', '')
                        if fig_bench:
                            st.plotly_chart(fig_bench, use_container_width=True)
                        st.success(f"Recommended normalization: **{best_nm.upper()}**")

    with tabs[1]:
        st.subheader("Model Training")
        if not getattr(st.session_state, 'ml_data_prepared', False):
            st.info("Complete Step  first."); return

        st.markdown("### Quick: Train All Models")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            use_optuna = st.checkbox(
                "Optuna Bayesian optimisation (slower, better results)",
                value=False, key="ml_use_optuna",
                help="Uses Optuna TPE sampler to find optimal hyperparameters automatically.")
            use_stacking = st.checkbox("Include Stacking Ensemble", True, key="ml_stacking")
        with col_q2:
            if use_optuna:
                n_trials = st.slider("Optuna trials per model:", 20, 100, 50, 10,
                                      key="ml_trials")
                st.caption("More trials = better params, longer runtime")
            else:
                n_trials = 50

        if st.button("Train ALL Models", type="primary", key="ml_train_all"):
            progress = st.progress(0, "Starting...")
            model_list = ['Decision Tree', 'Random Forest', 'XGBoost',
                          'LightGBM', 'SVM', 'Logistic Regression']
            if use_stacking: model_list.append('Stacking Ensemble')

            for i, mname in enumerate(model_list):
                progress.progress(int(i / len(model_list) * 90), f"Training {mname}...")
                with st.spinner(f"Training {mname}..."):
                    if 'Decision' in mname:
                        ml.train_decision_tree()
                    elif 'Forest' in mname:
                        ml.train_random_forest(optimize_params=use_optuna)
                    elif 'XGB' in mname:
                        ml.train_xgboost(optimize_params=use_optuna)
                    elif 'LightGBM' in mname:
                        ml.train_lightgbm(optimize_params=use_optuna)
                    elif 'SVM' in mname:
                        ml.train_svm(optimize_params=use_optuna)
                    elif 'Logistic' in mname:
                        ml.train_logistic_regression()
                    elif 'Stacking' in mname:
                        ml.train_stacking_ensemble()

            progress.progress(100, "Done!")
            st.session_state.ml_models_trained = True
            st.success(f"All models trained! ({len(ml.models)} models)")

        st.markdown("---")
        st.markdown("### Or Train Individually")
        ind_col1, ind_col2, ind_col3 = st.columns(3)

        with ind_col1:
            st.markdown("**Random Forest**")
            rf_n = st.slider("N trees:", 50, 500, 200, 50, key="ml_rf_n")
            rf_d = st.slider("Max depth:", 3, 25, 15, 1, key="ml_rf_d")
            if st.button("Train RF", key="ml_rf"):
                with st.spinner("Training RF..."):
                    ml.train_random_forest(n_estimators=rf_n, max_depth=rf_d,
                                            optimize_params=False)
                    st.session_state.ml_models_trained = True

        with ind_col2:
            st.markdown("**XGBoost**")
            xgb_n = st.slider("N estimators:", 50, 500, 200, 50, key="ml_xgb_n")
            xgb_lr = st.select_slider("Learning rate:",
                options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05, key="ml_xgb_lr")
            if st.button("Train XGBoost", key="ml_xgb"):
                with st.spinner("Training XGBoost..."):
                    ml.train_xgboost(n_estimators=xgb_n, learning_rate=xgb_lr)
                    st.session_state.ml_models_trained = True

        with ind_col3:
            st.markdown("**SVM (RBF)**")
            svm_c = st.select_slider("C (regularisation):",
                options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key="ml_svm_c")
            if st.button("Train SVM", key="ml_svm"):
                with st.spinner("Training SVM..."):
                    ml.train_svm(C=svm_c)
                    st.session_state.ml_models_trained = True

    with tabs[2]:
        st.subheader("Results & Metrics")
        if not getattr(st.session_state, 'ml_models_trained', False):
            st.info("Complete Step  first."); return
        if not ml.models:
            st.warning("No trained models."); return

        st.markdown("### Model Comparison Table")
        comp = ml.get_model_comparison()
        if comp is not None:

            st.dataframe(comp.style.highlight_max(
                subset=[c for c in comp.columns if c != 'Model'],
                color='#d4edda', axis=0
            ).format({c: '{:.4f}' for c in comp.columns if c != 'Model'}),
                use_container_width=True)

        fig_comp = ml.plot_model_comparison()
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True,
                             config={'displayModeBar': True,
                                     'toImageButtonOptions': {'format': 'png',
                                                               'filename': 'ml_comparison', 'scale': 2}})

        st.markdown("### Cross-Validation Score Distribution")
        fig_cv = ml.plot_cv_scores()
        if fig_cv:
            st.plotly_chart(fig_cv, use_container_width=True,
                             config={'displayModeBar': True})

        st.markdown("### Confusion Matrices")
        fig_cm = ml.plot_confusion_matrices()
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True,
                             config={'displayModeBar': True})

        r1, r2 = st.columns(2)
        with r1:
            fig_roc = ml.plot_roc_curves()
            if fig_roc:
                st.markdown("### ROC Curves")
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curves available for binary classification.")

        with r2:
            fig_pr = ml.plot_precision_recall_curves()
            if fig_pr:
                st.markdown("### Precision-Recall Curves")
                st.plotly_chart(fig_pr, use_container_width=True)

        fig_cal = ml.plot_calibration_curves()
        if fig_cal:
            st.markdown("### Calibration Curves (Reliability Diagram)")
            st.caption("Well-calibrated models follow the dashed diagonal.")
            st.plotly_chart(fig_cal, use_container_width=True)

        with st.expander("Detailed Classification Reports"):
            for name, res in ml.models.items():
                st.markdown(f"**{name.replace('_',' ').title()}**")
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Test Acc",  f"{res.get('test_accuracy', 0):.3f}")
                k2.metric("Balanced",  f"{res.get('balanced_accuracy', 0):.3f}")
                k3.metric("F1",        f"{res.get('f1_score', 0):.3f}")
                k4.metric("AUROC",     f"{res.get('roc_auc', 0):.3f}")
                k5.metric("CV",        f"{res.get('cv_mean', 0):.3f}±{res.get('cv_std', 0):.3f}")
                rpt = res.get('classification_report')
                if rpt:
                    st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
                st.markdown("---")

    with tabs[3]:
        st.subheader("SHAP Explainability")
        if not getattr(st.session_state, 'ml_models_trained', False):
            st.info("Complete Step  first."); return

        if not ml.shap_values:
            st.info("SHAP requires `shap` package: `pip install shap`. "
                    "Falling back to MDI / permutation importance.")
        else:
            shap_models = list(ml.shap_values.keys())
            sel_model = st.selectbox("Model for SHAP:", shap_models,
                                      format_func=lambda x: x.replace('_',' ').title(),
                                      key="ml_shap_model")
            top_n_shap = st.slider("Top N features:", 10, 50, 20, key="ml_shap_top")

            st.markdown("#### SHAP Beeswarm (Global Explanation)")
            st.caption("Each dot = one sample. X-axis = SHAP value = impact on prediction. "
                       "Colour = feature value (red=high, blue=low).")
            fig_bee = ml.plot_shap_beeswarm(model_name=sel_model, top_n=top_n_shap)
            if fig_bee:
                st.plotly_chart(fig_bee, use_container_width=True,
                                 config={'toImageButtonOptions':
                                          {'format': 'png', 'filename': 'shap_beeswarm', 'scale': 3}})

            st.markdown("#### SHAP Waterfall (Per-Sample Explanation)")
            n_samples = len(ml.shap_values[sel_model]['values'])
            sample_idx = st.slider("Sample index:", 0, max(0, n_samples - 1), 0,
                                    key="ml_shap_sample")
            fig_wf = ml.plot_shap_waterfall(sample_idx=sample_idx, model_name=sel_model)
            if fig_wf:
                st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("### Feature Importance (MDI + Permutation)")
        fi_top = st.slider("Top N:", 10, 50, 25, key="ml_fi_top")
        fig_fi = ml.plot_feature_importance(top_n=fi_top)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True,
                             config={'toImageButtonOptions':
                                      {'format': 'png', 'filename': 'feature_importance', 'scale': 3}})

        fig_perm = ml.plot_permutation_importance(top_n=fi_top)
        if fig_perm:
            st.markdown("#### Permutation Importance (Model-Agnostic)")
            st.plotly_chart(fig_perm, use_container_width=True)

        st.markdown("### Cross-Model Importance Heatmap")
        st.caption("Normalised importance across all models — rows that are "
                   "consistently dark are the most robust biomarkers.")
        fig_heat = ml.plot_feature_importance_heatmap(top_n=fi_top)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True,
                             config={'toImageButtonOptions':
                                      {'format': 'png', 'filename': 'fi_heatmap', 'scale': 3}})

    with tabs[4]:
        st.subheader("Biomarker Stability Analysis")
        st.markdown(
            "Biomarker stability measures how consistently a taxon is selected as "
            "important across different CV folds. Low coefficient of variation (CV_imp) "
            "= stable, reliable biomarker."
        )
        if not getattr(st.session_state, 'ml_models_trained', False):
            st.info("Complete Step  first."); return

        stab_model = st.selectbox(
            "Model:", [m for m in ml.stability_results if ml.stability_results[m] is not None],
            format_func=lambda x: x.replace('_', ' ').title(),
            key="ml_stab_model") if ml.stability_results else None

        if stab_model:
            top_stab = st.slider("Top N biomarkers:", 10, 50, 25, key="ml_stab_top")
            fig_stab = ml.plot_biomarker_stability(model_name=stab_model, top_n=top_stab)
            if fig_stab:
                st.plotly_chart(fig_stab, use_container_width=True,
                                 config={'toImageButtonOptions':
                                          {'format': 'png', 'filename': 'biomarker_stability', 'scale': 3}})

            stab_df = ml.stability_results[stab_model]
            if stab_df is not None:
                st.markdown("#### Biomarker Stability Table")
                display_df = stab_df.head(top_stab).copy()
                display_df['mean_imp']      = display_df['mean_imp'].round(4)
                display_df['std_imp']       = display_df['std_imp'].round(4)
                display_df['cv_imp']        = display_df['cv_imp'].round(3)
                display_df['selection_freq'] = (display_df['selection_freq'] * 100).round(1)
                display_df = display_df.rename(columns={
                    'feature': 'Taxon/Feature',
                    'mean_imp': 'Mean Importance',
                    'std_imp': 'SD',
                    'cv_imp': 'CV (lower=stable)',
                    'selection_freq': 'Selection Freq (%)'
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Stability CSV",
                    display_df.to_csv(index=False),
                    f"biomarker_stability_{stab_model}.csv", "text/csv",
                    key="ml_dl_stab")
        else:
            st.info("Stability is computed automatically for Random Forest during training. "
                    "Train RF first.")

    with tabs[5]:
        st.subheader("Dimensionality Reduction — Sample Space Explorer")
        st.markdown(
            "Visualize how well the microbiome data separates the classes in 2D. "
            "**UMAP** is best for preserving local structure; **t-SNE** for clusters; "
            "**PCA** for linear variance. Good separation predicts high model accuracy."
        )
        if not getattr(st.session_state, 'ml_data_prepared', False):
            st.info("Complete Step  first."); return

        dr_method = st.radio("Method:", ['umap', 'tsne', 'pca'],
                              format_func=lambda x: {'umap': 'UMAP (recommended)',
                                                      'tsne': 't-SNE',
                                                      'pca': 'PCA'}[x],
                              horizontal=True, key="ml_dr_method")

        dr_col1, dr_col2 = st.columns(2)
        with dr_col1:
            if dr_method == 'umap':
                n_neighbors = st.slider("n_neighbors:", 5, 50, 15, key="ml_umap_nn")
                min_dist    = st.select_slider("min_dist:",
                    options=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1,
                    key="ml_umap_md")
            elif dr_method == 'tsne':
                perplexity = st.slider("Perplexity:", 5, 50, 30, key="ml_tsne_perp")
                n_neighbors, min_dist = 15, 0.1
            else:
                n_neighbors, min_dist, perplexity = 15, 0.1, 30

        if st.button("Compute Projection", type="primary", key="ml_dr_run"):
            with st.spinner(f"Computing {dr_method.upper()}..."):
                perp = perplexity if dr_method == 'tsne' else 30
                fig_dr = ml.plot_dimensionality_reduction(
                    method=dr_method, n_neighbors=n_neighbors,
                    min_dist=min_dist, perplexity=perp)
                if fig_dr:
                    st.plotly_chart(fig_dr, use_container_width=True,
                                     config={'toImageButtonOptions':
                                              {'format': 'png', 'filename': f'{dr_method}_projection',
                                               'scale': 3}})
                else:
                    st.warning(f"Could not compute {dr_method.upper()}. "
                               f"For UMAP: pip install umap-learn")

    with tabs[6]:
        st.subheader("Advanced Scientific Methods")

        if not getattr(st.session_state, 'ml_data_prepared', False):
            st.info("Complete Step  first."); return

        adv_sections = st.tabs([
            "Permutation Test",
            "Learning Curves",
            "sPLS-DA",
            "LEfSe Analysis",
            "Prediction Confidence",
        ])

        with adv_sections[0]:
            st.markdown("### Permutation Test (Null Model Significance)")
            st.markdown(
                "**The most rigorous validation** that the model learned real biological "
                "signal, not noise. Labels are randomly shuffled N times and model is "
                "re-evaluated each time. Real score must be significantly better than "
                "null distribution. **p < 0.05 = model is statistically significant.**"
            )
            if not ml.models:
                st.info("Train at least one model first.");
            else:
                perm_model = st.selectbox("Model:", list(ml.models.keys()),
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="ml_perm_model")
                n_perm = st.slider("N permutations:", 50, 500, 100, 50,
                                    key="ml_n_perm",
                                    help="More permutations = more accurate p-value")
                if st.button("Run Permutation Test", type="primary", key="ml_perm_run"):
                    with st.spinner(f"Running {n_perm} permutations..."):
                        perm_res = ml.run_permutation_test(model_name=perm_model,
                                                            n_permutations=n_perm)
                        if perm_res:
                            pc1, pc2, pc3 = st.columns(3)
                            pc1.metric("Real Score", f"{perm_res['real_score']:.3f}")
                            pc2.metric("p-value", f"{perm_res['p_value']:.3f}")
                            pc3.metric("Significant?",
                                       "YES" if perm_res['significant'] else "NO")
                            fig_perm = perm_res.get('figure')
                            if fig_perm:
                                st.plotly_chart(fig_perm, use_container_width=True,
                                                 config={'toImageButtonOptions': {
                                                     'format': 'png', 'scale': 3,
                                                     'filename': 'permutation_test'}})

        with adv_sections[1]:
            st.markdown("### Learning Curves")
            st.markdown(
                "Shows how performance changes as training size grows. "
                "**Overfitting** = train >> val. **Underfitting** = both scores low. "
                "**Good fit** = scores converge. Critical for small microbiome datasets."
            )
            if not ml.models:
                st.info("Train at least one model first.")
            else:
                lc_model = st.selectbox("Model:", list(ml.models.keys()),
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="ml_lc_model")
                if st.button("Compute Learning Curves", type="primary", key="ml_lc_run"):
                    with st.spinner("Computing learning curves (this may take a moment)..."):
                        fig_lc = ml.plot_learning_curves(model_name=lc_model)
                        if fig_lc:
                            st.plotly_chart(fig_lc, use_container_width=True,
                                             config={'toImageButtonOptions': {
                                                 'format': 'png', 'scale': 3,
                                                 'filename': 'learning_curves'}})

        with adv_sections[2]:
            st.markdown("### sPLS-DA (Sparse PLS Discriminant Analysis)")
            st.markdown(
                "The **gold standard** multivariate method for microbiome classification, "
                "equivalent to mixOmics in R. Simultaneously classifies samples AND "
                "identifies key taxa that drive group separation. "
                "Score plot shows sample separation; loading plot shows taxa importance."
            )
            splsda_col1, splsda_col2 = st.columns(2)
            with splsda_col1:
                n_comp = st.slider("N components:", 2, 6, 2, key="ml_splsda_nc")
            with splsda_col2:
                top_load = st.slider("Top taxa in loadings:", 10, 40, 20, key="ml_splsda_top")

            if st.button("Train sPLS-DA", type="primary", key="ml_splsda_run"):
                with st.spinner("Training sPLS-DA..."):
                    res_spls = ml.train_splsda(n_components=n_comp)
                    if res_spls:
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Test Accuracy", f"{res_spls['test_accuracy']:.3f}")
                        sc2.metric("F1 Score",      f"{res_spls['f1_score']:.3f}")
                        sc3.metric("CV Balanced Acc", f"{res_spls['cv_mean']:.3f}")

            if 'splsda' in ml.models:
                fig_spls_scores = ml.plot_splsda_scores()
                if fig_spls_scores:
                    st.plotly_chart(fig_spls_scores, use_container_width=True,
                                     config={'toImageButtonOptions': {
                                         'format': 'png', 'scale': 3,
                                         'filename': 'splsda_scores'}})
                fig_spls_load = ml.plot_splsda_loadings(n_top=top_load)
                if fig_spls_load:
                    st.plotly_chart(fig_spls_load, use_container_width=True,
                                     config={'toImageButtonOptions': {
                                         'format': 'png', 'scale': 3,
                                         'filename': 'splsda_loadings'}})

                load_df = ml.models['splsda'].get('loadings')
                if load_df is not None:
                    st.download_button("Download sPLS-DA Loadings CSV",
                                        load_df.reset_index().to_csv(index=False),
                                        "splsda_loadings.csv", "text/csv",
                                        key="ml_dl_splsda")

        with adv_sections[3]:
            st.markdown("### LEfSe Analysis")
            st.markdown(
                "**Linear Discriminant Analysis Effect Size** (Segata et al., Genome Biology 2011). "
                "The classical microbiome biomarker discovery method. Combines Kruskal-Wallis "
                "non-parametric test with LDA effect size to find taxa enriched in each group. "
                "Colour shows which group each taxon is enriched in."
            )
            lefse_top = st.slider("Top N biomarkers:", 10, 50, 30, key="ml_lefse_top")
            if st.button("Run LEfSe Analysis", type="primary", key="ml_lefse_run"):
                with st.spinner("Running LEfSe..."):
                    lefse_res = ml.run_lefse_analysis(top_n=lefse_top)
                    if lefse_res:
                        fig_lefse = lefse_res.get('figure')
                        if fig_lefse:
                            st.plotly_chart(fig_lefse, use_container_width=True,
                                             config={'toImageButtonOptions': {
                                                 'format': 'png', 'scale': 3,
                                                 'filename': 'lefse_analysis'}})
                        top_tbl = lefse_res.get('top')
                        if top_tbl is not None:
                            st.dataframe(top_tbl[['Feature', 'LDA_score', 'KW_pvalue',
                                                   'Bonferroni_p', 'Enriched_in']].round(4),
                                          use_container_width=True, hide_index=True)
                            st.download_button("Download LEfSe Results CSV",
                                                top_tbl.to_csv(index=False),
                                                "lefse_results.csv", "text/csv",
                                                key="ml_dl_lefse")
                    else:
                        st.warning("LEfSe requires data preparation and metadata. "
                                   "Complete Step  first.")

        with adv_sections[4]:
            st.markdown("### Per-Sample Prediction Confidence")
            st.markdown(
                "For each sample: **confidence** = max predicted probability, "
                "**uncertainty** = Shannon entropy of all class probabilities. "
                "Misclassified samples tend to have low confidence + high entropy. "
                "Useful for identifying ambiguous or outlier samples."
            )
            if not ml.models:
                st.info("Train models first.")
            else:
                conf_model = st.selectbox("Model:", list(ml.models.keys()),
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="ml_conf_model")
                if st.button("Compute Prediction Confidence", type="primary",
                              key="ml_conf_run"):
                    with st.spinner("Computing confidence..."):
                        conf_res = ml.compute_prediction_confidence(model_name=conf_model)
                        if conf_res:
                            fig_conf = conf_res.get('figure')
                            if fig_conf:
                                st.plotly_chart(fig_conf, use_container_width=True,
                                                 config={'toImageButtonOptions': {
                                                     'format': 'png', 'scale': 3,
                                                     'filename': 'prediction_confidence'}})
                            conf_tbl = conf_res.get('table')
                            if conf_tbl is not None:
                                st.dataframe(conf_tbl, use_container_width=True,
                                              hide_index=True)
                                st.download_button("Download Confidence CSV",
                                                    conf_tbl.to_csv(index=False),
                                                    "prediction_confidence.csv",
                                                    "text/csv", key="ml_dl_conf")
                        else:
                            st.info("Confidence requires predict_proba — not available "
                                    "for this model.")

    with tabs[7]:
        st.subheader("Scientific Report & Auto-Methods Section")
        if not getattr(st.session_state, 'ml_data_prepared', False):
            st.info("Complete Step  first."); return

        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            if st.button("Generate Full ML Report", type="primary", key="ml_rpt"):
                with st.spinner("Generating..."):
                    report = ml.get_text_report()
                    st.markdown(report)
                    st.download_button("Download Report (Markdown)",
                                        report, "ml_report.md", "text/markdown",
                                        key="ml_dl_rpt")

        with rep_col2:
            if st.button("Generate Methods Section (for paper)", key="ml_methods"):
                with st.spinner("Generating methods text..."):
                    methods = ml.generate_methods_section()
                    st.markdown(methods)
                    st.download_button("Download Methods (Markdown)",
                                        methods, "ml_methods_section.md", "text/markdown",
                                        key="ml_dl_methods")
                    st.info("This text can be copy-pasted directly into your manuscript's "
                            "Methods section. References are included.")

def report_generation_tab():
    st.header("Report Generation")

    SESSION_MAP = {
        'alpha_analyzer': ('alpha',             'Alpha Diversity'),
        'beta_analyzer':  ('beta',              'Beta Diversity'),
        'comp_analyzer':  ('composition',       'Composition'),
        'heat_analyzer':  ('heatmap',           'Heatmap'),
        'clust_analyzer': ('clustered_heatmap', 'Clustered Heatmap'),
        'net_analyzer':   ('network',           'Network'),
        'ml_analyzer':    ('ml',                'Machine Learning'),
    }

    avail = {rk: st.session_state[sk]
             for sk, (rk, _) in SESSION_MAP.items()
             if sk in st.session_state}

    if not avail:
        st.warning("Run at least one analysis first."); return

    st.success(f"Available: {', '.join(avail.keys())}")
    include = st.multiselect("Include:", list(avail.keys()), list(avail.keys()), key="rpt_inc")
    title   = st.text_input("Title:", "BioMidas16s Report", key="rpt_title")

    if not st.button("Generate Report", type="primary", key="rpt_go"): return
    if not include:
        st.warning("Select sections."); return

    progress = st.progress(0, text="Starting...")
    t0     = time.time()
    loader = st.session_state.data_loader
    gc     = st.session_state.settings.get('group_column', '')
    parts  = []
    n_sec  = max(1, len(include))

    def _safe(fn, *a, **kw):
        try:    return fn(*a, **kw)
        except: return None

    def _df_ok(df):
        return df is not None and isinstance(df, pd.DataFrame) and not df.empty

    def _df_to_html(df, n=100):
        if not _df_ok(df):
            return ''
        try:
            d = df.head(n).copy()
            seen, new_cols = {}, []
            for col in d.columns:
                key = str(col)
                seen[key] = seen.get(key, 0) + 1
                new_cols.append(f'{key}.{seen[key]}' if seen[key] > 1 else key)
            d.columns = new_cols
            for col in d.columns:
                try:
                    s = d[col]
                    if isinstance(s, pd.Series) and s.dtype in ('float64', 'float32'):
                        d[col] = s.apply(lambda x: f'{x:.4f}' if pd.notnull(x) else '')
                except Exception:
                    pass
            return d.to_html(index=False, border=0, classes='tbl')
        except Exception:
            return ''

    def _h2(t):  parts.append(f'<h2>{t}</h2>')
    def _h3(t):  parts.append(f'<h3>{t}</h3>')
    def _bx(t):  parts.append(f'<div class="bx">{t}</div>')
    def _tbl(df, n=100):
        html = _df_to_html(df, n)
        if html:
            parts.append(html)

    def _upd(msg, i):
        progress.progress(min(95, int(10 + 80 * i / n_sec)), text=msg)

    ns, no = (loader.otu_table.shape if loader.otu_table is not None else (0, 0))
    gi = ''
    if gc and gc in loader.metadata.columns:
        vc = loader.metadata[gc].value_counts()
        gi = ' &nbsp;|&nbsp; ' + ', '.join(
            f'<b>{g}</b> (n={n})' for g, n in vc.items())

    parts.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:30px auto;color:#2C3E50;line-height:1.6;font-size:14px}}
h1{{border-bottom:3px solid #2980B9;padding-bottom:10px;color:#1a252f;margin-top:0}}
h2{{border-left:5px solid #2980B9;padding-left:14px;margin-top:40px;color:#2C3E50;font-size:1.25em}}
h3{{color:#555;margin-top:22px;font-size:1.05em;border-bottom:1px solid #ECF0F1;padding-bottom:4px}}
table{{border-collapse:collapse;margin:10px 0 18px 0;font-size:.82em;width:100%}}
th{{background:#2C3E50;color:#fff;padding:7px 10px;text-align:center;font-weight:600}}
td{{padding:5px 9px;border:1px solid #D5D8DC;text-align:center}}
tr:nth-child(even){{background:#F8F9FA}}
tr:hover{{background:#EBF5FB}}
.bx{{background:#EBF5FB;padding:11px 16px;border-radius:7px;margin:10px 0;
     border-left:4px solid #2980B9;font-size:.9em}}
@media print{{
  body{{margin:10px;font-size:10pt}}
  table{{font-size:8pt;page-break-inside:auto}}
  tr{{page-break-inside:avoid}}
  h2{{page-break-before:always}}
  h2:first-of-type{{page-break-before:auto}}
}}
</style></head><body>
<h1>{title}</h1>
<p style="color:#95A5A6;margin-top:-5px">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="bx">
  <b>Samples:</b> {ns} &nbsp;|&nbsp; <b>OTUs/Features:</b> {no} &nbsp;|&nbsp;
  <b>Group variable:</b> {gc if gc else '—'}{gi}
</div>""")

    # ═══════════════════════════════════════════════════════════════
    # 1. ALPHA DIVERSITY
    # ═══════════════════════════════════════════════════════════════
    if 'alpha' in include and 'alpha' in avail:
        _upd("Alpha diversity...", include.index('alpha'))
        a = avail['alpha']
        _h2('1. Alpha Diversity')

        stat = _safe(a.get_statistics_summary)
        if _df_ok(stat):
            _h3('1.1 Overall Statistics per Metric')
            _tbl(stat)

        grp = _safe(a.get_group_summary_table)
        if _df_ok(grp):
            _h3('1.2 Per-Group Summary')
            _tbl(grp)

        if hasattr(a, 'results') and a.results:
            try:
                frames = []
                metric_labels = getattr(a, 'METRIC_LABELS', {})
                for m, md in a.results.items():
                    if isinstance(md, pd.DataFrame) and m in md.columns:
                        label = metric_labels.get(m, m)
                        frames.append(md[[m]].rename(columns={m: label}))
                if frames:
                    combined = pd.concat(frames, axis=1)
                    for m, md in a.results.items():
                        if isinstance(md, pd.DataFrame) and gc and gc in md.columns:
                            combined.insert(0, gc, md[gc])
                            break
                    combined.index.name = 'Sample'
                    _h3('1.3 Per-Sample Values (all metrics)')
                    _tbl(combined.reset_index())
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # 2. BETA DIVERSITY
    # ═══════════════════════════════════════════════════════════════
    if 'beta' in include and 'beta' in avail:
        _upd("Beta diversity...", include.index('beta'))
        b = avail['beta']
        _h2('2. Beta Diversity')

        perm = _safe(b.get_permanova_summary)
        if _df_ok(perm):
            _h3('2.1 PERMANOVA')
            _tbl(perm)

        anos = _safe(b.get_anosim_summary)
        if _df_ok(anos):
            _h3('2.2 ANOSIM')
            _tbl(anos)

        if hasattr(b, 'distance_matrices') and b.distance_matrices:
            mn = getattr(b, 'METRIC_NAMES', {})
            for metric in b.distance_matrices:
                pair = _safe(b.get_pairwise_summary, metric)
                if _df_ok(pair):
                    _h3(f'2.3 Pairwise PERMANOVA — {mn.get(metric, metric)}')
                    _tbl(pair)

        if gc and hasattr(b, 'distance_matrices') and b.distance_matrices:
            mn = getattr(b, 'METRIC_NAMES', {})
            rows_dist = []
            for metric, dm in b.distance_matrices.items():
                try:
                    meta = loader.metadata.loc[dm.index, gc]
                    vals_w, vals_b = [], []
                    idx = list(dm.index)
                    for i in range(len(idx)):
                        for j in range(i + 1, len(idx)):
                            v = dm.iloc[i, j]
                            (vals_w if meta.iloc[i] == meta.iloc[j] else vals_b).append(v)
                    rows_dist.append({
                        'Metric':             mn.get(metric, metric),
                        'Mean within-group':  round(float(np.mean(vals_w)), 4) if vals_w else 'N/A',
                        'Mean between-group': round(float(np.mean(vals_b)), 4) if vals_b else 'N/A',
                        'SD within':          round(float(np.std(vals_w)),  4) if vals_w else 'N/A',
                        'SD between':         round(float(np.std(vals_b)),  4) if vals_b else 'N/A',
                    })
                except Exception:
                    pass
            if rows_dist:
                _h3('2.4 Within vs Between-Group Distances')
                _tbl(pd.DataFrame(rows_dist))

    # ═══════════════════════════════════════════════════════════════
    # 3. TAXONOMIC COMPOSITION
    # ═══════════════════════════════════════════════════════════════
    if 'composition' in include and 'composition' in avail:
        _upd("Composition...", include.index('composition'))
        c = avail['composition']
        _h2('3. Taxonomic Composition')

        summ = _safe(c.get_summary_statistics)
        if summ and isinstance(summ, dict):
            _bx(' &nbsp;|&nbsp; '.join(
                f'<b>{k}:</b> {round(v, 4) if isinstance(v, float) else v}'
                for k, v in summ.items()))

        pct = _safe(c.create_percentage_table, top_n=50)
        if _df_ok(pct):
            _h3('3.1 Top 50 Taxa — Relative Abundance (%) per Sample')
            _tbl(pct, n=50)

        tax_level = st.session_state.settings.get('taxonomic_level', '')
        if tax_level and hasattr(c, 'taxonomy_results') and tax_level in c.taxonomy_results:
            try:
                rel = c.taxonomy_results[tax_level]['relative_abundance']
                top50 = rel.mean(axis=1).sort_values(ascending=False).head(50).index
                rel_top = rel.loc[top50]
                if gc and gc in loader.metadata.columns:
                    grp_mean = (rel_top.T
                                .join(loader.metadata[[gc]])
                                .groupby(gc).mean().T)
                    grp_mean.index.name = 'Taxon'
                    _h3(f'3.2 Top 50 Taxa — Mean Abundance (%) by Group ({gc})')
                    _tbl(grp_mean.reset_index())
            except Exception:
                pass

        diff_result = _safe(c.create_differential_abundance, top_n=30, p_threshold=0.05)
        if isinstance(diff_result, tuple) and len(diff_result) == 2:
            _, diff_df = diff_result
            if _df_ok(diff_df):
                _h3('3.3 Differential Abundance (top 30 taxa)')
                _tbl(diff_df.drop(columns=['Significant'], errors='ignore'), n=30)

    # ═══════════════════════════════════════════════════════════════
    # 4. HEATMAP
    # ═══════════════════════════════════════════════════════════════
    if 'heatmap' in include and 'heatmap' in avail:
        _upd("Heatmap statistics...", include.index('heatmap'))
        h = avail['heatmap']
        _h2('4. Abundance Heatmap — Per-Taxon Statistics')

        stats_df = _safe(h.get_extended_statistics)
        if _df_ok(stats_df):
            _h3('4.1 Taxon Statistics (Mean, SD, Min, Max, Zeros, Prevalence)')
            _tbl(stats_df)

        if hasattr(h, 'heatmap_data') and _df_ok(getattr(h, 'heatmap_data', None)):
            try:
                hd = h.heatmap_data
                top50 = hd.mean(axis=1).sort_values(ascending=False).head(50)
                hd_top = hd.loc[top50.index].T.copy()
                hd_top.index.name = 'Sample'
                _h3('4.2 Top 50 Taxa Abundance Values per Sample')
                _tbl(hd_top.reset_index())
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # 5. CLUSTERED HEATMAP
    # ═══════════════════════════════════════════════════════════════
    if 'clustered_heatmap' in include and 'clustered_heatmap' in avail:
        _upd("Clustered heatmap...", include.index('clustered_heatmap'))
        ch = avail['clustered_heatmap']
        _h2('5. Clustered Heatmap')

        cs = _safe(ch.get_comprehensive_statistics)
        if cs and isinstance(cs, dict):
            _bx(' &nbsp;|&nbsp; '.join(
                f'<b>{k.replace("_", " ").title()}:</b> '
                f'{round(v, 4) if isinstance(v, float) else v}'
                for k, v in cs.items()))

        gm = getattr(ch, 'group_means', None)
        if _df_ok(gm):
            try:
                gm_out = gm.copy()
                gm_out.index.name = 'Taxon'
                _h3('5.1 Group Mean Abundance (%) — All Taxa')
                _tbl(gm_out.reset_index())
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # 6. NETWORK ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    if 'network' in include and 'network' in avail:
        _upd("Network analysis...", include.index('network'))
        net = avail['network']
        _h2('6. Co-occurrence Network Analysis')

        nm = _safe(net.calculate_network_metrics)
        if nm and isinstance(nm, dict):
            rows_nm = [{
                'Metric': k.replace('_', ' ').title(),
                'Value':  str(v) if isinstance(v, bool)
                          else (f'{v:.4f}' if isinstance(v, float) else v)
            } for k, v in nm.items()]
            _h3('6.1 Global Network Metrics')
            _tbl(pd.DataFrame(rows_nm))

        ks = _safe(net.identify_keystone_species, top_n=30)
        if _df_ok(ks):
            cols = [c for c in ['Taxon', 'Degree', 'Degree_Centrality',
                                 'Betweenness', 'Closeness', 'Eigenvector',
                                 'Abundance', 'Keystone_Score',
                                 'Positive_Edges', 'Negative_Edges', 'Community']
                    if c in ks.columns]
            _h3('6.2 Keystone Species (top 30)')
            _tbl(ks[cols].head(30))

        net_export = _safe(net.export_network_data)
        if isinstance(net_export, tuple) and len(net_export) == 2:
            nodes_df, edges_df = net_export
            if _df_ok(nodes_df):
                _h3('6.3 All Network Nodes')
                _tbl(nodes_df)
            if _df_ok(edges_df):
                _h3('6.4 Network Edges (significant correlations)')
                _tbl(edges_df)

        if hasattr(net, 'correlation_matrix') and net.correlation_matrix is not None:
            try:
                cm = net.correlation_matrix
                top40 = cm.abs().mean().sort_values(ascending=False).head(40).index
                sub = cm.loc[top40, top40].round(3).reset_index()
                sub.rename(columns={'index': 'Taxon'}, inplace=True)
                _h3('6.5 Correlation Matrix (top 40 taxa)')
                _tbl(sub)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # 7. MACHINE LEARNING
    # ═══════════════════════════════════════════════════════════════
    if 'ml' in include and 'ml' in avail:
        _upd("Machine learning...", include.index('ml'))
        ml = avail['ml']
        _h2('7. Machine Learning Classification')

        pd_obj = getattr(ml, 'processed_data', None)
        if pd_obj and isinstance(pd_obj, dict):
            _bx(
                f"<b>Target:</b> {pd_obj.get('target_name', 'N/A')} &nbsp;|&nbsp; "
                f"<b>Samples:</b> {len(pd_obj.get('sample_names', []))} &nbsp;|&nbsp; "
                f"<b>Features:</b> {len(pd_obj.get('feature_names', []))} &nbsp;|&nbsp; "
                f"<b>Classes:</b> {', '.join(str(x) for x in pd_obj.get('class_names', []))} &nbsp;|&nbsp; "
                f"<b>Normalization:</b> {pd_obj.get('normalization', 'N/A')} &nbsp;|&nbsp; "
                f"<b>CV:</b> {pd_obj.get('cv_strategy', 'N/A')}")

        imb = _safe(ml.detect_class_imbalance)
        if imb and isinstance(imb, dict) and imb.get('counts'):
            cnt_str = ', '.join(f'{k}: {v}' for k, v in imb['counts'].items())
            _bx(f"<b>Class counts:</b> {cnt_str} &nbsp;|&nbsp; "
                f"<b>Imbalance ratio:</b> {imb.get('ratio', 1):.2f} &nbsp;|&nbsp; "
                f"<b>Severity:</b> {imb.get('severity', 'N/A')}")

        comp = _safe(ml.get_model_comparison)
        if _df_ok(comp):
            _h3('7.1 Model Comparison')
            _tbl(comp)

        for method in ('shap', 'permutation', 'builtin'):
            fi = _safe(ml.get_feature_importance, top_n=30, method=method)
            if _df_ok(fi):
                _h3(f'7.2 Top-30 Feature Importances ({method.upper()})')
                _tbl(fi, n=30)
                break

        if hasattr(ml, 'models') and ml.models:
            cn = pd_obj.get('class_names', []) if pd_obj else []
            rows_cm = []
            for mname, mdata in ml.models.items():
                cm_arr = mdata.get('confusion_matrix')
                if cm_arr is None:
                    continue
                for i, row in enumerate(cm_arr):
                    for j, val in enumerate(row):
                        rows_cm.append({
                            'Model':     mname,
                            'True':      cn[i] if i < len(cn) else i,
                            'Predicted': cn[j] if j < len(cn) else j,
                            'Count':     int(val),
                        })
            if rows_cm:
                _h3('7.3 Confusion Matrices (all models)')
                _tbl(pd.DataFrame(rows_cm))

        if hasattr(ml, 'models') and ml.models:
            metric_keys = ['test_accuracy', 'balanced_accuracy', 'f1_score',
                           'auroc', 'mcc', 'cv_mean', 'cv_std']
            rows_met = []
            for mname, mdata in ml.models.items():
                row = {'Model': mname}
                for k in metric_keys:
                    v = mdata.get(k)
                    row[k.replace('_', ' ').title()] = (
                        round(float(v), 4) if isinstance(v, (int, float)) else v)
                rows_met.append(row)
            if rows_met:
                _h3('7.4 Detailed Model Metrics')
                _tbl(pd.DataFrame(rows_met))

        lefse = _safe(ml.run_lefse_analysis, top_n=30)
        if lefse and isinstance(lefse, dict):
            ldf = lefse.get('table')
            if not _df_ok(ldf):
                ldf = lefse.get('top')
            if _df_ok(ldf):
                cols_l = [c for c in ['Feature', 'LDA_score', 'KW_pvalue',
                                       'Bonferroni_p', 'Enriched_in', 'KW_stat']
                          if c in ldf.columns]
                _h3('7.5 LEfSe Biomarker Analysis (LDA Effect Size)')
                _tbl(ldf[cols_l].head(30))

    elapsed = time.time() - t0
    parts.append(
        f'<hr style="margin-top:40px">'
        f'<p style="color:#95A5A6;text-align:center;font-size:.82em">'
        f'Generated in {elapsed:.1f}s &nbsp;|&nbsp; '
        f'BioMidas16s &nbsp;|&nbsp; '
        f'Developer: Ablaev Aziz Yakubovich</p>'
        f'</body></html>')

    full_html = '\n'.join(parts)
    progress.progress(100, text="Done!")
    st.success(f"Done in **{elapsed:.1f}s** — {len(full_html) // 1024} KB")
    st.download_button("Download HTML Report", full_html,
                       "biomidas16s_report.html", "text/html",
                       type="primary", key="dl_html")

def export_results_tab():
    st.header("Export Results")
    available = []
    for key, label in [('alpha_analyzer', 'Alpha Diversity'), ('beta_analyzer', 'Beta Diversity'),
                       ('comp_analyzer', 'Composition'), ('heat_analyzer', 'Heatmap'), ('net_analyzer', 'Network')]:
        if key in st.session_state:
            available.append(label)
    if not available:
        st.warning("No results available. Run analyses first."); return
    st.info(f"Available: {', '.join(available)}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw Data")
        loader = st.session_state.data_loader
        csv_otu = loader.otu_table.to_csv()
        st.download_button("Download OTU Table", csv_otu, "otu_table.csv", "text/csv")
        csv_meta = loader.metadata.to_csv()
        st.download_button("Download Metadata", csv_meta, "metadata.csv", "text/csv")

    with c2:
        st.subheader("Analysis Results")
        if 'alpha_analyzer' in st.session_state:
            df = st.session_state['alpha_analyzer'].get_statistics_summary()
            if not df.empty:
                st.download_button("Alpha Diversity CSV", df.to_csv(index=False),
                                  "alpha_results.csv", "text/csv")
        if 'beta_analyzer' in st.session_state:
            df = st.session_state['beta_analyzer'].get_permanova_summary()
            if not df.empty:
                st.download_button("PERMANOVA CSV", df.to_csv(index=False),
                                  "permanova_results.csv", "text/csv")

if __name__ == "__main__":
    main()
