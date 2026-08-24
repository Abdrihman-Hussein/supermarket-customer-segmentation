import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import run_preprocessing

st.set_page_config(
    page_title="Supermarket Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #00d4aa;
    --accent-soft: rgba(0, 212, 170, 0.12);
    --card-bg: #16161f;
    --border: #262632;
    --muted: #8b8ba3;
    --bg: #0b0b12;
    --text: #f0f0f8;
    --text-soft: #d5d5e0;
}

html, body, [class*="css"], [class*="st"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text);
}

.stApp {
    background: linear-gradient(160deg, #0b0b12 0%, #101018 100%);
}

h1, h2, h3, h4 { color: var(--text); font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }
h1 { font-weight: 800; } h2 { font-weight: 700; }

[data-testid="stSidebar"] { background: #0f0f17; border-right: 1px solid var(--border); }

div[data-testid="stMetric"] {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.1rem 1.25rem; transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,212,170,0.08); }
div[data-testid="stMetric"] label { color: var(--muted); font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: var(--text); }

.app-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,212,170,0.02));
    border: 1px solid rgba(0,212,170,0.25); border-radius: 16px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.app-header .title {
    font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #4fd1c5, #81e6d9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-header .subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 0.25rem; }

.dashboard-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.dashboard-card h4 { margin-top: 0; font-size: 1rem; font-weight: 600; }

.section-title { font-size: 1.35rem; font-weight: 700; margin: 1.75rem 0 0.5rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); color: var(--text); }

.stButton > button { background: linear-gradient(135deg, #00d4aa, #00b894); color: #0b0b12; font-weight: 600; border: none; border-radius: 10px; padding: 0.6rem 1.4rem; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,212,170,0.25); }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.6rem 1.4rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

.app-footer { text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem 0 1rem; border-top: 1px solid var(--border); margin-top: 2.5rem; }

div[data-testid="stDataFrame"], .streamlit-expanderHeader { color: var(--text); }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #00a884;
    --accent-soft: rgba(0, 168, 132, 0.10);
    --card-bg: #ffffff;
    --border: #e5e7eb;
    --muted: #6b7280;
    --bg: #f8fafc;
    --text: #111827;
    --text-soft: #374151;
}

html, body, [class*="css"], [class*="st"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text);
}

.stApp { background: linear-gradient(160deg, #f8fafc 0%, #f1f5f9 100%); }

h1, h2, h3, h4 { color: var(--text); font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }
h1 { font-weight: 800; } h2 { font-weight: 700; }

[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--border); }

div[data-testid="stMetric"] {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,168,132,0.12); }
div[data-testid="stMetric"] label { color: var(--muted); font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: var(--text); }

.app-header {
    background: linear-gradient(135deg, rgba(0,168,132,0.12), rgba(0,168,132,0.03));
    border: 1px solid rgba(0,168,132,0.25); border-radius: 16px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem; background-color: #ffffff;
}
.app-header .title {
    font-size: 1.9rem; font-weight: 800; color: #0f172a;
}
.app-header .subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 0.25rem; }

.dashboard-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.dashboard-card h4 { margin-top: 0; font-size: 1rem; font-weight: 600; }

.section-title { font-size: 1.35rem; font-weight: 700; margin: 1.75rem 0 0.5rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); color: var(--text); }

.stButton > button { background: linear-gradient(135deg, #00a884, #0d9668); color: #ffffff; font-weight: 600; border: none; border-radius: 10px; padding: 0.6rem 1.4rem; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,168,132,0.25); }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.6rem 1.4rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

.app-footer { text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem 0 1rem; border-top: 1px solid var(--border); margin-top: 2.5rem; }
</style>
"""

# ── Theme Toggle ────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    theme = st.radio("🌓 Display Mode", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
    st.session_state.theme = theme

st.markdown(DARK_CSS if st.session_state.theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    X, feature_names, df, preprocessor = run_preprocessing()
    return X, feature_names, df, preprocessor


X, feature_names, df, preprocessor = load_data()


def compute_clusters(k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def compute_purity(labels, true_labels):
    if true_labels is None or len(true_labels) == 0:
        return 0.0
    ev = pd.DataFrame({'cluster': labels, 'true': true_labels})
    n = len(ev)
    pur = 0
    for i in range(labels.max() + 1):
        cluster_i = ev[ev['cluster'] == i]
        majority = cluster_i['true'].value_counts().iloc[0]
        pur += majority / n
    return pur / (labels.max() + 1)


# ── Sidebar (rest) ──────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    k_value = st.selectbox(
        "🧩 Number of Clusters",
        options=[2, 3, 4],
        index=0,
        help="Choose how many customer segments to create"
    )
    show_scatter = st.checkbox("📈 Show scatter analysis", value=True)
    show_products = st.checkbox("🛍️ Show product insights", value=True)
    show_comparison = st.checkbox("⚖️ Show K comparison", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    **Supermarket Customer Segmentation**
    
    Unsupervised K-Means clustering on 1000 transactions.
    
    Features: spend, quantity, price, rating, branch, payment, product line.
    """)

# ── Header ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="title">🛒 Supermarket Customer Segmentation</div>
    <div class="subtitle">Unsupervised ML · K-Means Clustering · 1,000 Transactions · 3 Branches</div>
</div>
""", unsafe_allow_html=True)

# ── Train model ─────────────────────────────────────────
kmeans, labels = compute_clusters(k_value)
df_with = df.copy()
df_with['cluster'] = labels

sil_score = silhouette_score(X, labels)
cluster_stats = df_with.groupby('cluster').mean(numeric_only=True)
true_labels = df_with.get('Customer type', None)
purity = compute_purity(labels, true_labels.values if hasattr(true_labels, 'values') else (true_labels if true_labels is not None else []))

cluster_counts = df_with['cluster'].value_counts().sort_index()
total_n = len(df_with)

# ── KPI Metrics ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Silhouette Score", f"{sil_score:.4f}")
col2.metric("👥 Total Transactions", f"{total_n:,}")
col3.metric("🎯 Cluster Purity", f"{purity:.1%}")
col4.metric("🧩 Segments", f"{k_value}")

# ── Tabs ────────────────────────────────────────────────
tab_overview, tab_clusters, tab_insights, tab_data = st.tabs(
    ["📊 Overview", "🧑‍🤝‍🧑 Cluster Profiles", "📈 Insights", "🗄️ Data"]
)

with tab_overview:
    st.markdown('<div class="section-title">Segment Distribution</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        pie = px.pie(
            values=cluster_counts.values,
            names=[f"Segment {i}" for i in cluster_counts.index],
            title=f"Transaction Distribution (k={k_value})",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.45
        )
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
            title_font_size=16
        )
        st.plotly_chart(pie, use_container_width=True)

    with col_right:
        bar_data = pd.DataFrame({
            'Segment': [f"Segment {i}" for i in cluster_counts.index],
            'Avg Total ($)': [cluster_stats.loc[i, 'Total'] if i in cluster_stats.index else 0 for i in cluster_counts.index],
            'Transactions': cluster_counts.values
        })
        bar = px.bar(
            bar_data, x='Segment', y='Avg Total ($)', color='Segment',
            text='Avg Total ($)', color_discrete_sequence=px.colors.qualitative.Set2,
            title="Average Transaction Value by Segment"
        )
        bar.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
            title_font_size=16, showlegend=False
        )
        st.plotly_chart(bar, use_container_width=True)

    st.markdown('<div class="section-title">Segment vs Customer Type</div>', unsafe_allow_html=True)
    if 'Customer type' in df_with.columns:
        crosstab = pd.crosstab(df_with['cluster'], df_with['Customer type'])
        heat = px.imshow(
            crosstab, text_auto=True, aspect="auto",
            color_continuous_scale="Tealgrn",
            title="Cluster vs Member/Normal Alignment"
        )
        heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
            title_font_size=16
        )
        st.plotly_chart(heat, use_container_width=True)

with tab_clusters:
    st.markdown('<div class="section-title">Customer Segment Profiles</div>', unsafe_allow_html=True)

    cols = st.columns(min(k_value, 3))
    for idx, c in enumerate(cluster_counts.index):
        with cols[idx % 3]:
            stats = cluster_stats.loc[c]
            size = cluster_counts[c]
            pct = size / total_n * 100
            st.markdown(f"""
            <div class="dashboard-card" style="border-top: 3px solid #00d4aa;">
                <h4>Segment {c} <span style="font-size:0.75rem;color:var(--muted);">· {pct:.1f}%</span></h4>
                <p style="color:var(--muted);font-size:0.82rem;margin-bottom:0.75rem;">
                    {size:,} transactions
                </p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.9rem;color:var(--text);">
                    <div><span style="color:var(--muted)">Avg Total</span><br><b>${stats['Total']:.2f}</b></div>
                    <div><span style="color:var(--muted)">Avg Items</span><br><b>{stats['Quantity']:.1f}</b></div>
                    <div><span style="color:var(--muted)">Avg Price</span><br><b>${stats['Unit price']:.2f}</b></div>
                    <div><span style="color:var(--muted)">Avg Rating</span><br><b>{stats['Rating']:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Detailed Comparison Table</div>', unsafe_allow_html=True)
    compare = cluster_stats[['Unit price', 'Quantity', 'Total', 'cogs', 'gross income', 'Rating']].copy()
    compare = compare.round(2)
    compare.index = [f"Segment {i}" for i in compare.index]
    compare.columns = ['Unit Price', 'Quantity', 'Total', 'COGS', 'Gross Income', 'Rating']
    st.dataframe(compare, use_container_width=True)

with tab_insights:
    if show_products and 'Product line' in df_with.columns:
        st.markdown('<div class="section-title">Product Preferences by Segment</div>', unsafe_allow_html=True)
        prod_counts = pd.crosstab(df_with['cluster'], df_with['Product line'])
        fig_prod = px.bar(
            prod_counts, barmode='group',
            title="Product Line Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_prod.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
            title_font_size=16, legend_title_text="Product Line"
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    if show_scatter:
        st.markdown('<div class="section-title">Spending Patterns</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            scat = px.scatter(
                df_with, x='Quantity', y='Total',
                color=df_with['cluster'].astype(str), size='Unit price', opacity=0.7,
                title="Quantity vs Total Spend", labels={'cluster': 'Segment'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
                title_font_size=16
            )
            st.plotly_chart(scat, use_container_width=True)

        with col_b:
            hist = px.histogram(
                df_with, x='Total', color=df_with['cluster'].astype(str), nbins=40,
                title="Total Spend Distribution", labels={'color': 'Segment'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
                title_font_size=16
            )
            st.plotly_chart(hist, use_container_width=True)

    if show_comparison:
        st.markdown('<div class="section-title">Silhouette Comparison (k values)</div>', unsafe_allow_html=True)
        sil_scores = []
        for k in [2, 3, 4, 5, 6]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(X)
            sil_scores.append(silhouette_score(X, labs))
        cmp_fig = px.line(
            x=[2, 3, 4, 5, 6], y=sil_scores, markers=True,
            title="Silhouette Score vs Number of Clusters",
            labels={'x': 'k (clusters)', 'y': 'Silhouette Score'}
        )
        cmp_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f8" if st.session_state.theme == "Dark" else "#111827"),
            title_font_size=16
        )
        st.plotly_chart(cmp_fig, use_container_width=True)

with tab_data:
    st.markdown('<div class="section-title">Processed Dataset with Cluster Labels</div>', unsafe_allow_html=True)
    st.dataframe(df_with.head(500), use_container_width=True)
    st.caption("Showing first 500 rows of 1,000 transactions")

st.markdown("""
<div class="app-footer">
    Supermarket Customer Segmentation · K-Means Clustering · Silhouette {sil} · Purity {pur}
</div>
""".format(sil=f"{sil_score:.4f}", pur=f"{purity:.1%}"), unsafe_allow_html=True)
