import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* main text */
h1, h2, h3, h4, p, label, div { color: #e0e0f0; }

/* metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.metric-card .label { font-size: 0.78rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 1px; }

/* result card */
.result-card {
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
}

/* segment badge */
.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* tip box */
.tip-box {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* predict button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(102,126,234,0.55) !important;
}

/* slider track */
[data-testid="stSlider"] { padding: 4px 0; }

/* divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* info/success/warning boxes */
.stAlert { border-radius: 12px !important; }

/* hide default streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Segment definitions ────────────────────────────────────────────────────────
SEGMENTS = {
    "HIGH VALUE - Premium Customers": {
        "emoji": "💎",
        "color": "#f59e0b",
        "bg": "rgba(245,158,11,0.12)",
        "border": "#f59e0b",
        "description": "High income, high spender — your most valuable customer.",
        "tips": [
            "Offer exclusive loyalty rewards and VIP memberships.",
            "Provide personalised premium product recommendations.",
            "Prioritise with early access to new arrivals and sales.",
        ],
        "retention_risk": "Low",
        "ltv": "Very High",
    },
    "HIGH INCOME - Cautious Spenders": {
        "emoji": "🏦",
        "color": "#3b82f6",
        "bg": "rgba(59,130,246,0.12)",
        "border": "#3b82f6",
        "description": "High earning but conservative — needs the right incentive to spend.",
        "tips": [
            "Highlight value-for-money and quality assurances.",
            "Use limited-time premium bundles to trigger purchases.",
            "Build trust through reviews, warranties, and guarantees.",
        ],
        "retention_risk": "Medium",
        "ltv": "High",
    },
    "MIDDLE INCOME - Enthusiastic Shoppers": {
        "emoji": "🛒",
        "color": "#10b981",
        "bg": "rgba(16,185,129,0.12)",
        "border": "#10b981",
        "description": "Average income but loves to shop — highly engaged and loyal.",
        "tips": [
            "Run frequent promotions and flash sales.",
            "Implement a points-based rewards programme.",
            "Cross-sell complementary products during checkout.",
        ],
        "retention_risk": "Low",
        "ltv": "Medium-High",
    },
    "LOW ENGAGEMENT - Budget Conscious": {
        "emoji": "💡",
        "color": "#8b5cf6",
        "bg": "rgba(139,92,246,0.12)",
        "border": "#8b5cf6",
        "description": "Lower income and cautious spending — price-sensitive segment.",
        "tips": [
            "Focus on discount campaigns and clearance events.",
            "Offer budget-friendly product lines and bundles.",
            "Use email/push notifications for deal alerts.",
        ],
        "retention_risk": "High",
        "ltv": "Low",
    },
}


# ── Model training (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on dataset...")
def load_and_train():
    df = pd.read_csv("data/Mall_Customers.csv")
    features = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    clf = LogisticRegression(penalty="l2", solver="lbfgs", random_state=42, max_iter=1000)
    clf.fit(features_scaled, labels)

    # Build cluster → segment map from actual data
    df["Cluster"] = labels
    segment_map = {}
    for cid in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cid]
        income = sub["Annual Income (k$)"].mean()
        spending = sub["Spending Score (1-100)"].mean()
        if income > 75 and spending > 50:
            name = "HIGH VALUE - Premium Customers"
        elif income > 75 and spending <= 50:
            name = "HIGH INCOME - Cautious Spenders"
        elif income <= 75 and spending > 50:
            name = "MIDDLE INCOME - Enthusiastic Shoppers"
        else:
            name = "LOW ENGAGEMENT - Budget Conscious"
        segment_map[cid] = name

    return df, scaler, kmeans, clf, labels, segment_map


def predict_segment(income, spending, scaler, kmeans, clf, segment_map):
    x = np.array([[income, spending]])
    x_scaled = scaler.transform(x)
    cluster_id = int(kmeans.predict(x_scaled)[0])
    proba = clf.predict_proba(x_scaled)[0]
    confidence = float(proba[cluster_id]) * 100
    segment = segment_map[cluster_id]
    return cluster_id, segment, confidence, x_scaled


def build_scatter(df, labels, scaler, user_scaled, segment_map):
    df_plot = df.copy()
    df_plot["Segment"] = [segment_map[c] for c in labels]

    color_map = {v["description"][:12]: v["color"] for v in SEGMENTS.values()}
    seg_colors = {seg: SEGMENTS[seg]["color"] for seg in SEGMENTS}

    fig = px.scatter(
        df_plot,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Segment",
        color_discrete_map=seg_colors,
        opacity=0.55,
        size_max=10,
        template="plotly_dark",
        title="Customer Segments — Where Do You Fall?",
    )

    # user point
    user_orig = scaler.inverse_transform(user_scaled)[0]
    fig.add_trace(go.Scatter(
        x=[user_orig[0]], y=[user_orig[1]],
        mode="markers",
        marker=dict(size=18, color="white", symbol="star",
                    line=dict(color="#667eea", width=3)),
        name="⭐ You",
        hovertemplate="<b>Your Input</b><br>Income: $%{x}k<br>Spending: %{y}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#e0e0f0", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        title_font_size=15,
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

df, scaler, kmeans, clf, labels, segment_map = load_and_train()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 32px 0 12px;'>
  <span style='font-size:3rem;'>🛍️</span>
  <h1 style='font-size:2.4rem; font-weight:700; margin:8px 0 4px;
             background:linear-gradient(90deg,#667eea,#f59e0b);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    Mall Customer Segmentation
  </h1>
  <p style='opacity:0.55; font-size:0.95rem; margin:0;'>
    Powered by K-Means Clustering &amp; Logistic Regression
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Profile")
    st.markdown("<p style='opacity:0.5; font-size:0.8rem;'>Fill in the details below and hit Predict.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    age = st.slider("🎂 Age", min_value=18, max_value=70, value=30, step=1)
    gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    income = st.slider("💰 Annual Income (k$)", min_value=15, max_value=137, value=60, step=1,
                       help="Annual income in thousands of dollars")
    spending = st.slider("🛒 Spending Score (1–100)", min_value=1, max_value=100, value=50, step=1,
                         help="Mall-assigned score based on purchase behaviour")

    st.markdown("---")
    predict_btn = st.button("🔍  Predict My Segment", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.75rem; opacity:0.4; text-align:center;'>
    Dataset: 200 Mall Customers<br>Algorithm: K-Means (k=4)<br>Classifier: Logistic Regression
    </p>""", unsafe_allow_html=True)

# ── Dataset overview metrics ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
stats = [
    ("200", "Total Customers", "#667eea"),
    ("4", "Segments", "#f59e0b"),
    (f"${df['Annual Income (k$)'].mean():.0f}k", "Avg Income", "#10b981"),
    (f"{df['Spending Score (1-100)'].mean():.0f}", "Avg Spend Score", "#f472b6"),
]
for col, (val, lbl, clr) in zip([col1, col2, col3, col4], stats):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value" style="color:{clr};">{val}</div>
          <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📊 Cluster Map")
    default_scaled = scaler.transform([[income, spending]])
    scatter_fig = build_scatter(df, labels, scaler, default_scaled, segment_map)
    st.plotly_chart(scatter_fig, use_container_width=True)

with right:
    st.markdown("### 🧠 Prediction Result")

    if predict_btn:
        cluster_id, segment, confidence, user_scaled = predict_segment(
            income, spending, scaler, kmeans, clf, segment_map
        )
        seg_info = SEGMENTS[segment]

        # Live-update scatter with actual prediction
        with left:
            updated_fig = build_scatter(df, labels, scaler, user_scaled, segment_map)
            st.plotly_chart(updated_fig, use_container_width=True, key="updated")

        # Segment result card
        st.markdown(f"""
        <div class="result-card" style="background:{seg_info['bg']};border-color:{seg_info['border']};">
          <div class="badge" style="background:{seg_info['color']}22;color:{seg_info['color']};border:1px solid {seg_info['color']}44;">
            Cluster {cluster_id}
          </div>
          <h2 style="font-size:1.5rem;font-weight:700;margin:4px 0 8px;color:{seg_info['color']};">
            {seg_info['emoji']}  {segment}
          </h2>
          <p style="opacity:0.75;margin:0 0 16px;font-size:0.9rem;">{seg_info['description']}</p>
          <div style="display:flex;gap:24px;">
            <div>
              <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;">Confidence</div>
              <div style="font-size:1.3rem;font-weight:700;color:{seg_info['color']};">{confidence:.1f}%</div>
            </div>
            <div>
              <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;">Lifetime Value</div>
              <div style="font-size:1.3rem;font-weight:700;color:{seg_info['color']};">{seg_info['ltv']}</div>
            </div>
            <div>
              <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;">Retention Risk</div>
              <div style="font-size:1.3rem;font-weight:700;color:{seg_info['color']};">{seg_info['retention_risk']}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:14px 18px;
                    display:flex;gap:32px;margin:8px 0;">
          <div><span style="opacity:0.5;font-size:0.75rem;">AGE</span><br><b>{age}</b></div>
          <div><span style="opacity:0.5;font-size:0.75rem;">GENDER</span><br><b>{gender}</b></div>
          <div><span style="opacity:0.5;font-size:0.75rem;">INCOME</span><br><b>${income}k</b></div>
          <div><span style="opacity:0.5;font-size:0.75rem;">SPEND SCORE</span><br><b>{spending}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Marketing tips
        st.markdown("#### 💡 Marketing Recommendations")
        for tip in seg_info["tips"]:
            st.markdown(f"""
            <div class="tip-box" style="border-color:{seg_info['color']};">
              {tip}
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04);border:1px dashed rgba(255,255,255,0.15);
                    border-radius:20px;padding:48px 32px;text-align:center;margin-top:8px;">
          <div style="font-size:3rem;margin-bottom:16px;">🎯</div>
          <h3 style="font-weight:600;margin:0 0 8px;">Enter your profile</h3>
          <p style="opacity:0.45;font-size:0.88rem;margin:0;">
            Adjust the sliders in the sidebar and click<br>
            <b>Predict My Segment</b> to see your result.
          </p>
        </div>""", unsafe_allow_html=True)

# ── Segment overview table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗂️ All Customer Segments")

seg_cols = st.columns(4)
for col, (seg_name, info) in zip(seg_cols, SEGMENTS.items()):
    with col:
        sub = df.copy()
        sub["Segment"] = [segment_map[c] for c in labels]
        sub_seg = sub[sub["Segment"] == seg_name]
        count = len(sub_seg)
        avg_inc = sub_seg["Annual Income (k$)"].mean() if count else 0
        avg_sp = sub_seg["Spending Score (1-100)"].mean() if count else 0

        st.markdown(f"""
        <div class="metric-card" style="border-color:{info['border']}44;">
          <div style="font-size:2rem;">{info['emoji']}</div>
          <div style="font-size:0.8rem;font-weight:600;color:{info['color']};margin:8px 0 4px;">
            {seg_name.split(' - ')[1]}
          </div>
          <div style="font-size:0.72rem;opacity:0.5;margin-bottom:12px;">{seg_name.split(' - ')[0]}</div>
          <div style="display:flex;justify-content:space-around;font-size:0.8rem;">
            <div><b>{count}</b><br><span style="opacity:0.5;font-size:0.7rem;">customers</span></div>
            <div><b>${avg_inc:.0f}k</b><br><span style="opacity:0.5;font-size:0.7rem;">avg income</span></div>
            <div><b>{avg_sp:.0f}</b><br><span style="opacity:0.5;font-size:0.7rem;">avg score</span></div>
          </div>
        </div>""", unsafe_allow_html=True)
