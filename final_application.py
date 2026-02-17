import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

import google.generativeai as genai
st.set_page_config(
    page_title="ML Inspector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
    .main { background-color: #070B14 !important; }
    
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #0A0F1A 100%) !important;
        border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
    }
    [data-testid="stSidebar"] * { color: #CBD5E0 !important; }
    
    /* ── Nav Menu Override ── */
    .nav-link { border-radius: 10px !important; margin: 2px 0 !important; }
    .nav-link-selected {
        background: linear-gradient(135deg, #1A3A5C, #1E4976) !important;
        border-left: 3px solid #63B3ED !important;
        color: #E2F0FF !important;
    }
    .nav-link:hover { background: rgba(99, 179, 237, 0.08) !important; }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0F1923 0%, #131E2E 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 14px;
        padding: 16px 20px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(99,179,237,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,179,237,0.15);
    }
    div[data-testid="stMetricLabel"] > div { color: #6B8BAE !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] > div { color: #63B3ED !important; font-size: 28px !important; font-weight: 700 !important; font-family: 'JetBrains Mono' !important; }
    div[data-testid="stMetricDelta"] > div { color: #68D391 !important; font-size: 12px !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1A3A5C, #1E4976) !important;
        color: #E2F0FF !important;
        border: 1px solid rgba(99, 179, 237, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk' !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.2s ease !important;
        padding: 0.4rem 1.2rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E4976, #2B5F94) !important;
        border-color: rgba(99, 179, 237, 0.6) !important;
        box-shadow: 0 0 20px rgba(99, 179, 237, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="baseButton-primary"] > button, button[kind="primary"] {
        background: linear-gradient(135deg, #2B6CB0, #2C5282) !important;
        box-shadow: 0 0 20px rgba(99, 179, 237, 0.3) !important;
    }

    /* ── Page Headers ── */
    .page-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #0F2035 100%);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .page-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #63B3ED, transparent);
    }
    .page-header h1 {
        color: #E2F0FF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    .page-header p {
        color: #6B8BAE !important;
        font-size: 14px !important;
        margin: 6px 0 0 0 !important;
    }

    /* ── Cards / Containers ── */
    .card {
        background: linear-gradient(135deg, #0F1923 0%, #111824 100%);
        border: 1px solid rgba(99, 179, 237, 0.12);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-accent {
        border-left: 3px solid #63B3ED;
    }

    /* ── Stat Pill ── */
    .stat-pill {
        display: inline-block;
        background: rgba(99, 179, 237, 0.1);
        color: #63B3ED;
        border: 1px solid rgba(99, 179, 237, 0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 3px;
    }

    /* ── Risk Badge ── */
    .risk-critical { background: rgba(245,101,101,0.15); color: #FC8181; border: 1px solid rgba(245,101,101,0.3); border-radius: 8px; padding: 8px 16px; font-weight: 700; }
    .risk-high     { background: rgba(237,137,54,0.15);  color: #F6AD55; border: 1px solid rgba(237,137,54,0.3);  border-radius: 8px; padding: 8px 16px; font-weight: 700; }
    .risk-medium   { background: rgba(214,158,46,0.15);  color: #ECC94B; border: 1px solid rgba(214,158,46,0.3);  border-radius: 8px; padding: 8px 16px; font-weight: 700; }
    .risk-low      { background: rgba(72,187,120,0.15);  color: #68D391; border: 1px solid rgba(72,187,120,0.3);  border-radius: 8px; padding: 8px 16px; font-weight: 700; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden; }
    iframe { border-radius: 12px !important; }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2B6CB0, #63B3ED) !important;
        border-radius: 4px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 25, 35, 0.8) !important;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #6B8BAE !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1A3A5C, #1E4976) !important;
        color: #E2F0FF !important;
    }

    /* ── Selectbox / Sliders ── */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #0F1923 !important;
        border: 1px solid rgba(99, 179, 237, 0.2) !important;
        border-radius: 10px !important;
        color: #CBD5E0 !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #2B6CB0, #63B3ED) !important;
    }

    /* ── Chat ── */
    .stChatMessage {
        background: #0F1923 !important;
        border: 1px solid rgba(99, 179, 237, 0.12) !important;
        border-radius: 14px !important;
        padding: 14px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatInput"] {
        background: #0F1923 !important;
        border: 1px solid rgba(99, 179, 237, 0.25) !important;
        border-radius: 12px !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(99, 179, 237, 0.1) !important; }

    /* ── Info/Warning/Success boxes ── */
    .stAlert { border-radius: 10px !important; border-left-width: 3px !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #070B14; }
    ::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2B6CB0; }

    /* ── Section Label ── */
    .section-label {
        color: #6B8BAE;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    /* ── Tournament Table ── */
    .gold   { color: #ECC94B !important; font-weight: 700; }
    .silver { color: #CBD5E0 !important; font-weight: 700; }
    .bronze { color: #F6AD55 !important; font-weight: 700; }

    /* ── Sidebar logo area ── */
    .logo-area {
        padding: 12px 8px 20px 8px;
        border-bottom: 1px solid rgba(99, 179, 237, 0.1);
        margin-bottom: 12px;
        text-align: center;
    }
    .logo-title { color: #E2F0FF !important; font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
    .logo-sub   { color: #6B8BAE !important; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }

    /* ── Simulation widget ── */
    .sim-result-box {
        background: linear-gradient(135deg, #0D1B2A, #0F2035);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
    }
    .sim-pred-value { font-size: 48px; font-weight: 700; font-family: 'JetBrains Mono'; color: #63B3ED; }
    .sim-label { color: #6B8BAE; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }

    /* ── Hover animation for feature pills ── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 rgba(99,179,237,0); }
        50% { box-shadow: 0 0 12px rgba(99,179,237,0.3); }
    }
    
    /* ── Input file ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(99, 179, 237, 0.25) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        background: rgba(15, 25, 35, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_dark"

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(10,15,26,0)',
    plot_bgcolor='rgba(10,15,26,0)',
    font=dict(family='Space Grotesk', color='#CBD5E0'),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='rgba(99,179,237,0.08)', linecolor='rgba(99,179,237,0.15)'),
    yaxis=dict(gridcolor='rgba(99,179,237,0.08)', linecolor='rgba(99,179,237,0.15)'),
)

def apply_layout(fig):
    fig.update_layout(**CHART_LAYOUT)
    return fig

#LLM
def get_gemini_chat_response(history, context):
    """Call Gemini and return a response string. Never cached."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            return (
                "**API Key Missing** — please add your Gemini key.\n\n"
                "Create `.streamlit/secrets.toml` and add:\n"
                "```toml\nGEMINI_API_KEY = 'your_key_here'\n```\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=api_key)
        valid_model = "gemini-1.5-flash"
        try:
            for m_item in genai.list_models():
                if 'generateContent' in m_item.supported_generation_methods:
                    if 'flash' in m_item.name.lower():
                        valid_model = m_item.name
                        break
        except Exception:
            pass

        model_obj = genai.GenerativeModel(valid_model)

        sys_prompt = (
            "You are InspectorML, a Senior Data Scientist and ML Governance Expert "
            "embedded inside InspectorML Pro.\n\n"
            f"CURRENT APP CONTEXT:\n{context}\n\n"
            "YOUR ROLE:\n"
            "- Explain ML results, metrics, and simulation outcomes clearly\n"
            "- Reference actual numbers from the context when they exist\n"
            "- Give actionable, practical recommendations\n"
            "- Be concise (under 300 words unless the question is complex)\n\n"
            "FORMAT: Use markdown. Use bullet points for lists. Use **bold** for key terms."
        )
        gemini_history = []
        for msg in history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat_session = model_obj.start_chat(history=gemini_history)
        full_prompt = f"[SYSTEM]\n{sys_prompt}\n\n[USER QUESTION]\n{history[-1]['content']}"
        response = chat_session.send_message(full_prompt)
        return response.text

    except Exception as e:
        err = str(e)
        if "API_KEY" in err.upper() or "credential" in err.lower() or "invalid" in err.lower():
            return "**Invalid API Key** — check your Gemini key at https://aistudio.google.com/app/apikey"
        elif "quota" in err.lower() or "429" in err:
            return "⏱**Rate Limited** — too many requests, please wait a moment then try again."
        else:
            return f"**AI Error**: {err}"


#session state
defaults = {
    "df_cleaned": None, "model": None, "sim_params": {},
    "chat_history": [], "sim_result_context": "No simulation run yet.",
    "model_type": None, "feature_names": [], "target_name": None,
    "X_test": None, "y_test": None, "train_score": None,
    "scaler": None, "gov_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-title">InspectorML Pro</div>
        <div class="logo-sub">ML Behavior Inspector</div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=["Upload & Clean", "EDA & Outliers", "Model Training", "Evaluation", "Auto-Governance", "Simulation Lab", "AI Consultant"],
        icons=["cloud-upload", "bar-chart-line", "cpu", "check-circle", "trophy", "virus2", "robot"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background": "transparent"},
            "icon": {"color": "#63B3ED", "font-size": "15px"},
            "nav-link": {"font-size": "13px", "color": "#8BA3BE", "padding": "10px 14px", "font-weight": "500"},
            "nav-link-selected": {"background": "linear-gradient(135deg,#1A3A5C,#1E4976)", "color": "#E2F0FF", "font-weight": "600"},
        }
    )

    st.markdown("---")
    st.markdown('<div class="section-label">📂 Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file and st.session_state.df_cleaned is None:
        st.session_state.df_cleaned = pd.read_csv(uploaded_file)

    if st.session_state.df_cleaned is not None:
        df_info = st.session_state.df_cleaned
        missing_count = int(df_info.isnull().sum().sum())
        missing_color = '#FC8181' if missing_count > 0 else '#68D391'
        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:12px;margin-top:8px;">
            <div style="color:#6B8BAE;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Loaded Dataset</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#CBD5E0;font-size:12px;">Rows</span>
                <span style="color:#63B3ED;font-weight:600;font-size:12px;font-family:'JetBrains Mono'">{df_info.shape[0]:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#CBD5E0;font-size:12px;">Columns</span>
                <span style="color:#63B3ED;font-weight:600;font-size:12px;font-family:'JetBrains Mono'">{df_info.shape[1]}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#CBD5E0;font-size:12px;">Missing</span>
                <span style="color:{missing_color};font-weight:600;font-size:12px;font-family:'JetBrains Mono'">{missing_count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.model is not None:
        score_val = st.session_state.train_score
        score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
        model_name_str = type(st.session_state.model).__name__
        model_type_str = st.session_state.model_type or "Unknown"
        st.markdown(f"""
        <div style="background:rgba(72,187,120,0.06);border:1px solid rgba(72,187,120,0.2);border-radius:10px;padding:12px;margin-top:10px;">
            <div style="color:#6B8BAE;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Active Model</div>
            <div style="color:#68D391;font-size:12px;font-weight:600;">✓ {model_name_str}</div>
            <div style="color:#CBD5E0;font-size:11px;margin-top:2px;">{model_type_str} • Score: {score_str}</div>
        </div>
        """, unsafe_allow_html=True)

# P-1:upload and clean
if selected_page == "Upload & Clean":
    st.markdown("""
    <div class="page-header">
        <h1>☁️ Data Setup Center</h1>
        <p>Upload your dataset, inspect quality, and prepare it for analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum(),
                    delta="Clean ✓" if df.isnull().sum().sum() == 0 else f"In {df.isnull().any().sum()} cols")
        col4.metric("Duplicates", df.duplicated().sum())

        st.markdown('<div class="section-label" style="margin-top:20px;">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=280)

        st.markdown('<div class="section-label" style="margin-top:20px;">Data Operations</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Auto-Clean (Mean/Mode)", use_container_width=True):
                df_temp = df.copy()
                for c in df_temp.columns:
                    if df_temp[c].dtype == 'object':
                        df_temp[c] = df_temp[c].fillna(df_temp[c].mode()[0] if not df_temp[c].mode().empty else 'Unknown')
                    else:
                        df_temp[c] = df_temp[c].fillna(df_temp[c].mean())
                    df_temp = df_temp.drop_duplicates()
                st.session_state.df_cleaned = df_temp
                st.success(f"✓ Cleaned! {df.isnull().sum().sum()} missing values filled, {df.duplicated().sum()} duplicates removed.")
                st.rerun()
        with c2:
            if st.button("🔄 Reset Dataset", use_container_width=True):
                st.session_state.df_cleaned = pd.read_csv(uploaded_file) if uploaded_file else None
                st.rerun()
        with c3:
            cols_to_drop = st.multiselect("Select columns to drop", df.columns, label_visibility="collapsed",
                                           placeholder="Select columns to drop...")
            if cols_to_drop and st.button(f"Drop {len(cols_to_drop)} column(s)", use_container_width=True):
                st.session_state.df_cleaned = df.drop(columns=cols_to_drop)
                st.success(f"✓ Dropped: {', '.join(cols_to_drop)}")
                st.rerun()

        # Column type summary
        st.markdown('<div class="section-label" style="margin-top:20px;">Column Overview</div>', unsafe_allow_html=True)
        col_info = []
        for col in df.columns:
            col_info.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": df[col].count(),
                "Null %": f"{df[col].isnull().mean()*100:.1f}%",
                "Unique": df[col].nunique(),
                "Sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,25,35,0.5);border:1px dashed rgba(99,179,237,0.2);border-radius:16px;">
            <div style="font-size:48px;margin-bottom:16px;">📁</div>
            <div style="color:#E2F0FF;font-size:18px;font-weight:600;margin-bottom:8px;">No Dataset Loaded</div>
            <div style="color:#6B8BAE;font-size:14px;">Upload a CSV file from the sidebar to get started</div>
        </div>
        """, unsafe_allow_html=True)


# p2:eda
if selected_page == "EDA & Outliers":
    st.markdown("""
    <div class="page-header">
        <h1>Exploratory Analysis</h1>
        <p>Visualize distributions, correlations, and outliers data</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_cleaned
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Stats", "Correlations", "Outliers", "Pair Plot", "Violin & Balance"])

    with tab1:
        c1, c2 = st.columns([1, 3])
        with c1:
            feat = st.selectbox("Select Feature", df.columns)
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:12px;">
                <div class="section-label">Stats</div>
                <div style="font-family:'JetBrains Mono';font-size:12px;color:#CBD5E0;line-height:1.8;">
                    {''.join([f'<div><span style="color:#6B8BAE">{k}:</span> <span style="color:#63B3ED">{v}</span></div>'
                    for k, v in df[feat].describe().round(3).items()])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            
            chart_colors = ['#7B61FF', '#00D4FF', '#FF6B9D', '#00FFB2', '#FFB347', '#FF4757', '#2ED573']
            feat_idx = list(df.columns).index(feat) % len(chart_colors)
            feat_color = chart_colors[feat_idx]
            fig = px.histogram(df, x=feat, marginal="box", color_discrete_sequence=[feat_color],
                               title=f"Distribution of {feat}")
            fig.update_traces(marker_line_width=0, opacity=0.85)
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Feature Importance / Correlation</div>', unsafe_allow_html=True)

        shown = False
        if st.session_state.model and st.session_state.feature_names:
            model = st.session_state.model
            cols = st.session_state.feature_names
            imps = None
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            if imps is not None and len(imps) == len(cols):
                imp_df = pd.DataFrame({"Feature": cols, "Importance": imps}).sort_values("Importance", ascending=True).tail(15)
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                             title="Model Feature Importance",
                             color="Importance", color_continuous_scale=["#7B61FF", "#00D4FF", "#00FFB2"])
                apply_layout(fig)
                fig.update_coloraxes(showscale=True, colorbar=dict(tickfont=dict(color='#CBD5E0')))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                shown = True

        if not shown:
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                t = st.selectbox("Target for Correlation", num_df.columns, index=len(num_df.columns)-1)
                corr = num_df.corr()[t].drop(t).sort_values()
                colors = ['#FC8181' if v < 0 else '#63B3ED' for v in corr.values]
                fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation='h',
                                       marker_color=colors, opacity=0.9))
                fig.update_layout(title=f"Correlation with {t}", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            corr_matrix = num_df.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f",
                            color_continuous_scale=["#FF4757", "#0F1923", "#7B61FF"],
                            title="Feature Correlation Heatmap",
                            zmin=-1, zmax=1)
            apply_layout(fig)
            fig.update_layout(height=520)
            fig.update_traces(textfont=dict(size=10, color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab3:
        st.markdown('<div class="section-label">Outlier Inspector</div>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            c1, c2 = st.columns([1, 3])
            with c1:
                col_sel = st.selectbox("Select Column", num_cols)
                q1 = df[col_sel].quantile(0.25)
                q3 = df[col_sel].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col_sel] < q1 - 1.5*iqr) | (df[col_sel] > q3 + 1.5*iqr)]
                outlier_color = '#FC8181' if len(outliers) > 0 else '#68D391'
                outlier_pct = len(outliers)/len(df)*100
                st.markdown(f"""
                <div class="card card-accent" style="margin-top:12px;">
                    <div class="section-label">IQR Analysis</div>
                    <div style="font-family:'JetBrains Mono';font-size:12px;color:#CBD5E0;line-height:2.0;">
                        <div><span style="color:#6B8BAE">Q1:</span> <span style="color:#63B3ED">{q1:.2f}</span></div>
                        <div><span style="color:#6B8BAE">Q3:</span> <span style="color:#63B3ED">{q3:.2f}</span></div>
                        <div><span style="color:#6B8BAE">IQR:</span> <span style="color:#63B3ED">{iqr:.2f}</span></div>
                        <div><span style="color:#6B8BAE">Outliers:</span> <span style="color:{outlier_color}">{len(outliers)} rows</span></div>
                        <div><span style="color:#6B8BAE">Outlier %:</span> <span style="color:#ECC94B">{outlier_pct:.1f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                fig_box = px.box(df, y=col_sel, points="outliers", title=f"Box Plot: {col_sel}",
                                 color_discrete_sequence=['#7B61FF'])
                fig_box.update_traces(
                    marker=dict(size=7, color='#FF4757', symbol='x', opacity=0.85,
                                line=dict(width=2, color='#FF4757')),
                    fillcolor='rgba(123,97,255,0.2)',
                    line=dict(color='#7B61FF', width=2),
                    whiskerwidth=0.5
                )
                apply_layout(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)
            fig_scatter = px.scatter(df, y=col_sel, title=f"Scatter Spread: {col_sel}",
                                     color_discrete_sequence=['#00D4FF'], opacity=0.4)
            if len(outliers) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=outliers.index, y=outliers[col_sel],
                    mode='markers', name=f'Outliers ({len(outliers)})',
                    marker=dict(color='#FF4757', size=9, symbol='x', line=dict(width=2, color='#FF4757'))
                ))
            apply_layout(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)


    with tab4:
        st.markdown("""
        <div class="card" style="margin-bottom:16px;">
            <div style="color:#E2F0FF;font-weight:600;margin-bottom:4px;">Scatter Matrix (Pair Plot)</div>
            <div style="color:#6B8BAE;font-size:13px;">Shows relationships between all selected numeric features simultaneously. 
            Diagonal shows distributions; off-diagonal shows feature-vs-feature scatter. 
            Strong diagonal patterns = linear relationships.</div>
        </div>
        """, unsafe_allow_html=True)

        num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_all) >= 2:
            c1, c2 = st.columns([1, 3])
            with c1:
                max_feats = min(6, len(num_cols_all))
                selected_feats = st.multiselect(
                    "Select features (max 6)",
                    num_cols_all,
                    default=num_cols_all[:min(4, len(num_cols_all))],
                    max_selections=6
                )
                color_by = None
                if st.session_state.target_name and st.session_state.target_name in df.columns:
                    tgt = df[st.session_state.target_name]
                    if tgt.dtype == 'object' or tgt.nunique() <= 10:
                        color_by = st.session_state.target_name
                        st.success(f"✓ Colored by: {color_by}")
                    else:
                        st.info("Target is continuous — showing uncolored.")

            with c2:
                if len(selected_feats) >= 2:
                    plot_df = df[selected_feats + ([color_by] if color_by and color_by not in selected_feats else [])].dropna()
                    fig_pair = px.scatter_matrix(
                        plot_df,
                        dimensions=selected_feats,
                        color=color_by if color_by else None,
                        color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D', '#FFB347', '#00D4FF', '#FF4757'],
                        title="Pair Plot — Feature Relationships",
                        opacity=0.6
                    )
                    fig_pair.update_traces(
                        diagonal_visible=True,
                        showupperhalf=True,
                        marker=dict(size=3)
                    )
                    apply_layout(fig_pair)
                    fig_pair.update_layout(height=560)
                    st.plotly_chart(fig_pair, use_container_width=True)
                else:
                    st.warning("Select at least 2 features.")
        else:
            st.info("Need at least 2 numeric columns for a pair plot.")

    
    with tab5:
        num_cols_v = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_v = df.select_dtypes(include='object').columns.tolist()

        st.markdown('<div class="section-label">Violin Plot</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="margin-bottom:12px;">
            <div style="color:#6B8BAE;font-size:13px;">Violin plots show the full distribution shape (like a KDE) combined with a box plot. 
            Wider sections = more data points at that value. Great for comparing distributions across categories.</div>
        </div>
        """, unsafe_allow_html=True)

        cv1, cv2 = st.columns(2)
        with cv1:
            violin_feat = st.selectbox("Numeric Feature (Y-axis)", num_cols_v, key="violin_feat")
        with cv2:
            violin_group = st.selectbox(
                "Group by (X-axis, optional)",
                ["None"] + cat_cols_v,
                key="violin_group"
            )

        if violin_group == "None":
            fig_vio = px.violin(df, y=violin_feat, box=True, points="outliers",
                                title=f" Violin: {violin_feat}",
                                color_discrete_sequence=['#7B61FF'])
        else:
            
            fig_vio = px.violin(df, y=violin_feat, x=violin_group, box=True, points="outliers",
                                title=f"{violin_feat} by {violin_group}",
                                color=violin_group,
                                color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D',
                                                          '#FFB347', '#00D4FF', '#FF4757'])

        fig_vio.update_traces(meanline_visible=True)
        apply_layout(fig_vio)
        fig_vio.update_layout(height=420)
        st.plotly_chart(fig_vio, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Class / Value Balance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="margin-bottom:12px;">
            <div style="color:#6B8BAE;font-size:13px;">Shows how balanced your target variable is. 
            For classification: class imbalance can heavily bias model predictions. 
            For regression: the distribution of the target value across the data.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.target_name and st.session_state.target_name in df.columns:
            tgt_col = st.session_state.target_name
            tgt_series = df[tgt_col]

            cb1, cb2 = st.columns(2)
            with cb1:
                if tgt_series.dtype == 'object' or tgt_series.nunique() <= 15:
                    counts = tgt_series.value_counts().reset_index()
                    counts.columns = ["Class", "Count"]
                    fig_pie = px.pie(counts, values="Count", names="Class",
                                     title=f"Class Balance: {tgt_col}",
                                     color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D',
                                                               '#FFB347', '#00D4FF', '#FF4757', '#2ED573'],
                                     hole=0.45)
                    fig_pie.update_traces(textposition='outside',
                                          textinfo='percent+label',
                                          textfont=dict(size=12, color='white'))
                    apply_layout(fig_pie)
                    fig_pie.update_layout(height=380, showlegend=True)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    fig_hist_tgt = px.histogram(df, x=tgt_col, nbins=40,
                                                 title=f"Target Distribution: {tgt_col}",
                                                 color_discrete_sequence=['#7B61FF'])
                    fig_hist_tgt.update_traces(marker_line_width=0, opacity=0.85)
                    apply_layout(fig_hist_tgt)
                    fig_hist_tgt.update_layout(height=380)
                    st.plotly_chart(fig_hist_tgt, use_container_width=True)

            with cb2:
                st.markdown('<div class="section-label" style="margin-top:4px;">Missing Value Map</div>', unsafe_allow_html=True)
                null_df = df.isnull().astype(int)
                if null_df.sum().sum() > 0:
                    
                    sample_null = null_df.sample(min(200, len(null_df)), random_state=42)
                    fig_null = px.imshow(sample_null.T,
                                         color_continuous_scale=["#0F1923", "#FF4757"],
                                         title="Missing Value Heatmap",
                                         aspect="auto")
                    apply_layout(fig_null)
                    fig_null.update_layout(height=380)
                    fig_null.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_null, use_container_width=True)
                else:
                    st.success("No missing values detected in the dataset!")
                    dtype_counts = df.dtypes.value_counts().reset_index()
                    dtype_counts.columns = ["Type", "Count"]
                    dtype_counts["Type"] = dtype_counts["Type"].astype(str)
                    fig_dtype = px.bar(dtype_counts, x="Type", y="Count",
                                        title="Column Data Types",
                                        color="Count",
                                        color_continuous_scale=["#7B61FF", "#00FFB2"])
                    apply_layout(fig_dtype)
                    fig_dtype.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_dtype, use_container_width=True)
        else:
            st.info("Train a model first to see the target variable's class balance.")


#p3:model training
if selected_page == "Model Training":
    st.markdown("""
    <div class="page-header">
        <h1>Model Training Studio</h1>
        <p>Configure and train machine learning models on your dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_cleaned.copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Target & Problem Type</div>', unsafe_allow_html=True)
        target = st.selectbox("Target Variable", df.columns)
        dtype_str = str(df[target].dtype)
        n_unique = df[target].nunique()
        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.06);border-radius:8px;padding:10px 14px;margin:8px 0;">
            <span class="stat-pill">{dtype_str}</span>
            <span class="stat-pill">{n_unique} unique values</span>
            <span class="stat-pill">{df[target].isnull().sum()} nulls</span>
        </div>
        """, unsafe_allow_html=True)
        p_type = st.radio("Problem Type", ["Classification", "Regression"], horizontal=True)

    with c2:
        st.markdown('<div class="section-label">Algorithm Selection</div>', unsafe_allow_html=True)
        if p_type == "Classification":
            model_name = st.selectbox("Algorithm", [
                "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (SVC)", "Neural Network (MLP)"
            ])
        else:
            model_name = st.selectbox("Algorithm", [
                "Linear Regression", "Random Forest", "Gradient Boosting", "SVM (SVR)", "Neural Network (MLP)"
            ])

        val_method = st.radio("Validation Method", ["Train-Test Split", "K-Fold Cross-Validation"], horizontal=True)
        if val_method == "Train-Test Split":
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5,
                                  help="Percentage of data reserved for testing")
            st.caption(f"Using {test_size}% test data → {100-test_size}% training data")
            test_size = test_size / 100.0 
        else:
            k_folds = st.slider("K Folds", 3, 10, 5)

    if st.button("Start Training", type="primary", use_container_width=True):
        with st.spinner("Preparing data and training model..."):
            prog = st.progress(0)
            X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
            y = df[target].copy()
            if p_type == "Classification" and y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            prog.progress(20)

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            st.session_state.scaler = scaler
            st.session_state.feature_names = X.columns.tolist()
            prog.progress(40)

            if p_type == "Classification":
                m_map = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "SVM (SVC)": SVC(probability=True),
                    "Neural Network (MLP)": MLPClassifier(max_iter=500)
                }
            else:
                m_map = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "SVM (SVR)": SVR(),
                    "Neural Network (MLP)": MLPRegressor(max_iter=500)
                }
            m = m_map[model_name]
            prog.progress(60)

            if val_method == "Train-Test Split":
                X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=test_size, random_state=42)
                m.fit(X_tr, y_tr)
                score = m.score(X_te, y_te)
                st.session_state.X_test = X_te
                st.session_state.y_test = y_te
            else:
                score = cross_val_score(m, X_sc, y, cv=k_folds).mean()
                m.fit(X_sc, y)
                st.session_state.X_test = X_sc
                st.session_state.y_test = y

            prog.progress(90)
            st.session_state.model = m
            st.session_state.model_type = p_type
            st.session_state.train_score = score
            st.session_state.target_name = target
            prog.progress(100)

        label = "Accuracy" if p_type == "Classification" else "R² Score"
        color = "#68D391" if score > 0.8 else "#ECC94B" if score > 0.6 else "#FC8181"
        st.markdown(f"""
        <div style="background:rgba(15,25,35,0.8);border:1px solid {color}33;border-radius:14px;padding:24px;text-align:center;margin-top:16px;">
            <div style="color:#6B8BAE;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Training Complete — {label}</div>
            <div style="font-size:48px;font-weight:700;font-family:'JetBrains Mono';color:{color};">{score:.4f}</div>
            <div style="color:#6B8BAE;font-size:13px;margin-top:8px;">{model_name} • {val_method}</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# p4:evaluation
if selected_page == "Evaluation":
    st.markdown("""
    <div class="page-header">
        <h1>Performance Dashboard</h1>
        <p>Detailed evaluation metrics, confusion matrix, and diagnostic plots</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Please train a model first.")
        st.stop()

    m = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = m.predict(X_test)

    if st.session_state.model_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2%}", delta=f"{'Good' if acc > 0.8 else 'Fair'}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")

        t1, t2, t3 = st.tabs(["Confusion Matrix", "ROC Curve", "Class Report"])

        with t1:
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(np.unique(y_test))
            fig = px.imshow(cm, text_auto=True,
                            color_continuous_scale=["#0F1923", "#7B61FF", "#00FFB2"],
                            x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                            title="Confusion Matrix")
            apply_layout(fig)
            fig.update_layout(height=420)
            fig.update_traces(textfont=dict(size=14, color='white', family='JetBrains Mono'))
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            if hasattr(m, "predict_proba") and len(np.unique(y_test)) == 2:
                probs = m.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy',
                                         fillcolor='rgba(123,97,255,0.2)',
                                         line=dict(color='#7B61FF', width=3),
                                         name=f"Model (AUC={roc_auc:.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                         line=dict(color='#FF4757', dash='dash', width=1.5),
                                         name='Random Classifier'))
                fig.add_annotation(x=0.6, y=0.3,
                                   text=f"AUC = {roc_auc:.3f}",
                                   font=dict(size=20, color='#00FFB2', family='JetBrains Mono'),
                                   showarrow=False,
                                   bgcolor='rgba(0,0,0,0.5)', bordercolor='#7B61FF', borderwidth=1)
                fig.update_layout(title="📈 ROC Curve", **CHART_LAYOUT,
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC Curve requires binary classification with probability support.")

        with t3:
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                         use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-label">Advanced Diagnostics</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            if hasattr(m, "predict_proba"):
                all_probs = m.predict_proba(X_test)
                if all_probs.shape[1] == 2:
                    conf_vals = all_probs[:, 1]
                else:
                    conf_vals = np.max(all_probs, axis=1)
                fig_conf = go.Figure()
                correct_mask = (y_pred == np.array(y_test))
                fig_conf.add_trace(go.Histogram(
                    x=conf_vals[correct_mask], name="Correct Predictions",
                    marker_color='#00FFB2', opacity=0.75, nbinsx=30
                ))
                fig_conf.add_trace(go.Histogram(
                    x=conf_vals[~correct_mask], name="Wrong Predictions",
                    marker_color='#FF4757', opacity=0.75, nbinsx=30
                ))
                fig_conf.update_layout(
                    title="📊 Prediction Confidence Distribution",
                    barmode='overlay',
                    xaxis_title="Model Confidence (Probability)",
                    yaxis_title="Count",
                    **CHART_LAYOUT, height=340
                )
                fig_conf.add_vline(x=0.5, line_color='#ECC94B', line_dash='dash', line_width=2,
                                   annotation_text="Decision Threshold",
                                   annotation_font_color='#ECC94B')
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("Confidence distribution requires a model with probability support.")
        with d2:
            from sklearn.model_selection import learning_curve
            try:
                with st.spinner("Computing learning curve..."):
                    train_sizes_frac = np.linspace(0.1, 1.0, 5)
                    train_sizes, train_scores, val_scores = learning_curve(
                        m, X_test, y_test, cv=3,
                        train_sizes=train_sizes_frac,
                        n_jobs=-1, error_score='raise'
                    )
                train_mean = train_scores.mean(axis=1)
                val_mean = val_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                val_std = val_scores.std(axis=1)

                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=train_mean,
                    name="Training Score", line=dict(color='#00D4FF', width=2),
                    mode='lines+markers', marker=dict(size=7)
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=val_mean,
                    name="Validation Score", line=dict(color='#FF6B9D', width=2, dash='dot'),
                    mode='lines+markers', marker=dict(size=7)
                ))
                fig_lc.add_trace(go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                    fill='toself', fillcolor='rgba(255,107,157,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Val Band'
                ))
                fig_lc.update_layout(
                    title="Learning Curve",
                    xaxis_title="Training Set Size",
                    yaxis_title="Score",
                    **CHART_LAYOUT, height=340
                )
                st.plotly_chart(fig_lc, use_container_width=True)
                gap = float(train_mean[-1] - val_mean[-1])
                if gap > 0.15:
                    st.warning("⚠️High gap between train and validation → model may be **overfitting**. Try more data or regularization.")
                elif val_mean[-1] < 0.6:
                    st.warning("⚠️ Both scores are low → model may be **underfitting**. Try a more complex model.")
                else:
                    st.success("✅ Train and validation scores are close — model is **well-fitted**.")
            except Exception as e:
                st.info(f"Learning curve unavailable for this model/data combination.")
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{r2:.4f}", delta=f"{'Excellent' if r2>0.85 else 'Good' if r2>0.7 else 'Fair'}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("MSE", f"{mse:.4f}")
        c4.metric("MAE", f"{mae:.4f}")

        t1, t2 = st.tabs(["Actual vs Predicted", "Residuals"])

        with t1:
            n = min(100, len(y_test))
            y_test_arr = np.array(y_test)[:n]
            y_pred_arr = y_pred[:n]
            comp = pd.DataFrame({"Index": range(n), "Actual": y_test_arr, "Predicted": y_pred_arr})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comp["Index"], y=comp["Actual"], mode='lines+markers',
                                     name="Actual", line=dict(color='#00D4FF', width=2),
                                     marker=dict(size=5, color='#00D4FF')))
            fig.add_trace(go.Scatter(x=comp["Index"], y=comp["Predicted"], mode='lines+markers',
                                     name="Predicted", line=dict(color='#FF6B9D', dash='dot', width=2),
                                     marker=dict(size=5, color='#FF6B9D')))
            # Fill gap between them
            fig.add_trace(go.Scatter(
                x=list(comp["Index"]) + list(comp["Index"])[::-1],
                y=list(comp["Actual"]) + list(comp["Predicted"])[::-1],
                fill='toself', fillcolor='rgba(123,97,255,0.07)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Gap'
            ))
            fig.update_layout(title="Actual vs Predicted", **CHART_LAYOUT,
                               xaxis_title="Sample Index", yaxis_title="Value", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            residuals = np.array(y_test) - y_pred
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                           marker=dict(color='#7B61FF', opacity=0.85,
                                                       line=dict(color='rgba(123,97,255,0.3)', width=0.5)),
                                           name="Residuals"))
                fig.add_vline(x=0, line_color='#00FFB2', line_dash='dash', line_width=2,
                              annotation_text="Zero Error", annotation_font_color='#00FFB2')
                fig.update_layout(title="📉 Residual Distribution", **CHART_LAYOUT,
                                   xaxis_title="Residual (Actual − Predicted)",
                                   yaxis_title="Frequency", height=360)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Scatter(
                    x=y_pred, y=residuals, mode='markers',
                    marker=dict(color=residuals,
                                colorscale=["#00D4FF", "#7B61FF", "#FF4757"],
                                size=6, opacity=0.75,
                                colorbar=dict(title="Error", tickfont=dict(color='#CBD5E0'))),
                    name="Residuals"
                ))
                fig2.add_hline(y=0, line_color='#00FFB2', line_dash='dash', line_width=2)
                fig2.update_layout(title="Residuals vs Predicted", **CHART_LAYOUT,
                                    xaxis_title="Predicted Value",
                                    yaxis_title="Residual", height=360)
                st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-label">Advanced Diagnostics</div>', unsafe_allow_html=True)
        da1, da2 = st.columns(2)
        with da1:
            from sklearn.model_selection import learning_curve
            try:
                with st.spinner("Computing learning curve..."):
                    train_sizes, train_scores, val_scores = learning_curve(
                        m, X_test, y_test, cv=3,
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        n_jobs=-1
                    )
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=train_scores.mean(axis=1),
                    name="Training Score", line=dict(color='#00D4FF', width=2),
                    mode='lines+markers', marker=dict(size=7)
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=val_scores.mean(axis=1),
                    name="Validation Score", line=dict(color='#FF6B9D', width=2, dash='dot'),
                    mode='lines+markers', marker=dict(size=7)
                ))
                fig_lc.update_layout(
                    title="Learning Curve",
                    xaxis_title="Training Set Size",
                    yaxis_title="Score",
                    **CHART_LAYOUT, height=340
                )
                st.plotly_chart(fig_lc, use_container_width=True)

                gap = float(train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1])
                if gap > 0.15:
                    st.warning("⚠️ Possible **overfitting** — large gap between train and validation.")
                elif val_scores.mean(axis=1)[-1] < 0.5:
                    st.warning("⚠️ Possible **underfitting** — both scores are low.")
                else:
                    st.success("✅ Model appears **well-fitted** — train and validation scores are close.")
            except Exception:
                st.info("Learning curve unavailable for this model/data size combination.")

        with da2:
            fig_pva = go.Figure()
            sample_n = min(500, len(y_test))
            y_samp = np.array(y_test)[:sample_n]
            p_samp = y_pred[:sample_n]
            fig_pva.add_trace(go.Scatter(
                x=y_samp, y=p_samp, mode='markers',
                marker=dict(color=np.abs(y_samp - p_samp),
                            colorscale=['#00FFB2', '#7B61FF', '#FF4757'],
                            size=6, opacity=0.7,
                            colorbar=dict(title="|Error|", tickfont=dict(color='#CBD5E0'))),
                name="Predictions"
            ))
            # Perfect prediction line
            min_val = float(min(y_samp.min(), p_samp.min()))
            max_val = float(max(y_samp.max(), p_samp.max()))
            fig_pva.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='#ECC94B', dash='dash', width=2),
                name='Perfect Fit'
            ))
            fig_pva.update_layout(
                title="Predicted vs Actual Scatter",
                xaxis_title="Actual Value",
                yaxis_title="Predicted Value",
                **CHART_LAYOUT, height=340
            )
            st.plotly_chart(fig_pva, use_container_width=True)


# p5:model ranking
if selected_page == "Auto-Governance":
    st.markdown("""
    <div class="page-header">
        <h1>Models List</h1>
        <p>Automatically compare multiple algorithms to detect the best model</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df_full = st.session_state.df_cleaned

    c1, c2 = st.columns(2)
    with c1:
        target_g = st.selectbox("Target Variable", df_full.columns)
    with c2:
        p_type_g = st.radio("Problem Type", ["Classification", "Regression"], horizontal=True)

    st.markdown("""
    <div class="card" style="margin:12px 0;">
        <div class="section-label">Tournament Settings</div>
        <p style="color:#6B8BAE;font-size:13px;margin:0;">Runs Random Forest, Logistic/Linear Regression, SVM, and MLP. 
        Uses 3-fold cross-validation.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Tournament", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        df_curr = df_full.sample(min(2000, len(df_full)), random_state=42)
        X = pd.get_dummies(df_curr.drop(columns=[target_g]), drop_first=True)
        y = df_curr[target_g].copy()
        if p_type_g == "Classification" and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        X_sc = StandardScaler().fit_transform(X)

        if p_type_g == "Classification":
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "SVM": SVC(probability=True),
                "Neural Network": MLPClassifier(max_iter=200, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
            }
        else:
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                "Linear Regression": LinearRegression(),
                "SVM": SVR(),
                "Neural Network": MLPRegressor(max_iter=200, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
            }

        results = []
        model_objects = {}
        total = len(models)
        for i, (name, m_obj) in enumerate(models.items()):
            status.markdown(f'<div style="color:#63B3ED;font-size:13px;">Training {name}...</div>', unsafe_allow_html=True)
            try:
                scores = cross_val_score(m_obj, X_sc, y, cv=3, n_jobs=-1)
                results.append({
                    "Model": name,
                    "Mean Score": scores.mean(),
                    "Std Dev": scores.std(),
                    "Min": scores.min(),
                    "Max": scores.max()
                })
                model_objects[name] = m_obj
            except Exception as e:
                results.append({"Model": name, "Mean Score": 0.0, "Std Dev": 0.0, "Min": 0.0, "Max": 0.0})
            progress_bar.progress(int((i+1)/total * 100))

        status.empty()
        res_df = pd.DataFrame(results).sort_values("Mean Score", ascending=False).reset_index(drop=True)
        st.session_state.gov_results = res_df
        st.session_state.gov_model_objects = model_objects
        st.session_state.gov_target = target_g
        st.session_state.gov_ptype = p_type_g

    if st.session_state.gov_results is not None:
        res_df = st.session_state.gov_results
        medals = ["1", "2", "3", "4", "5"]
        st.markdown('<div class="section-label" style="margin-top:16px;">Leaderboard</div>', unsafe_allow_html=True)
        for i, row in res_df.iterrows():
            score_pct = row["Mean Score"]
            bar_width = max(5, int(score_pct * 100)) if score_pct <= 1 else min(100, int(score_pct))
            rank_color = ["#ECC94B", "#CBD5E0", "#F6AD55", "#63B3ED", "#63B3ED"][i]
            st.markdown(f"""
            <div style="background:{'rgba(99,179,237,0.08)' if i==0 else 'rgba(15,25,35,0.8)'};
                        border:1px solid {'rgba(99,179,237,0.3)' if i==0 else 'rgba(99,179,237,0.1)'};
                        border-radius:12px;padding:14px 20px;margin-bottom:8px;display:flex;align-items:center;gap:16px;">
                <span style="font-size:20px;width:30px">{medals[i]}</span>
                <div style="flex:1;">
                    <div style="color:#E2F0FF;font-weight:600;font-size:14px;">{row['Model']}</div>
                    <div style="background:rgba(10,15,26,0.6);border-radius:4px;height:6px;margin-top:6px;overflow:hidden;">
                        <div style="width:{bar_width}%;height:100%;background:linear-gradient(90deg,{rank_color}88,{rank_color});border-radius:4px;"></div>
                    </div>
                </div>
                <div style="text-align:right;min-width:100px;">
                    <div style="color:{rank_color};font-size:20px;font-weight:700;font-family:'JetBrains Mono';">{row['Mean Score']:.4f}</div>
                    <div style="color:#6B8BAE;font-size:11px;">±{row['Std Dev']:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        fig = px.bar(res_df, x="Model", y="Mean Score", error_y="Std Dev",
                     title="Tournament Results Comparison",
                     color="Mean Score",
                     color_continuous_scale=["#FF4757", "#FFB347", "#00FFB2"],
                     text=res_df["Mean Score"].round(4))
        fig.update_traces(textposition='outside', textfont=dict(color='#E2F0FF', size=11,
                                                                  family='JetBrains Mono'))
        apply_layout(fig)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(xaxis_title="", yaxis_title="Cross-Val Score", height=380)
        st.plotly_chart(fig, use_container_width=True)

        winner_name = res_df.iloc[0]["Model"]
        st.markdown(f"""
        <div style="background:rgba(104,211,145,0.08);border:1px solid rgba(104,211,145,0.25);border-radius:12px;
                    padding:16px 24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
            <div>
                <div style="color:#68D391;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Tournament Winner</div>
                <div style="color:#E2F0FF;font-size:18px;font-weight:700;">{winner_name}</div>
                <div style="color:#6B8BAE;font-size:13px;">Score: {res_df.iloc[0]['Mean Score']:.4f}</div>
            </div>
            <div style="font-size:40px;"></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Activate best model", type="primary", use_container_width=True):
            target_g = st.session_state.gov_target
            p_type_g = st.session_state.gov_ptype
            winner_model = st.session_state.gov_model_objects[winner_name]

            X_full = pd.get_dummies(df_full.drop(columns=[target_g]), drop_first=True)
            y_full = df_full[target_g].copy()
            if p_type_g == "Classification" and y_full.dtype == 'object':
                y_full = LabelEncoder().fit_transform(y_full)

            scaler_full = StandardScaler()
            X_sc_full = scaler_full.fit_transform(X_full)
            winner_model.fit(X_sc_full, y_full)

            st.session_state.model = winner_model
            st.session_state.scaler = scaler_full
            st.session_state.feature_names = X_full.columns.tolist()
            st.session_state.target_name = target_g
            st.session_state.model_type = p_type_g
            st.session_state.train_score = res_df.iloc[0]["Mean Score"]
            st.session_state.X_test = X_sc_full
            st.session_state.y_test = y_full

            st.success(f"✓ {winner_name} is now the active model!")
            st.balloons()


# p6:simulation
if selected_page == "Simulation Lab":
    st.markdown("""
    <div class="page-header">
        <h1>Risk & Simulation Lab</h1>
        <p>Interactively simulate scenarios and predict outcomes with risk scoring</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Please train a model first.")
        st.stop()

    df = st.session_state.df_cleaned
    target = st.session_state.target_name
    X_str = df.drop(columns=[target])
    tgt_min = float(df[target].min()) if df[target].dtype != 'object' else 0
    tgt_max = float(df[target].max()) if df[target].dtype != 'object' else 1
    is_classifier = st.session_state.model_type == "Classification"

    if is_classifier:
        risk_explanation = (
            "For <b>classification</b>: Risk Score = probability of the positive or highest class "
            
        )
    else:
        risk_explanation = (
            f"For <b>regression</b>: Outcome Intensity describes how the prediction sits within the "
            f"full target range [{tgt_min:.1f} → {tgt_max:.1f}]. "
            
        )

    st.markdown(f"""
    <div class="card" style="border-left:3px solid #7B61FF;margin-bottom:16px;">
        <div style="color:#E2F0FF;font-weight:600;margin-bottom:6px;">📡 What does the Risk / Outcome Score mean?</div>
        <div style="color:#8BA3BE;font-size:13px;line-height:1.6;">{risk_explanation}</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("Load Random Case", use_container_width=True):
        rand_idx = np.random.randint(0, len(df))
        for col in X_str.columns:
            st.session_state.sim_params[col] = X_str.iloc[rand_idx][col]
        st.rerun()

    if c2.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.sim_params = {}
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Input Parameters</div>', unsafe_allow_html=True)

    inputs = {}
    num_cols_sim = [c for c in X_str.columns if pd.api.types.is_numeric_dtype(X_str[c])]
    cat_cols_sim = [c for c in X_str.columns if not pd.api.types.is_numeric_dtype(X_str[c])]

    if num_cols_sim:
        n_per_row = 3
        for i in range(0, len(num_cols_sim), n_per_row):
            batch = num_cols_sim[i:i+n_per_row]
            cols_ui = st.columns(len(batch))
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    is_int = pd.api.types.is_integer_dtype(X_str[col])
                    mn = int(X_str[col].min()) if is_int else float(X_str[col].min())
                    mx = int(X_str[col].max()) if is_int else float(X_str[col].max())
                    avg = int(X_str[col].mean()) if is_int else float(X_str[col].mean())
                    val = st.session_state.sim_params.get(col, avg)
                    val = int(val) if is_int else float(val)
                    val = max(mn, min(mx, val))  # clamp to range
                    inputs[col] = st.slider(col, mn, mx, val, key=f"sim_{col}")
                    st.session_state.sim_params[col] = inputs[col]

    if cat_cols_sim:
        st.markdown('<div class="section-label" style="margin-top:12px;">Categorical Inputs</div>', unsafe_allow_html=True)
        n_per_row = 3
        for i in range(0, len(cat_cols_sim), n_per_row):
            batch = cat_cols_sim[i:i+n_per_row]
            cols_ui = st.columns(len(batch))
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    opts = X_str[col].dropna().unique().tolist()
                    val = st.session_state.sim_params.get(col, opts[0])
                    if val not in opts:
                        val = opts[0]
                    inputs[col] = st.selectbox(col, opts, index=opts.index(val), key=f"sim_{col}")
                    st.session_state.sim_params[col] = inputs[col]

    st.markdown("---")
    if st.button("Simulate Outcome", type="primary", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        input_enc = pd.get_dummies(input_df, drop_first=True)

        for c in st.session_state.feature_names:
            if c not in input_enc.columns:
                input_enc[c] = 0
        final_in = st.session_state.scaler.transform(input_enc[st.session_state.feature_names])

        pred = st.session_state.model.predict(final_in)[0]
        risk = 0.0
        if hasattr(st.session_state.model, "predict_proba"):
            probs = st.session_state.model.predict_proba(final_in)[0]
            if len(probs) == 2:
                risk = float(probs[1]) 
            else:
                risk = float(np.max(probs))  
        else:
            
            t_vals = df[target]
            t_min, t_max = float(t_vals.min()), float(t_vals.max())
            if t_max != t_min:
                risk = (float(pred) - t_min) / (t_max - t_min)
            else:
                risk = 0.5
            risk = float(np.clip(risk, 0.0, 1.0))
        if risk >= 0.75:
            risk_label, risk_class, risk_emoji = "CRITICAL", "risk-critical", "🔴"
        elif risk >= 0.5:
            risk_label, risk_class, risk_emoji = "HIGH", "risk-high", "🟠"
        elif risk >= 0.25:
            risk_label, risk_class, risk_emoji = "MEDIUM", "risk-medium", "🟡"
        else:
            risk_label, risk_class, risk_emoji = "LOW", "risk-low", "🟢"
        st.session_state.sim_result_context = (
            f"Prediction: {pred} | Risk Score: {risk:.2%} ({risk_label}) | "
            f"Inputs: {inputs} | Target: {target} | Model: {type(st.session_state.model).__name__}"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", f"{pred:.3f}" if isinstance(pred, float) else str(pred))
        c2.metric("Risk Score", f"{risk:.1%}")
        c3.metric("Risk Level", f"{risk_emoji} {risk_label}")

        st.markdown(f"""
        <div style="margin:16px 0 8px 0;">
            <div style="display:flex;justify-content:space-between;color:#6B8BAE;font-size:11px;margin-bottom:4px;">
                <span>LOW RISK</span><span>CRITICAL RISK</span>
            </div>
            <div style="background:#0F1923;border-radius:8px;height:12px;overflow:hidden;border:1px solid rgba(99,179,237,0.15);">
                <div style="width:{risk*100:.1f}%;height:100%;background:linear-gradient(90deg,
                    #68D391 0%,#ECC94B 40%,#F6AD55 65%,#FC8181 100%);
                    border-radius:8px;transition:width 0.5s ease;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        gauge_bar_color = '#00FFB2' if risk < 0.25 else '#ECC94B' if risk < 0.5 else '#FF6B35' if risk < 0.75 else '#FF4757'
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk * 100,
            number={'suffix': '%', 'font': {'size': 36, 'family': 'JetBrains Mono', 'color': gauge_bar_color}},
            delta={'reference': 50, 'valueformat': '.1f',
                   'increasing': {'color': '#FF4757'}, 'decreasing': {'color': '#00FFB2'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{risk_emoji} {risk_label} RISK</b>", 'font': {'color': gauge_bar_color, 'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6B8BAE',
                         'tickfont': {'color': '#6B8BAE', 'size': 10}},
                'bar': {'color': gauge_bar_color, 'thickness': 0.25},
                'bgcolor': 'rgba(15,25,35,0.8)',
                'bordercolor': 'rgba(99,179,237,0.15)',
                'borderwidth': 1,
                'steps': [
                    {'range': [0, 25],  'color': 'rgba(0,255,178,0.12)'},
                    {'range': [25, 50], 'color': 'rgba(236,201,75,0.12)'},
                    {'range': [50, 75], 'color': 'rgba(255,107,53,0.12)'},
                    {'range': [75, 100],'color': 'rgba(255,71,87,0.15)'},
                ],
                'threshold': {
                    'line': {'color': gauge_bar_color, 'width': 4},
                    'thickness': 0.8,
                    'value': risk * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=280,
            margin=dict(l=30, r=30, t=40, b=10),
            font=dict(family='Space Grotesk', color='#CBD5E0')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if hasattr(st.session_state.model, 'feature_importances_') or hasattr(st.session_state.model, 'coef_'):
            model_obj = st.session_state.model
            imps = None
            if hasattr(model_obj, 'feature_importances_'):
                imps = model_obj.feature_importances_
            elif hasattr(model_obj, 'coef_'):
                imps = np.abs(model_obj.coef_[0]) if len(model_obj.coef_.shape) > 1 else np.abs(model_obj.coef_)

            if imps is not None and len(imps) == len(st.session_state.feature_names):
                imp_df = pd.DataFrame({
                    "Feature": st.session_state.feature_names,
                    "Importance": imps
                }).sort_values("Importance", ascending=False).head(5)

                s1, s2 = st.columns(2)
                with s1:
                    st.markdown('<div class="section-label" style="margin-top:12px;">Top Influencing Features</div>', unsafe_allow_html=True)
                    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                     color="Importance",
                                     color_continuous_scale=["#FF4757", "#FFB347", "#00FFB2"],
                                     title="Prediction Drivers")
                    apply_layout(fig_imp)
                    fig_imp.update_coloraxes(showscale=False)
                    fig_imp.update_layout(height=280, margin=dict(l=0, r=20, t=40, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

                with s2:
                    top_feat_name = imp_df.iloc[0]["Feature"]
                    orig_num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    match_col = None
                    for oc in orig_num_cols:
                        if oc != target and (oc in top_feat_name or top_feat_name.startswith(oc)):
                            match_col = oc
                            break
                    if match_col is None and len(orig_num_cols) > 0:
                        
                        match_col = [c for c in orig_num_cols if c != target][0] if len([c for c in orig_num_cols if c != target]) > 0 else None

                    st.markdown('<div class="section-label" style="margin-top:12px;">Feature vs Target</div>', unsafe_allow_html=True)
                    if match_col and df[target].dtype != 'object':
                        sample_df = df[[match_col, target]].dropna().sample(min(500, len(df)), random_state=42)
                        fig_fvt = go.Figure()
                        fig_fvt.add_trace(go.Scatter(
                            x=sample_df[match_col], y=sample_df[target],
                            mode='markers',
                            marker=dict(color=sample_df[target],
                                        colorscale=['#7B61FF', '#00D4FF', '#00FFB2'],
                                        size=5, opacity=0.6,
                                        colorbar=dict(title=target, tickfont=dict(color='#CBD5E0'))),
                            name=f"{match_col} vs {target}"
                        ))
                        
                        if match_col in inputs:
                            fig_fvt.add_trace(go.Scatter(
                                x=[inputs[match_col]], y=[pred],
                                mode='markers',
                                marker=dict(color='#FF4757', size=16, symbol='star',
                                            line=dict(color='white', width=2)),
                                name="Your Input ★"
                            ))
                        fig_fvt.update_layout(
                            title=f"{match_col} vs {target}",
                            xaxis_title=match_col, yaxis_title=target,
                            **CHART_LAYOUT, height=280
                        )
                        st.plotly_chart(fig_fvt, use_container_width=True)
                    else:
                        st.info("Feature vs Target chart available for numeric targets.")


# p7:ai consultant
if selected_page == "AI Consultant":
    st.markdown("""
    <div class="page-header">
        <h1>AI Decision Consultant</h1>
        <p>Powered by Gemini — ask anything about your model, data, and risk scenarios</p>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.model is not None:
        score_disp = f"{st.session_state.train_score:.4f}" if st.session_state.train_score is not None else "N/A"
        m_info = f"Model: {type(st.session_state.model).__name__} ({st.session_state.model_type}), Score: {score_disp}"
    else:
        m_info = "No model trained yet."

    if st.session_state.df_cleaned is not None:
        data_info = (
            f"Dataset: {st.session_state.df_cleaned.shape[0]} rows, "
            f"{st.session_state.df_cleaned.shape[1]} columns, "
            f"Target variable: {st.session_state.target_name}"
        )
    else:
        data_info = "No dataset loaded."

    sim_info = st.session_state.sim_result_context
    full_ctx = f"{m_info} | {data_info} | Last Simulation: {sim_info}"
    st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    quick_prompt = None
    if c1.button("Explain Last Simulation", use_container_width=True):
        quick_prompt = "Explain my last simulation results in detail. What does the risk score mean? What should I do based on these results?"
    if c2.button("Analyze Model Performance", use_container_width=True):
        quick_prompt = "Analyze my current model's performance score. Is it good for this type of problem? What are the main weaknesses and how can I improve it?"
    if c3.button("Generate Risk Strategy", use_container_width=True):
        quick_prompt = "Based on the simulation results and risk score, suggest a concrete, actionable risk mitigation strategy with specific steps."

    if quick_prompt:
        st.session_state.chat_history.append({"role": "user", "content": quick_prompt})
        with st.spinner("InspectorML is thinking..."):
            ai_response = get_gemini_chat_response(st.session_state.chat_history, full_ctx)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask InspectorML anything about your data, model, or simulation..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_response = get_gemini_chat_response(st.session_state.chat_history, full_ctx)
            st.markdown(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()