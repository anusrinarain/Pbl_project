import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InspectorML Pro",
    page_icon="lightning",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
.main { background-color: #070B14 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #0A0F1A 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15) !important;
}
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }

.nav-link { border-radius: 8px !important; margin: 2px 4px !important; color: #8BA3BE !important; font-size: 13px !important; }
.nav-link-selected {
    background: linear-gradient(135deg, #1A3A5C, #1E4976) !important;
    border-left: 3px solid #63B3ED !important;
    color: #E2F0FF !important;
}
.nav-link:hover { background: rgba(99,179,237,0.08) !important; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0F1923 0%, #131E2E 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 16px 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
div[data-testid="stMetricLabel"] > div { color: #6B8BAE !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetricValue"] > div { color: #63B3ED !important; font-size: 28px !important; font-weight: 700 !important; font-family: 'JetBrains Mono' !important; }
div[data-testid="stMetricDelta"] > div { color: #68D391 !important; font-size: 12px !important; }

.stButton > button {
    background: linear-gradient(135deg, #1A3A5C, #1E4976) !important;
    color: #E2F0FF !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk' !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1E4976, #2B5F94) !important;
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.25) !important;
    transform: translateY(-1px) !important;
}

.page-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0F2035 100%);
    border: 1px solid rgba(99,179,237,0.15);
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
.page-header h1 { color: #E2F0FF; font-size: 26px; font-weight: 700; margin: 0 0 6px 0; }
.page-header p { color: #6B8BAE; font-size: 14px; margin: 0; }

.card {
    background: linear-gradient(135deg, #0F1923, #111D2E);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-accent { border-left: 3px solid #63B3ED !important; }
.card-gold   { border-left: 3px solid #ECC94B !important; }
.card-green  { border-left: 3px solid #68D391 !important; }

.section-label {
    color: #6B8BAE;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 10px;
}

.stat-pill {
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #63B3ED;
    font-family: 'JetBrains Mono';
    margin-right: 6px;
}

.math-block {
    background: rgba(10,15,26,0.8);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono';
    font-size: 13px;
    color: #A0C4E8;
    margin: 10px 0;
    line-height: 1.8;
}

div[data-testid="stTabs"] button { color: #6B8BAE !important; font-family: 'Space Grotesk' !important; font-weight: 500 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #63B3ED !important; border-bottom: 2px solid #63B3ED !important; }

div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    background: #0F1923 !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 8px !important;
    color: #CBD5E0 !important;
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(99,179,237,0.25) !important;
    border-radius: 12px !important;
    background: rgba(15,25,35,0.5) !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #2B6CB0, #63B3ED) !important; border-radius: 4px !important; }
[data-testid="stChatMessage"] {
    background: #0F1923 !important;
    border: 1px solid rgba(99,179,237,0.12) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
hr { border-color: rgba(99,179,237,0.1) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070B14; }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }
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

# ─────────────────────────────────────────────────────────────
# SESSION SAVE / LOAD  (joblib-based persistence)
# ─────────────────────────────────────────────────────────────
SESSION_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".inspectorml_sessions")
SESSION_FILE = os.path.join(SESSION_DIR, "last_session.joblib")
META_FILE    = os.path.join(SESSION_DIR, "last_session_meta.json")

os.makedirs(SESSION_DIR, exist_ok=True)

def save_session():
    """
    Persist the current session to disk using joblib.
    Saves: dataframe, trained model, scaler, feature names,
           target name, model type, train score, governance results.
    Everything needed to restore a full working session without retraining.
    """
    try:
        payload = {
            "df_cleaned":    st.session_state.df_cleaned,
            "model":         st.session_state.model,
            "scaler":        st.session_state.scaler,
            "feature_names": st.session_state.feature_names,
            "target_name":   st.session_state.target_name,
            "model_type":    st.session_state.model_type,
            "train_score":   st.session_state.train_score,
            "gov_results":   st.session_state.gov_results,
            "gov_target":    st.session_state.gov_target,
            "gov_ptype":     st.session_state.gov_ptype,
        }
        joblib.dump(payload, SESSION_FILE, compress=3)
        # Save readable metadata separately so the UI can show what's saved
        meta = {
            "rows":        int(st.session_state.df_cleaned.shape[0]) if st.session_state.df_cleaned is not None else 0,
            "cols":        int(st.session_state.df_cleaned.shape[1]) if st.session_state.df_cleaned is not None else 0,
            "model":       type(st.session_state.model).__name__ if st.session_state.model is not None else "None",
            "target":      st.session_state.target_name or "None",
            "score":       round(float(st.session_state.train_score), 4) if st.session_state.train_score is not None else 0,
            "model_type":  st.session_state.model_type or "None",
        }
        with open(META_FILE, "w") as f:
            json.dump(meta, f)
        return True
    except Exception as e:
        return str(e)

def load_session():
    """Restore a previously saved session from disk."""
    try:
        if not os.path.exists(SESSION_FILE):
            return False, "No saved session found."
        payload = joblib.load(SESSION_FILE)
        for key, value in payload.items():
            st.session_state[key] = value
        return True, "Session restored."
    except Exception as e:
        return False, str(e)

def get_session_meta():
    """Return saved session metadata for display, or None if nothing saved."""
    try:
        if not os.path.exists(META_FILE):
            return None
        with open(META_FILE) as f:
            return json.load(f)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# GEMINI HELPER
# ─────────────────────────────────────────────────────────────
def get_gemini_response(history, context):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            return (
                "No API key found. Add GEMINI_API_KEY to `.streamlit/secrets.toml`:\n\n"
                "```toml\nGEMINI_API_KEY = 'your_key_here'\n```\n\n"
                "Get one free at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        model_name = "gemini-1.5-flash"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                    model_name = m.name
                    break
        except Exception:
            pass
        model = genai.GenerativeModel(model_name)
        system = (
            "You are InspectorML, a senior data scientist embedded in InspectorML Pro.\n\n"
            f"Current session context:\n{context}\n\n"
            "Answer questions about this project's ML models, metrics, and simulation results. "
            "Be specific — use numbers from the context when available. Keep answers concise and practical. "
            "Use markdown formatting."
        )
        gem_history = []
        for msg in history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gem_history.append({"role": role, "parts": [msg["content"]]})
        chat = model.start_chat(history=gem_history)
        response = chat.send_message(f"[Context]\n{system}\n\n[Question]\n{history[-1]['content']}")
        return response.text
    except Exception as e:
        err = str(e)
        if "API_KEY" in err.upper() or "invalid" in err.lower():
            return "Invalid API key. Check your Gemini key at https://aistudio.google.com/app/apikey"
        elif "quota" in err.lower() or "429" in err:
            return "Rate limited. Wait a moment and try again."
        return f"Error calling Gemini: {err}"

# ─────────────────────────────────────────────────────────────
# GOVERNANCE SCORE ENGINE
# ─────────────────────────────────────────────────────────────
def compute_governance_score(accuracy, cv_std, fairness_score,
                              w_acc=0.50, w_stab=0.25, w_fair=0.25):
    stability = max(0.0, 1.0 - cv_std)
    g_score = w_acc * accuracy + w_stab * stability + w_fair * fairness_score
    return {
        "accuracy": accuracy,
        "stability": stability,
        "fairness": fairness_score,
        "governance_score": g_score,
    }

def compute_fairness(model, X, y, sensitive_col_idx=None):
    try:
        preds = model.predict(X)
        if sensitive_col_idx is None or sensitive_col_idx >= X.shape[1]:
            return 1.0
        group = X[:, sensitive_col_idx] > np.median(X[:, sensitive_col_idx])
        g0 = preds[~group]
        g1 = preds[group]
        if len(g0) == 0 or len(g1) == 0:
            return 1.0
        unique_vals = np.unique(preds)
        if len(unique_vals) <= 10:
            pos_val = unique_vals[-1]
            p0 = np.mean(g0 == pos_val)
            p1 = np.mean(g1 == pos_val)
        else:
            overall_range = float(preds.max() - preds.min())
            if overall_range == 0:
                return 1.0
            p0 = float(np.mean(g0)) / overall_range
            p1 = float(np.mean(g1)) / overall_range
        dpd = abs(p0 - p1)
        return float(np.clip(1.0 - dpd, 0.0, 1.0))
    except Exception:
        return 1.0

def build_stacking_model(problem_type):
    if problem_type == "Classification":
        estimators = [
            ("rf",  RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)),
            ("gb",  GradientBoostingClassifier(n_estimators=60, random_state=42)),
            ("lr",  LogisticRegression(max_iter=1000, C=1.0)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3, n_jobs=-1, passthrough=False
        )
    else:
        estimators = [
            ("rf",  RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)),
            ("gb",  GradientBoostingRegressor(n_estimators=60, random_state=42)),
            ("lr",  Ridge(alpha=1.0)),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3, n_jobs=-1, passthrough=False
        )

# ─────────────────────────────────────────────────────────────
# COUNTERFACTUAL ENGINE  (FIXED)
# ─────────────────────────────────────────────────────────────
def _get_risk(model, scaler, feature_names, raw_vec, is_classifier, t_min=0.0, t_max=1.0):
    """
    Always returns a value in [0, 1].

    Classification : probability of the positive / highest class.
    Regression     : (prediction - t_min) / (t_max - t_min), clipped to [0, 1].

    The t_min / t_max parameters are the actual min and max of the target
    column in the dataset so regression predictions are on the same 0-1 scale
    as classifier probabilities.  Without this normalisation, raw predictions
    (e.g. exam score = 67) are treated as percentages, causing the impossible
    5000 % 'improvements' seen previously.
    """
    arr = np.array(raw_vec, dtype=float).reshape(1, -1)
    sc  = scaler.transform(arr)
    if is_classifier and hasattr(model, "predict_proba"):
        probs = model.predict_proba(sc)[0]
        risk  = float(probs[-1]) if len(probs) > 1 else 0.5
    else:
        pred = float(model.predict(sc)[0])
        denom = t_max - t_min
        risk  = (pred - t_min) / denom if denom > 0 else 0.5
    return float(np.clip(risk, 0.0, 1.0))


def counterfactual_search(model, scaler, feature_names, input_vec,
                           target_threshold, feature_bounds,
                           is_classifier, t_min=0.0, t_max=1.0, n_steps=200):
    """
    Greedy counterfactual search.

    Math:  x* = argmin ||x - x'||_2   subject to   f_norm(x') <= tau

    All risk values are kept in [0, 1] throughout (via _get_risk) so the
    'improvement' figures shown to the user are genuine percentage-point drops
    in the normalised outcome score — never more than 100 % in total.

    Only changes that actually reduce the risk by at least 0.1 pp are kept.
    """
    suggestions  = []
    x            = list(input_vec)          # plain list — avoid numpy copy issues
    current_risk = _get_risk(model, scaler, feature_names, x,
                             is_classifier, t_min, t_max)

    if current_risk <= target_threshold:
        return suggestions, current_risk

    # Order features by importance (highest first = biggest lever first)
    if hasattr(model, "feature_importances_"):
        imps = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw  = np.array(model.coef_)
        imps = np.abs(raw[0]) if raw.ndim > 1 else np.abs(raw)
        if len(imps) != len(feature_names):
            imps = np.ones(len(feature_names))
    else:
        imps = np.ones(len(feature_names))

    feat_order = np.argsort(imps)[::-1]

    for fi in feat_order:
        if fi >= len(feature_names):
            continue
        fname = feature_names[fi]
        if fname not in feature_bounds:
            continue
        lo, hi, _ = feature_bounds[fname]
        if lo >= hi:
            continue

        best_val  = x[fi]
        best_risk = current_risk   # only accept improvements from this baseline

        # Scan the full valid range for this feature
        for step in np.linspace(lo, hi, n_steps):
            x_try    = list(x)
            x_try[fi] = float(step)
            r = _get_risk(model, scaler, feature_names, x_try,
                         is_classifier, t_min, t_max)
            if r < best_risk:
                best_risk = r
                best_val  = float(step)

        improvement = current_risk - best_risk

        # Only record genuinely useful changes (> 0.1 pp improvement)
        if improvement > 0.001 and abs(best_val - x[fi]) > 1e-6:
            suggestions.append({
                "feature":     fname,
                "original":    float(x[fi]),
                "suggested":   best_val,
                "risk_after":  float(np.clip(best_risk,  0.0, 1.0)),
                "improvement": float(np.clip(improvement, 0.0, 1.0)),
            })
            x[fi]        = best_val
            current_risk = best_risk

        if current_risk <= target_threshold:
            break

    return suggestions, float(np.clip(current_risk, 0.0, 1.0))

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "df_cleaned": None, "model": None, "sim_params": {}, "cf_params": {},
    "chat_history": [], "sim_result_context": "No simulation run yet.",
    "sim_followup_history": [],
    "model_type": None, "feature_names": [], "target_name": None,
    "X_test": None, "y_test": None, "train_score": None,
    "scaler": None, "gov_results": None, "gov_target": None,
    "gov_ptype": None, "gov_model_objects": {},
    "raw_input_vec": None,
    "_sim_pred": None, "_sim_risk": None, "_sim_risk_label": None,
    "_sim_input_enc": None, "_sim_inputs": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 8px 12px 8px;text-align:center;border-bottom:1px solid rgba(99,179,237,0.12);margin-bottom:12px;">
        <div style="color:#63B3ED;font-size:18px;font-weight:700;letter-spacing:1px;">InspectorML Pro</div>
        <div style="color:#4A6580;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-top:2px;">Decision Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=["Upload & Clean", "EDA & Outliers", "Model Training",
                 "Evaluation", "Governance Engine", "Simulation & Analysis",
                 "Counterfactual", "AI Consultant"],
        icons=["cloud-upload", "bar-chart", "cpu", "check-circle",
               "shield-check", "sliders", "arrow-repeat", "robot"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "0"},
            "icon": {"color": "#4A6580", "font-size": "13px"},
            "nav-link": {"font-size": "13px", "color": "#8BA3BE", "padding": "8px 12px"},
            "nav-link-selected": {"background": "linear-gradient(135deg,#1A3A5C,#1E4976)", "color": "#E2F0FF"},
        }
    )

    st.markdown('<div class="section-label" style="margin-top:20px;padding:0 8px;">Dataset</div>', unsafe_allow_html=True)
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
        score_str = f"{st.session_state.train_score:.4f}" if st.session_state.train_score is not None else "N/A"
        st.markdown(f"""
        <div style="background:rgba(72,187,120,0.06);border:1px solid rgba(72,187,120,0.2);border-radius:10px;padding:12px;margin-top:10px;">
            <div style="color:#6B8BAE;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Active Model</div>
            <div style="color:#68D391;font-size:12px;font-weight:600;">&#10003; {type(st.session_state.model).__name__}</div>
            <div style="color:#CBD5E0;font-size:11px;margin-top:2px;">{st.session_state.model_type} &bull; Score: {score_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Session persistence ──
    st.markdown('<div class="section-label" style="margin-top:20px;padding:0 8px;">Session</div>',
                unsafe_allow_html=True)

    meta = get_session_meta()
    if meta:
        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.04);border:1px solid rgba(99,179,237,0.12);
                    border-radius:10px;padding:10px 12px;margin-bottom:8px;">
            <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Last Saved</div>
            <div style="color:#CBD5E0;font-size:11px;line-height:1.7;">
                <div>{meta.get('rows',0):,} rows &bull; {meta.get('cols',0)} cols</div>
                <div>Model: {meta.get('model','—')} ({meta.get('model_type','—')})</div>
                <div>Target: {meta.get('target','—')} &bull; Score: {meta.get('score',0)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    sb1, sb2 = st.columns(2)
    with sb1:
        if st.button("Save session", use_container_width=True,
                     help="Saves your dataset, trained model, and scaler to disk so you don't have to re-upload or retrain next time"):
            if st.session_state.df_cleaned is None and st.session_state.model is None:
                st.warning("Nothing to save yet.")
            else:
                result = save_session()
                if result is True:
                    st.success("Saved.")
                else:
                    st.error(f"Save failed: {result}")
    with sb2:
        if st.button("Load session", use_container_width=True,
                     help="Restores your last saved dataset and trained model — no need to re-upload or retrain"):
            ok, msg = load_session()
            if ok:
                st.success("Loaded.")
                st.rerun()
            else:
                st.error(msg)

# ─────────────────────────────────────────────────────────────
# PAGE 1: UPLOAD & CLEAN
# ─────────────────────────────────────────────────────────────
if selected_page == "Upload & Clean":
    st.markdown("""
    <div class="page-header">
        <h1>Data Setup</h1>
        <p>Upload your CSV, review data quality, and clean it before analysis.</p>
    </div>
    """, unsafe_allow_html=True)



    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())

        st.markdown('<div class="section-label" style="margin-top:16px;">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=260)

        st.markdown('<div class="section-label" style="margin-top:16px;">Data Controls</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Auto-clean missing values", use_container_width=True):
                df_temp = df.copy()
                for col in df_temp.columns:
                    if df_temp[col].dtype == 'object':
                        df_temp[col] = df_temp[col].fillna(
                            df_temp[col].mode()[0] if not df_temp[col].mode().empty else "Unknown"
                        )
                    else:
                        df_temp[col] = df_temp[col].fillna(df_temp[col].mean())
                st.session_state.df_cleaned = df_temp
                st.success(f"Filled {df.isnull().sum().sum()} missing values.")
                st.rerun()
        with c2:
            if st.button("Reset to original", use_container_width=True):
                st.session_state.df_cleaned = pd.read_csv(uploaded_file) if uploaded_file else None
                st.rerun()
        with c3:
            cols_to_drop = st.multiselect("Drop columns", df.columns, label_visibility="collapsed",
                                           placeholder="Select columns to remove...")
            if cols_to_drop and st.button(f"Drop {len(cols_to_drop)} column(s)", use_container_width=True):
                st.session_state.df_cleaned = df.drop(columns=cols_to_drop)
                st.rerun()

        st.markdown('<div class="section-label" style="margin-top:20px;">Column Overview</div>', unsafe_allow_html=True)
        col_info = []
        for col in df.columns:
            col_info.append({
                "Column": col, "Type": str(df[col].dtype),
                "Non-Null": df[col].count(),
                "Null %": f"{df[col].isnull().mean()*100:.1f}%",
                "Unique": df[col].nunique(),
                "Sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,25,35,0.5);border:1px dashed rgba(99,179,237,0.2);border-radius:16px;">
            <div style="color:#E2F0FF;font-size:18px;font-weight:600;margin-bottom:8px;">No Dataset Loaded</div>
            <div style="color:#6B8BAE;font-size:14px;">Upload a CSV from the sidebar to begin.</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2: EDA & OUTLIERS
# ─────────────────────────────────────────────────────────────
if selected_page == "EDA & Outliers":
    st.markdown("""
    <div class="page-header">
        <h1>Exploratory Analysis</h1>
        <p>Distributions, correlations, outliers, and feature relationships across your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.df_cleaned
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Feature Stats", "Correlations", "Outlier Inspector", "Pair Plot", "Violin & Balance"
    ])

    with tab1:
        c1, c2 = st.columns([1, 3])
        with c1:
            feat = st.selectbox("Select Feature", df.columns)
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:12px;">
                <div class="section-label">Descriptive Stats</div>
                <div style="font-family:'JetBrains Mono';font-size:12px;color:#CBD5E0;line-height:1.8;">
                    {''.join([f'<div><span style="color:#6B8BAE">{k}:</span> <span style="color:#63B3ED">{v}</span></div>'
                    for k, v in df[feat].describe().round(3).items()])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            chart_colors = ['#7B61FF', '#00D4FF', '#FF6B9D', '#00FFB2', '#FFB347', '#FF4757', '#2ED573']
            feat_color = chart_colors[list(df.columns).index(feat) % len(chart_colors)]
            fig = px.histogram(df, x=feat, marginal="box",
                               color_discrete_sequence=[feat_color],
                               title=f"Distribution — {feat}")
            fig.update_traces(marker_line_width=0, opacity=0.85)
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Distribution of {feat} across all {len(df):,} rows. The box plot at the top shows median, IQR, and outliers. A symmetric bell shape means the data is normally distributed.")

        st.markdown("---")
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        shown = False
        if st.session_state.model and st.session_state.feature_names:
            model = st.session_state.model
            imps = None
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            if imps is not None and len(imps) == len(st.session_state.feature_names):
                imp_df = pd.DataFrame({"Feature": st.session_state.feature_names, "Importance": imps})
                imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                             title="Model Feature Importance",
                             color="Importance", color_continuous_scale=["#7B61FF", "#00D4FF", "#00FFB2"])
                apply_layout(fig)
                fig.update_coloraxes(showscale=True, colorbar=dict(tickfont=dict(color='#CBD5E0')))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Longer bars indicate features the model relies on most when making predictions. These are the levers with the highest potential for change in the counterfactual analysis.")
                shown = True
        if not shown:
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                t = st.selectbox("Correlate with", num_df.columns, index=len(num_df.columns)-1)
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
                            title="Feature Correlation Heatmap", zmin=-1, zmax=1)
            apply_layout(fig)
            fig.update_layout(height=520)
            fig.update_traces(textfont=dict(size=10, color='white'))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Values range from −1 (perfect negative correlation) to +1 (perfect positive correlation). Values near 0 mean two features share little relationship. Bright purple cells indicate features that tend to move together.")
        else:
            st.info("No numeric columns found.")

    with tab3:
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
                outlier_pct = len(outliers) / len(df) * 100
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
                fig_box = px.box(df, y=col_sel, points="outliers",
                                 title=f"Box Plot — {col_sel}",
                                 color_discrete_sequence=['#7B61FF'])
                fig_box.update_traces(
                    marker=dict(size=7, color='#FF4757', symbol='x', opacity=0.85,
                                line=dict(width=2, color='#FF4757')),
                    fillcolor='rgba(123,97,255,0.2)',
                    line=dict(color='#7B61FF', width=2)
                )
                apply_layout(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption(f"The box shows the middle 50% of values (IQR). The line inside is the median. Red × marks are statistical outliers — values more than 1.5× the IQR beyond the box edges.")

            fig_scatter = px.scatter(df, y=col_sel, title=f"Scatter Spread — {col_sel}",
                                     color_discrete_sequence=['#00D4FF'], opacity=0.4)
            if len(outliers) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=outliers.index, y=outliers[col_sel], mode='markers',
                    name=f'Outliers ({len(outliers)})',
                    marker=dict(color='#FF4757', size=9, symbol='x', line=dict(width=2, color='#FF4757'))
                ))
            apply_layout(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"Each dot is one row in your dataset. Red × marks are the {len(outliers)} outliers ({outlier_pct:.1f}% of data) identified by the IQR method. These rows may need investigation or removal before training.")

    with tab4:
        num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols_all) >= 2:
            c1, c2 = st.columns([1, 3])
            with c1:
                selected_feats = st.multiselect("Features (pick 2–6)", num_cols_all,
                    default=num_cols_all[:min(4, len(num_cols_all))], max_selections=6)
                color_by = None
                if st.session_state.target_name and st.session_state.target_name in df.columns:
                    tgt = df[st.session_state.target_name]
                    if tgt.dtype == 'object' or tgt.nunique() <= 10:
                        color_by = st.session_state.target_name
                        st.success(f"Coloured by: {color_by}")
                    else:
                        st.info("Target is continuous — no colour grouping.")
            with c2:
                if len(selected_feats) >= 2:
                    plot_df = df[selected_feats + ([color_by] if color_by and color_by not in selected_feats else [])].dropna()
                    fig_pair = px.scatter_matrix(plot_df, dimensions=selected_feats,
                        color=color_by if color_by else None,
                        color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D', '#FFB347', '#00D4FF', '#FF4757'],
                        title="Pair Plot", opacity=0.6)
                    fig_pair.update_traces(diagonal_visible=True, marker=dict(size=3))
                    apply_layout(fig_pair)
                    fig_pair.update_layout(height=560)
                    st.plotly_chart(fig_pair, use_container_width=True)
                    st.caption("Each cell shows the relationship between two features. Diagonal cells show each feature's own distribution. Colour groups by the target variable when it has few unique values.")
                else:
                    st.warning("Select at least 2 features.")
        else:
            st.info("Need at least 2 numeric columns.")

    with tab5:
        num_cols_v = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_v = df.select_dtypes(include='object').columns.tolist()

        st.markdown('<div class="section-label">Violin Plot</div>', unsafe_allow_html=True)
        cv1, cv2 = st.columns(2)
        with cv1:
            violin_feat = st.selectbox("Numeric feature", num_cols_v, key="violin_feat")
        with cv2:
            violin_group = st.selectbox("Group by (optional)", ["None"] + cat_cols_v, key="violin_group")

        if violin_group == "None":
            fig_vio = px.violin(df, y=violin_feat, box=True, points="outliers",
                                title=f"Violin — {violin_feat}",
                                color_discrete_sequence=['#7B61FF'])
        else:
            fig_vio = px.violin(df, y=violin_feat, x=violin_group, box=True, points="outliers",
                                title=f"{violin_feat} by {violin_group}", color=violin_group,
                                color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D', '#FFB347', '#00D4FF', '#FF4757'])
        fig_vio.update_traces(meanline_visible=True)
        apply_layout(fig_vio)
        fig_vio.update_layout(height=420)
        st.plotly_chart(fig_vio, use_container_width=True)
        st.caption("Width of the shape shows where data is densest. The embedded box shows median and IQR. The white dot marks the median. Useful for spotting bimodal distributions or comparing groups.")

        st.markdown("---")
        st.markdown('<div class="section-label">Target Variable Balance</div>', unsafe_allow_html=True)
        if st.session_state.target_name and st.session_state.target_name in df.columns:
            tgt_col = st.session_state.target_name
            tgt_series = df[tgt_col]
            cb1, cb2 = st.columns(2)
            with cb1:
                if tgt_series.dtype == 'object' or tgt_series.nunique() <= 15:
                    counts = tgt_series.value_counts().reset_index()
                    counts.columns = ["Class", "Count"]
                    fig_pie = px.pie(counts, values="Count", names="Class",
                                     title=f"Class Balance — {tgt_col}",
                                     color_discrete_sequence=['#7B61FF', '#00FFB2', '#FF6B9D',
                                                               '#FFB347', '#00D4FF', '#FF4757'],
                                     hole=0.45)
                    fig_pie.update_traces(textposition='outside', textinfo='percent+label',
                                          textfont=dict(size=12, color='white'))
                    apply_layout(fig_pie)
                    fig_pie.update_layout(height=380)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    fig_hist_tgt = px.histogram(df, x=tgt_col, nbins=40,
                                                 title=f"Target Distribution — {tgt_col}",
                                                 color_discrete_sequence=['#7B61FF'])
                    fig_hist_tgt.update_traces(marker_line_width=0, opacity=0.85)
                    apply_layout(fig_hist_tgt)
                    fig_hist_tgt.update_layout(height=380)
                    st.plotly_chart(fig_hist_tgt, use_container_width=True)
            with cb2:
                st.markdown('<div class="section-label">Missing Value Map</div>', unsafe_allow_html=True)
                null_df = df.isnull().astype(int)
                if null_df.sum().sum() > 0:
                    sample_null = null_df.sample(min(200, len(null_df)), random_state=42)
                    fig_null = px.imshow(sample_null.T, color_continuous_scale=["#0F1923", "#FF4757"],
                                         title="Missing values (sample of 200 rows)", aspect="auto")
                    apply_layout(fig_null)
                    fig_null.update_layout(height=380)
                    fig_null.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_null, use_container_width=True)
                else:
                    st.success("No missing values in the dataset.")
        else:
            st.info("Train a model first so the target variable can be identified.")

# ─────────────────────────────────────────────────────────────
# PAGE 3: MODEL TRAINING
# ─────────────────────────────────────────────────────────────
if selected_page == "Model Training":
    st.markdown("""
    <div class="page-header">
        <h1>Model Training Studio</h1>
        <p>Select your target variable, choose an algorithm, and train. Results appear immediately below.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.df_cleaned.copy()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-label">Target & Problem Type</div>', unsafe_allow_html=True)
        target = st.selectbox("Target Variable", df.columns)
        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.06);border-radius:8px;padding:10px 14px;margin:8px 0;">
            <span class="stat-pill">{str(df[target].dtype)}</span>
            <span class="stat-pill">{df[target].nunique()} unique</span>
            <span class="stat-pill">{df[target].isnull().sum()} nulls</span>
        </div>
        """, unsafe_allow_html=True)
        p_type = st.radio("Problem Type", ["Classification", "Regression"], horizontal=True)

    with c2:
        st.markdown('<div class="section-label">Algorithm</div>', unsafe_allow_html=True)
        algo_options_clf = [
            "Hybrid Stacking Model (RF + GB + LR)",
            "Random Forest", "Gradient Boosting",
            "Logistic Regression", "SVM (SVC)", "Neural Network (MLP)"
        ]
        algo_options_reg = [
            "Hybrid Stacking Model (RF + GB + Ridge)",
            "Random Forest", "Gradient Boosting",
            "Linear Regression", "SVM (SVR)", "Neural Network (MLP)"
        ]
        model_name = st.selectbox("Algorithm", algo_options_clf if p_type == "Classification" else algo_options_reg)

        val_method = st.radio("Validation", ["Train-Test Split", "K-Fold Cross-Validation"], horizontal=True)
        if val_method == "Train-Test Split":
            test_pct = st.slider("Test set size (%)", 10, 50, 20, 5)
            st.caption(f"Training on {100 - test_pct}% of data, testing on {test_pct}%.")
            test_size = test_pct / 100.0
        else:
            k_folds = st.slider("Number of folds", 3, 10, 5)

    if st.button("Start Training", type="primary", use_container_width=True):
        with st.spinner("Training..."):
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

            if "Hybrid" in model_name:
                m = build_stacking_model(p_type)
            elif p_type == "Classification":
                m_map = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM (SVC)": SVC(probability=True),
                    "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=42)
                }
                m = m_map[model_name]
            else:
                m_map = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression(),
                    "SVM (SVR)": SVR(),
                    "Neural Network (MLP)": MLPRegressor(max_iter=500, random_state=42)
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
                score = cross_val_score(m, X_sc, y, cv=k_folds, n_jobs=-1).mean()
                m.fit(X_sc, y)
                st.session_state.X_test = X_sc
                st.session_state.y_test = y

            prog.progress(90)
            st.session_state.model = m
            st.session_state.model_type = p_type
            st.session_state.train_score = score
            st.session_state.target_name = target
            prog.progress(100)

        label = "Accuracy" if p_type == "Classification" else "R2 Score"
        color = "#68D391" if score > 0.8 else "#ECC94B" if score > 0.6 else "#FC8181"
        st.markdown(f"""
        <div style="background:rgba(15,25,35,0.8);border:1px solid {color}33;border-radius:14px;padding:24px;text-align:center;margin-top:16px;">
            <div style="color:#6B8BAE;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Training Complete &mdash; {label}</div>
            <div style="font-size:48px;font-weight:700;font-family:'JetBrains Mono';color:{color};">{score:.4f}</div>
            <div style="color:#6B8BAE;font-size:13px;margin-top:8px;">{model_name} &bull; {val_method}</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# ─────────────────────────────────────────────────────────────
# PAGE 4: EVALUATION
# ─────────────────────────────────────────────────────────────
if selected_page == "Evaluation":
    st.markdown("""
    <div class="page-header">
        <h1>Model Evaluation</h1>
        <p>Performance metrics, diagnostic charts, and learning behaviour for your trained model.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    m = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = m.predict(X_test)

    if st.session_state.model_type == "Classification":
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc:.2%}",  delta="Good" if acc > 0.8 else "Fair")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall",    f"{rec:.3f}")
        c4.metric("F1 Score",  f"{f1:.3f}")

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
                                         name='Random classifier'))
                fig.add_annotation(x=0.6, y=0.3, text=f"AUC = {roc_auc:.3f}",
                                   font=dict(size=20, color='#00FFB2', family='JetBrains Mono'),
                                   showarrow=False, bgcolor='rgba(0,0,0,0.5)',
                                   bordercolor='#7B61FF', borderwidth=1)
                fig.update_layout(title="ROC Curve", **CHART_LAYOUT,
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC curve requires binary classification with probability support.")

        with t3:
            from sklearn.metrics import classification_report
            report    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df.style.background_gradient(cmap='Blues',
                         subset=['precision', 'recall', 'f1-score']), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Advanced Diagnostics</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            if hasattr(m, "predict_proba"):
                all_probs    = m.predict_proba(X_test)
                conf_vals    = all_probs[:, 1] if all_probs.shape[1] == 2 else np.max(all_probs, axis=1)
                correct_mask = (y_pred == np.array(y_test))
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Histogram(x=conf_vals[correct_mask],  name="Correct",
                                                 marker_color='#00FFB2', opacity=0.75, nbinsx=30))
                fig_conf.add_trace(go.Histogram(x=conf_vals[~correct_mask], name="Wrong",
                                                 marker_color='#FF4757', opacity=0.75, nbinsx=30))
                fig_conf.add_vline(x=0.5, line_color='#ECC94B', line_dash='dash', line_width=2,
                                   annotation_text="Decision threshold",
                                   annotation_font_color='#ECC94B')
                fig_conf.update_layout(title="Prediction Confidence Distribution",
                                        barmode='overlay', **CHART_LAYOUT, height=340,
                                        xaxis_title="Confidence", yaxis_title="Count")
                st.plotly_chart(fig_conf, use_container_width=True)
                st.caption("Green bars = correctly predicted samples, red bars = incorrect. Correct predictions should cluster toward 1.0; incorrect ones near 0.5 indicate the model was uncertain — that's expected behaviour.")
        with d2:
            try:
                with st.spinner("Computing learning curve..."):
                    ts, tr_sc, v_sc = learning_curve(m, X_test, y_test, cv=3,
                                                      train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=ts, y=tr_sc.mean(axis=1), name="Training",
                                             line=dict(color='#00D4FF', width=2),
                                             mode='lines+markers', marker=dict(size=7)))
                fig_lc.add_trace(go.Scatter(x=ts, y=v_sc.mean(axis=1), name="Validation",
                                             line=dict(color='#FF6B9D', width=2, dash='dot'),
                                             mode='lines+markers', marker=dict(size=7)))
                fig_lc.update_layout(title="Learning Curve", xaxis_title="Training set size",
                                      yaxis_title="Score", **CHART_LAYOUT, height=340)
                st.plotly_chart(fig_lc, use_container_width=True)
                st.caption("A small gap between training and validation scores means the model generalises well. A large gap means it may be memorising training data (overfitting).")
                gap = float(tr_sc.mean(axis=1)[-1] - v_sc.mean(axis=1)[-1])
                if gap > 0.15:
                    st.warning("Large gap between train and validation — the model may be overfitting.")
                elif v_sc.mean(axis=1)[-1] < 0.6:
                    st.warning("Both scores are low — possible underfitting. Try a more complex model.")
                else:
                    st.success("Train and validation scores are close. The model is generalising well.")
            except Exception:
                st.info("Learning curve unavailable for this model/data combination.")

    else:
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R2 Score", f"{r2:.4f}",   delta="Excellent" if r2 > 0.85 else "Good" if r2 > 0.7 else "Fair")
        c2.metric("RMSE",     f"{rmse:.4f}")
        c3.metric("MSE",      f"{mse:.4f}")
        c4.metric("MAE",      f"{mae:.4f}")

        t1, t2 = st.tabs(["Actual vs Predicted", "Residuals"])
        with t1:
            n    = min(100, len(y_test))
            y_ta = np.array(y_test)[:n]
            y_pa = y_pred[:n]
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n)), y=y_ta, mode='lines+markers',
                                     name="Actual", line=dict(color='#00D4FF', width=2)))
            fig.add_trace(go.Scatter(x=list(range(n)), y=y_pa, mode='lines+markers',
                                     name="Predicted", line=dict(color='#FF6B9D', dash='dot', width=2)))
            fig.update_layout(title="Actual vs Predicted (first 100)", **CHART_LAYOUT,
                               xaxis_title="Sample", yaxis_title="Value", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Blue line = ground truth values from your test set. Pink dashed = what the model predicted. Lines that track closely indicate a well-calibrated model.")
        with t2:
            residuals = np.array(y_test) - y_pred
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                           marker=dict(color='#7B61FF', opacity=0.85), name="Residuals"))
                fig.add_vline(x=0, line_color='#00FFB2', line_dash='dash', line_width=2,
                              annotation_text="Zero error", annotation_font_color='#00FFB2')
                fig.update_layout(title="Residual Distribution", **CHART_LAYOUT,
                                   xaxis_title="Residual", yaxis_title="Count", height=360)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Residual = actual − predicted. A bell curve centred at 0 means errors are random — which is ideal. Skew or heavy tails indicate systematic over- or under-prediction.")
            with c2:
                fig2 = go.Figure(go.Scatter(x=y_pred, y=residuals, mode='markers',
                    marker=dict(color=residuals, colorscale=["#00D4FF", "#7B61FF", "#FF4757"],
                                size=6, opacity=0.75,
                                colorbar=dict(title="Error", tickfont=dict(color='#CBD5E0')))))
                fig2.add_hline(y=0, line_color='#00FFB2', line_dash='dash', line_width=2)
                fig2.update_layout(title="Residuals vs Predicted", **CHART_LAYOUT,
                                    xaxis_title="Predicted", yaxis_title="Residual", height=360)
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Points should scatter randomly around the green zero line. A funnel shape (wider at higher predictions) indicates heteroscedasticity — the model is less reliable at extreme values.")

        st.markdown("---")
        st.markdown('<div class="section-label">Advanced Diagnostics</div>', unsafe_allow_html=True)
        da1, da2 = st.columns(2)
        with da1:
            try:
                with st.spinner("Computing learning curve..."):
                    ts, tr_sc, v_sc = learning_curve(m, X_test, y_test, cv=3,
                                                      train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=ts, y=tr_sc.mean(axis=1), name="Training",
                                             line=dict(color='#00D4FF', width=2), mode='lines+markers'))
                fig_lc.add_trace(go.Scatter(x=ts, y=v_sc.mean(axis=1), name="Validation",
                                             line=dict(color='#FF6B9D', width=2, dash='dot'),
                                             mode='lines+markers'))
                fig_lc.update_layout(title="Learning Curve", xaxis_title="Training set size",
                                      yaxis_title="Score", **CHART_LAYOUT, height=340)
                st.plotly_chart(fig_lc, use_container_width=True)
            except Exception:
                st.info("Learning curve unavailable for this model.")
        with da2:
            sn = min(500, len(y_test))
            ys = np.array(y_test)[:sn]
            ps = y_pred[:sn]
            fig_pva = go.Figure()
            fig_pva.add_trace(go.Scatter(x=ys, y=ps, mode='markers',
                marker=dict(color=np.abs(ys - ps), colorscale=['#00FFB2', '#7B61FF', '#FF4757'],
                            size=6, opacity=0.7,
                            colorbar=dict(title="|Error|", tickfont=dict(color='#CBD5E0')))))
            mv = float(min(ys.min(), ps.min()))
            xv = float(max(ys.max(), ps.max()))
            fig_pva.add_trace(go.Scatter(x=[mv, xv], y=[mv, xv], mode='lines',
                                          line=dict(color='#ECC94B', dash='dash', width=2),
                                          name='Perfect fit'))
            fig_pva.update_layout(title="Predicted vs Actual", xaxis_title="Actual",
                                   yaxis_title="Predicted", **CHART_LAYOUT, height=340)
            st.plotly_chart(fig_pva, use_container_width=True)
            st.caption("Points on the diagonal = perfect prediction. Colour shows error magnitude.")

# ─────────────────────────────────────────────────────────────
# PAGE 5: GOVERNANCE ENGINE
# ─────────────────────────────────────────────────────────────
if selected_page == "Governance Engine":
    st.markdown("""
    <div class="page-header">
        <h1>Responsible AI Governance Engine</h1>
        <p>Models ranked by a composite Governance Score across accuracy, stability, and fairness.
           This goes beyond single-metric selection to reflect real-world deployment requirements.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df_full = st.session_state.df_cleaned
    c1, c2 = st.columns(2)
    with c1:
        target_g = st.selectbox("Target Variable", df_full.columns)
    with c2:
        p_type_g = st.radio("Problem Type", ["Classification", "Regression"], horizontal=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        w_acc = st.slider("Weight: Accuracy", 0.1, 0.8, 0.5, 0.05)
    with c4:
        w_stab = st.slider("Weight: Stability", 0.1, 0.6, 0.25, 0.05)
    with c5:
        w_fair = max(0.05, round(1.0 - w_acc - w_stab, 2))
        st.metric("Weight: Fairness (auto)", f"{w_fair:.2f}",
                  delta="Remaining weight after accuracy + stability")

    if st.button("Run Governance Analysis", type="primary", use_container_width=True):
        df_sample = df_full.sample(min(2000, len(df_full)), random_state=42)
        X_g = pd.get_dummies(df_sample.drop(columns=[target_g]), drop_first=True)
        y_g = df_sample[target_g].copy()
        if p_type_g == "Classification" and y_g.dtype == 'object':
            y_g = LabelEncoder().fit_transform(y_g)
        sc_g    = StandardScaler()
        X_sc_g  = sc_g.fit_transform(X_g)
        sens_idx = 0

        if p_type_g == "Classification":
            candidates = {
                "Hybrid Stacking":    build_stacking_model("Classification"),
                "Random Forest":      RandomForestClassifier(n_estimators=50, random_state=42),
                "Gradient Boosting":  GradientBoostingClassifier(n_estimators=50, random_state=42),
                "Logistic Regression":LogisticRegression(max_iter=1000),
                "Neural Network":     MLPClassifier(max_iter=200, random_state=42),
            }
        else:
            candidates = {
                "Hybrid Stacking":   build_stacking_model("Regression"),
                "Random Forest":     RandomForestRegressor(n_estimators=50, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
                "Linear/Ridge":      Ridge(alpha=1.0),
                "Neural Network":    MLPRegressor(max_iter=200, random_state=42),
            }

        results       = []
        model_objects = {}
        prog          = st.progress(0)
        status_ph     = st.empty()

        for i, (name, clf) in enumerate(candidates.items()):
            status_ph.markdown(f'<div style="color:#63B3ED;font-size:13px;">Evaluating {name}...</div>',
                               unsafe_allow_html=True)
            try:
                cv_scores  = cross_val_score(clf, X_sc_g, y_g, cv=3, n_jobs=-1)
                clf.fit(X_sc_g, y_g)
                fair_score = compute_fairness(clf, X_sc_g, y_g, sens_idx)
                g = compute_governance_score(float(cv_scores.mean()), float(cv_scores.std()),
                                             fair_score, w_acc=w_acc, w_stab=w_stab, w_fair=w_fair)
                results.append({
                    "Model": name,
                    "Accuracy":         round(g["accuracy"],         4),
                    "Stability":        round(g["stability"],        4),
                    "Fairness":         round(g["fairness"],         4),
                    "Governance Score": round(g["governance_score"], 4),
                })
                model_objects[name] = clf
            except Exception:
                results.append({"Model": name, "Accuracy": 0.0, "Stability": 0.0,
                                "Fairness": 0.0, "Governance Score": 0.0})
            prog.progress(int((i + 1) / len(candidates) * 100))

        status_ph.empty()
        res_df = pd.DataFrame(results).sort_values("Governance Score", ascending=False).reset_index(drop=True)
        st.session_state.gov_results       = res_df
        st.session_state.gov_target        = target_g
        st.session_state.gov_ptype         = p_type_g
        st.session_state.gov_model_objects = model_objects

    if st.session_state.gov_results is not None:
        res_df = st.session_state.gov_results
        st.markdown('<div class="section-label" style="margin-top:20px;">Governance Leaderboard</div>',
                    unsafe_allow_html=True)

        medals      = ["1st", "2nd", "3rd", "4th", "5th"]
        rank_colors = ["#ECC94B", "#CBD5E0", "#F6AD55", "#63B3ED", "#63B3ED"]

        for i, row in res_df.iterrows():
            rc    = rank_colors[i] if i < len(rank_colors) else "#63B3ED"
            g_pct = row["Governance Score"] / max(res_df["Governance Score"].max(), 0.01) * 100
            st.markdown(f"""
            <div class="card" style="padding:14px 20px;margin-bottom:8px;
                 {'border-left:3px solid #ECC94B;' if i==0 else ''}">
                <div style="display:flex;align-items:center;gap:16px;">
                    <div style="color:{rc};font-size:12px;font-family:'JetBrains Mono';width:36px;">{medals[i]}</div>
                    <div style="flex:1;">
                        <div style="color:#E2F0FF;font-size:14px;font-weight:600;margin-bottom:6px;">{row['Model']}</div>
                        <div style="display:flex;gap:16px;font-size:11px;color:#6B8BAE;margin-bottom:6px;">
                            <span>Accuracy: <span style="color:#63B3ED">{row['Accuracy']:.4f}</span></span>
                            <span>Stability: <span style="color:#00FFB2">{row['Stability']:.4f}</span></span>
                            <span>Fairness: <span style="color:#FFB347">{row['Fairness']:.4f}</span></span>
                        </div>
                        <div style="background:#0A0F1A;border-radius:4px;height:6px;overflow:hidden;">
                            <div style="width:{g_pct:.1f}%;height:100%;background:{rc};border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="text-align:right;min-width:80px;">
                        <div style="color:{rc};font-size:20px;font-weight:700;font-family:'JetBrains Mono';">{row['Governance Score']:.4f}</div>
                        <div style="color:#4A6580;font-size:10px;">Governance Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        fig_radar   = go.Figure()
        dims        = ["Accuracy", "Stability", "Fairness"]
        colors_r    = ['#7B61FF', '#00FFB2', '#FF6B9D', '#FFB347', '#00D4FF']
        for i, row in res_df.iterrows():
            vals = [row["Accuracy"], row["Stability"], row["Fairness"], row["Accuracy"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=dims + [dims[0]],
                fill='toself', name=row["Model"],
                line=dict(color=colors_r[i % len(colors_r)], width=2),
                fillcolor='rgba(0,0,0,0)'
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(10,15,26,0.5)',
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#6B8BAE', size=9)),
                angularaxis=dict(tickfont=dict(color='#CBD5E0', size=12))
            ),
            paper_bgcolor='rgba(10,15,26,0)',
            title="Governance Dimensions — All Models",
            font=dict(family='Space Grotesk', color='#CBD5E0'),
            height=450, margin=dict(l=40, r=40, t=60, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Each axis represents one governance dimension scored 0–1. A model that fills all three axes equally is the most balanced. The winner is the model with the largest overall area across all three dimensions.")

        fig_bar = go.Figure()
        for dim, col in {"Accuracy": "#63B3ED", "Stability": "#00FFB2", "Fairness": "#FFB347"}.items():
            fig_bar.add_trace(go.Bar(name=dim, x=res_df["Model"], y=res_df[dim],
                                      marker_color=col, opacity=0.85))
        fig_bar.add_trace(go.Bar(name="Governance Score", x=res_df["Model"],
                                  y=res_df["Governance Score"], marker_color="#FF6B9D", opacity=0.95))
        fig_bar.update_layout(barmode='group', title="Score Breakdown by Model",
                               **CHART_LAYOUT, height=380, yaxis_title="Score")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Side-by-side comparison of all four dimensions for every model. The pink Governance Score bar is the weighted composite of the other three. A model with high accuracy but low fairness will have a shorter pink bar than expected.")

        winner_name = res_df.iloc[0]["Model"]
        st.markdown(f"""
        <div style="background:rgba(236,201,75,0.06);border:1px solid rgba(236,201,75,0.25);border-radius:12px;
                    padding:16px 24px;margin-bottom:16px;">
            <div style="color:#ECC94B;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Governance Winner</div>
            <div style="color:#E2F0FF;font-size:18px;font-weight:700;">{winner_name}</div>
            <div style="color:#6B8BAE;font-size:13px;margin-top:4px;">
                Governance Score: {res_df.iloc[0]['Governance Score']:.4f} &nbsp;|&nbsp;
                Accuracy: {res_df.iloc[0]['Accuracy']:.4f} &nbsp;|&nbsp;
                Stability: {res_df.iloc[0]['Stability']:.4f} &nbsp;|&nbsp;
                Fairness: {res_df.iloc[0]['Fairness']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Deploy Governance Winner as Active Model", type="primary", use_container_width=True):
            target_g     = st.session_state.gov_target
            p_type_g     = st.session_state.gov_ptype
            winner_model = st.session_state.gov_model_objects[winner_name]
            X_full       = pd.get_dummies(df_full.drop(columns=[target_g]), drop_first=True)
            y_full       = df_full[target_g].copy()
            if p_type_g == "Classification" and y_full.dtype == 'object':
                y_full = LabelEncoder().fit_transform(y_full)
            sc_full    = StandardScaler()
            X_sc_full  = sc_full.fit_transform(X_full)
            winner_model.fit(X_sc_full, y_full)
            st.session_state.model         = winner_model
            st.session_state.scaler        = sc_full
            st.session_state.feature_names = X_full.columns.tolist()
            st.session_state.target_name   = target_g
            st.session_state.model_type    = p_type_g
            st.session_state.train_score   = res_df.iloc[0]["Governance Score"]
            st.session_state.X_test        = X_sc_full
            st.session_state.y_test        = y_full
            st.success(f"{winner_name} is now the active model.")
            st.balloons()

# ─────────────────────────────────────────────────────────────
# PAGE 6: SIMULATION & ANALYSIS  (merged)
# ─────────────────────────────────────────────────────────────
if selected_page == "Simulation & Analysis":
    st.markdown("""
    <div class="page-header">
        <h1>Simulation & Analysis</h1>
        <p>Set input values, run a prediction, get an AI explanation, then find the minimum changes
           needed to reach a better outcome — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    df            = st.session_state.df_cleaned
    target        = st.session_state.target_name
    X_str         = df.drop(columns=[target])
    is_classifier = st.session_state.model_type == "Classification"
    num_cols_sim  = [c for c in X_str.columns if pd.api.types.is_numeric_dtype(X_str[c])]
    cat_cols_sim  = [c for c in X_str.columns if not pd.api.types.is_numeric_dtype(X_str[c])]

    # ── Score meaning explanation ──
    if is_classifier:
        risk_explanation = (
            "The outcome score is the model's <b>confidence</b> that this input belongs to the positive class. "
            "A score of 80% means the model is 80% sure. Whether a high score is good or bad depends "
            "entirely on what you are predicting."
        )
    else:
        tgt_min = float(df[target].min())
        tgt_max = float(df[target].max())
        risk_explanation = (
            f"The outcome score shows where this prediction sits <b>relative to the full target range</b> "
            f"({tgt_min:.1f} to {tgt_max:.1f} in your dataset). "
            f"A 70% score means the prediction is in the upper third of all possible outcomes. "
            f"It is not inherently good or bad — it is context for how extreme this result is."
        )

    st.caption(risk_explanation.replace('<b>', '').replace('</b>', ''))

    # ── Quick loaders ──
    st.markdown('<div class="section-label">Quick Scenario Loaders</div>', unsafe_allow_html=True)
    ql1, ql2, ql3, ql4 = st.columns(4)

    if ql1.button("Load a random row", use_container_width=True):
        rand_idx = np.random.randint(0, len(df))
        for col in num_cols_sim:
            raw_v  = X_str.iloc[rand_idx][col]
            is_int = pd.api.types.is_integer_dtype(X_str[col])
            v      = float(np.clip(float(raw_v), float(X_str[col].min()), float(X_str[col].max())))
            v      = int(v) if is_int else v
            st.session_state.sim_params[col] = v
            st.session_state[f"sim_{col}"]   = v
        for col in cat_cols_sim:
            opts  = X_str[col].dropna().unique().tolist()
            raw_v = X_str.iloc[rand_idx][col]
            v     = raw_v if raw_v in opts else opts[0]
            st.session_state.sim_params[col] = v
            st.session_state[f"sim_{col}"]   = v
        st.rerun()

    if ql2.button("Worst case scenario", use_container_width=True,
                  help="Loads the actual row in your dataset that produced the lowest model prediction — the true worst case according to the model"):
        try:
            # Find the row that actually produces the lowest prediction from the model
            X_all = pd.get_dummies(X_str, drop_first=True)
            for col in st.session_state.feature_names:
                if col not in X_all.columns:
                    X_all[col] = 0
            X_all_sc  = st.session_state.scaler.transform(X_all[st.session_state.feature_names])
            preds_all  = st.session_state.model.predict(X_all_sc)
            worst_idx  = int(np.argmin(preds_all))
            for col in num_cols_sim:
                is_int = pd.api.types.is_integer_dtype(X_str[col])
                v = X_str.iloc[worst_idx][col]
                v = int(v) if is_int else float(v)
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
            for col in cat_cols_sim:
                opts = X_str[col].dropna().unique().tolist()
                raw_v = X_str.iloc[worst_idx][col]
                v = raw_v if raw_v in opts else opts[0]
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
        except Exception:
            # Fallback: all minimums
            for col in num_cols_sim:
                is_int = pd.api.types.is_integer_dtype(X_str[col])
                v = int(X_str[col].min()) if is_int else float(X_str[col].min())
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
            for col in cat_cols_sim:
                opts = X_str[col].dropna().unique().tolist()
                st.session_state.sim_params[col] = opts[0]
                st.session_state[f"sim_{col}"]   = opts[0]
        st.rerun()

    if ql3.button("Best case scenario", use_container_width=True,
                  help="Loads the actual row in your dataset that produced the highest model prediction — the true best case according to the model"):
        try:
            # Find the row that actually produces the highest prediction from the model
            X_all = pd.get_dummies(X_str, drop_first=True)
            for col in st.session_state.feature_names:
                if col not in X_all.columns:
                    X_all[col] = 0
            X_all_sc  = st.session_state.scaler.transform(X_all[st.session_state.feature_names])
            preds_all  = st.session_state.model.predict(X_all_sc)
            best_idx   = int(np.argmax(preds_all))
            for col in num_cols_sim:
                is_int = pd.api.types.is_integer_dtype(X_str[col])
                v = X_str.iloc[best_idx][col]
                v = int(v) if is_int else float(v)
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
            for col in cat_cols_sim:
                opts = X_str[col].dropna().unique().tolist()
                raw_v = X_str.iloc[best_idx][col]
                v = raw_v if raw_v in opts else opts[-1]
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
        except Exception:
            # Fallback: all maximums
            for col in num_cols_sim:
                is_int = pd.api.types.is_integer_dtype(X_str[col])
                v = int(X_str[col].max()) if is_int else float(X_str[col].max())
                st.session_state.sim_params[col] = v
                st.session_state[f"sim_{col}"]   = v
            for col in cat_cols_sim:
                opts = X_str[col].dropna().unique().tolist()
                st.session_state.sim_params[col] = opts[-1]
                st.session_state[f"sim_{col}"]   = opts[-1]
        st.rerun()

    if ql4.button("Reset to averages", use_container_width=True):
        for col in num_cols_sim:
            is_int = pd.api.types.is_integer_dtype(X_str[col])
            v = int(X_str[col].mean()) if is_int else float(X_str[col].mean())
            st.session_state.sim_params[col] = v
            st.session_state[f"sim_{col}"]   = v
        for col in cat_cols_sim:
            opts = X_str[col].dropna().unique().tolist()
            st.session_state.sim_params[col] = opts[0]
            st.session_state[f"sim_{col}"]   = opts[0]
        st.rerun()

    # ── Input sliders ──
    st.markdown("---")
    st.markdown('<div class="section-label">Input Values</div>', unsafe_allow_html=True)
    inputs = {}

    if num_cols_sim:
        for i in range(0, len(num_cols_sim), 3):
            batch   = num_cols_sim[i:i + 3]
            cols_ui = st.columns(len(batch))
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    is_int = pd.api.types.is_integer_dtype(X_str[col])
                    mn  = int(X_str[col].min())  if is_int else float(X_str[col].min())
                    mx  = int(X_str[col].max())  if is_int else float(X_str[col].max())
                    avg = int(X_str[col].mean()) if is_int else float(X_str[col].mean())
                    val = st.session_state.sim_params.get(col, avg)
                    val = int(val) if is_int else float(val)
                    val = max(mn, min(mx, val))
                    inputs[col] = st.slider(col, mn, mx, val, key=f"sim_{col}")
                    st.session_state.sim_params[col] = inputs[col]

    if cat_cols_sim:
        st.markdown('<div class="section-label" style="margin-top:12px;">Categorical Inputs</div>',
                    unsafe_allow_html=True)
        for i in range(0, len(cat_cols_sim), 3):
            batch   = cat_cols_sim[i:i + 3]
            cols_ui = st.columns(len(batch))
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    opts = X_str[col].dropna().unique().tolist()
                    val  = st.session_state.sim_params.get(col, opts[0])
                    if val not in opts:
                        val = opts[0]
                    inputs[col] = st.selectbox(col, opts, index=opts.index(val), key=f"sim_{col}")
                    st.session_state.sim_params[col] = inputs[col]

    # ══════════════════════════════════════════════════════════
    # SECTION A — SIMULATION
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    if st.button("Run Simulation", type="primary", use_container_width=True):
        try:
            input_row = pd.DataFrame([inputs])
            input_enc = pd.get_dummies(input_row, drop_first=True)
            for col in st.session_state.feature_names:
                if col not in input_enc.columns:
                    input_enc[col] = 0
            final_in = st.session_state.scaler.transform(input_enc[st.session_state.feature_names])
            pred     = st.session_state.model.predict(final_in)[0]

            if hasattr(st.session_state.model, "predict_proba"):
                probs = st.session_state.model.predict_proba(final_in)[0]
                risk  = float(probs[1]) if len(probs) == 2 else float(np.max(probs))
            else:
                t_min = float(df[target].min())
                t_max = float(df[target].max())
                risk  = (float(pred) - t_min) / (t_max - t_min) if t_max != t_min else 0.5
            risk = float(np.clip(risk, 0.0, 1.0))

            risk_label = ("CRITICAL" if risk >= 0.75 else "HIGH" if risk >= 0.5
                          else "MEDIUM" if risk >= 0.25 else "LOW")

            st.session_state.raw_input_vec      = input_enc[st.session_state.feature_names].values[0].tolist()
            st.session_state.sim_result_context = (
                f"Prediction: {pred} | Score: {risk:.2%} ({risk_label}) | "
                f"Inputs: {inputs} | Target: {target} | Model: {type(st.session_state.model).__name__}"
            )

            # Store sim result so counterfactual section below can read it without re-running
            st.session_state["_sim_pred"]       = pred
            st.session_state["_sim_risk"]       = risk
            st.session_state["_sim_risk_label"] = risk_label
            st.session_state["_sim_input_enc"]  = input_enc[st.session_state.feature_names].values[0].tolist()
            st.session_state["_sim_inputs"]     = dict(inputs)

        except Exception as e:
            st.error(f"Simulation failed: {e}")

    # ── Show result if available ──
    if st.session_state.get("_sim_risk") is not None:
        pred       = st.session_state["_sim_pred"]
        risk       = st.session_state["_sim_risk"]
        risk_label = st.session_state["_sim_risk_label"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction",    f"{pred:.3f}" if isinstance(pred, float) else str(pred))
        c2.metric("Outcome Score", f"{risk:.1%}")
        c3.metric("Level",         risk_label)

        st.markdown(f"""
        <div style="margin:16px 0 8px 0;">
            <div style="display:flex;justify-content:space-between;color:#6B8BAE;font-size:11px;margin-bottom:4px;">
                <span>Low</span><span>High</span>
            </div>
            <div style="background:#0F1923;border-radius:8px;height:12px;overflow:hidden;border:1px solid rgba(99,179,237,0.15);">
                <div style="width:{risk*100:.1f}%;height:100%;background:linear-gradient(90deg,
                    #00FFB2 0%,#ECC94B 40%,#FF6B35 65%,#FF4757 100%);border-radius:8px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        gc = '#00FFB2' if risk < 0.25 else '#ECC94B' if risk < 0.5 else '#FF6B35' if risk < 0.75 else '#FF4757'
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk * 100,
            number={'suffix': '%', 'font': {'size': 36, 'family': 'JetBrains Mono', 'color': gc}},
            delta={'reference': 50, 'valueformat': '.1f',
                   'increasing': {'color': '#FF4757'}, 'decreasing': {'color': '#00FFB2'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{risk_label}</b>", 'font': {'color': gc, 'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6B8BAE',
                         'tickfont': {'color': '#6B8BAE', 'size': 10}},
                'bar': {'color': gc, 'thickness': 0.25},
                'bgcolor': 'rgba(15,25,35,0.8)',
                'bordercolor': 'rgba(99,179,237,0.15)',
                'steps': [
                    {'range': [0,  25], 'color': 'rgba(0,255,178,0.12)'},
                    {'range': [25, 50], 'color': 'rgba(236,201,75,0.12)'},
                    {'range': [50, 75], 'color': 'rgba(255,107,53,0.12)'},
                    {'range': [75,100], 'color': 'rgba(255,71,87,0.15)'},
                ],
                'threshold': {'line': {'color': gc, 'width': 4}, 'value': risk * 100}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                                 margin=dict(l=30, r=30, t=40, b=10),
                                 font=dict(family='Space Grotesk', color='#CBD5E0'))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance + Feature vs Target
        model_obj = st.session_state.model
        imps = None
        if hasattr(model_obj, 'feature_importances_'):
            imps = model_obj.feature_importances_
        elif hasattr(model_obj, 'coef_'):
            imps = np.abs(model_obj.coef_[0]) if len(model_obj.coef_.shape) > 1 else np.abs(model_obj.coef_)

        top_features_text = ""
        if imps is not None and len(imps) == len(st.session_state.feature_names):
            imp_df = pd.DataFrame({
                "Feature": st.session_state.feature_names, "Importance": imps
            }).sort_values("Importance", ascending=False).head(5)
            top_features_text = ", ".join(imp_df["Feature"].tolist())

            s1, s2 = st.columns(2)
            with s1:
                st.markdown('<div class="section-label">Top Influencing Features</div>', unsafe_allow_html=True)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance",
                                 color_continuous_scale=["#FF4757", "#FFB347", "#00FFB2"],
                                 title="What drove this prediction")
                apply_layout(fig_imp)
                fig_imp.update_coloraxes(showscale=False)
                fig_imp.update_layout(height=280)
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption("Features ranked by their influence on this specific prediction. The top feature is the biggest lever — changing it would have the most impact on the outcome.")
            with s2:
                top_feat      = imp_df.iloc[0]["Feature"]
                orig_num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
                match_col     = next((c for c in orig_num_cols if top_feat.startswith(c)), None)
                if match_col is None and orig_num_cols:
                    match_col = orig_num_cols[0]
                st.markdown('<div class="section-label">Feature vs Target</div>', unsafe_allow_html=True)
                if match_col and df[target].dtype != 'object':
                    sdf = df[[match_col, target]].dropna().sample(min(500, len(df)), random_state=42)
                    fig_fvt = go.Figure()
                    fig_fvt.add_trace(go.Scatter(x=sdf[match_col], y=sdf[target], mode='markers',
                        marker=dict(color=sdf[target], colorscale=['#7B61FF', '#00D4FF', '#00FFB2'],
                                    size=5, opacity=0.6)))
                    saved_inputs = st.session_state.get("_sim_inputs", {})
                    if match_col in saved_inputs:
                        fig_fvt.add_trace(go.Scatter(
                            x=[saved_inputs[match_col]], y=[pred], mode='markers',
                            marker=dict(color='#FF4757', size=16, symbol='star',
                                        line=dict(color='white', width=2)), name="Your input"))
                    fig_fvt.update_layout(title=f"{match_col} vs {target}",
                                           xaxis_title=match_col, yaxis_title=target,
                                           **CHART_LAYOUT, height=280)
                    st.plotly_chart(fig_fvt, use_container_width=True)
                    st.caption(f"Each dot is one record from your dataset. The red ★ marks where your current simulation input sits. This shows whether your scenario is typical or extreme relative to the full data distribution.")

        # ── AI Interpretation (button-driven) ──
        st.markdown("---")
        st.markdown('<div class="section-label">AI Interpretation</div>', unsafe_allow_html=True)

        pred_display  = f"{pred:.2f}" if isinstance(pred, float) else str(pred)
        saved_inputs  = st.session_state.get("_sim_inputs", inputs)
        tgt_min_str   = float(df[target].min())
        tgt_max_str   = float(df[target].max())
        percentile_pos = int(((float(pred) - tgt_min_str) / (tgt_max_str - tgt_min_str) * 100)) if not is_classifier and tgt_max_str != tgt_min_str else int(risk * 100)

        # Context-aware performance descriptor
        if risk_label == "CRITICAL":
            perf_descriptor = f"among the highest {100 - percentile_pos}% of all outcomes in the dataset"
        elif risk_label == "HIGH":
            perf_descriptor = f"above average — better than approximately {percentile_pos}% of cases in the dataset"
        elif risk_label == "MEDIUM":
            perf_descriptor = f"around the middle — better than approximately {percentile_pos}% of cases in the dataset"
        else:
            perf_descriptor = f"in the lower range — better than only {percentile_pos}% of cases in the dataset"

        if st.button("Generate AI Explanation", use_container_width=True):
            ai_sim_prompt = (
                f"You are InspectorML, a senior AI analyst embedded in a decision intelligence dashboard. "
                f"Write a detailed, warm, and clear explanation of this prediction result for a non-technical audience "
                f"(teacher, HR manager, doctor, or business analyst). "
                f"Structure your response in exactly 6 sentences. No markdown headers, no bullet points.\n\n"
                f"Sentence 1 — Main finding: State what the model is predicting and the actual predicted value clearly "
                f"(e.g. 'Based on these inputs, the model predicts an Exam Score of 72.4 out of a possible 101.').\n"
                f"Sentence 2 — Context: Explain where this result sits relative to all cases in the dataset. "
                f"Use the percentile position to contextualise it (e.g. 'This places this student in the top 35% of all students in the dataset, with a score higher than approximately 65% of their peers.').\n"
                f"Sentence 3 — Key driver 1: Name the single most important feature driving this prediction and explain its direction "
                f"(e.g. 'The strongest driver of this prediction is Attendance at 55%, which is significantly below the dataset average of 79% and is pulling the predicted score down considerably.').\n"
                f"Sentence 4 — Key driver 2: Name the second most important feature and its effect "
                f"(e.g. 'Hours_Studied at just 1 hour per week is also a major factor, as students with similar low study hours consistently score in the lower quartile.').\n"
                f"Sentence 5 — Risk summary: Summarise what this combination of factors means for the overall outcome and risk level "
                f"(e.g. 'Taken together, this profile represents a HIGH risk scenario where without intervention the predicted outcome is unlikely to improve significantly.').\n"
                f"Sentence 6 — Actionable recommendation: Give one specific, realistic recommendation that a decision-maker can act on immediately "
                f"(e.g. 'The most impactful single action would be to increase weekly study hours to at least 15-20 hours, which the model suggests could improve the predicted score by approximately 8-12 points.').\n\n"
                f"ACTUAL DATA TO USE:\n"
                f"- Target being predicted: {target}\n"
                f"- Predicted value: {pred_display}\n"
                f"- Dataset range for {target}: {tgt_min_str:.1f} (min) to {tgt_max_str:.1f} (max)\n"
                f"- Dataset average for {target}: {df[target].mean():.1f}\n"
                f"- Outcome score (normalised 0-100%): {risk:.1%} — {perf_descriptor}\n"
                f"- Risk level: {risk_label}\n"
                f"- Top influencing features: {top_features_text if top_features_text else 'see feature importance chart'}\n"
                f"- All current input values: {', '.join([f'{k}={v}' for k, v in list(saved_inputs.items())])}\n"
                f"- Model type: {type(st.session_state.model).__name__}, Problem: {st.session_state.model_type}"
            )

            with st.spinner("Generating detailed explanation..."):
                ai_explanation = get_gemini_response(
                    [{"role": "user", "content": ai_sim_prompt}],
                    st.session_state.sim_result_context
                )

            st.session_state["_ai_explanation"] = ai_explanation
            st.session_state["_ai_gc"]          = gc
            st.session_state["_ai_risk_label"]  = risk_label

        if st.session_state.get("_ai_explanation") and st.session_state.get("_ai_risk_label") == risk_label:
            ai_explanation = st.session_state["_ai_explanation"]
            stored_gc      = st.session_state.get("_ai_gc", gc)
            sentences = [s.strip() for s in ai_explanation.replace('\n', ' ').split('. ') if s.strip()]
            formatted = '. '.join(sentences)
            if formatted and not formatted.endswith('.'):
                formatted += '.'
            # Split into pairs for two-column card layout
            sentence_list = [s.strip() + '.' for s in ai_explanation.replace('\n', ' ').split('.') if s.strip()]
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {stored_gc};margin-top:4px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                    <div style="background:{stored_gc};border-radius:50%;width:10px;height:10px;flex-shrink:0;"></div>
                    <div style="color:{stored_gc};font-weight:700;font-size:15px;">
                        {risk_label} — Detailed Analysis
                    </div>
                </div>
                <div style="color:#CBD5E0;font-size:14px;line-height:2.0;">
                    {formatted.replace('. ', '.<br><br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Follow-up chat
        st.markdown('<div class="section-label" style="margin-top:20px;">Ask a follow-up question</div>',
                    unsafe_allow_html=True)
        if "sim_followup_history" not in st.session_state:
            st.session_state.sim_followup_history = []

        for msg in st.session_state.sim_followup_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if followup := st.chat_input("Ask about this prediction...", key="sim_chat_input"):
            st.session_state.sim_followup_history.append({"role": "user", "content": followup})
            with st.chat_message("user"):
                st.markdown(followup)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    followup_response = get_gemini_response(
                        st.session_state.sim_followup_history,
                        st.session_state.sim_result_context
                    )
                st.markdown(followup_response)
            st.session_state.sim_followup_history.append(
                {"role": "assistant", "content": followup_response})

        # ── Next Step card ──
        st.markdown("---")
        st.markdown(f"""
        <div class="card" style="border-left:3px solid #00FFB2;margin-top:8px;">
            <div style="color:#00FFB2;font-weight:700;font-size:14px;margin-bottom:8px;">Ready for intervention planning?</div>
            <div style="color:#CBD5E0;font-size:13px;line-height:1.8;">
                Your current outcome score is <span style="color:{gc};font-family:'JetBrains Mono';font-weight:600;">{risk:.1%}</span>.
                Go to the <b>Counterfactual</b> page to find the minimum changes needed to reach a target score —
                both improving a low score or reducing a high one.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 7: COUNTERFACTUAL
# ─────────────────────────────────────────────────────────────
if selected_page == "Counterfactual":
    st.markdown("""
    <div class="page-header">
        <h1>Counterfactual Analysis</h1>
        <p>Find the minimum changes to your inputs that would push the outcome score
           above a goal threshold or below a risk threshold — in either direction.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    if st.session_state.get("_sim_risk") is None:
        st.info("Run a simulation first on the Simulation & Analysis page to set up your input scenario, then come here.")
        st.stop()

    df            = st.session_state.df_cleaned
    target        = st.session_state.target_name
    X_str         = df.drop(columns=[target])
    is_classifier = st.session_state.model_type == "Classification"
    num_cols_cf   = [c for c in X_str.columns if pd.api.types.is_numeric_dtype(X_str[c])]
    cat_cols_cf   = [c for c in X_str.columns if not pd.api.types.is_numeric_dtype(X_str[c])]

    current_risk  = st.session_state["_sim_risk"]
    risk_label    = st.session_state["_sim_risk_label"]
    raw_vec       = st.session_state["_sim_input_enc"]
    current_pred  = st.session_state["_sim_pred"]
    gc_cf         = '#00FFB2' if current_risk < 0.25 else '#ECC94B' if current_risk < 0.5 else '#FF6B35' if current_risk < 0.75 else '#FF4757'

    t_min_cf = float(df[target].min()) if not is_classifier else 0.0
    t_max_cf = float(df[target].max()) if not is_classifier else 1.0

    # ── Current prediction display ──
    pred_display_cf = f"{float(current_pred):.2f}" if not is_classifier else str(current_pred)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px;">
        <div class="card" style="text-align:center;padding:16px;">
            <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Predicted {target}</div>
            <div style="font-size:32px;font-weight:700;font-family:'JetBrains Mono';color:{gc_cf};">{pred_display_cf}</div>
            <div style="color:#6B8BAE;font-size:11px;margin-top:2px;">Range: {t_min_cf:.1f} – {t_max_cf:.1f}</div>
        </div>
        <div class="card" style="text-align:center;padding:16px;">
            <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Outcome Score</div>
            <div style="font-size:32px;font-weight:700;font-family:'JetBrains Mono';color:{gc_cf};">{current_risk:.1%}</div>
            <div style="color:#6B8BAE;font-size:11px;margin-top:2px;">Normalised position</div>
        </div>
        <div class="card" style="text-align:center;padding:16px;">
            <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Risk Level</div>
            <div style="font-size:32px;font-weight:700;font-family:'JetBrains Mono';color:{gc_cf};">{risk_label}</div>
            <div style="color:#6B8BAE;font-size:11px;margin-top:2px;">From last simulation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Direction toggle ──
    st.markdown('<div class="section-label">What do you want to do?</div>', unsafe_allow_html=True)
    direction_mode = st.radio(
        "Direction",
        ["Reduce — bring the predicted value DOWN (risk reduction)",
         "Improve — bring the predicted value UP (goal setting)"],
        horizontal=False,
        label_visibility="collapsed"
    )
    is_reduce = direction_mode.startswith("Reduce")

    # Threshold as actual target value (not normalised %)
    if not is_classifier:
        current_actual = float(current_pred)
        if is_reduce:
            default_target_val = max(t_min_cf, round(current_actual - (t_max_cf - t_min_cf) * 0.20, 1))
            tau_label   = f"Target: I want the predicted {target} to drop BELOW"
            tau_caption = f"Current prediction is {current_actual:.1f}. Set your target below that."
            direction_color = "#63B3ED"
        else:
            default_target_val = min(t_max_cf, round(current_actual + (t_max_cf - t_min_cf) * 0.20, 1))
            tau_label   = f"Target: I want the predicted {target} to rise ABOVE"
            tau_caption = f"Current prediction is {current_actual:.1f}. Set your target above that."
            direction_color = "#00FFB2"

        target_val = st.slider(tau_label, float(t_min_cf), float(t_max_cf), float(default_target_val), float((t_max_cf - t_min_cf) / 50))
        st.caption(tau_caption)
        # Convert actual target value → normalised threshold
        tau = float(np.clip((target_val - t_min_cf) / (t_max_cf - t_min_cf), 0.0, 1.0)) if t_max_cf != t_min_cf else 0.5
        target_display = f"{target_val:.1f}"
    else:
        if is_reduce:
            default_tau = max(0.05, round(current_risk - 0.20, 2))
            tau_label   = "Target probability — drop BELOW"
            tau_caption = f"Current probability is {current_risk:.1%}. Set threshold below that."
            direction_color = "#63B3ED"
        else:
            default_tau = min(0.90, round(current_risk + 0.20, 2))
            tau_label   = "Target probability — rise ABOVE"
            tau_caption = f"Current probability is {current_risk:.1%}. Set threshold above that."
            direction_color = "#00FFB2"
        tau = st.slider(tau_label, 0.05, 0.95, default_tau, 0.05)
        st.caption(tau_caption)
        target_display = f"{tau:.0%}"


    if st.button("Find What Needs to Change", type="primary", use_container_width=True):
        try:
            # Check if threshold is already met
            already_met = (current_risk <= tau) if is_reduce else (current_risk >= tau)
            if already_met:
                st.success(
                    f"The current prediction of {pred_display_cf} already {'meets' if is_reduce else 'exceeds'} "
                    f"your target of {target_display}. Try moving the target further from the current value."
                )
            else:
                feat_bounds = {}
                for col in num_cols_cf:
                    if col in st.session_state.feature_names:
                        idx = st.session_state.feature_names.index(col)
                        feat_bounds[col] = (float(X_str[col].min()), float(X_str[col].max()), raw_vec[idx])

                if is_reduce:
                    # Standard reduce mode — existing counterfactual_search
                    with st.spinner(f"Searching for changes to bring {current_risk:.1%} below {tau:.0%}..."):
                        suggestions, final_risk = counterfactual_search(
                            model=st.session_state.model, scaler=st.session_state.scaler,
                            feature_names=st.session_state.feature_names, input_vec=raw_vec,
                            target_threshold=tau, feature_bounds=feat_bounds,
                            is_classifier=is_classifier, t_min=t_min_cf, t_max=t_max_cf, n_steps=200,
                        )
                else:
                    # Improve mode — scan for maximum instead of minimum
                    with st.spinner(f"Searching for changes to bring {current_risk:.1%} above {tau:.0%}..."):
                        suggestions = []
                        x = list(raw_vec)
                        cur = _get_risk(st.session_state.model, st.session_state.scaler,
                                        st.session_state.feature_names, x, is_classifier, t_min_cf, t_max_cf)

                        # Build feature importance order
                        model_obj = st.session_state.model
                        if hasattr(model_obj, "feature_importances_"):
                            imps = np.array(model_obj.feature_importances_, dtype=float)
                        elif hasattr(model_obj, "coef_"):
                            raw_c = np.array(model_obj.coef_)
                            imps  = np.abs(raw_c[0]) if raw_c.ndim > 1 else np.abs(raw_c)
                            if len(imps) != len(st.session_state.feature_names):
                                imps = np.ones(len(st.session_state.feature_names))
                        else:
                            imps = np.ones(len(st.session_state.feature_names))

                        feat_order = np.argsort(imps)[::-1]

                        for fi in feat_order:
                            if fi >= len(st.session_state.feature_names):
                                continue
                            fname = st.session_state.feature_names[fi]
                            if fname not in feat_bounds:
                                continue
                            lo, hi, _ = feat_bounds[fname]
                            if lo >= hi:
                                continue

                            best_val  = x[fi]
                            best_risk = cur

                            for step in np.linspace(lo, hi, 200):
                                x_try     = list(x)
                                x_try[fi] = float(step)
                                r = _get_risk(st.session_state.model, st.session_state.scaler,
                                              st.session_state.feature_names, x_try,
                                              is_classifier, t_min_cf, t_max_cf)
                                if r > best_risk:  # maximise
                                    best_risk = r
                                    best_val  = float(step)

                            improvement = best_risk - cur
                            if improvement > 0.001 and abs(best_val - x[fi]) > 1e-6:
                                suggestions.append({
                                    "feature":     fname,
                                    "original":    float(x[fi]),
                                    "suggested":   best_val,
                                    "risk_after":  float(np.clip(best_risk, 0.0, 1.0)),
                                    "improvement": float(np.clip(improvement, 0.0, 1.0)),
                                })
                                x[fi] = best_val
                                cur   = best_risk

                            if cur >= tau:
                                break

                        final_risk = float(np.clip(cur, 0.0, 1.0))

                if not suggestions:
                    st.warning(
                        "Could not find numeric changes that reach your target. "
                        "Try adjusting categorical inputs manually or moving the target closer to the current prediction."
                    )
                else:
                    n_changes   = len(suggestions)
                    change_word = "change" if n_changes == 1 else "changes"
                    action_verb = "reduce" if is_reduce else "improve"
                    # Compute final actual prediction from final normalised risk
                    final_pred_actual = final_risk * (t_max_cf - t_min_cf) + t_min_cf if not is_classifier else final_risk

                    st.markdown(f"""
                    <div style="background:rgba({'0,255,178' if not is_reduce else '99,179,237'},0.06);
                                border:1px solid rgba({'0,255,178' if not is_reduce else '99,179,237'},0.25);
                                border-radius:12px;padding:14px 20px;margin-bottom:16px;">
                        <div style="color:{direction_color};font-size:13px;font-weight:600;">
                            Found {n_changes} {change_word} that would {action_verb} the predicted {target}
                            from <span style="font-family:'JetBrains Mono';">{pred_display_cf}</span>
                            {'below' if is_reduce else 'above'} your target of <span style="font-family:'JetBrains Mono';">{target_display}</span>.
                        </div>
                        <div style="color:#8BA3BE;font-size:12px;margin-top:4px;">
                            Apply these steps in order. Most impactful change is listed first.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Recommended Changes — Apply in this order</div>',
                                unsafe_allow_html=True)

                    for rank_i, s in enumerate(suggestions):
                        direction_word = "Increase" if s["suggested"] > s["original"] else "Decrease"
                        dir_verb       = "up" if s["suggested"] > s["original"] else "down"
                        delta          = abs(s["suggested"] - s["original"])
                        improvement_pct = s["improvement"] * 100
                        # Convert risk_after back to actual predicted value for display
                        pred_after_actual = s["risk_after"] * (t_max_cf - t_min_cf) + t_min_cf if not is_classifier else s["risk_after"]

                        if improvement_pct >= 20:
                            impact_label = "Very high impact"; impact_color = "#00FFB2"
                        elif improvement_pct >= 10:
                            impact_label = "High impact"; impact_color = "#63B3ED"
                        elif improvement_pct >= 5:
                            impact_label = "Medium impact"; impact_color = "#ECC94B"
                        else:
                            impact_label = "Low impact"; impact_color = "#8BA3BE"

                        score_label  = f"Predicted {target} after step"
                        change_label = f"{target} reduction" if is_reduce else f"{target} gain"
                        change_sign  = "&minus;" if is_reduce else "+"
                        change_actual = s["improvement"] * (t_max_cf - t_min_cf) if not is_classifier else improvement_pct / 100

                        st.markdown(f"""
                        <div class="card card-green" style="padding:18px 22px;margin-bottom:10px;">
                            <div style="flex:1;">
                                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
                                    <div style="background:{direction_color};color:#0A0F1A;font-size:10px;font-weight:700;
                                         border-radius:4px;padding:2px 8px;">Step {rank_i+1}</div>
                                    <div style="color:#E2F0FF;font-weight:700;font-size:15px;">{s['feature']}</div>
                                    <div style="background:rgba(99,179,237,0.1);border:1px solid rgba(99,179,237,0.2);
                                         border-radius:20px;padding:2px 10px;font-size:11px;color:{impact_color};">
                                        {impact_label}
                                    </div>
                                </div>
                                <div style="color:#CBD5E0;font-size:14px;line-height:1.8;margin-bottom:8px;">
                                    {direction_word} <b style="color:#E2F0FF;">{s['feature']}</b>
                                    from <span style="color:#FC8181;font-family:'JetBrains Mono';font-weight:600;">{s['original']:.2f}</span>
                                    &nbsp;&rarr;&nbsp;
                                    <span style="color:{direction_color};font-family:'JetBrains Mono';font-weight:600;">{s['suggested']:.2f}</span>
                                    &nbsp;<span style="color:#6B8BAE;font-size:12px;">(move {dir_verb} by {delta:.2f})</span>
                                </div>
                                <div style="display:flex;gap:16px;flex-wrap:wrap;">
                                    <div style="background:rgba(10,15,26,0.6);border-radius:8px;padding:8px 14px;">
                                        <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;">{score_label}</div>
                                        <div style="color:#ECC94B;font-family:'JetBrains Mono';font-size:16px;font-weight:600;margin-top:2px;">{pred_after_actual:.2f}</div>
                                    </div>
                                    <div style="background:rgba(10,15,26,0.6);border-radius:8px;padding:8px 14px;">
                                        <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;">{change_label}</div>
                                        <div style="color:{direction_color};font-family:'JetBrains Mono';font-size:16px;font-weight:600;margin-top:2px;">{change_sign}{change_actual:.2f}</div>
                                    </div>
                                    <div style="background:rgba(10,15,26,0.6);border-radius:8px;padding:8px 14px;">
                                        <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Score shift</div>
                                        <div style="color:{impact_color};font-family:'JetBrains Mono';font-size:16px;font-weight:600;margin-top:2px;">{change_sign}{improvement_pct:.1f}%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Summary card using actual values ──
                    fa_color    = ('#00FFB2' if final_risk < 0.25 else '#ECC94B' if final_risk < 0.50 else '#FF6B35' if final_risk < 0.75 else '#FF4757')
                    reached     = (final_risk <= tau) if is_reduce else (final_risk >= tau)
                    total_delta_actual = abs(float(current_pred) - final_pred_actual) if not is_classifier else abs(current_risk - final_risk) * 100

                    if reached:
                        summary_headline = f"Target achieved — these changes are sufficient."
                        summary_color    = "#00FFB2"
                        summary_detail   = (f"Applying all {n_changes} {change_word} moves the predicted {target} "
                                            f"from {pred_display_cf} to {final_pred_actual:.2f} — "
                                            f"a total {'reduction' if is_reduce else 'improvement'} of {total_delta_actual:.2f} units.")
                    else:
                        summary_headline = f"Partial progress — target of {target_display} not fully reached."
                        summary_color    = "#FFB347"
                        summary_detail   = (f"These changes move the predicted {target} from {pred_display_cf} to {final_pred_actual:.2f}. "
                                            f"Try also adjusting categorical inputs to close the remaining gap.")

                    st.markdown(f"""
                    <div class="card" style="margin-top:8px;padding:24px;border-left:3px solid {summary_color};">
                        <div style="color:{summary_color};font-weight:700;font-size:15px;margin-bottom:10px;">{summary_headline}</div>
                        <div style="color:#CBD5E0;font-size:14px;line-height:1.8;margin-bottom:16px;">{summary_detail}</div>
                        <div style="display:flex;justify-content:space-around;align-items:center;gap:20px;flex-wrap:wrap;
                                    background:rgba(10,15,26,0.5);border-radius:10px;padding:16px;">
                            <div style="text-align:center;">
                                <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Current {target}</div>
                                <div style="font-size:30px;font-weight:700;font-family:'JetBrains Mono';color:{gc_cf};">{pred_display_cf}</div>
                            </div>
                            <div style="font-size:24px;color:#6B8BAE;">{'&darr;' if is_reduce else '&uarr;'}</div>
                            <div style="text-align:center;">
                                <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Predicted {target} after</div>
                                <div style="font-size:30px;font-weight:700;font-family:'JetBrains Mono';color:{fa_color};">{final_pred_actual:.2f}</div>
                            </div>
                            <div style="font-size:24px;color:#6B8BAE;">=</div>
                            <div style="text-align:center;">
                                <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Total {'reduction' if is_reduce else 'gain'}</div>
                                <div style="font-size:30px;font-weight:700;font-family:'JetBrains Mono';color:{direction_color};">{'&minus;' if is_reduce else '+'}{total_delta_actual:.2f}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#6B8BAE;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Your target</div>
                                <div style="font-size:30px;font-weight:700;font-family:'JetBrains Mono';color:#ECC94B;">{target_display}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Charts
                    st.markdown('<div class="section-label" style="margin-top:20px;">Visual breakdown</div>', unsafe_allow_html=True)

                    fig_cf = go.Figure()
                    fig_cf.add_trace(go.Bar(name="Current value", x=[s["feature"] for s in suggestions],
                                             y=[s["original"] for s in suggestions], marker_color='#7B61FF', opacity=0.85))
                    fig_cf.add_trace(go.Bar(name="Suggested value", x=[s["feature"] for s in suggestions],
                                             y=[s["suggested"] for s in suggestions],
                                             marker_color='#00FFB2' if not is_reduce else '#63B3ED', opacity=0.85))
                    fig_cf.update_layout(barmode='group', title="Current vs Suggested Feature Values",
                                          **CHART_LAYOUT, height=320, yaxis_title="Feature Value")
                    st.plotly_chart(fig_cf, use_container_width=True)
                    st.caption(f"Each pair of bars shows one recommended change. Purple = your current input value. {'Blue' if is_reduce else 'Green'} = the suggested new value the model recommends.")

                    # Waterfall in actual target units
                    if not is_classifier:
                        wf_start  = float(current_pred)
                        wf_steps  = [(s["improvement"] * (t_max_cf - t_min_cf)) * (-1 if is_reduce else 1) for s in suggestions]
                        wf_end    = final_pred_actual
                        wf_y_vals = [wf_start] + wf_steps + [wf_end]
                        wf_labels = [f"{wf_start:.1f}"] + [f"{'+' if v > 0 else ''}{v:.1f}" for v in wf_steps] + [f"{wf_end:.1f}"]
                        yaxis_title_wf = f"Predicted {target}"
                    else:
                        wf_y_vals = [current_risk * 100] + [(s["improvement"] * 100) * (-1 if is_reduce else 1) for s in suggestions] + [final_risk * 100]
                        wf_labels = [f"{v:.1f}%" for v in wf_y_vals]
                        yaxis_title_wf = "Probability (%)"

                    wf_x = [f"Start: {pred_display_cf}"] + [f"Step {i+1}: {s['feature']}" for i, s in enumerate(suggestions)] + [f"Final: {final_pred_actual:.1f}" if not is_classifier else f"Final: {final_risk:.1%}"]
                    wf_m = ["absolute"] + ["relative"] * len(suggestions) + ["total"]

                    fig_wf = go.Figure(go.Waterfall(
                        orientation="v", measure=wf_m, x=wf_x, y=wf_y_vals,
                        connector={"line": {"color": "rgba(99,179,237,0.3)"}},
                        decreasing={"marker": {"color": "#00FFB2" if is_reduce else "#FF4757"}},
                        increasing={"marker": {"color": "#FF4757" if is_reduce else "#00FFB2"}},
                        totals={"marker": {"color": fa_color}},
                        text=wf_labels, textposition="outside"
                    ))
                    fig_wf.update_layout(
                        title=f"How the predicted {target} {'drops' if is_reduce else 'rises'} step by step",
                        **CHART_LAYOUT, height=380, yaxis_title=yaxis_title_wf, showlegend=False
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)
                    st.caption(f"Each bar shows the contribution of one feature change to the final predicted {target}. The last bar is your final predicted value after all changes are applied.")

        except Exception as e:
            st.error(f"Counterfactual search failed: {e}")

# ─────────────────────────────────────────────────────────────
# PAGE 8: AI CONSULTANT
# ─────────────────────────────────────────────────────────────
if selected_page == "AI Consultant":
    st.markdown("""
    <div class="page-header">
        <h1>AI Consultant</h1>
        <p>Powered by Gemini. Ask questions about your results, or generate a full plain-English analysis report.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is not None:
        score_disp = f"{st.session_state.train_score:.4f}" if st.session_state.train_score else "N/A"
        m_info     = f"Model: {type(st.session_state.model).__name__} ({st.session_state.model_type}), Score: {score_disp}"
    else:
        m_info = "No model trained yet."

    if st.session_state.df_cleaned is not None:
        data_info = (f"Dataset: {st.session_state.df_cleaned.shape[0]} rows, "
                     f"{st.session_state.df_cleaned.shape[1]} columns, "
                     f"Target: {st.session_state.target_name}")
    else:
        data_info = "No dataset loaded."

    gov_info = ""
    if st.session_state.gov_results is not None:
        top      = st.session_state.gov_results.iloc[0]
        gov_info = (f"Governance winner: {top['Model']} with score {top['Governance Score']:.4f} "
                    f"(Accuracy {top['Accuracy']:.4f}, Stability {top['Stability']:.4f}, "
                    f"Fairness {top['Fairness']:.4f})")

    full_ctx = f"{m_info} | {data_info} | {gov_info} | Last simulation: {st.session_state.sim_result_context}"

    # ── NLP Report Generation ────────────────────────────────────
    st.markdown('<div class="section-label">Generate Analysis Report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="margin-bottom:16px;">
        <div style="color:#E2F0FF;font-weight:600;margin-bottom:6px;">Full Plain-English Report</div>
        <div style="color:#8BA3BE;font-size:13px;line-height:1.7;">
            Generates a complete narrative summary of your entire analysis — dataset overview,
            model performance, governance scores, and simulation results — written in plain English
            that any non-technical stakeholder can read and understand. You can copy and paste this
            into a report or presentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Full Analysis Report", use_container_width=True):
        if st.session_state.model is None:
            st.warning("Train a model first before generating a report.")
        else:
            # Build a comprehensive context for the report
            df_ctx = st.session_state.df_cleaned
            num_numeric = len(df_ctx.select_dtypes(include='number').columns) if df_ctx is not None else 0
            num_categorical = len(df_ctx.select_dtypes(include='object').columns) if df_ctx is not None else 0

            gov_ctx = ""
            if st.session_state.gov_results is not None:
                rows = []
                for _, row in st.session_state.gov_results.iterrows():
                    rows.append(f"{row['Model']}: governance={row['Governance Score']:.4f}, "
                                f"accuracy={row['Accuracy']:.4f}, stability={row['Stability']:.4f}, "
                                f"fairness={row['Fairness']:.4f}")
                gov_ctx = "Governance results: " + " | ".join(rows)

            report_prompt = (
                f"You are InspectorML, an AI analyst. Write a complete, professional analysis report "
                f"in plain English based on the following session data. "
                f"The report should be readable by a non-technical person — a manager, teacher, or policy-maker. "
                f"Structure it with clear sections. Write in full sentences and paragraphs, not bullet points. "
                f"Be specific — use the actual numbers provided.\n\n"
                f"SESSION DATA:\n"
                f"- Dataset: {st.session_state.df_cleaned.shape[0]:,} rows, "
                f"{st.session_state.df_cleaned.shape[1]} columns "
                f"({num_numeric} numeric, {num_categorical} categorical)\n"
                f"- Target variable: {st.session_state.target_name}\n"
                f"- Problem type: {st.session_state.model_type}\n"
                f"- Trained model: {type(st.session_state.model).__name__}\n"
                f"- Model score: {st.session_state.train_score:.4f}\n"
                f"- {gov_ctx}\n"
                f"- Last simulation: {st.session_state.sim_result_context}\n\n"
                f"Write the report with these sections:\n"
                f"1. Executive Summary (2-3 sentences — what this analysis is about and the key finding)\n"
                f"2. Dataset Overview (what data was used, how many records, what was being predicted)\n"
                f"3. Model Performance (how well the model performed, what the score means in plain English)\n"
                f"4. Governance Assessment (if available — which model won, why, what the fairness and stability scores mean)\n"
                f"5. Simulation Results (what the last simulation showed, what it means for a real decision)\n"
                f"6. Recommendations (3 concrete, actionable recommendations based on the findings)\n\n"
                f"Keep the total length to around 400-500 words."
            )

            with st.spinner("Writing your report..."):
                report_text = get_gemini_response(
                    [{"role": "user", "content": report_prompt}],
                    full_ctx
                )

            st.markdown(f"""
            <div class="card card-accent" style="margin-top:8px;">
                <div style="color:#63B3ED;font-weight:600;font-size:13px;margin-bottom:14px;">
                    Analysis Report — InspectorML Pro
                </div>
                <div style="color:#CBD5E0;font-size:14px;line-height:1.9;">
                    {report_text.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Offer plain text copy
            st.download_button(
                label="Download report as text file",
                data=report_text,
                file_name="inspectorml_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("---")

    # ── Quick Questions ──────────────────────────────────────────
    st.markdown('<div class="section-label">Quick Questions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    quick_prompt = None
    if c1.button("Explain the last simulation",   use_container_width=True):
        quick_prompt = "Explain my last simulation in plain English. What does the outcome score mean and what should I take away from it?"
    if c2.button("Interpret the governance scores", use_container_width=True):
        quick_prompt = "Explain the governance scores. Which model should I use and why? What do the stability and fairness numbers mean in practice?"
    if c3.button("What should I improve?",        use_container_width=True):
        quick_prompt = "Based on what you can see of my project, what should I improve — in the model, the data, or the analysis?"

    if quick_prompt:
        st.session_state.chat_history.append({"role": "user", "content": quick_prompt})
        with st.spinner("Thinking..."):
            ai_response = get_gemini_response(st.session_state.chat_history, full_ctx)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your model, data, governance scores, or counterfactual results..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_response = get_gemini_response(st.session_state.chat_history, full_ctx)
            st.markdown(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()