"""
app.py — Main entry point for the Disease Prediction Streamlit App.

Run with:
    streamlit run app.py

Navigation flow:
    Sidebar → Login  →  Register  →  Predict
"""

import streamlit as st
from auth import init_db, login_user, register_user
from utils.preprocess import get_display_symptoms, symptoms_to_feature_vector, validate_symptom_count
from utils.predictor import predict_disease, get_full_disease_info

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MediPredict — Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize DB (creates users table if missing) ────────────────────────────
init_db()

# ── Session state defaults ────────────────────────────────────────────────────
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "username"    not in st.session_state: st.session_state.username    = ""
if "active_page" not in st.session_state: st.session_state.active_page = "Login"


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Serif+Display&display=swap');

        /* ── Root palette ── */
        :root {
            --bg:        #0d1117;
            --surface:   #161b22;
            --border:    #30363d;
            --accent:    #2ea043;
            --accent2:   #1f6feb;
            --warn:      #d29922;
            --danger:    #f85149;
            --text:      #e6edf3;
            --muted:     #8b949e;
            --card-bg:   #1c2128;
        }

        /* ── Global reset ── */
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Sora', sans-serif !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }

        /* ── Headings ── */
        h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
        h1 { font-size: 2.6rem !important; }

        /* ── Inputs ── */
        input, textarea, [data-baseweb="input"] input {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        input:focus { border-color: var(--accent2) !important; }

        /* ── Buttons ── */
        .stButton > button {
            background: var(--accent2) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 2rem !important;
            font-family: 'Sora', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.03em !important;
            transition: filter 0.2s, transform 0.15s !important;
        }
        .stButton > button:hover {
            filter: brightness(1.15) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Cards ── */
        .medi-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .medi-card h4 {
            font-family: 'DM Serif Display', serif;
            font-size: 1.15rem;
            margin-bottom: 0.6rem;
            color: var(--text);
        }

        /* ── Badge / Chip ── */
        .chip {
            display: inline-block;
            background: #21262d;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.82rem;
            margin: 3px;
            color: var(--text);
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; }

        /* ── Disease result banner ── */
        .disease-banner {
            background: linear-gradient(135deg, #1a2a3a 0%, #0d2137 100%);
            border: 1px solid var(--accent2);
            border-radius: 16px;
            padding: 1.8rem 2rem;
            text-align: center;
            margin-bottom: 1.8rem;
        }
        .disease-banner .label {
            color: var(--muted);
            font-size: 0.85rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .disease-banner .disease-name {
            font-family: 'DM Serif Display', serif;
            font-size: 2.2rem;
            color: #58a6ff;
        }
        .disease-banner .confidence {
            color: var(--accent);
            font-size: 1rem;
            margin-top: 0.4rem;
        }

        /* ── Auth card ── */
        .auth-card {
            max-width: 460px;
            margin: 3rem auto;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 2.5rem 2.8rem;
        }
        .auth-card h2 {
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .auth-subtitle {
            text-align: center;
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 1.8rem;
        }

        /* ── Multiselect tags ── */
        [data-baseweb="tag"] {
            background: #1f6feb22 !important;
            border: 1px solid var(--accent2) !important;
            border-radius: 6px !important;
        }

        /* ── Metric delta ── */
        [data-testid="metric-container"] {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
        }

        /* ── Expander ── */
        [data-testid="stExpander"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        /* ── Disclaimer box ── */
        .disclaimer {
            background: #2d1b00;
            border: 1px solid var(--warn);
            border-radius: 10px;
            padding: 0.9rem 1.2rem;
            font-size: 0.85rem;
            color: #f0c060;
            margin-top: 1.5rem;
        }

        /* hide streamlit default menu & footer in prod */
        #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
                <span style='font-size:2.8rem;'>🩺</span><br>
                <span style='font-family:"DM Serif Display",serif;
                             font-size:1.5rem; color:#e6edf3;'>MediPredict</span><br>
                <span style='color:#8b949e; font-size:0.78rem;'>
                    AI-Powered Disease Prediction
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.session_state.logged_in:
            st.markdown(
                f"<div style='text-align:center; margin-bottom:1rem;'>"
                f"<span style='color:#2ea043; font-size:0.85rem;'>● Logged in as </span>"
                f"<b style='color:#e6edf3;'>{st.session_state.username}</b></div>",
                unsafe_allow_html=True,
            )
            nav_options = ["🔬 Predict Disease", "🚪 Logout"]
        else:
            nav_options = ["🔑 Login", "📝 Register"]

        choice = st.segmented_control("Navigation", nav_options, label_visibility="collapsed")

        if choice == "🔬 Predict Disease":
            st.session_state.active_page = "Predict"
        elif choice == "🔑 Login":
            st.session_state.active_page = "Login"
        elif choice == "📝 Register":
            st.session_state.active_page = "Register"
        elif choice == "🚪 Logout":
            st.session_state.logged_in   = False
            st.session_state.username    = ""
            st.session_state.active_page = "Login"
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='color:#8b949e; font-size:0.75rem; text-align:center;'>"
            "⚠️ For educational purposes only.<br>Not a substitute for medical advice."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_login() -> None:
    st.markdown(
        "<div class='auth-card'>"
        "<h2>Welcome Back 👋</h2>"
        "<div class='auth-subtitle'>Sign in to your MediPredict account</div>",
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        ok, msg = login_user(username, password)
        if ok:
            st.session_state.logged_in   = True
            st.session_state.username    = username.strip()
            st.session_state.active_page = "Predict"
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.markdown(
        "<div style='text-align:center; margin-top:1rem; color:#8b949e; font-size:0.88rem;'>"
        "Don't have an account? Use the <b>Register</b> option in the sidebar."
        "</div></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTER PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_register() -> None:
    st.markdown(
        "<div class='auth-card'>"
        "<h2>Create Account 🧬</h2>"
        "<div class='auth-subtitle'>Join MediPredict — it's free</div>",
        unsafe_allow_html=True,
    )

    with st.form("register_form", clear_on_submit=True):
        username  = st.text_input("Username", placeholder="Choose a username (min 3 chars)")
        password  = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)")
        password2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        if password != password2:
            st.error("Passwords do not match. Please try again.")
        else:
            ok, msg = register_user(username, password)
            if ok:
                st.success(msg + " ✅")
                st.info("Head to the Login page to sign in.")
            else:
                st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_info_card(icon: str, title: str, items: list[str], accent_color: str = "#1f6feb") -> None:
    """Render a stylised card for descriptions, diet, meds, etc."""
    items_html = "".join(
        f"<div style='display:flex; align-items:flex-start; gap:8px; "
        f"margin-bottom:6px;'>"
        f"<span style='color:{accent_color}; margin-top:2px;'>◆</span>"
        f"<span style='color:#c9d1d9; font-size:0.9rem;'>{item}</span>"
        f"</div>"
        for item in items
    )
    st.markdown(
        f"""
        <div class='medi-card'>
            <h4>{icon} {title}</h4>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_predict() -> None:
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='margin-bottom:0.1rem;'>Disease Prediction</h1>"
        f"<p style='color:#8b949e; margin-top:0;'>Select your symptoms and let the AI analyse them</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Symptom selector ─────────────────────────────────────────────────────
    all_display_symptoms = get_display_symptoms()

    col_sel, col_info = st.columns([3, 1])
    with col_sel:
        st.markdown("#### 🔍 Select Symptoms")
        selected_symptoms = st.multiselect(
            label="Choose all that apply:",
            options=all_display_symptoms,
            placeholder="Type to search symptoms…",
            help="Select at least 2 symptoms for a reliable prediction.",
        )

    with col_info:
        st.markdown("#### 📊 Selection")
        st.metric("Symptoms selected", len(selected_symptoms))

    # Show chips of selected symptoms
    if selected_symptoms:
        chips = " ".join(f"<span class='chip'>{s}</span>" for s in selected_symptoms)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("")

    # ── Predict button ────────────────────────────────────────────────────────
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("🔬 Analyse Symptoms", use_container_width=True)

    # ── Prediction logic ──────────────────────────────────────────────────────
    if predict_btn:
        valid, err = validate_symptom_count(selected_symptoms)
        if not valid:
            st.warning(err)
            return

        with st.spinner("Running AI analysis…"):
            feature_vec            = symptoms_to_feature_vector(selected_symptoms)
            predicted_disease, top5 = predict_disease(feature_vec)
            info                   = get_full_disease_info(predicted_disease)

        st.markdown("---")

        # ── Disease banner ────────────────────────────────────────────────────
        top_conf = top5.get(predicted_disease, 0.0)
        st.markdown(
            f"""
            <div class='disease-banner'>
                <div class='label'>AI Prediction Result</div>
                <div class='disease-name'>{predicted_disease.title()}</div>
                <div class='confidence'>Confidence: {top_conf:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-5 alternatives (expander) ──────────────────────────────────────
        with st.expander("📈 View Top-5 Predictions"):
            for disease, prob in top5.items():
                bar_color = "#1f6feb" if disease == predicted_disease else "#30363d"
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:10px; margin:6px 0;'>"
                    f"<span style='width:220px; color:#c9d1d9; font-size:0.88rem;'>{disease.title()}</span>"
                    f"<div style='flex:1; background:#21262d; border-radius:6px; height:14px; overflow:hidden;'>"
                    f"<div style='width:{prob}%; background:{bar_color}; height:100%; border-radius:6px;'></div>"
                    f"</div>"
                    f"<span style='width:55px; text-align:right; color:#8b949e; font-size:0.85rem;'>{prob:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Info cards ────────────────────────────────────────────────────────
        # Description — full width
        render_info_card(
            "📋", "Description",
            [info["description"]],
            accent_color="#58a6ff",
        )

        col1, col2 = st.columns(2)
        with col1:
            render_info_card("🥗", "Recommended Diet",        info["diet"],        "#2ea043")
            render_info_card("⚠️",  "Precautions",             info["precautions"], "#d29922")
        with col2:
            render_info_card("💊", "Medications",             info["medications"], "#f85149")
            render_info_card("🏃", "Suggested Workouts",      info["workouts"],    "#a371f7")

        # ── Medical disclaimer ────────────────────────────────────────────────
        st.markdown(
            "<div class='disclaimer'>"
            "⚠️ <b>Medical Disclaimer:</b> This tool is intended for educational and "
            "informational purposes only. It is <b>not</b> a substitute for professional "
            "medical advice, diagnosis, or treatment. Always consult a qualified "
            "healthcare provider with any questions you may have regarding a medical condition."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    render_sidebar()

    page = st.session_state.active_page

    if page == "Login":
        page_login()
    elif page == "Register":
        page_register()
    elif page == "Predict":
        if st.session_state.logged_in:
            page_predict()
        else:
            st.warning("🔒 Please log in to access the prediction tool.")
            page_login()


if __name__ == "__main__":
    main()
