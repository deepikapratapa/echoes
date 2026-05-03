import sys
sys.path.insert(0, '/Users/saturnine/echoes')

import streamlit as st
from dotenv import load_dotenv
load_dotenv('/Users/saturnine/echoes/.env', override=True)

st.set_page_config(
    page_title="echoes",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("/Users/saturnine/echoes/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 echoes")
    st.markdown("*your music, reflected back*")
    st.divider()

    username = st.text_input(
        "last.fm username",
        value=st.session_state.get('username', ''),
        placeholder="e.g. thesaturnineeee",
        key="sidebar_username"
    )
    if username:
        st.session_state['username'] = username
        st.success(f"✓ {username}")

    st.divider()
    st.markdown("### navigate")
    st.page_link("app.py",              label="🏠 home")
    st.page_link("pages/01_profile.py", label="📊 your profile")
    st.page_link("pages/02_generate.py",label="✨ generate playlist")
    st.page_link("pages/03_result.py",  label="🎧 your playlist")
    st.page_link("pages/04_social.py",  label="👥 compare tastes")

# ── home ──────────────────────────────────────────────────────
st.markdown("# echoes")
st.markdown("### *your music, reflected back*")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 📊 profile")
    st.markdown("Visualize your listening universe — taste clusters, personality archetype, listening heatmap")
    if st.button("explore profile →", key="btn_profile"):
        st.switch_page("pages/01_profile.py")

with col2:
    st.markdown("#### ✨ generate")
    st.markdown("Name your playlist, describe the vibe — get AI-curated recommendations with moodboard")
    if st.button("generate playlist →", key="btn_generate"):
        st.switch_page("pages/02_generate.py")

with col3:
    st.markdown("#### 👥 social")
    st.markdown("Compare taste profiles with a friend — find your overlap and generate a shared playlist")
    if st.button("compare tastes →", key="btn_social"):
        st.switch_page("pages/04_social.py")

st.divider()
st.markdown(
    "<p style='text-align:center;color:#333;font-size:12px'>"
    "built with Last.fm API · sentence-transformers · scikit-learn · hugging face · streamlit"
    "</p>",
    unsafe_allow_html=True
)
