import sys
sys.path.insert(0, '/Users/saturnine/echoes')

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/Users/saturnine/echoes/.env', override=True)

st.set_page_config(page_title="echoes — generate", page_icon="✨", layout="wide")

with open("/Users/saturnine/echoes/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 echoes")
    st.markdown("*your music, reflected back*")
    st.divider()
    username_input = st.text_input(
        "last.fm username",
        value=st.session_state.get('username', ''),
        placeholder="e.g. thesaturnineeee",
        key="sidebar_username"
    )
    if username_input:
        st.session_state['username'] = username_input
        st.success(f"✓ {username_input}")
    st.divider()
    st.page_link("app.py",               label="🏠 home")
    st.page_link("pages/01_profile.py",  label="📊 your profile")
    st.page_link("pages/02_generate.py", label="✨ generate playlist")
    st.page_link("pages/03_result.py",   label="🎧 your playlist")
    st.page_link("pages/04_social.py",   label="👥 compare tastes")

# ── main ──────────────────────────────────────────────────────
st.markdown("# ✨ generate playlist")
st.markdown("*describe the vibe, get a playlist curated for you*")
st.divider()

username = st.session_state.get('username', '')
if not username:
    st.info("enter your last.fm username in the sidebar first")
    st.stop()

st.markdown(f"generating for **{username}**")
st.divider()

# ── form ──────────────────────────────────────────────────────
playlist_name = st.text_input(
    "### name your playlist",
    value=st.session_state.get('playlist_name', ''),
    placeholder="late night drive...",
)

vibe_text = st.text_area(
    "### describe the vibe",
    value=st.session_state.get('vibe_text', ''),
    placeholder="e.g. slow, introspective, windows down at 3am, a little broken but hopeful...",
    height=120,
)

st.markdown("**quick moods** — click to append:")
mood_cols = st.columns(8)
moods = ["late night","chill","hype","heartbreak","focus","euphoric","nostalgic","sad hours"]
for i, mood in enumerate(moods):
    with mood_cols[i]:
        if st.button(mood, key=f"mood_{i}"):
            current = st.session_state.get('vibe_text', '')
            st.session_state['vibe_text'] = f"{current}, {mood}".strip(', ')
            st.rerun()

n_tracks = st.slider("how many tracks?", min_value=8, max_value=20, value=12)

st.divider()

if st.button("✨ generate playlist", key="btn_gen", type="primary"):
    if not vibe_text and not playlist_name:
        st.warning("add a playlist name or vibe description first")
        st.stop()

    # persist inputs
    st.session_state['playlist_name'] = playlist_name or "my playlist"
    st.session_state['vibe_text']     = vibe_text or ""
    st.session_state['n_tracks']      = n_tracks

    # parse vibe
    with st.spinner("parsing your vibe..."):
        from utils.vibe_parser import parse_vibe, get_feature_scores
        feature_scores = get_feature_scores(vibe_text) if vibe_text else {}
        vibe_weights   = parse_vibe(vibe_text) if vibe_text else {}

    # show decoded vibe
    if feature_scores:
        st.markdown("#### vibe decoded")
        feature_labels = {
            'energy':'⚡ energy','valence':'😊 mood','acousticness':'🎸 acoustic',
            'danceability':'💃 dance','tempo':'🥁 tempo','darkness':'🌙 darkness',
        }
        fcols = st.columns(6)
        for i, (feat, score) in enumerate(feature_scores.items()):
            with fcols[i]:
                st.metric(feature_labels.get(feat, feat), f"{int(score*100)}%")

    st.session_state['feature_scores'] = feature_scores
    st.session_state['vibe_weights']   = vibe_weights

    # generate recommendations
    with st.spinner("building your playlist... (~2 minutes)"):
        from utils.recommender import hybrid_recommend
        recs = hybrid_recommend(
            username,
            vibe_scores=vibe_weights,
            top_n=n_tracks,
        )

    if recs is not None and len(recs) > 0:
        st.session_state['recommendations'] = recs.to_dict('records')
        st.success(f"✓ {len(recs)} tracks curated for you!")
        st.balloons()
        st.markdown("---")
        st.markdown("### ✅ playlist ready!")
        st.markdown("click **🎧 your playlist** in the sidebar to view results")
    else:
        st.error("something went wrong — try again")
