import sys
sys.path.insert(0, '/Users/saturnine/echoes')

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv('/Users/saturnine/echoes/.env', override=True)

st.set_page_config(page_title="echoes — playlist", page_icon="🎧", layout="wide")

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
st.markdown("# 🎧 your playlist")
st.markdown("*curated for your vibe*")
st.divider()

recs           = st.session_state.get('recommendations', None)
playlist_name  = st.session_state.get('playlist_name', 'my playlist')
vibe_text      = st.session_state.get('vibe_text', '')
feature_scores = st.session_state.get('feature_scores', {})

if not recs:
    st.info("no playlist generated yet")
    if st.button("✨ generate a playlist →"):
        st.switch_page("pages/02_generate.py")
    st.stop()

df = pd.DataFrame(recs)

# ── playlist header ───────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#4c1d95,#1D9E75);"
        "width:110px;height:110px;border-radius:12px;display:flex;"
        "align-items:center;justify-content:center;font-size:44px'>🎵</div>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(f"## {playlist_name}")
    if vibe_text:
        st.markdown(
            f"<p style='color:#666;font-style:italic;font-size:14px'>"
            f"\"{vibe_text[:100]}{'...' if len(vibe_text)>100 else ''}\""
            f"</p>", unsafe_allow_html=True
        )
    st.markdown(
        f"<span style='color:#444;font-size:13px'>{len(df)} tracks · echoes</span>",
        unsafe_allow_html=True
    )

st.divider()

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎵 tracks", "📊 vibe DNA", "🎨 moodboard"])

with tab1:
    # fetch top track per recommended artist
    if 'track_list' not in st.session_state or \
       st.session_state.get('track_list_playlist') != playlist_name:
        with st.spinner("fetching top tracks..."):
            from utils.lastfm import get_artist_top_track
            import time
            track_list = []
            for _, row in df.iterrows():
                track = get_artist_top_track(row['artist'])
                track['tags']         = row.get('tags', [])
                track['reason']       = row.get('reason', '')
                track['via']          = row.get('via', [])
                track['final_score']  = row.get('final_score', 0)
                track['cb_pct']       = row.get('cb_pct', 0)
                track['cf_pct']       = row.get('cf_pct', 0)
                track['vibe_pct']     = row.get('vibe_pct', 0)
                track_list.append(track)
                time.sleep(0.15)
            st.session_state['track_list'] = track_list
            st.session_state['track_list_playlist'] = playlist_name
    else:
        track_list = st.session_state['track_list']

    for i, track in enumerate(track_list):
        tags = track.get('tags', [])
        if isinstance(tags, str):
            import ast
            try:    tags = ast.literal_eval(tags)
            except: tags = []
        tags_str = ', '.join(tags[:3]) if tags else ''

        via = track.get('via', [])
        if isinstance(via, str):
            import ast
            try:    via = ast.literal_eval(via)
            except: via = []
        via_str = ', '.join(via[:2]) if via else 'tag match'

        fin = float(track.get('final_score', 0))
        cb  = int(track.get('cb_pct', 0))
        cf  = int(track.get('cf_pct', 0))
        vb  = int(track.get('vibe_pct', 0))
        url = track.get('url', '')

        title_html = (
            f"<a href='{url}' target='_blank' "
            f"style='color:#e8e8e8;text-decoration:none'>{track['title']}</a>"
            if url else track['title']
        )

        st.markdown(
            f"<div style='background:#111;border:1px solid #1a1a1a;border-radius:12px;"
            f"padding:0.9rem 1.25rem;margin-bottom:8px;display:flex;"
            f"align-items:center;gap:16px'>"
            f"<div style='color:#444;font-size:13px;min-width:24px'>{i+1}</div>"
            f"<div style='flex:1'>"
            f"<div style='font-size:15px;font-weight:500'>{title_html}</div>"
            f"<div style='color:#666;font-size:13px'>{track['artist']}</div>"
            f"<div style='color:#444;font-size:11px;margin-top:2px'>{tags_str}</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div style='color:#666;font-size:11px'>{track.get('reason','')}</div>"
            f"<div style='color:#444;font-size:11px'>via {via_str[:35]}</div>"
            f"</div>"
            f"<div style='text-align:right;min-width:80px'>"
            f"<div style='color:#8b5cf6;font-size:14px;font-weight:500'>{fin:.3f}</div>"
            f"<div style='color:#333;font-size:10px'>cb:{cb}% cf:{cf}% v:{vb}%</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

with tab2:
    if feature_scores:
        st.markdown("#### vibe radar")
        feature_labels = {
            'energy':'⚡ energy','valence':'😊 mood','acousticness':'🎸 acoustic',
            'danceability':'💃 dance','tempo':'🥁 tempo','darkness':'🌙 darkness',
        }
        cats  = [feature_labels.get(k, k) for k in feature_scores]
        vals  = list(feature_scores.values())
        vals += vals[:1]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill='toself',
            fillcolor='rgba(139,92,246,0.2)',
            line=dict(color='#8b5cf6', width=2),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='#111111',
                radialaxis=dict(visible=True, range=[0,1],
                               gridcolor='#222', tickfont=dict(color='#444')),
                angularaxis=dict(gridcolor='#222',
                                tickfont=dict(color='#aaa', size=11)),
            ),
            paper_bgcolor='#0a0a0a', showlegend=False,
            height=380, margin=dict(l=80,r=80,t=40,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        fcols = st.columns(6)
        for i, (feat, score) in enumerate(feature_scores.items()):
            with fcols[i]:
                st.metric(feature_labels.get(feat, feat), f"{int(score*100)}%")

        st.divider()
        st.markdown(
            "<div style='background:#111;border-radius:12px;padding:1rem 1.5rem;"
            "font-size:13px;color:#888;line-height:2.2'>"
            "🟣 <b style='color:#8b5cf6'>content</b> — tag similarity to your top artists<br>"
            "🟢 <b style='color:#1D9E75'>collab</b> — listener overlap signal from Last.fm<br>"
            "🌙 <b style='color:#AFA9EC'>vibe</b> — mood alignment with your description"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("generate a playlist with a vibe description to see DNA breakdown")

with tab3:
    st.markdown("#### playlist moodboard")
    st.markdown("*a visual interpretation of your vibe*")

    # generate color palette from vibe
    vibe_lower = (vibe_text or playlist_name).lower()

    # mood → color mapping
    if any(w in vibe_lower for w in ['sad','broken','rain','night','dark','melanchol']):
        colors = ['#0d0d1a','#1a1035','#2d1b69','#1a0a2e','#0a0a1a','#16213e']
        mood_label = "dark & introspective"
    elif any(w in vibe_lower for w in ['happy','summer','party','alive','golden','bright']):
        colors = ['#ff6b35','#f7c59f','#efefd0','#004e89','#1a936f','#88d498']
        mood_label = "warm & euphoric"
    elif any(w in vibe_lower for w in ['focus','study','concentrate','work','calm']):
        colors = ['#2d3047','#419d78','#e0a458','#ffddd2','#93b7be','#1b1b2f']
        mood_label = "focused & clear"
    elif any(w in vibe_lower for w in ['hype','energy','gym','intense','hard']):
        colors = ['#ff0054','#ff5400','#ffbd00','#bf0603','#8d0801','#240046']
        mood_label = "high energy"
    else:
        colors = ['#1a1a2e','#16213e','#0f3460','#533483','#e94560','#2b2d42']
        mood_label = "cinematic"

    # render moodboard as colored tiles with vibe words
    vibe_words = (vibe_text or playlist_name).replace(',', ' ').split()[:12]
    top_tracks_display = [row['artist'] for _, row in df.head(6).iterrows()]

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;"
        f"margin:1rem 0'>",
        unsafe_allow_html=True
    )

    tile_contents = [
        (colors[0], playlist_name,         "18px", "600"),
        (colors[1], vibe_words[0] if vibe_words else "◆", "32px", "700"),
        (colors[2], top_tracks_display[0] if top_tracks_display else "◆", "14px", "400"),
        (colors[3], vibe_words[1] if len(vibe_words)>1 else "◆", "28px", "600"),
        (colors[4], top_tracks_display[1] if len(top_tracks_display)>1 else "◆","13px","400"),
        (colors[5], mood_label, "15px", "500"),
    ]

    tiles_html = "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:1rem 0'>"
    for color, text, size, weight in tile_contents:
        tiles_html += (
            f"<div style='background:{color};border-radius:12px;height:160px;"
            f"display:flex;align-items:center;justify-content:center;padding:1rem;"
            f"text-align:center'>"
            f"<span style='color:#ffffff;font-size:{size};font-weight:{weight};"
            f"opacity:0.9;font-family:Space Grotesk'>{text}</span>"
            f"</div>"
        )
    tiles_html += "</div>"
    st.markdown(tiles_html, unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:#444;font-size:12px;text-align:center'>"
        f"mood: {mood_label} · palette generated from your vibe</p>",
        unsafe_allow_html=True
    )

    # add HF moodboard as optional upgrade
    with st.expander("🚀 upgrade: generate AI images (requires HF_TOKEN)"):
        st.markdown(
            "Add `HF_TOKEN=your_token` to your `.env` file for AI-generated "
            "cinematic moodboard images using Stable Diffusion XL.\n\n"
            "Get a free token at [huggingface.co/settings/tokens]"
            "(https://huggingface.co/settings/tokens)"
        )

st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("← back to profile"):
        st.switch_page("pages/01_profile.py")
with c2:
    if st.button("✨ generate new playlist"):
        st.switch_page("pages/02_generate.py")
