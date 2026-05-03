import sys
sys.path.insert(0, '/Users/saturnine/echoes')

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
load_dotenv('/Users/saturnine/echoes/.env', override=True)

st.set_page_config(page_title="echoes — social", page_icon="👥", layout="wide")

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
st.markdown("# 👥 compare tastes")
st.markdown("*find your musical overlap*")
st.divider()

username1 = st.session_state.get('username', '')
col1, col2 = st.columns(2)
with col1:
    u1 = st.text_input("your username", value=username1,
                        placeholder="thesaturnineeee")
with col2:
    u2 = st.text_input("friend's username", placeholder="their last.fm username")

if not u1 or not u2:
    st.info("enter both usernames to compare taste profiles")
    st.stop()

if st.button("🔍 compare", key="btn_compare", type="primary"):
    from utils.lastfm import get_top_artists

    with st.spinner(f"fetching {u1}'s top artists..."):
        try:
            df1 = get_top_artists(u1, limit=50)
            df1['source'] = u1
        except Exception as e:
            st.error(f"could not fetch {u1}: {e}")
            st.stop()

    with st.spinner(f"fetching {u2}'s top artists..."):
        try:
            df2 = get_top_artists(u2, limit=50)
            df2['source'] = u2
        except Exception as e:
            st.error(f"could not fetch {u2}: {e}")
            st.stop()

    artists1 = set(df1['name'].str.lower())
    artists2 = set(df2['name'].str.lower())

    shared   = artists1 & artists2
    only1    = artists1 - artists2
    only2    = artists2 - artists1

    overlap_pct = len(shared) / len(artists1 | artists2) * 100

    # compatibility label
    if overlap_pct > 50:
        label, color = "Sonic Soulmates 💜", "#8b5cf6"
    elif overlap_pct > 30:
        label, color = "Kindred Listeners 🎵", "#1D9E75"
    elif overlap_pct > 15:
        label, color = "Curious Crossover 🌊", "#378ADD"
    else:
        label, color = "Beautifully Opposite 🌙", "#F0997B"

    st.divider()

    # ── overlap score ─────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("shared artists",  len(shared))
    c2.metric("taste overlap",   f"{overlap_pct:.1f}%")
    c3.metric("compatibility",   label)

    st.markdown(
        f"<div style='background:#111;border:1px solid #2a2a2a;border-radius:12px;"
        f"padding:1.5rem;text-align:center;margin:1rem 0'>"
        f"<div style='font-size:32px;font-weight:600;color:{color}'>{label}</div>"
        f"<div style='color:#666;font-size:14px;margin-top:0.5rem'>"
        f"{overlap_pct:.1f}% taste overlap · {len(shared)} shared artists"
        f"</div></div>",
        unsafe_allow_html=True
    )

    # ── venn breakdown ────────────────────────────────────────
    st.markdown("#### taste breakdown")
    tc1, tc2, tc3 = st.columns(3)

    with tc1:
        st.markdown(f"**only {u1}** ({len(only1)})")
        only1_artists = df1[df1['name'].str.lower().isin(only1)].head(8)
        for _, row in only1_artists.iterrows():
            st.markdown(
                f"<div style='color:#aaa;font-size:13px;padding:2px 0'>"
                f"🎵 {row['name']}</div>",
                unsafe_allow_html=True
            )

    with tc2:
        st.markdown(f"**shared** ({len(shared)})")
        shared_artists = df1[df1['name'].str.lower().isin(shared)].head(8)
        for _, row in shared_artists.iterrows():
            st.markdown(
                f"<div style='color:#8b5cf6;font-size:13px;padding:2px 0'>"
                f"💜 {row['name']}</div>",
                unsafe_allow_html=True
            )

    with tc3:
        st.markdown(f"**only {u2}** ({len(only2)})")
        only2_artists = df2[df2['name'].str.lower().isin(only2)].head(8)
        for _, row in only2_artists.iterrows():
            st.markdown(
                f"<div style='color:#aaa;font-size:13px;padding:2px 0'>"
                f"🎵 {row['name']}</div>",
                unsafe_allow_html=True
            )

    # ── playcount comparison ──────────────────────────────────
    st.divider()
    st.markdown("#### playcount comparison — top 10 shared artists")

    if shared:
        shared_df1 = df1[df1['name'].str.lower().isin(shared)][['name','playcount']].copy()
        shared_df1.columns = ['name', u1]
        shared_df2 = df2[df2['name'].str.lower().isin(shared)][['name','playcount']].copy()
        shared_df2.columns = ['name', u2]

        merged = shared_df1.merge(shared_df2, on='name').head(10)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name=u1, x=merged['name'], y=merged[u1],
            marker_color='#8b5cf6'
        ))
        fig_bar.add_trace(go.Bar(
            name=u2, x=merged['name'], y=merged[u2],
            marker_color='#1D9E75'
        ))
        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
            font=dict(color='#e8e8e8'),
            xaxis=dict(gridcolor='#222', tickangle=-30),
            yaxis=dict(gridcolor='#222', title='plays'),
            legend=dict(bgcolor='#111111'),
            height=400,
            margin=dict(l=10, r=10, t=10, b=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    if st.button("✨ generate shared playlist", key="btn_shared"):
        # store shared context and redirect to generate
        shared_vibe = f"blend of {u1} and {u2}'s taste — {label}"
        st.session_state['playlist_name'] = f"{u1} × {u2}"
        st.session_state['vibe_text']     = shared_vibe
        st.switch_page("pages/02_generate.py")
