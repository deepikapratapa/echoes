import sys
sys.path.insert(0, '/Users/saturnine/echoes')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
load_dotenv('/Users/saturnine/echoes/.env', override=True)

# ── page config ───────────────────────────────────────────────
st.set_page_config(page_title="echoes — profile", page_icon="📊", layout="wide")

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
st.markdown("# 📊 your profile")
st.markdown("*a deep dive into your listening universe*")
st.divider()

username = st.session_state.get('username', '')
if not username:
    st.info("enter your last.fm username in the sidebar to load your profile")
    st.stop()

DATA_DIR = '/Users/saturnine/echoes/data'

@st.cache_data
def load_profile_data(uname):
    history     = pd.read_csv(f'{DATA_DIR}/history.csv')
    top_artists = pd.read_csv(f'{DATA_DIR}/top_artists.csv')
    cluster_df  = pd.read_csv(f'{DATA_DIR}/artist_clusters.csv')
    history['datetime'] = pd.to_datetime(history['datetime'])
    history['date']     = pd.to_datetime(history['date'])
    return history, top_artists, cluster_df

with st.spinner("loading your listening data..."):
    history, top_artists, cluster_df = load_profile_data(username)

from utils.personality import get_listener_personality
personality = get_listener_personality(username)
stats       = personality['stats']
archetype   = personality['archetype']
radar       = personality['radar']

# ── stat cards ────────────────────────────────────────────────
st.markdown("### at a glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("total scrobbles", f"{stats['total_scrobbles']:,}")
c2.metric("unique artists",  f"{stats['unique_artists']}")
c3.metric("avg plays / day", f"{stats['avg_daily_plays']}")
c4.metric("peak hour",       f"{stats['peak_hour']}:00")
st.divider()

# ── archetype + radar ─────────────────────────────────────────
st.markdown("### your listener archetype")
col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown(
        f"<div style='background:#111;border:1px solid #2a2a2a;border-radius:16px;"
        f"padding:2rem;text-align:center'>"
        f"<div style='font-size:56px'>{archetype['emoji']}</div>"
        f"<div style='font-family:Space Grotesk;font-size:20px;font-weight:600;"
        f"color:{archetype['color']};margin-top:0.5rem'>{archetype['name']}</div>"
        f"<div style='color:#666;font-size:13px;margin-top:0.5rem'>"
        f"{archetype['description']}</div></div>",
        unsafe_allow_html=True
    )

with col_b:
    cats  = list(radar.keys())
    vals  = list(radar.values())
    vals += vals[:1]
    cats_d = [c.replace(' ', '\n') for c in cats]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=cats_d + [cats_d[0]],
        fill='toself',
        fillcolor='rgba(139,92,246,0.2)',
        line=dict(color='#8b5cf6', width=2),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='#111111',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#222',
                           tickfont=dict(color='#444')),
            angularaxis=dict(gridcolor='#222', tickfont=dict(color='#aaa', size=11)),
        ),
        paper_bgcolor='#0a0a0a', showlegend=False,
        height=350, margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 top artists", "🔥 heatmap", "🌍 taste clusters", "📈 timeline"
])

with tab1:
    st.markdown("#### top 20 artists (all time)")
    fig_a = px.bar(
        top_artists.head(20), x='playcount', y='name',
        orientation='h', color='playcount',
        color_continuous_scale=['#2a2a2a', '#8b5cf6'],
        labels={'playcount': 'plays', 'name': ''},
    )
    fig_a.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(color='#e8e8e8'), yaxis=dict(autorange='reversed'),
        coloraxis_showscale=False, height=600,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_a.update_xaxes(gridcolor='#222')
    fig_a.update_yaxes(gridcolor='#222')
    st.plotly_chart(fig_a, use_container_width=True)

with tab2:
    st.markdown("#### when do you listen?")
    pivot = history.groupby(['day_num','hour']).size().unstack(fill_value=0)
    day_labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot.index = [day_labels[i] for i in pivot.index]

    fig_h = px.imshow(
        pivot,
        color_continuous_scale=['#0a0a0a','#4c1d95','#8b5cf6'],
        labels=dict(x='hour of day', y='', color='plays'),
        aspect='auto',
    )
    fig_h.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(color='#e8e8e8'), height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_h, use_container_width=True)
    peak_hour = int(history['hour'].value_counts().idxmax())
    peak_day  = history['day'].value_counts().idxmax()
    st.markdown(
        f"<p style='color:#8b5cf6;font-size:14px'>peak: {peak_day}s at {peak_hour}:00</p>",
        unsafe_allow_html=True
    )

with tab3:
    st.markdown("#### your taste clusters")
    cluster_names  = {0:'Bollywood Soul',1:'Vocal Traditionalist',
                      2:'Alternative Wanderer',3:'Pop-Rock Crossover',4:'Cinematic Composer'}
    cluster_colors = ['#F0997B','#1D9E75','#8b5cf6','#378ADD','#EF9F27']
    artist_plays   = history['artist'].value_counts().to_dict()
    cluster_plays  = []
    for c in range(5):
        artists = cluster_df[cluster_df['cluster']==c]['artist'].tolist()
        plays   = sum(artist_plays.get(a,0) for a in artists)
        cluster_plays.append({'cluster': cluster_names[c], 'plays': plays})

    df_c = pd.DataFrame(cluster_plays)
    fig_p = px.pie(
        df_c, values='plays', names='cluster',
        color_discrete_sequence=cluster_colors, hole=0.5,
    )
    fig_p.update_layout(
        paper_bgcolor='#0a0a0a', font=dict(color='#e8e8e8'),
        legend=dict(bgcolor='#111111'), height=400,
    )
    fig_p.update_traces(textfont_color='#e8e8e8')
    st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    st.markdown("#### listening over time")
    daily = history.groupby('date').size().reset_index(name='plays')
    daily['rolling'] = daily['plays'].rolling(7, center=True).mean()

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=daily['date'], y=daily['plays'],
        fill='tozeroy', fillcolor='rgba(139,92,246,0.15)',
        line=dict(color='rgba(139,92,246,0.3)', width=1), name='daily plays',
    ))
    fig_t.add_trace(go.Scatter(
        x=daily['date'], y=daily['rolling'],
        line=dict(color='#8b5cf6', width=2), name='7-day avg',
    ))
    fig_t.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(color='#e8e8e8'),
        xaxis=dict(gridcolor='#222'), yaxis=dict(gridcolor='#222'),
        legend=dict(bgcolor='#111111'), height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_t, use_container_width=True)

st.divider()
if st.button("✨ generate a playlist based on your profile →"):
    st.switch_page("pages/02_generate.py")
