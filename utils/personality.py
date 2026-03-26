import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR_DEFAULT = '/Users/saturnine/echoes/data'

ARCHETYPES = {
    0: {
        'name':        'The Bollywood Soul',
        'emoji':       '🎬',
        'description': 'Cinematic Indian scores, melodic Bollywood, emotional depth',
        'color':       '#F0997B',
    },
    1: {
        'name':        'The Vocal Traditionalist',
        'emoji':       '🎙️',
        'description': 'Classical playback tradition, pure vocal artistry',
        'color':       '#1D9E75',
    },
    2: {
        'name':        'The Alternative Wanderer',
        'emoji':       '🌙',
        'description': 'Western indie, R&B, electronic, post-punk — the moody lane',
        'color':       '#8b5cf6',
    },
    3: {
        'name':        'The Pop-Rock Crossover',
        'emoji':       '🎸',
        'description': 'Guitar-driven pop-rock bridging east and west',
        'color':       '#378ADD',
    },
    4: {
        'name':        'The Cinematic Composer',
        'emoji':       '🎹',
        'description': 'Film composers, fusion, cinematic storytelling',
        'color':       '#EF9F27',
    },
}

def get_listener_personality(username: str, data_dir: str = DATA_DIR_DEFAULT) -> dict:
    """
    Derive listener personality from cluster distribution.

    Returns dict with:
        archetype     — dominant taste cluster
        cluster_dist  — % of plays in each cluster
        radar_scores  — 6-axis personality radar values
        stats         — key listening statistics
    """
    try:
        history     = pd.read_csv(f'{data_dir}/history.csv')
        top_artists = pd.read_csv(f'{data_dir}/top_artists.csv')
        cluster_df  = pd.read_csv(f'{data_dir}/artist_clusters.csv')
    except FileNotFoundError:
        return {}

    history['datetime'] = pd.to_datetime(history['datetime'])
    total_plays = len(history)

    # cluster distribution by plays
    artist_plays   = history['artist'].value_counts().to_dict()
    cluster_plays  = {i: 0 for i in range(5)}

    for _, row in cluster_df.iterrows():
        plays = artist_plays.get(row['artist'], 0)
        cluster_plays[row['cluster']] = cluster_plays.get(row['cluster'], 0) + plays

    cluster_pct = {
        ARCHETYPES[k]['name']: round(v / total_plays * 100, 1)
        for k, v in cluster_plays.items()
    }

    # dominant archetype
    dominant_cluster = max(cluster_plays, key=cluster_plays.get)
    dominant         = ARCHETYPES[dominant_cluster]

    # radar scores
    bollywood_artists = cluster_df[cluster_df['cluster'] == 0]['artist'].tolist()
    western_artists   = cluster_df[cluster_df['cluster'] == 2]['artist'].tolist()

    bollywood_plays = history[history['artist'].isin(bollywood_artists)].shape[0]
    western_plays   = history[history['artist'].isin(western_artists)].shape[0]

    top3_plays   = history['artist'].value_counts().head(3).sum()
    night_plays  = history[history['hour'].isin(list(range(20,24)) + list(range(0,3)))].shape[0]
    max_track    = history.groupby('title').size().max()

    # replace the entire radar dict with this
    radar = {
    'bollywood affinity':  round(float(min(bollywood_plays / total_plays, 1.0)), 3),
    'western independent': round(float(min(western_plays / total_plays * 3, 1.0)), 3),
    'artist loyalty':      round(float(min(top3_plays / total_plays, 1.0)), 3),
    'night owl':           round(float(min(night_plays / total_plays * 1.5, 1.0)), 3),
    'explorer':            round(float(min(history['artist'].nunique() / 300, 1.0)), 3),
    'track obsession':     round(float(min(max_track / 50, 1.0)), 3),
   }

    # key stats
    days = (history['datetime'].max() - history['datetime'].min()).days or 1
    stats = {
        'total_scrobbles':  int(history.shape[0]),
        'unique_artists':   int(history['artist'].nunique()),
        'unique_tracks':    int(history['title'].nunique()),
        'avg_daily_plays':  round(total_plays / days, 1),
        'peak_hour':        int(history['hour'].value_counts().idxmax()),
        'peak_day':         history['day'].value_counts().idxmax(),
        'top_artist':       history['artist'].value_counts().idxmax(),
        'most_played_track': history['title'].value_counts().idxmax(),
    }

    return {
        'username':        username,
        'archetype':       dominant,
        'cluster_dist':    cluster_pct,
        'radar':           radar,
        'stats':           stats,
        'dominant_cluster': dominant_cluster,
    }