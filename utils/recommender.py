import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from utils.lastfm import get_similar_artists, get_artist_tags

load_dotenv('/Users/saturnine/echoes/.env', override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TAG_ENRICHMENT = {
    'Pritam':            ['bollywood', 'hindi', 'soundtrack', 'romantic', 'melody'],
    'A.R. Rahman':       ['bollywood', 'soundtrack', 'indian', 'orchestral', 'melodic'],
    'Shankar Mahadevan': ['bollywood', 'hindi', 'classical', 'devotional'],
    'Siddharth Pandit':  ['hindi', 'bollywood', 'romantic', 'melody'],
    'DIGV':              ['hindi', 'indie', 'bollywood', 'romantic'],
    'Ana Rehman':        ['pakistani', 'soft rock', 'romantic', 'acoustic'],
    'Prithvi Gandharv':  ['hindi', 'indie', 'romantic', 'folk'],
    'Amit Sharma':       ['hindi', 'bollywood', 'romantic', 'melody'],
    'Mohit Chauhan':     ['hindi', 'bollywood', 'indie', 'acoustic', 'romantic'],
    'Rochak Kohli':      ['hindi', 'bollywood', 'romantic', 'pop'],
    'Armaan Malik':      ['hindi', 'bollywood', 'pop', 'romantic'],
    'Darshan Raval':     ['hindi', 'indie', 'romantic', 'pop'],
    'Ankit Tiwari':      ['hindi', 'bollywood', 'romantic', 'pop'],
    'Rashid Ali':        ['hindi', 'bollywood', 'sufi', 'classical'],
    'Karthik':           ['bollywood', 'hindi', 'playback', 'melodic'],
    'Kushagra':          ['metal', 'experimental', 'rock', 'alternative'],
    'Mitraz':            ['indie', 'hindi', 'pop', 'acoustic', 'chill'],
}

NOISE_REFINED = {
    'all', 'music', 'pritam', 'shankar', 'rahman', 'atif',
    'himesh', 'kailash', 'abhijeet', 'buddamat', 'enrique',
    'kot', 'ost', 'bollybolly', 'ofwgkta', 'bangla',
    'shankar mahadevan', 'shankar ehsaan loy', 'amit trivedi',
    'tollywood', 'mondiovision', 'radio bombay',
    "can't spell isai without sai", 'goat abhyankkar',
    'snake', 'taylor swift', 'a r rahman', 'mohit chauhan',
    'dev d', 'udaan', 'golden', 'steampunk', 'bharat',
    'bollywood soundtrack', 'film music composer',
    'bollywood film', 'desi filmi', 'indian music director',
}

def load_assets():
    """Load all precomputed data assets."""
    with open(os.path.join(DATA_DIR, 'artists_final.json')) as f:
        artists_final = json.load(f)
    with open(os.path.join(DATA_DIR, 'artist_tags_final.json')) as f:
        artist_tags_final = json.load(f)

    artist_tags_enriched = {}
    for artist in artists_final:
        existing = artist_tags_final.get(artist, [])
        extra    = TAG_ENRICHMENT.get(artist, [])
        artist_tags_enriched[artist] = list(dict.fromkeys(existing + extra))

    top_artists = pd.read_csv(os.path.join(DATA_DIR, 'top_artists.csv'))
    history     = pd.read_csv(os.path.join(DATA_DIR, 'history.csv'))

    return artists_final, artist_tags_enriched, top_artists, history

def build_candidate_pool(top_artists, history, top_n_sources=8, candidates_per=15):
    """Fetch external artists not in user library as candidates."""
    known      = set(history['artist'].str.lower().unique())
    candidates = {}

    for _, row in top_artists.head(top_n_sources).iterrows():
        similar = get_similar_artists(row['name'], limit=candidates_per)
        new     = [a for a in similar if a.lower() not in known]
        for artist in new:
            if artist not in candidates:
                candidates[artist] = []
        time.sleep(0.2)

    for i, artist in enumerate(list(candidates.keys())):
        candidates[artist] = get_artist_tags(artist, limit=10)
        time.sleep(0.15)

    return candidates

def hybrid_recommend(
        username,
        vibe_scores=None,
        top_n=15,
        alpha=0.35,
        beta=0.45,
        gamma=0.20,
        ):
    """
    Full hybrid recommendation pipeline.

    Args:
        username:    Last.fm username
        vibe_scores: dict of {tag: weight} from vibe parser (optional)
        top_n:       number of recommendations to return
        alpha:       content-based weight
        beta:        collaborative weight
        gamma:       vibe weight (only used if vibe_scores provided)

    Returns:
        pd.DataFrame with ranked recommendations and explanations
    """
    artists_final, artist_tags_enriched, top_artists, history = load_assets()
    artist_to_idx = {a: i for i, a in enumerate(artists_final)}

    # build candidate pool
    candidates = build_candidate_pool(top_artists, history)
    if not candidates:
        return pd.DataFrame()

    # build unified TF-IDF space
    all_tag_strs = []
    for artist in artists_final:
        tags = artist_tags_enriched.get(artist, [])
        tags = [t for t in tags
                if t.lower() not in NOISE_REFINED and len(t) > 2]
        all_tag_strs.append(" ".join(tags) if tags else "pop")
    for artist, tags in candidates.items():
        clean = [t for t in tags if len(t) > 2]
        all_tag_strs.append(" ".join(clean) if clean else "pop")

    # add vibe as pseudo-document if provided
    if vibe_scores:
        vibe_tags = list(vibe_scores.keys())[:15]
        all_tag_strs.append(" ".join(vibe_tags))

    tfidf_u  = TfidfVectorizer(max_features=150, ngram_range=(1, 1))
    matrix_u = normalize(tfidf_u.fit_transform(all_tag_strs).toarray())

    n_yours          = len(artists_final)
    your_matrix      = matrix_u[:n_yours]
    end_idx          = n_yours + len(candidates)
    candidate_matrix = matrix_u[n_yours:end_idx]
    vibe_vector      = matrix_u[end_idx:end_idx+1] if vibe_scores else None

    # content scores
    top10       = top_artists.head(10)
    valid_pairs = [(row['name'], row['playcount'])
                   for _, row in top10.iterrows()
                   if row['name'] in artist_to_idx]
    if valid_pairs:
        valid_idx     = [artist_to_idx[a] for a, _ in valid_pairs]
        valid_weights = np.array([p for _, p in valid_pairs])
        valid_weights = valid_weights / valid_weights.sum()
        cb_raw = cosine_similarity(
            candidate_matrix, your_matrix[valid_idx]
        ) @ valid_weights
    else:
        cb_raw = np.zeros(len(candidates))

    # collab scores
    cf_raw     = []
    cf_sources = {}
    for artist in candidates.keys():
        score, sources = 0, []
        for _, row in top_artists.head(10).iterrows():
            similar = get_similar_artists(row['name'], limit=15)
            weight  = row['playcount'] / top_artists['playcount'].sum()
            if artist in similar:
                score += weight
                sources.append(row['name'])
        cf_raw.append(score)
        cf_sources[artist] = sources

    cf_raw = np.array(cf_raw)

    # vibe scores
    if vibe_scores and vibe_vector is not None:
        vibe_raw  = cosine_similarity(candidate_matrix, vibe_vector).flatten()
        vibe_norm = vibe_raw / (vibe_raw.max() + 1e-9)
    else:
        vibe_norm = np.zeros(len(candidates))
        gamma     = 0.0
        alpha     = 0.4
        beta      = 0.6

    # normalize
    cb_norm = cb_raw / (cb_raw.max() + 1e-9)
    cf_norm = cf_raw / (cf_raw.max() + 1e-9)

    # final scores
    final = alpha * cb_norm + beta * cf_norm + gamma * vibe_norm

    # build results
    results = []
    for i, artist in enumerate(candidates.keys()):
        cb, cf, vb = float(cb_norm[i]), float(cf_norm[i]), float(vibe_norm[i])
        fin        = float(final[i])
        total      = cb + cf + vb + 1e-9

        if cb > 0.1 and cf > 0.1:
            reason = "tag match + listener overlap"
        elif cb > cf:
            reason = "tag similarity"
        else:
            reason = "listener overlap"

        results.append({
            'artist':        artist,
            'final_score':   round(fin, 4),
            'content_score': round(cb, 4),
            'collab_score':  round(cf, 4),
            'vibe_score':    round(vb, 4),
            'cb_pct':        round(cb / total * 100),
            'cf_pct':        round(cf / total * 100),
            'vibe_pct':      round(vb / total * 100),
            'reason':        reason,
            'via':           cf_sources.get(artist, [])[:2],
            'tags':          candidates[artist][:4],
        })

    df = pd.DataFrame(results).sort_values(
        'final_score', ascending=False
    ).reset_index(drop=True)

    return df.head(top_n)