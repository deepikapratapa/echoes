import os
import pylast
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

API_KEY = os.getenv("LASTFM_API_KEY")
API_SECRET = os.getenv("LASTFM_SECRET")

def get_network():
    return pylast.LastFMNetwork(
        api_key=API_KEY,
        api_secret=API_SECRET
    )

def get_user(username: str):
    network = get_network()
    return network.get_user(username)

# ── Top artists ────────────────────────────────────────────────
def get_top_artists(username: str, limit: int = 20) -> pd.DataFrame:
    user = get_user(username)
    top = user.get_top_artists(period=pylast.PERIOD_OVERALL, limit=limit)
    
    rows = []
    for item in top:
        artist = item.item
        rows.append({
            "name": artist.name,
            "playcount": int(item.weight),
            "url": artist.get_url(),
            "image": get_artist_image(artist),
        })
    return pd.DataFrame(rows)

# ── Top tracks ─────────────────────────────────────────────────
def get_top_tracks(username: str, limit: int = 50) -> pd.DataFrame:
    user = get_user(username)
    top = user.get_top_tracks(period=pylast.PERIOD_OVERALL, limit=limit)
    
    rows = []
    for item in top:
        track = item.item
        rows.append({
            "title": track.title,
            "artist": track.artist.name,
            "playcount": int(item.weight),
            "url": track.get_url(),
        })
    return pd.DataFrame(rows)

# ── Recent tracks ──────────────────────────────────────────────
def get_recent_tracks(username: str, limit: int = 200) -> pd.DataFrame:
    user = get_user(username)
    recent = user.get_recent_tracks(limit=limit)
    
    rows = []
    for track in recent:
        rows.append({
            "title": track.track.title,
            "artist": track.track.artist.name,
            "timestamp": track.timestamp,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day_name()
        df["month"] = df["datetime"].dt.to_period("M").astype(str)
    return df

# ── Artist tags ────────────────────────────────────────────────
def get_artist_tags(artist_name: str, limit: int = 10) -> list[str]:
    try:
        network = get_network()
        artist = network.get_artist(artist_name)
        tags = artist.get_top_tags(limit=limit)
        return [t.item.name.lower() for t in tags]
    except Exception:
        return []

# ── Similar artists ────────────────────────────────────────────
def get_similar_artists(artist_name: str, limit: int = 10) -> list[str]:
    try:
        network = get_network()
        artist = network.get_artist(artist_name)
        similar = artist.get_similar(limit=limit)
        return [s.item.name for s in similar]
    except Exception:
        return []

# ── Similar users (for collaborative filtering) ────────────────
def get_similar_users(username: str, limit: int = 10) -> list[str]:
    try:
        user = get_user(username)
        neighbors = user.get_neighbours(limit=limit)
        return [n.item.name for n in neighbors]
    except Exception:
        return []

# ── Artist image ───────────────────────────────────────────────
def get_artist_image(artist) -> str:
    try:
        return artist.get_cover_image()
    except Exception:
        return ""

# ── User info ──────────────────────────────────────────────────
def get_user_info(username: str) -> dict:
    try:
        user = get_user(username)
        return {
            "username": username,
            "real_name": user.get_name(),
            "playcount": user.get_playcount(),
            "registered": user.get_registered(),
            "image": user.get_image(),
            "url": user.get_url(),
        }
    except Exception:
        return {"username": username}

# ── Build full user profile (single call for the app) ─────────
def build_user_profile(username: str) -> dict:
    print(f"Fetching profile for {username}...")
    return {
        "info":         get_user_info(username),
        "top_artists":  get_top_artists(username, limit=20),
        "top_tracks":   get_top_tracks(username, limit=50),
        "recent":       get_recent_tracks(username, limit=200),
    }
```