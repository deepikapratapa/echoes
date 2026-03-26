import os
import pylast
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("6ab2ebcbe6d3ed50f425f0c85b2793e8")
API_SECRET = os.getenv("7383bf8331dce497f86efd67013633c1")

def get_network():
    return pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

def get_user(username):
    return get_network().get_user(username)

def get_top_artists(username, limit=20):
    user = get_user(username)
    top = user.get_top_artists(period=pylast.PERIOD_OVERALL, limit=limit)
    rows = []
    for item in top:
        artist = item.item
        try:
            image = artist.get_cover_image()
        except:
            image = ""
        rows.append({
            "name": artist.name,
            "playcount": int(item.weight),
            "url": artist.get_url(),
            "image": image,
        })
    return pd.DataFrame(rows)

def get_top_tracks(username, limit=50):
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

def get_recent_tracks(username, limit=200):
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

def get_artist_tags(artist_name, limit=10):
    try:
        artist = get_network().get_artist(artist_name)
        tags = artist.get_top_tags(limit=limit)
        return [t.item.name.lower() for t in tags]
    except:
        return []

def get_similar_artists(artist_name, limit=10):
    try:
        artist = get_network().get_artist(artist_name)
        similar = artist.get_similar(limit=limit)
        return [s.item.name for s in similar]
    except:
        return []

def get_similar_users(username, limit=10):
    try:
        user = get_user(username)
        neighbors = user.get_neighbours(limit=limit)
        return [n.item.name for n in neighbors]
    except:
        return []

def get_user_info(username):
    try:
        user = get_user(username)
        return {
            "username": username,
            "playcount": user.get_playcount(),
            "registered": user.get_registered(),
            "image": user.get_image(),
            "url": user.get_url(),
        }
    except:
        return {"username": username}

def build_user_profile(username):
    print(f"Fetching profile for {username}...")
    return {
        "info":        get_user_info(username),
        "top_artists": get_top_artists(username, limit=20),
        "top_tracks":  get_top_tracks(username, limit=50),
        "recent":      get_recent_tracks(username, limit=200),
    }
