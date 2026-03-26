import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = None  # set at runtime

FEATURE_ANCHORS = {
    'energy': {
        'high': "aggressive intense high energy powerful loud explosive",
        'low':  "calm quiet peaceful soft gentle slow relaxing",
    },
    'valence': {
        'high': "happy joyful uplifting positive cheerful euphoric",
        'low':  "sad melancholic heartbreak depression dark hopeless",
    },
    'acousticness': {
        'high': "acoustic guitar folk unplugged intimate raw organic",
        'low':  "electronic synthesizer digital produced beats drops",
    },
    'danceability': {
        'high': "dance groove rhythm party club beat move body",
        'low':  "still slow ambient cinematic atmospheric no beat",
    },
    'tempo': {
        'high': "fast rushing urgent driving momentum quick rapid",
        'low':  "slow dragging languid floating drifting unhurried",
    },
    'darkness': {
        'high': "dark brooding sinister haunting ominous noir shadow",
        'low':  "bright light sunny warm radiant open airy",
    },
}

TAG_PREFERENCE_MAP = {
    'energy_low':        ['ambient', 'chillout', 'acoustic', 'soft', 'calm'],
    'energy_high':       ['electronic', 'rock', 'punk', 'intense', 'power'],
    'valence_low':       ['sadcore', 'melancholic', 'dark', 'emotional', 'depression'],
    'valence_high':      ['happy', 'upbeat', 'feel good', 'summer', 'fun'],
    'acousticness_high': ['acoustic', 'folk', 'singer-songwriter', 'unplugged'],
    'acousticness_low':  ['electronic', 'synth', 'digital', 'edm', 'produced'],
    'danceability_high': ['dance', 'groove', 'club', 'party', 'rhythm'],
    'danceability_low':  ['ambient', 'post-rock', 'instrumental', 'atmospheric'],
    'tempo_low':         ['slow', 'ballad', 'downtempo', 'dream pop', 'shoegaze'],
    'tempo_high':        ['fast', 'punk', 'drum and bass', 'uptempo'],
    'darkness_high':     ['dark', 'gothic', 'noir', 'brooding', 'haunting'],
    'darkness_low':      ['bright', 'indie pop', 'sunshine pop', 'warm'],
}

_model          = None
_anchor_embeds  = None

def _load_model():
    global _model, _anchor_embeds
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        _anchor_embeds = {}
        for feature, anchors in FEATURE_ANCHORS.items():
            _anchor_embeds[feature] = {
                'high': _model.encode(anchors['high']),
                'low':  _model.encode(anchors['low']),
            }

def parse_vibe(vibe_text: str, contrast_strength: float = 2.5) -> dict:
    """
    Parse free-text vibe description into audio feature weights.

    Args:
        vibe_text:         user's mood description
        contrast_strength: amplification factor (default 2.5)

    Returns:
        dict of {tag: weight} for boosting recommendations
    """
    _load_model()
    vibe_vec = _model.encode(vibe_text)
    scores   = {}

    for feature, anchors in _anchor_embeds.items():
        sim_high = np.dot(vibe_vec, anchors['high']) / (
            np.linalg.norm(vibe_vec) * np.linalg.norm(anchors['high']) + 1e-9)
        sim_low  = np.dot(vibe_vec, anchors['low']) / (
            np.linalg.norm(vibe_vec) * np.linalg.norm(anchors['low']) + 1e-9)
        raw      = sim_high - sim_low
        score    = (raw + 1) / 2
        centered = score - 0.5
        amp      = max(0.0, min(1.0, 0.5 + centered * contrast_strength))
        scores[feature] = round(float(amp), 4)

    # map to tag weights
    tag_weights = {}
    for feat, score in scores.items():
        if score > 0.6:
            direction, weight = 'high', score
        elif score < 0.4:
            direction, weight = 'low', 1 - score
        else:
            continue
        for tag in TAG_PREFERENCE_MAP.get(f"{feat}_{direction}", []):
            tag_weights[tag] = tag_weights.get(tag, 0) + weight

    if tag_weights:
        max_w = max(tag_weights.values())
        tag_weights = {
            k: round(v / max_w, 4)
            for k, v in sorted(
                tag_weights.items(), key=lambda x: x[1], reverse=True
            )
        }

    return tag_weights

def get_feature_scores(vibe_text: str) -> dict:
    """Return raw amplified feature scores for visualization."""
    _load_model()
    vibe_vec = _model.encode(vibe_text)
    scores   = {}

    for feature, anchors in _anchor_embeds.items():
        sim_high = np.dot(vibe_vec, anchors['high']) / (
            np.linalg.norm(vibe_vec) * np.linalg.norm(anchors['high']) + 1e-9)
        sim_low  = np.dot(vibe_vec, anchors['low']) / (
            np.linalg.norm(vibe_vec) * np.linalg.norm(anchors['low']) + 1e-9)
        raw      = sim_high - sim_low
        score    = (raw + 1) / 2
        centered = score - 0.5
        amp      = max(0.0, min(1.0, 0.5 + centered * 2.5))
        scores[feature] = round(float(amp), 4)

    return scores