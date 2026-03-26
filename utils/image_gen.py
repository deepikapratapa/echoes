import os
import asyncio
import aiohttp
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv('/Users/saturnine/echoes/.env', override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

def build_image_prompts(
        vibe: str,
        top_artists: list,
        playlist_name: str = "",
) -> list[str]:
    """
    Build 7 cinematic image prompts from vibe + artist context.
    1 hero image + 6 moodboard panels.
    No LLM needed — pure f-string engineering.
    """
    style   = "cinematic 35mm film, muted color palette, A24 aesthetic, no text, no watermark, no people"
    artists = ", ".join(top_artists[:3]) if top_artists else "indie artists"
    mood    = vibe[:80] if vibe else "cinematic mood"

    return [
        # hero
        f"Ultra cinematic cover art, {mood}, inspired by {artists}, square format, {style}",
        # 6 moodboard panels
        f"Empty atmospheric scene, {mood}, abandoned place at night, {style}",
        f"Abstract texture closeup, {mood} mood, macro photography, film grain, {style}",
        f"Lone silhouette emotional scene, {mood}, rear view, neon or moonlight, {style}",
        f"Symbolic object representing {mood}, moody still life, dramatic lighting, {style}",
        f"Cinematic landscape, {mood} atmosphere, golden hour or deep night, {style}",
        f"Pure light and color abstraction, {mood} feeling, painterly, {style}",
    ]

def generate_image_sync(prompt: str) -> bytes | None:
    """Generate a single image synchronously."""
    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping image generation")
        return None
    try:
        response = requests.post(
            MODEL_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
            timeout=60,
        )
        if response.status_code == 200:
            return response.content
        else:
            print(f"Image gen failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Image gen error: {e}")
        return None

async def _generate_one(session, prompt: str, idx: int) -> tuple[int, bytes | None]:
    """Async single image generation."""
    try:
        async with session.post(
            MODEL_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                print(f"  ✓ image {idx+1} generated")
                return idx, data
            else:
                print(f"  ✗ image {idx+1} failed: {resp.status}")
                return idx, None
    except Exception as e:
        print(f"  ✗ image {idx+1} error: {e}")
        return idx, None

async def _generate_all_async(prompts: list[str]) -> list[bytes | None]:
    """Generate all images in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [_generate_one(session, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
    ordered = [None] * len(prompts)
    for idx, data in results:
        ordered[idx] = data
    return ordered

def generate_moodboard(
        vibe: str,
        top_artists: list,
        playlist_name: str = "",
) -> dict:
    """
    Generate hero image + 6 moodboard images in parallel.

    Returns:
        {
            'hero':      bytes | None,
            'moodboard': [bytes | None] × 6,
            'prompts':   list[str],
        }
    """
    prompts = build_image_prompts(vibe, top_artists, playlist_name)
    print(f"Generating {len(prompts)} images in parallel...")

    images  = asyncio.run(_generate_all_async(prompts))

    return {
        'hero':      images[0],
        'moodboard': images[1:],
        'prompts':   prompts,
    }

def stitch_moodboard(image_bytes_list: list) -> bytes | None:
    """
    Stitch 6 images into a 2×3 grid PNG.
    Returns PNG bytes for download.
    """
    images = []
    for b in image_bytes_list:
        if b:
            try:
                img = Image.open(BytesIO(b)).convert('RGB')
                img = img.resize((512, 512))
                images.append(img)
            except Exception:
                images.append(Image.new('RGB', (512, 512), color='#111111'))
        else:
            images.append(Image.new('RGB', (512, 512), color='#111111'))

    # 2 rows × 3 cols
    grid = Image.new('RGB', (512 * 3, 512 * 2))
    positions = [(0,0),(512,0),(1024,0),(0,512),(512,512),(1024,512)]

    for img, pos in zip(images[:6], positions):
        grid.paste(img, pos)

    output = BytesIO()
    grid.save(output, format='PNG', quality=95)
    return output.getvalue()