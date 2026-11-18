"""
tshirt-studio  -  image sidecar for the n8n "AI T-Shirt Design Studio" workflow.

n8n cannot run Pillow / rembg / Tesseract, so this small service does the heavy lifting:
  * image generation through swappable provider adapters (ComfyUI, Cloudflare, Pollinations, AI Horde)
  * real background removal (flood-fill keying for flat backgrounds, rembg as fallback)
  * upscale + centre on a transparent canvas (default 4500x5400)
  * quality control that only reports checks that were really executed (incl. OCR spelling check)
  * storage in /data/tshirt-generator/<run>/{PNG,ZIP,PROMPTS,SEO,THUMBS} + manifest.json
  * CSV / prompts / ZIP generation

Zero paid APIs. Providers that need credentials are only used when you configure them.
"""
import base64
import csv
import difflib
import io
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field
from scipy import ndimage as ndi

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAVE_TESS = True
except Exception:  # tesseract binary missing
    HAVE_TESS = False

try:
    from rembg import new_session, remove as rembg_remove  # optional
    HAVE_REMBG = True
except Exception:
    HAVE_REMBG = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tshirt-studio")
Image.MAX_IMAGE_PIXELS = None

DATA_DIR = Path(os.getenv("DATA_DIR", "/data/tshirt-generator"))
API_KEY = os.getenv("API_KEY", "")
SUBDIRS = ["PNG", "ZIP", "PROMPTS", "SEO", "THUMBS"]
_locks: Dict[str, threading.Lock] = {}
_rembg_session = None


def auth(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid API key")


app = FastAPI(title="tshirt-studio", dependencies=[Depends(auth)])


# --------------------------------------------------------------------------- storage helpers
def lock_for(run_id: str) -> threading.Lock:
    return _locks.setdefault(run_id, threading.Lock())


def resolve_run(run_id: str) -> Path:
    if run_id == "latest":
        p = DATA_DIR / "latest.txt"
        if p.exists():
            run_id = p.read_text().strip()
        else:
            dirs = sorted([d.name for d in DATA_DIR.glob("20*") if d.is_dir()]) if DATA_DIR.exists() else []
            if not dirs:
                raise HTTPException(404, "no run yet")
            run_id = dirs[-1]
    run_id = os.path.basename(run_id)
    d = DATA_DIR / run_id
    if not d.is_dir():
        raise HTTPException(404, f"run {run_id} not found")
    return d


def read_manifest(d: Path) -> Dict[str, Any]:
    f = d / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_manifest(d: Path, m: Dict[str, Any]):
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    tmp.replace(d / "manifest.json")


def safe_slug(s: str, default="Design") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return (s or default)[:40]


# --------------------------------------------------------------------------- image providers
class ProviderError(Exception):
    pass


def gen_size(canvas_w: int, canvas_h: int):
    long_side = int(os.getenv("GEN_LONG_SIDE", "1024"))
    if canvas_h >= canvas_w:
        h = long_side
        w = max(512, round(long_side * canvas_w / canvas_h / 64) * 64)
    else:
        w = long_side
        h = max(512, round(long_side * canvas_h / canvas_w / 64) * 64)
    return int(w), int(h)


def gen_pollinations(prompt, w, h, seed):
    """Free, no key. Anonymous tier ~1 request / 15 s and may add a watermark (set POLLINATIONS_TOKEN to remove)."""
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:1500], safe="")
    params = {"width": w, "height": h, "model": "flux", "seed": seed, "nologo": "true", "private": "true"}
    headers = {}
    if os.getenv("POLLINATIONS_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["POLLINATIONS_TOKEN"]
    last = ""
    for i in range(4):
        r = requests.get(url, params=params, headers=headers, timeout=240)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return r.content
        last = f"{r.status_code} {r.text[:120]}"
        if r.status_code in (429, 502, 503, 524):
            time.sleep(16 * (i + 1))
            continue
        break
    raise ProviderError("pollinations: " + last)


def gen_cloudflare(prompt, w, h, seed):
    """Workers AI FLUX.1-schnell. Free daily allowance (needs CF_ACCOUNT_ID + CF_API_TOKEN). Output is ~1024px."""
    acc, tok = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
    if not (acc and tok):
        raise ProviderError("cloudflare: CF_ACCOUNT_ID / CF_API_TOKEN not set")
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell",
        headers={"Authorization": f"Bearer {tok}"},
        json={"prompt": prompt[:2000], "steps": 6, "seed": seed},
        timeout=240,
    )
    try:
        data = r.json()
        return base64.b64decode(data["result"]["image"])
    except Exception:
        raise ProviderError(f"cloudflare: {r.status_code} {r.text[:200]}")


def gen_comfyui(prompt, w, h, seed):
    """Your own ComfyUI. Default graph = FLUX.1-schnell (fp8 all-in-one checkpoint). Override with COMFY_WORKFLOW_FILE
    (API-format JSON containing __PROMPT__, __SEED__, __WIDTH__, __HEIGHT__)."""
    base = os.getenv("COMFYUI_URL", "").rstrip("/")
    if not base:
        raise ProviderError("comfyui: COMFYUI_URL not set")
    wf_file = os.getenv("COMFY_WORKFLOW_FILE")
    if wf_file and Path(wf_file).exists():
        txt = Path(wf_file).read_text()
        txt = (txt.replace("__PROMPT__", json.dumps(prompt)[1:-1]).replace("__SEED__", str(seed))
               .replace("__WIDTH__", str(w)).replace("__HEIGHT__", str(h)))
        graph = json.loads(txt)
    else:
        ckpt = os.getenv("COMFY_CKPT", "flux1-schnell-fp8.safetensors")
        graph = {
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": int(os.getenv("COMFY_STEPS", "4")), "cfg": 1.0, "sampler_name": "euler",
                "scheduler": "simple", "denoise": 1, "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "tshirt"}},
        }
    r = requests.post(base + "/prompt", json={"prompt": graph}, timeout=60)
    if r.status_code != 200:
        raise ProviderError(f"comfyui: {r.status_code} {r.text[:300]}")
    pid = r.json()["prompt_id"]
    deadline = time.time() + int(os.getenv("COMFY_TIMEOUT", "1500"))
    while time.time() < deadline:
        time.sleep(3)
        h_ = requests.get(f"{base}/history/{pid}", timeout=30).json()
        if pid in h_:
            outs = h_[pid].get("outputs", {})
            for node in outs.values():
                for im in node.get("images", []):
                    q = {"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")}
                    return requests.get(base + "/view", params=q, timeout=60).content
            raise ProviderError("comfyui: finished without images " + json.dumps(h_[pid].get("status", {}))[:200])
    raise ProviderError("comfyui: timeout")


def gen_horde(prompt, w, h, seed):
    """AI Horde (free, crowdsourced). Anonymous key 0000000000. Speed/model availability vary."""
    key = os.getenv("HORDE_API_KEY", "0000000000")
    hdr = {"apikey": key, "Client-Agent": "tshirt-studio:1.0:n8n"}
    models = json.loads(os.getenv("HORDE_MODELS", '["Flux.1-Schnell fp8 (Compact)"]'))
    body = {"prompt": prompt, "models": models, "nsfw": False, "censor_nsfw": True, "r2": True, "shared": False,
            "slow_workers": True,
            "params": {"width": min(w, 1024) // 64 * 64, "height": min(h, 1024) // 64 * 64, "n": 1,
                       "steps": int(os.getenv("HORDE_STEPS", "8")), "cfg_scale": float(os.getenv("HORDE_CFG", "1.0")),
                       "sampler_name": "k_euler", "seed": str(seed)}}
    r = requests.post("https://aihorde.net/api/v2/generate/async", headers=hdr, json=body, timeout=60)
    if r.status_code not in (200, 202):
        raise ProviderError(f"horde: {r.status_code} {r.text[:200]}")
    jid = r.json()["id"]
    deadline = time.time() + int(os.getenv("HORDE_TIMEOUT", "900"))
    while time.time() < deadline:
        time.sleep(6)
        c = requests.get(f"https://aihorde.net/api/v2/generate/check/{jid}", timeout=30).json()
        if c.get("faulted"):
            raise ProviderError("horde: request faulted")
        if c.get("done"):
            s = requests.get(f"https://aihorde.net/api/v2/generate/status/{jid}", timeout=60).json()
            img = s["generations"][0]["img"]
            if img.startswith("http"):
                return requests.get(img, timeout=120).content
            return base64.b64decode(img)
    raise ProviderError("horde: timeout")


PROVIDERS = {"comfyui": gen_comfyui, "cloudflare": gen_cloudflare, "pollinations": gen_pollinations, "aihorde": gen_horde}


# --------------------------------------------------------------------------- background removal
def remove_background(img: Image.Image, clear_enclosed: bool = False):
    """Return (RGBA image, info). Flat-background keying preserves text perfectly; rembg is the fallback."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    if (arr[..., 3] < 10).mean() > 0.05:
        return rgba, {"method": "native_alpha"}
    rgb = arr[..., :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    uniform = float((np.linalg.norm(border - bg, axis=1) < 28).mean())
    if uniform >= 0.85:
        dist = np.linalg.norm(rgb - bg, axis=2)
        lo, hi = 22.0, 70.0
        lab, _ = ndi.label(dist < hi)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        bgmask = np.isin(lab, edge[edge > 0])
        if clear_enclosed:  # also clear small enclosed background-coloured islands (letter counters such as the hole in O/A/B)
            lab2, n2 = ndi.label(dist < lo)
            if n2:
                areas = ndi.sum(np.ones_like(dist), lab2, index=np.arange(1, n2 + 1))
                small_ids = np.where(areas < 0.008 * dist.size)[0] + 1
                bgmask |= np.isin(lab2, small_ids)
        alpha = np.ones(dist.shape, np.float32)
        ramp = np.clip((dist - lo) / (hi - lo), 0, 1)
        alpha[bgmask] = ramp[bgmask]
        band = ndi.binary_dilation(bgmask, iterations=2) & ~bgmask
        core = ~bgmask & ~band
        if core.sum() < 50:
            raise ValueError("empty artwork after background removal")
        idx = ndi.distance_transform_edt(~core, return_distances=False, return_indices=True)
        near = rgb[idx[0], idx[1]]
        dc = np.linalg.norm(near - bg, axis=2)
        est = np.clip(dist / np.maximum(dc, 1.0), 0, 1)
        est[dc < 40] = 1.0  # core colour too close to background: cannot unmix
        alpha[band] = est[band]
        out = rgb.copy()
        soft = alpha < 1.0
        out[soft] = near[soft]  # remove background colour bleed on semi-transparent pixels
        res = np.dstack([out, alpha * 255]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(res, "RGBA"), {"method": "flat_background_key", "bg_rgb": [int(x) for x in bg], "border_uniformity": round(uniform, 3)}
    if HAVE_REMBG:
        global _rembg_session
        if _rembg_session is None:
            _rembg_session = new_session(os.getenv("REMBG_MODEL", "isnet-general-use"))
        out = rembg_remove(img.convert("RGB"), session=_rembg_session)
        return out.convert("RGBA"), {"method": "rembg", "border_uniformity": round(uniform, 3)}
    raise ValueError(f"background is not flat (uniformity {uniform:.2f}) and rembg is not installed")


# --------------------------------------------------------------------------- OCR text check
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s.upper())).strip()


def ocr_check(art: Image.Image, slogan: str, rec_thr: float, word_thr: float):
    """Spelling check. Tesseract struggles with big illustrations, so we OCR the full artwork AND horizontal bands,
    on white and on black. An expected word counts only if it is read EXACTLY in at least one variant."""
    if not HAVE_TESS:
        return {"performed": False, "reason": "tesseract not installed"}
    exp = _norm(slogan)
    exp_words = exp.split()
    if not exp_words:
        return {"performed": False, "reason": "no slogan given"}
    scale = 1600 / max(art.size)
    small = art.resize((max(1, int(art.width * scale)), max(1, int(art.height * scale))), Image.LANCZOS)
    w, h = small.size
    crops = [(small, 11), (small, 6)]
    for a_, b_ in ((0, 0.4), (0.3, 0.7), (0.6, 1.0), (0.0, 0.2), (0.8, 1.0)):
        crops.append((small.crop((0, int(h * a_), w, int(h * b_))), 6))
    found, best_recall, texts = set(), 0.0, []
    a = exp.replace(" ", "")
    try:
        for bgc in ((255, 255, 255), (0, 0, 0)):
            for crop, psm in crops:
                comp = Image.new("RGB", crop.size, bgc)
                comp.paste(crop, mask=crop.split()[3])
                got = _norm(pytesseract.image_to_string(comp, config=f"--psm {psm}"))
                if not got:
                    continue
                texts.append(got)
                gw = set(got.split())
                found |= {x for x in exp_words if x in gw}
                sm = difflib.SequenceMatcher(None, a, got.replace(" ", ""), autojunk=False)
                best_recall = max(best_recall, sum(m.size for m in sm.get_matching_blocks()) / max(1, len(a)))
            if len(found) == len(exp_words):
                break  # everything read exactly on white, no need for the black pass
    except Exception as e:  # pragma: no cover
        return {"performed": False, "reason": f"tesseract error: {e}"}
    allw = {x for t in texts for x in t.split()}
    extra = sorted(x for x in allw if x not in exp_words and len(x) >= 3)
    res = {"performed": True, "word_exact": round(len(found) / len(exp_words), 3), "recall": round(best_recall, 3),
           "extra_words": extra[:8], "ocr_text": (max(texts, key=len) if texts else "")[:160]}
    res["passed"] = res["recall"] >= rec_thr and res["word_exact"] >= word_thr
    return res


# --------------------------------------------------------------------------- processing + QC
def process_image(raw: bytes, slogan: str, W: int, H: int, opts: "DesignReq"):
    qc: Dict[str, Any] = {"png_exists": None, "warnings": [], "failures": []}
    src = Image.open(io.BytesIO(raw))
    src.load()
    qc["source_size"] = list(src.size)
    rgba, info = remove_background(src, opts.clear_enclosed)
    qc["background_removal"] = info

    alpha = np.array(rgba.split()[3])
    strips = np.concatenate([alpha[:2].ravel(), alpha[-2:].ravel(), alpha[:, :2].ravel(), alpha[:, -2:].ravel()])
    edge_ratio = float((strips > 40).mean())
    qc["not_cropped"] = edge_ratio < 0.003
    if not qc["not_cropped"]:
        qc["failures"].append("artwork touches the image border (cropped)")

    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        qc["failures"].append("no visible artwork")
        return None, qc
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = int(x1 - x0), int(y1 - y0)
    frame_cover = (bw * bh) / (alpha.shape[0] * alpha.shape[1])
    fill = float((alpha[y0:y1, x0:x1] > 16).mean())
    qc["artwork_frame_coverage"] = round(frame_cover, 3)
    qc["artwork_fill_ratio"] = round(fill, 3)
    qc["empty_space_ok"] = frame_cover >= opts.min_frame_coverage and fill >= 0.12
    if not qc["empty_space_ok"]:
        qc["failures"].append(f"too much empty space (coverage {frame_cover:.2f}, fill {fill:.2f})")

    art = rgba.crop((int(x0), int(y0), int(x1), int(y1)))
    margin = 0.03
    scale = min(W * (1 - 2 * margin) / bw, H * (1 - 2 * margin) / bh)
    qc["upscale_factor"] = round(scale, 2)
    if scale > 3.2:
        qc["warnings"].append(f"artwork upscaled x{scale:.1f}: edges/text will be soft; use ComfyUI high-res or an ESRGAN upscaler for print")
    nw, nh = max(1, round(bw * scale)), max(1, round(bh * scale))
    big = art.resize((nw, nh), Image.LANCZOS)  # Pillow resamples RGBA premultiplied
    r, g, b, a = big.split()
    rgb_img = Image.merge("RGB", (r, g, b)).filter(ImageFilter.UnsharpMask(radius=2.2, percent=55, threshold=3))
    a_np = np.array(a).astype(np.float32) / 255.0
    a_np = np.clip((a_np - 0.25) / 0.5, 0, 1)  # re-sharpen the soft alpha edge produced by upscaling
    a_np = a_np * a_np * (3 - 2 * a_np)
    final_art = Image.merge("RGBA", (*rgb_img.split(), Image.fromarray((a_np * 255).astype(np.uint8))))
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(final_art, ((W - nw) // 2, (H - nh) // 2), final_art)

    fa = np.array(canvas.split()[3])
    corners = [fa[0, 0], fa[0, -1], fa[-1, 0], fa[-1, -1]]
    transparent_ratio = float((fa < 8).mean())
    semi = float(((fa >= 8) & (fa < 247)).mean())
    qc.update({
        "alpha_channel": canvas.mode == "RGBA",
        "corners_transparent": all(c == 0 for c in corners),
        "transparent_ratio": round(transparent_ratio, 3),
        "semi_transparent_ratio": round(semi, 4),
        "resolution": [W, H],
        "resolution_ok": W >= opts.min_width and H >= opts.min_height,
    })
    if not (qc["alpha_channel"] and qc["corners_transparent"] and transparent_ratio > 0.08):
        qc["failures"].append("transparency validation failed (alpha channel / corners / ratio)")
    if not qc["resolution_ok"]:
        qc["failures"].append("resolution below minimum")

    # shirt-colour compatibility (informational)
    px = np.array(canvas)[fa > 200][:, :3].astype(np.float32)
    if len(px):
        lum = px @ np.array([0.299, 0.587, 0.114], np.float32)
        light, dark = float((lum > 225).mean()), float((lum < 30).mean())
        qc["light_pixel_share"], qc["dark_pixel_share"] = round(light, 3), round(dark, 3)
        if light > 0.55:
            qc["warnings"].append("mostly very light colours: weak on white shirts")
        if dark > 0.55:
            qc["warnings"].append("mostly very dark colours: weak on black shirts (add a light outline)")

    # text check on the artwork only
    if opts.ocr_check:
        art_for_ocr = canvas.crop(((W - nw) // 2, (H - nh) // 2, (W - nw) // 2 + nw, (H - nh) // 2 + nh))
        t = ocr_check(art_for_ocr, slogan, opts.ocr_recall, opts.ocr_word_match)
    else:
        t = {"performed": False, "reason": "disabled by request"}
    qc["text_check"] = t
    if t.get("performed") and not t.get("passed"):
        qc["failures"].append(f"slogan text check failed (word match {t['word_exact']}, recall {t['recall']}, OCR read '{t['ocr_text']}')")
    if t.get("performed") and len(t.get("extra_words", [])) >= 2:
        qc["warnings"].append("unexpected extra text detected: " + ", ".join(t["extra_words"]))
    return canvas, qc


def rank_of(status: str) -> int:
    return {"approved": 3, "needs_review": 2, "failed": 1}.get(status, 0)


def status_from_qc(qc: Dict[str, Any]) -> str:
    tech_fail = [f for f in qc["failures"] if not f.startswith("slogan text")]
    if tech_fail:
        return "failed"
    t = qc.get("text_check", {})
    if not t.get("performed"):
        return "needs_review"  # never claim a pass for a check that was not performed
    return "approved" if t.get("passed") else "needs_review"


# --------------------------------------------------------------------------- API models
class RunReq(BaseModel):
    params: Dict[str, Any] = {}


class ManifestReq(BaseModel):
    params: Dict[str, Any] = {}
    designs: List[Dict[str, Any]] = []
    status: str = "running"


class DesignReq(BaseModel):
    slogan: str
    niche: str = "design"
    slug: Optional[str] = None
    prompt: str
    providers: List[str] = ["pollinations"]
    max_attempts: int = Field(2, ge=1, le=6)
    width: int = 4500
    height: int = 5400
    min_width: int = 3000
    min_height: int = 3000
    min_frame_coverage: float = 0.28
    ocr_check: bool = True
    clear_enclosed: bool = False
    ocr_recall: float = 0.9
    ocr_word_match: float = 0.8
    seed: Optional[int] = None


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True, "data_dir": str(DATA_DIR), "writable": os.access(DATA_DIR, os.W_OK),
        "tesseract": HAVE_TESS, "rembg": HAVE_REMBG,
        "providers_configured": {
            "pollinations": True,
            "aihorde": True,
            "comfyui": bool(os.getenv("COMFYUI_URL")),
            "cloudflare": bool(os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_API_TOKEN")),
        },
    }


@app.post("/runs")
def create_run(req: RunReq):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_id, n = base, 1
    while (DATA_DIR / run_id).exists():  # never overwrite a previous generation
        n += 1
        run_id = f"{base}-{n}"
    d = DATA_DIR / run_id
    for s in SUBDIRS:
        (d / s).mkdir(parents=True)
    write_manifest(d, {"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                       "status": "running", "params": req.params, "designs": [], "log": []})
    (DATA_DIR / "latest.txt").write_text(run_id)
    log.info("run %s created", run_id)
    return {"run_id": run_id, "dir": str(d)}


@app.put("/runs/{run_id}/manifest")
def put_manifest(run_id: str, req: ManifestReq):
    d = resolve_run(run_id)
    with lock_for(d.name):
        m = read_manifest(d)
        m["params"] = req.params or m.get("params", {})
        m["status"] = req.status
        old = {x.get("n"): x for x in m.get("designs", [])}
        merged = []
        for c in req.designs:
            prev = old.get(c.get("n"), {})
            row = {**c, "status": prev.get("status", "pending")}
            for k in ("file", "thumb", "qc", "attempts", "provider_used", "error", "attempt_log"):
                if k in prev:
                    row[k] = prev[k]
            merged.append(row)
        m["designs"] = merged
        write_manifest(d, m)
    return {"ok": True, "designs": len(req.designs)}


@app.get("/runs/{run_id}/manifest")
def get_manifest(run_id: str):
    return read_manifest(resolve_run(run_id))


@app.post("/runs/{run_id}/designs/{n}/generate")
def generate_design(run_id: str, n: int, req: DesignReq):
    d = resolve_run(run_id)
    W, H = req.width, req.height
    gw, gh = gen_size(W, H)
    base_seed = req.seed if req.seed is not None else random.randint(1, 2**31 - 1)
    plan = [p for p in req.providers for _ in range(req.max_attempts)]
    best = None  # (rank, score, canvas, qc, provider)
    attempt_log: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, prov in enumerate(plan, start=1):
        fn = PROVIDERS.get(prov)
        if not fn:
            attempt_log.append({"attempt": i, "provider": prov, "error": "unknown provider"})
            continue
        seed = (base_seed + i * 7919) % (2**31 - 1)
        try:
            log.info("design %s attempt %s via %s", n, i, prov)
            raw = fn(req.prompt, gw, gh, seed)
            canvas, qc = process_image(raw, req.slogan, W, H, req)
        except Exception as e:
            log.warning("design %s attempt %s failed: %s", n, i, e)
            attempt_log.append({"attempt": i, "provider": prov, "error": str(e)[:300]})
            continue
        status = status_from_qc(qc) if canvas is not None else "failed"
        attempt_log.append({"attempt": i, "provider": prov, "status": status, "failures": qc["failures"]})
        if canvas is not None:
            t = qc.get("text_check", {})
            score = (t.get("word_exact", 0), t.get("recall", 0)) if t.get("performed") else (0, 0)
            key = (rank_of(status), score)
            if best is None or key > best[0]:
                best = (key, canvas, qc, prov, status, i)
        if status == "approved":
            break

    result: Dict[str, Any] = {"n": n, "attempt_log": attempt_log, "seconds": round(time.time() - t0, 1)}
    with lock_for(d.name):
        m = read_manifest(d)
        row = next((x for x in m.get("designs", []) if x.get("n") == n), None)
        if row is None:
            row = {"n": n, "slogan": req.slogan, "niche": req.niche, "prompt": req.prompt, "status": "pending"}
            m.setdefault("designs", []).append(row)
        prev_rank = rank_of(row.get("status", "pending")) if row.get("file") else 0
        row["attempts"] = row.get("attempts", 0) + len(attempt_log)
        row["attempt_log"] = (row.get("attempt_log", []) + attempt_log)[-12:]
        if best is not None:
            key, canvas, qc, prov, status, _ = best
            if key[0] >= prev_rank:
                slug = safe_slug(req.slug or req.niche)
                fname = f"{n:02d}_{slug}_Design.png"
                canvas.save(d / "PNG" / fname, "PNG", dpi=(300, 300), compress_level=4)
                th = canvas.copy()
                th.thumbnail((640, 768), Image.LANCZOS)
                th.save(d / "THUMBS" / fname, "PNG")
                qc["png_exists"] = (d / "PNG" / fname).exists()
                if not qc["png_exists"]:
                    qc["failures"].append("PNG file missing after save")
                    status = "failed"
                row.update({"file": fname, "thumb": fname, "qc": qc, "provider_used": prov, "status": status})
        else:
            if not row.get("file"):
                row["status"] = "failed"
            row["error"] = "; ".join(a.get("error", "") for a in attempt_log if a.get("error"))[:400] or "no usable image"
        m.setdefault("log", []).append(f"{datetime.now().strftime('%H:%M:%S')} design {n}: {row['status']} ({len(attempt_log)} attempts)")
        write_manifest(d, m)
        result.update({"status": row["status"], "file": row.get("file"), "provider_used": row.get("provider_used"),
                       "qc_failures": (row.get("qc") or {}).get("failures", []), "error": row.get("error")})
    return result


@app.post("/runs/{run_id}/finalize")
def finalize(run_id: str):
    d = resolve_run(run_id)
    with lock_for(d.name):
        m = read_manifest(d)
        designs = sorted(m.get("designs", []), key=lambda x: x.get("n", 0))
        have = [x for x in designs if x.get("file")]
        approved = [x for x in designs if x.get("status") == "approved"]
        m["status"] = "complete" if len(approved) == len(designs) and designs else ("partial" if have else "failed")

        cols = ["design", "file", "status", "niche", "audience", "slogan", "seo_title", "tags", "illustration_concept",
                "style", "palette", "opportunity_score_ai_estimate", "score_basis", "measured_signals", "provider", "prompt"]
        with open(d / "SEO" / "designs.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for x in designs:
                w.writerow([x.get("n"), x.get("file", ""), x.get("status"), x.get("niche"), x.get("audience"),
                            x.get("slogan"), x.get("seo_title"), ", ".join(x.get("tags", [])), x.get("concept"),
                            x.get("style"), x.get("palette"), x.get("score"), x.get("score_basis"),
                            " | ".join(x.get("measured_signals", [])), x.get("provider_used", ""), x.get("prompt")])
        with open(d / "PROMPTS" / "prompts.txt", "w", encoding="utf-8") as f:
            for x in designs:
                f.write(f"=== Design {x.get('n'):02d} | {x.get('niche')} | \"{x.get('slogan')}\" ===\n{x.get('prompt')}\n\n")
        for x in designs:
            slug = safe_slug(x.get("niche"))
            (d / "SEO" / f"{x.get('n'):02d}_{slug}_seo.txt").write_text(
                f"TITLE:\n{x.get('seo_title')}\n\nTAGS ({len(x.get('tags', []))}):\n{', '.join(x.get('tags', []))}\n\n"
                f"SLOGAN: {x.get('slogan')}\nAUDIENCE: {x.get('audience')}\nPOTENTIAL: {x.get('potential')}\n"
                f"SCORE (AI estimate, not measured): {x.get('score')}\n", encoding="utf-8")
        status_txt = "\n".join(f"{x.get('n'):02d} {x.get('status'):13s} {x.get('file', '-')}  "
                               f"{'; '.join((x.get('qc') or {}).get('failures', []))}" for x in designs)
        zpath = d / "ZIP" / f"{d.name}_tshirt-designs.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
            for x in have:
                z.write(d / "PNG" / x["file"], f"PNG/{x['file']}")
            z.write(d / "SEO" / "designs.csv", "designs.csv")
            z.write(d / "PROMPTS" / "prompts.txt", "prompts.txt")
            z.writestr("STATUS.txt", "QC status per design (approved = every check was performed and passed)\n" + status_txt)
        m["zip"] = zpath.name
        m["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_manifest(d, m)
    return {"ok": True, "status": m["status"], "approved": len(approved), "with_png": len(have), "total": len(designs), "zip": m["zip"]}


@app.get("/runs/{run_id}/file")
def get_file(run_id: str, kind: str = "png", name: str = ""):
    d = resolve_run(run_id)
    folder = {"png": "PNG", "thumb": "THUMBS", "zip": "ZIP", "csv": "SEO", "prompts": "PROMPTS"}.get(kind)
    if not folder:
        raise HTTPException(400, "bad kind")
    name = os.path.basename(name)
    if not name:
        files = sorted((d / folder).glob("*"))
        if not files:
            raise HTTPException(404, "no file yet")
        p = files[0]
    else:
        p = d / folder / name
    if not p.is_file():
        raise HTTPException(404, "file not found")
    mt = {".png": "image/png", ".zip": "application/zip", ".csv": "text/csv", ".txt": "text/plain"}.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=mt, filename=p.name)
