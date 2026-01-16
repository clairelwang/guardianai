from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ExifTags
import io
import tempfile
import numpy as np
import cv2
import torch

from transformers import AutoImageProcessor, AutoModelForImageClassification


# ======================
# App setup
# ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# Forensics helpers
# ======================
COMMON_AI_SIZES = {
    (512, 512), (768, 768), (1024, 1024),
    (512, 768), (768, 512),
    (832, 1216), (1216, 832),
    (1024, 1536), (1536, 1024),
    (1152, 896), (896, 1152),
}

SOFTWARE_SUSPECT_KEYWORDS = [
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dalle",
    "diffusion", "comfyui", "automatic1111", "invokeai", "sdxl",
    "gfpgan", "codeformer", "faceswap", "deepfake", "reface"
]


def extract_image_metadata(pil_img: Image.Image) -> dict:
    meta = {
        "exif_present": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
    }

    exif = pil_img.getexif()
    if not exif:
        return meta

    meta["exif_present"] = True
    named = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

    meta["camera_make"] = named.get("Make")
    meta["camera_model"] = named.get("Model")
    meta["software"] = named.get("Software")

    return meta


def analyze_image_properties(pil_img: Image.Image) -> dict:
    w, h = pil_img.size
    return {
        "width": w,
        "height": h,
        "megapixels": round((w * h) / 1_000_000, 3),
        "aspect_ratio": round(w / h, 4) if h else None,
        "matches_common_ai_size": (w, h) in COMMON_AI_SIZES,
    }


def interpret_forensics(metadata: dict, props: dict) -> dict:
    """
    Converts raw forensics into weak signals. Never used as sole proof.
    """
    software = (metadata.get("software") or "")
    software_l = str(software).lower()

    software_suspect = any(k in software_l for k in SOFTWARE_SUSPECT_KEYWORDS)

    # "real camera" hint: EXIF present + make/model looks like phone/camera
    has_real_camera_hint = bool(metadata.get("exif_present") and (metadata.get("camera_make") or metadata.get("camera_model")))

    return {
        "software_suspect": bool(software_suspect),
        "has_real_camera_hint": bool(has_real_camera_hint),
        "matches_common_ai_size": bool(props.get("matches_common_ai_size")),
    }


# ======================
# Face detection
# ======================
def detect_faces(pil_img: Image.Image):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def crop_face(pil_img: Image.Image, face_box, pad=0.35):
    x, y, w, h = face_box
    W, H = pil_img.size
    px, py = int(w * pad), int(h * pad)
    return pil_img.crop((
        max(0, x - px),
        max(0, y - py),
        min(W, x + w + px),
        min(H, y + h + py),
    ))


# ======================
# Multi-crop stability
# ======================
def multi_crops(pil_img: Image.Image):
    W, H = pil_img.size
    s = int(min(W, H) * 0.85)
    if s < 64:
        return [pil_img]

    def c(x, y):
        return pil_img.crop((x, y, x + s, y + s))

    return [
        pil_img,
        c(0, 0),
        c(W - s, 0),
        c(0, H - s),
        c(W - s, H - s),
        c((W - s) // 2, (H - s) // 2),
    ]


def jpeg_reencode_variants(pil_img: Image.Image):
    """
    Cheap 'temporal smoothing' for a single image:
    re-encode at different JPEG qualities to see if model flips.
    """
    variants = [pil_img]
    for q in (95, 85, 70):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        variants.append(Image.open(buf).convert("RGB"))
    return variants


# ======================
# Model wrapper
# ======================
class ModelWrapper:
    def __init__(self, name, expected_size):
        self.name = name
        self.size = expected_size
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(DEVICE)
        self.model.eval()
        self.fake_idx = self._find_fake_index()

    def _find_fake_index(self):
        for k, v in self.model.config.id2label.items():
            s = v.lower()
            if any(x in s for x in ["fake", "ai", "synthetic", "generated"]):
                return int(k)
        return 1

    def predict(self, pil_img: Image.Image) -> float:
        img = pil_img.resize((self.size, self.size), Image.BICUBIC)
        inputs = self.processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        # move tensors to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
        return float(probs[self.fake_idx].item())


# ======================
# Models (heterogeneous)
# ======================
MODEL_AI = ModelWrapper("dima806/deepfake_vs_real_image_detection", expected_size=224)
MODEL_FORENSIC = ModelWrapper("buildborderless/CommunityForensics-DeepfakeDet-ViT", expected_size=384)
MODEL_FACE = ModelWrapper("prithivMLmods/Deep-Fake-Detector-v2-Model", expected_size=224)


# ======================
# Ensemble logic
# ======================
def ensemble_predict(pil_img: Image.Image, face_detected: bool):
    crops = multi_crops(pil_img)

    scores = {
        "ai": float(np.mean([MODEL_AI.predict(c) for c in crops])),
        "forensic": float(np.mean([MODEL_FORENSIC.predict(c) for c in crops])),
    }

    if face_detected:
        scores["face"] = float(np.mean([MODEL_FACE.predict(c) for c in crops]))

    weights = {
        "ai": 0.40,
        "forensic": 0.20,
        "face": 0.40 if face_detected else 0.0,
    }

    total = sum(weights[k] for k in scores)
    final_score = sum(scores[k] * weights[k] for k in scores) / total
    disagreement = float(np.std(list(scores.values())))

    return {
        "final_score": float(final_score),
        "disagreement": disagreement,
        "per_model": scores,
        "n_models_used": len(scores),
        "n_crops": len(crops),
    }


def explain_ensemble(ens: dict) -> list[str]:
    explanations = []
    for name, score in ens["per_model"].items():
        if score >= 0.70:
            explanations.append(f"{name}: strong AI-like signal")
        elif score >= 0.55:
            explanations.append(f"{name}: moderate AI-like signal")
        elif score >= 0.45:
            explanations.append(f"{name}: weak AI-like signal")
        else:
            explanations.append(f"{name}: mostly real-like signal")
    return explanations


def decide_label(score: float, ens: dict, metadata: dict, forensic_signals: dict, stability: dict):
    """
    Priority 1+2 style decision system:
    - abstain when evidence conflicts
    - use forensics as veto/consistency checks
    - use stability to detect randomness
    """
    disagreement = ens.get("disagreement")
    if disagreement is None:
        return "UNCERTAIN", "Missing disagreement signal."

    # Stability gates (if tiny changes cause big swings, abstain)
    if stability["std"] > 0.12:
        return "UNCERTAIN", "Unstable prediction under minor re-encodes."

    # High model disagreement → abstain
    if disagreement > 0.18:
        return "UNCERTAIN", "Models disagree significantly."

    # Near boundary → abstain
    if 0.42 <= score <= 0.58:
        return "UNCERTAIN", "Score near decision boundary."

    # Strong forensics conflict: real camera hint but high fake score
    if forensic_signals.get("has_real_camera_hint") and score > 0.65:
        return "UNCERTAIN", "Camera metadata conflicts with ML signals."

    # Strong forensics support: explicit software tag hints
    if forensic_signals.get("software_suspect") and score < 0.55:
        return "UNCERTAIN", "Software metadata suggests editing/generation, but ML is unsure."

    # Otherwise decide
    label = "FAKE" if score >= 0.5 else "REAL"
    return label, "Signals are sufficiently consistent."


def image_pipeline(pil_img: Image.Image) -> dict:
    """
    Runs your full robust image pipeline:
    - face gate + face crop
    - ensemble on multi-crops
    - stability check via JPEG re-encodes
    - forensics extraction + interpretation
    - abstention decision
    """
    metadata = extract_image_metadata(pil_img)
    props = analyze_image_properties(pil_img)
    forensic_signals = interpret_forensics(metadata, props)

    faces = detect_faces(pil_img)
    infer_img = crop_face(pil_img, max(faces, key=lambda b: b[2] * b[3])) if faces else pil_img

    # Stability: run on jpeg re-encodes, aggregate
    variant_scores = []
    variant_disagreements = []
    last_ens = None
    for v in jpeg_reencode_variants(infer_img):
        ens = ensemble_predict(v, face_detected=bool(faces))
        last_ens = ens
        variant_scores.append(ens["final_score"])
        variant_disagreements.append(ens["disagreement"])

    stability = {
        "mean": float(np.mean(variant_scores)),
        "std": float(np.std(variant_scores)),
        "min": float(np.min(variant_scores)),
        "max": float(np.max(variant_scores)),
    }

    # Use the mean of variant scores as final score for robustness
    score = stability["mean"]
    ens_summary = {
        # per-model from the last run (good enough for UI)
        "per_model": last_ens["per_model"],
        "disagreement": float(np.mean(variant_disagreements)),
        "n_models_used": last_ens["n_models_used"],
        "n_crops": last_ens["n_crops"],
    }

    # Confidence band from disagreement + stability
    if ens_summary["disagreement"] < 0.08 and stability["std"] < 0.06:
        confidence = "High"
    elif ens_summary["disagreement"] < 0.16 and stability["std"] < 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"

    label, decision_reason = decide_label(score, ens_summary, metadata, forensic_signals, stability)

    return {
        "label": label,
        "score": score,
        "decision_reason": decision_reason,
        "confidence_band": confidence,
        "forensics": {
            "metadata": metadata,
            "image_properties": props,
            "signals": forensic_signals,
            "faces_detected": len(faces),
        },
        "ensemble": ens_summary,
        "model_signals": explain_ensemble(ens_summary),
        "stability": stability,
    }


# ======================
# Video support
# ======================
def extract_video_frames(video_bytes: bytes, max_frames: int = 20, sample_every_seconds: float = 1.0):
    """
    Extract up to max_frames frames at ~1 fps (configurable).
    Requires OpenCV to be able to decode the video (ffmpeg installed helps).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        f.write(video_bytes)
        f.flush()

        cap = cv2.VideoCapture(f.name)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * sample_every_seconds)))

        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1

        cap.release()
        return frames


def video_pipeline(video_bytes: bytes) -> dict:
    frames = extract_video_frames(video_bytes, max_frames=20, sample_every_seconds=1.0)
    if not frames:
        return {
            "label": "ERROR",
            "score": 0.0,
            "confidence_band": "Low",
            "decision_reason": "Could not decode video or no frames extracted.",
            "temporal": {},
        }

    frame_results = []
    for i, frame in enumerate(frames):
        r = image_pipeline(frame)
        frame_results.append({
            "i": i,
            "score": r["score"],
            "label": r["label"],
            "confidence_band": r["confidence_band"],
            "disagreement": r["ensemble"]["disagreement"],
            "faces_detected": r["forensics"]["faces_detected"],
        })

    scores = np.array([x["score"] for x in frame_results], dtype=float)
    disagreements = np.array([x["disagreement"] for x in frame_results], dtype=float)

    # Robust aggregation: 75th percentile catches "spiky" fake frames
    video_score = float(np.percentile(scores, 75))
    volatility = float(np.std(scores))
    mean_disagreement = float(np.mean(disagreements))

    # Decision rules (simple, defensible)
    if volatility > 0.20:
        label = "UNCERTAIN"
        reason = "High frame-to-frame volatility."
    elif mean_disagreement > 0.18:
        label = "UNCERTAIN"
        reason = "High average model disagreement across frames."
    elif video_score >= 0.65:
        label = "FAKE"
        reason = "Many frames show strong manipulation signals."
    elif 0.45 <= video_score <= 0.58:
        label = "UNCERTAIN"
        reason = "Video score near decision boundary."
    else:
        label = "REAL"
        reason = "Frames mostly consistent with real content."

    # Confidence band (based on volatility + disagreement)
    if label != "UNCERTAIN" and volatility < 0.12 and mean_disagreement < 0.12:
        confidence = "High"
    elif volatility < 0.18 and mean_disagreement < 0.16:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "label": label,
        "score": video_score,
        "confidence_band": confidence,
        "decision_reason": reason,
        "temporal": {
            "n_frames": len(frames),
            "volatility_std": volatility,
            "mean_disagreement": mean_disagreement,
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
        },
        "frame_results_preview": frame_results[:10],  # keep response small
    }


# ======================
# API endpoint (image + video)
# ======================
@app.post("/api/deepfake")
async def deepfake(file: UploadFile = File(...)):
    content = await file.read()
    ctype = (file.content_type or "").lower()

    # Images
    if ctype.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        out = image_pipeline(img)
        out["details"] = "Image: ensemble + forensics + stability"
        return out

    # Videos
    if ctype.startswith("video/"):
        out = video_pipeline(content)
        out["details"] = "Video: frame sampling + temporal aggregation"
        return out

    return JSONResponse(
        {"error": "Unsupported file type. Upload an image or video."},
        status_code=400,
    )