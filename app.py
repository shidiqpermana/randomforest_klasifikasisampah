import io
import os
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from flask import Flask, render_template, request


# ==========================================
# Inference-only Flask App (modernized)
# ==========================================

IMAGE_SIZE = (100, 100)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL_PATH = "model.pkl"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to flattened grayscale feature vector."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format file valid.")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, IMAGE_SIZE)
    features = image_resized.flatten().astype(np.float32)
    return features.reshape(1, -1)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{model_path}'. Jalankan notebook pelatihan untuk membuatnya."
        )
    return joblib.load(model_path)


def get_top_predictions(model, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        class_labels = getattr(model, "classes_", None)
        if class_labels is None:
            # Fallback: buat label numerik bila tidak tersedia
            class_labels = [str(i) for i in range(len(probs))]
        ranked = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    # Fallback tanpa proba
    pred = model.predict(features)[0]
    return [(str(pred), 1.0)]


app = Flask(__name__)
model = load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    top_probs: List[Tuple[str, float]] = []
    error_message = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error_message = "Silakan pilih file gambar."
        elif not allowed_file(file.filename):
            error_message = "Format file tidak didukung. Gunakan JPG/JPEG/PNG."
        else:
            try:
                image_bytes = file.read()
                features = preprocess_image_bytes(image_bytes)
                top_probs = get_top_predictions(model, features, top_k=3)
                prediction = top_probs[0][0] if top_probs else None
            except Exception as exc:
                error_message = f"Terjadi kesalahan saat memproses gambar: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        top_probs=top_probs,
        error_message=error_message,
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    # Set host untuk akses jaringan lokal; matikan debug di produksi
    app.run(host="0.0.0.0", port=5000, debug=True)
