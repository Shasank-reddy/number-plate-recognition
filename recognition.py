"""
ALPR (Automatic License Plate Recognition) - Notebook Friendly Template
Compatible with Jupyter Notebook / Google Colab and Jetson Nano (with local camera).

Features
- Notebook UI: upload image or use camera (if available).
- Two pipelines:
    1) If a Keras/TensorFlow plate detector model and a character recognizer are provided,
       the code will attempt to use them.
    2) Otherwise, a robust OpenCV-based detection + optional pytesseract OCR fallback is used.
- Defensive imports: missing optional libraries won't crash the notebook.
- Visual, colorful outputs using IPython.display.
- Designed as a template: replace model paths or adapt detector decode logic for your models.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import time
import string
import random

# -------------------------
# Optional imports (defensive)
# -------------------------
_have_tf = False
_have_tesseract = False
try:
    import tensorflow as tf
    from tensorflow import keras
    _have_tf = True
except Exception:
    _have_tf = False

try:
    import pytesseract
    _have_tesseract = True
except Exception:
    _have_tesseract = False

# -------------------------
# Configuration
# -------------------------
# If you have Keras models, set these paths (or leave None to use OpenCV/Tesseract fallback)
PLATE_DETECTOR_MODEL = None   # e.g., "plate_detector.h5"
CHAR_RECOGNIZER_MODEL = None  # e.g., "char_recognizer.h5"

# Visualization helpers
def colorful(msg, color="#2b7cff", size=14):
    display(HTML(f"<p style='color:{color}; font-size:{size}px; margin:4px 0;'>{msg}</p>"))

def show_image_bgr(img, title=None, figsize=(10,6)):
    """Display BGR image in notebook using matplotlib (converts to RGB)."""
    if img is None:
        colorful("No image to display.", color="red")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(img_rgb)
    plt.show()

# -------------------------
# Model loading (if provided)
# -------------------------
plate_model = None
char_model = None

def try_load_models():
    global plate_model, char_model
    if _have_tf and PLATE_DETECTOR_MODEL and os.path.exists(PLATE_DETECTOR_MODEL):
        try:
            plate_model = keras.models.load_model(PLATE_DETECTOR_MODEL)
            colorful("Loaded plate detector model.", color="green")
        except Exception as e:
            colorful(f"Failed to load plate detector model: {e}", color="red")
            plate_model = None
    else:
        if PLATE_DETECTOR_MODEL:
            colorful(f"Plate detector model not found at {PLATE_DETECTOR_MODEL}. Using fallback.", color="orange")

    if _have_tf and CHAR_RECOGNIZER_MODEL and os.path.exists(CHAR_RECOGNIZER_MODEL):
        try:
            char_model = keras.models.load_model(CHAR_RECOGNIZER_MODEL)
            colorful("Loaded character recognizer model.", color="green")
        except Exception as e:
            colorful(f"Failed to load character recognizer model: {e}", color="red")
            char_model = None
    else:
        if CHAR_RECOGNIZER_MODEL:
            colorful(f"Character recognizer model not found at {CHAR_RECOGNIZER_MODEL}. Using fallback.", color="orange")

try_load_models()

# -------------------------
# Utility image processing functions
# -------------------------
def resize_keep_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def preprocess_plate_for_ocr(plate_img):
    """Preprocess plate image for contour segmentation or OCR."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Contrast enhancement
    gray = cv2.equalizeHist(gray)
    # Bilateral filter to reduce noise while keeping edges
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive threshold (invert for easier contour detection)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    return th

def find_plate_candidates_by_morphology(img_bgr):
    """
    Heuristic OpenCV pipeline to find rectangular plate-like regions.
    Returns list of bounding boxes (x, y, w, h) in image coordinates.
    """
    img = resize_keep_aspect(img_bgr, width=800)
    ratio = img_bgr.shape[1] / img.shape[1]  # to map back to original size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Top-hat to emphasize bright regions on dark background
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kern)
    # Sobel to get vertical edges
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    if maxVal - minVal != 0:
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    else:
        gradX = gradX.astype("uint8")
    # Blur and threshold
    gradX = cv2.GaussianBlur(gradX, (5,5), 0)
    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Morph close to connect regions
    sq_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kern)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        area = w * h
        # Heuristics for plate-like shapes
        if area > 2000 and 2.0 < aspect < 6.5 and h > 15:
            # Map back to original image coordinates
            x0 = int(x * ratio)
            y0 = int(y * ratio)
            w0 = int(w * ratio)
            h0 = int(h * ratio)
            candidates.append((x0, y0, w0, h0))
    # Sort by area (largest first)
    candidates = sorted(candidates, key=lambda b: b[2]*b[3], reverse=True)
    return candidates

def segment_characters_from_plate(th_plate):
    """
    Segment characters from a thresholded plate image.
    Returns list of character images (grayscale) sorted left-to-right.
    """
    # Find contours on thresholded plate
    contours, _ = cv2.findContours(th_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    h_plate = th_plate.shape[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by size relative to plate
        if h > 0.4 * h_plate and w > 0.02 * th_plate.shape[1]:
            regions.append((x, y, w, h))
    regions = sorted(regions, key=lambda r: r[0])  # left to right
    char_images = []
    for (x, y, w, h) in regions:
        roi = th_plate[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)  # channel
        char_images.append(roi)
    return char_images

def recognize_with_char_model(char_model, char_images):
    """Use a Keras char_model to predict characters. Mapping must match training."""
    if not char_images:
        return ""
    X = np.array(char_images)
    preds = char_model.predict(X)
    labels = np.argmax(preds, axis=1)
    # Example mapping: digits then uppercase letters
    idx_to_char = list(string.digits + string.ascii_uppercase)
    plate_text = "".join(idx_to_char[i] if i < len(idx_to_char) else "?" for i in labels)
    return plate_text

def ocr_with_tesseract(plate_img):
    """Use pytesseract to read text from plate image (requires pytesseract installed)."""
    if not _have_tesseract:
        return ""
    # Convert to RGB for pytesseract
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    # Optional config: only alphanumeric, single line
    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7"
    try:
        text = pytesseract.image_to_string(plate_rgb, config=config)
        # Clean text
        text = "".join(ch for ch in text if ch.isalnum())
        return text
    except Exception:
        return ""

# -------------------------
# Main recognition function
# -------------------------
def recognize_plate_from_image(img_bgr, visualize=True):
    """
    High-level function:
    - If plate_model is available, use it (user must adapt decode logic).
    - Else use OpenCV morphological detection to find candidate plates.
    - For each candidate, try char_model recognition, then pytesseract fallback.
    Returns list of dicts: [{'bbox':(x,y,w,h), 'plate':text, 'plate_img':img}, ...]
    """
    results = []
    # 1) If a plate detector model is available, use it (placeholder logic)
    if plate_model is not None:
        try:
            # NOTE: This block is model-specific. Replace with your model's preprocessing & decoding.
            inp = cv2.resize(img_bgr, (224,224)).astype("float32") / 255.0
            inp = np.expand_dims(inp, axis=0)
            pred = plate_model.predict(inp)
            # Example: assume model returns [x, y, w, h, score] normalized
            if pred.shape[-1] >= 5:
                x, y, w, h, score = pred[0][:5]
                if score > 0.4:
                    # Convert normalized to pixel coords
                    H, W = img_bgr.shape[:2]
                    cx = int(x * W); cy = int(y * H)
                    bw = int(w * W); bh = int(h * H)
                    x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
                    x2 = min(W, cx + bw//2); y2 = min(H, cy + bh//2)
                    plate_img = img_bgr[y1:y2, x1:x2]
                    th = preprocess_plate_for_ocr(plate_img)
                    plate_text = ""
                    if char_model is not None:
                        char_imgs = segment_characters_from_plate(th)
                        plate_text = recognize_with_char_model(char_model, char_imgs)
                    if not plate_text and _have_tesseract:
                        plate_text = ocr_with_tesseract(plate_img)
                    results.append({'bbox':(x1,y1,x2-x1,y2-y1), 'plate':plate_text, 'plate_img':plate_img})
        except Exception as e:
            colorful(f"Plate model inference failed: {e}", color="orange")

    # 2) Fallback OpenCV morphological detection
    candidates = find_plate_candidates_by_morphology(img_bgr)
    used = set()
    for (x,y,w,h) in candidates:
        # Avoid duplicates if model already found similar bbox
        key = (x//10, y//10, w//10, h//10)
        if key in used:
            continue
        used.add(key)
        plate_img = img_bgr[y:y+h, x:x+w].copy()
        if plate_img.size == 0:
            continue
        th = preprocess_plate_for_ocr(plate_img)
        plate_text = ""
        # Try char model first
        if char_model is not None:
            char_imgs = segment_characters_from_plate(th)
            plate_text = recognize_with_char_model(char_model, char_imgs)
        # Fallback to pytesseract
        if not plate_text and _have_tesseract:
            plate_text = ocr_with_tesseract(plate_img)
        # If still empty, try a simple heuristic: find contours and join digits (very rough)
        if not plate_text:
            # Attempt to segment characters and do simple template matching (not implemented)
            plate_text = ""  # leave blank if unknown
        results.append({'bbox':(x,y,w,h), 'plate':plate_text, 'plate_img':plate_img})

    # Optionally visualize
    if visualize:
        vis = img_bgr.copy()
        for res in results:
            x,y,w,h = res['bbox']
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            label = res['plate'] if res['plate'] else "UNKNOWN"
            cv2.putText(vis, label, (x, max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        show_image_bgr(vis, title="Detected Plates (green boxes)")
    return results

# -------------------------
# Notebook UI: upload image or use camera
# -------------------------
upload = widgets.FileUpload(accept='image/*', multiple=False)
btn_process = widgets.Button(description="Process Uploaded Image", button_style="success")
out = widgets.Output()

def on_process_clicked(b):
    with out:
        clear_output()
        if not upload.value:
            colorful("Please upload an image first.", color="red")
            return
        # Get uploaded file bytes
        uploaded_filename = list(upload.value.keys())[0]
        content = upload.value[uploaded_filename]['content']
        # Convert to numpy array and decode
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            colorful("Failed to decode uploaded image.", color="red")
            return
        colorful(f"Processing {uploaded_filename} ...", color="#2b7cff")
        results = recognize_plate_from_image(img, visualize=True)
        if not results:
            colorful("No plate candidates found.", color="orange")
            return
        # Show cropped plates and recognized text
        for i, r in enumerate(results, 1):
            colorful(f"Candidate {i}: Detected text = <b>{r['plate'] or 'UNKNOWN'}</b>", color="#0b6623")
            show_image_bgr(r['plate_img'], title=f"Plate candidate {i}", figsize=(6,2))

btn_process.on_click(on_process_clicked)

# Camera capture (optional) - works if cv2.VideoCapture is available and camera accessible
btn_camera = widgets.Button(description="Capture from Camera (one frame)", button_style="info")
def on_camera_clicked(b):
    with out:
        clear_output()
        colorful("Capturing from camera...", color="#2b7cff")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            colorful("Camera not accessible. Make sure a camera is connected and accessible.", color="red")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            colorful("Failed to capture frame from camera.", color="red")
            return
        show_image_bgr(frame, title="Captured Frame")
        results = recognize_plate_from_image(frame, visualize=True)
        if not results:
            colorful("No plate candidates found.", color="orange")
            return
        for i, r in enumerate(results, 1):
            colorful(f"Candidate {i}: Detected text = <b>{r['plate'] or 'UNKNOWN'}</b>", color="#0b6623")
            show_image_bgr(r['plate_img'], title=f"Plate candidate {i}", figsize=(6,2))

btn_camera.on_click(on_camera_clicked)

# Display UI
ui = widgets.VBox([
    widgets.HTML("<h3 style='color:#2b7cff'>ALPR Notebook Interface</h3>"),
    widgets.HBox([widgets.Label("Upload image:"), upload, btn_process, btn_camera]),
    out
])
display(ui)

# -------------------------
# Example usage note
# -------------------------
colorful("Notes: If you have trained Keras models, set PLATE_DETECTOR_MODEL and CHAR_RECOGNIZER_MODEL variables at the top of this cell and re-run. "
         "If pytesseract is not installed, install it (and Tesseract engine) to enable OCR fallback.", color="#555555")
