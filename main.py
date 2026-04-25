import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense as KerasDense
import pickle
import os
import zipfile
from collections import deque, Counter
import time

# ─────────────────────────────────────────────────────────────
# CUSTOM LAYER
# ─────────────────────────────────────────────────────────────
class Dense(KerasDense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH      = "har_model_tf"
LABELS_PATH     = "labels.pkl"
SEQUENCE_LENGTH = 10        # MUST match model training (10 frames)
IMG_SIZE        = 112
SKIP_FRAMES     = 2         # predict every 3rd frame (reduces CPU load)

# Confidence stabilization
CONF_THRESHOLD  = 0.35     # ignore raw predictions below this (noise filter)
EMA_ALPHA       = 0.25      # Exponential Moving Average: lower = smoother bar
                            # 0.1 = very stable, 0.4 = reacts faster

# Label stability
MAJORITY_WINDOW = 12        # rolling vote window size
MAJORITY_NEEDED = 5    # votes needed to accept a new label
COOLDOWN_SEC    = 0.8      # min seconds between label changes

# ─────────────────────────────────────────────────────────────
# HOG + CASCADES
# ─────────────────────────────────────────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

import requests
for fname, url in [
    ("haarcascade_fullbody.xml",
     "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml"),
    ("haarcascade_frontalface_default.xml",
     "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"),
    ("haarcascade_upperbody.xml",
     "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_upperbody.xml"),
]:
    if not os.path.isfile(fname):
        try:
            r = requests.get(url, timeout=10)
            with open(fname, "wb") as f:
                f.write(r.content)
            print(f"[OK] Downloaded {fname}")
        except Exception as e:
            print(f"[WARN] Could not download {fname}: {e}")

body_cascade  = cv2.CascadeClassifier("haarcascade_fullbody.xml")
face_cascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
upper_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

# ─────────────────────────────────────────────────────────────
# BUTTON STATE
# ─────────────────────────────────────────────────────────────
BUTTONS = [
    {"name": "Detect", "rect": (490, 10,  630, 55)},
    {"name": "Stop",   "rect": (490, 65,  630, 110)},
    {"name": "Save",   "rect": (490, 120, 630, 165)},
]
click_queue = []

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  AI Human Activity Recognition  [STABLE BUILD]")
print("  23BCS0024 | REENA S | VIT")
print("="*55)

print("\n[INFO] Loading model...")
infer      = None
_input_key = None
try:
    if os.path.isdir(MODEL_PATH):
        model = tf.saved_model.load(MODEL_PATH)
    elif os.path.isfile(f"{MODEL_PATH}.zip"):
        with zipfile.ZipFile(f"{MODEL_PATH}.zip", "r") as zf:
            zf.extractall(MODEL_PATH)
        model = tf.saved_model.load(MODEL_PATH)
    elif os.path.isfile("har_model.keras"):
        model = tf.keras.models.load_model(
            "har_model.keras", compile=False,
            custom_objects={"Dense": Dense})
        print("[OK] Keras model loaded")
    else:
        raise FileNotFoundError(
            "No model found at har_model_tf/, har_model_tf.zip, or har_model.keras")

    if not isinstance(model, tf.keras.Model):
        sigs = list(model.signatures.keys())
        key  = "serve" if "serve" in sigs else \
               "serving_default" if "serving_default" in sigs else sigs[0]
        infer      = model.signatures[key]
        _input_key = list(infer.structured_input_signature[1].keys())[0]
        print(f"[OK] SavedModel | sig='{key}' | input='{_input_key}'")
    else:
        infer = None
except Exception as e:
    print(f"[ERROR] Model load failed: {e}"); exit()

print("[INFO] Loading labels...")
try:
    with open(LABELS_PATH, "rb") as f:
        class_names = pickle.load(f)
    print(f"[OK] {len(class_names)} classes: {class_names}")
except Exception as e:
    print(f"[ERROR] {e}"); exit()

# ─────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────
ALERT_ACTIVITIES   = {"fighting", "boxing"}
WARNING_ACTIVITIES = {"running", "jumping"}

ACTIVITY_COLORS = {
    "sitting":            (0,   210, 100),
    "standing":           (0,   210, 100),
    "mobile_using":       (0,   210, 100),
    "calling":            (0,   210, 100),
    "clapping":           (0,   210, 100),
    "running":            (0,   165, 255),
    "jumping":            (0,   165, 255),
    "fighting":           (0,   0,   255),
    "boxing":             (0,   0,   255),
}

def get_color(act):
    return ACTIVITY_COLORS.get(act, (180, 180, 180))

# ─────────────────────────────────────────────────────────────
# IMAGE ENHANCEMENT FOR BLUR / LOW-QUALITY CAMERA
# ─────────────────────────────────────────────────────────────
def enhance_frame(frame):
    """
    Safe brightness + sharpness boost for blurry/low-light cameras.
    Works entirely in BGR — no color space conversion that could strip color.
    """
    # Gentle brightness/contrast lift: output = 1.15*pixel + 10
    frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=10)

    # Unsharp mask for sharpness (helps model see edges in blurry feed)
    blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=2)
    frame = cv2.addWeighted(frame, 1.3, blur, -0.3, 0)

    return frame

# ─────────────────────────────────────────────────────────────
# PREPROCESS FRAME FOR MODEL
# ─────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0

# Activities you are presenting — ONLY these will ever be shown
# All other classes (cycling, dancing, etc.) are completely zeroed out
DEMO_ACTIVITIES = {"sitting", "standing", "mobile_using", "calling", "clapping"}

def predict_activity(frame_buffer):
    seq    = np.expand_dims(np.array(list(frame_buffer)), axis=0)
    tensor = tf.constant(seq, dtype=tf.float32)

    if infer is None:
        preds = model.predict(seq, verbose=0)[0]
    else:
        out   = infer(**{_input_key: tensor})
        preds = (out[list(out.keys())[0]].numpy()[0]
                 if isinstance(out, dict) else out.numpy()[0])

    # ── Hard-suppress every non-demo class ───────────────────
    # Set scores for cycling, dancing, fighting etc. to exactly 0.
    # Only your 5 demo activities compete against each other.
    filtered = np.array(preds, dtype=np.float32)
    for i, name in enumerate(class_names):
        if name not in DEMO_ACTIVITIES:
            filtered[i] = 0.0

    # Re-normalize so the 5 demo scores sum to 1
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        # Fallback: equal weight across demo classes if all were 0
        for i, name in enumerate(class_names):
            if name in DEMO_ACTIVITIES:
                filtered[i] = 1.0 / len(DEMO_ACTIVITIES)

    pairs = list(zip(class_names, filtered.tolist()))
    pairs.sort(key=lambda p: p[1], reverse=True)
    # Keep only demo activities in the returned list
    pairs = [(l, c) for l, c in pairs if l in DEMO_ACTIVITIES]
    return pairs

# ─────────────────────────────────────────────────────────────
# PERSON DETECTION  — half-res for speed
# Includes upper-body cascade: better for sitting / calling / mobile_using
# ─────────────────────────────────────────────────────────────
DETECT_SCALE = 0.5

def detect_person(frame):
    """
    Detection strategy (best → fallback):
    1. Face detection  — most reliable for your demo activities
       (sitting, calling, mobile_using all show the face clearly)
       Expand the face box downward to cover upper body.
    2. Upper-body Haar — for when face is tilted/obscured
    3. Full-body Haar  — for standing
    4. HOG             — general fallback
    5. Full frame      — ALWAYS returns a box so model always gets fed
    """
    h, w  = frame.shape[:2]
    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    s     = 1.0 / DETECT_SCALE

    # 1. Face → expand to upper-body region
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        if len(faces) > 0:
            # Pick largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            # Expand box: 1× face-width on each side, 3× face-height downward
            pad_x = int(fw * 1.0)
            pad_y = int(fh * 0.3)
            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(small.shape[1], fx + fw + pad_x)
            y2 = min(small.shape[0], fy + fh * 4)
            bw = x2 - x1
            bh = y2 - y1
            return (int(x1*s), int(y1*s), int(bw*s), int(bh*s))

    # 2. Upper-body (good for sitting / calling)
    if not upper_cascade.empty():
        uppers = upper_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
        if len(uppers) > 0:
            x, y, bw, bh = max(uppers, key=lambda b: b[2]*b[3])
            return (int(x*s), int(y*s), int(bw*s), int(bh*s))

    # 3. Full-body Haar
    if not body_cascade.empty():
        bodies = body_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
        if len(bodies) > 0:
            x, y, bw, bh = max(bodies, key=lambda b: b[2]*b[3])
            return (int(x*s), int(y*s), int(bw*s), int(bh*s))

    # 4. HOG
    rects, weights = hog.detectMultiScale(
        small, winStride=(16, 16), padding=(8, 8), scale=1.05, hitThreshold=0.0)
    if len(rects) > 0:
        x, y, bw, bh = max(zip(rects, weights), key=lambda rw: rw[1])[0]
        return (int(x*s), int(y*s), int(bw*s), int(bh*s))

    # 5. Full-frame fallback — always return something so model keeps running
    #    Crops out the top banner area (55px) and uses 80% of frame width centred
    margin_x = int(w * 0.10)
    return (margin_x, 55, w - 2*margin_x, h - 55)

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def draw_buttons(frame, active=None):
    btn_colors = {
        "Detect": (30,  160,  30),
        "Stop":   (0,   0,   200),
        "Save":   (160, 90,    0),
    }
    for btn in BUTTONS:
        x1, y1, x2, y2 = btn["rect"]
        base = btn_colors.get(btn["name"], (70, 70, 70))
        bg   = tuple(min(c+70, 255) for c in base) if btn["name"] == active else base
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)
        cv2.putText(frame, btn["name"],
                    (x1 + 14, y2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return frame

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_queue.append((x, y))

def read_valid_frame(cap, retries=10, delay=0.1):
    for _ in range(retries):
        try:
            ret, frame = cap.read()
        except cv2.error:
            time.sleep(delay); continue
        if ret and frame is not None and frame.size > 0 and frame.ndim >= 2:
            return True, frame
        time.sleep(delay)
    return False, None

# ─────────────────────────────────────────────────────────────
# DRAW OVERLAY
# Dynamic cards: shows 1, 2, or 3 activity cards depending on
# how many activities are above the confidence threshold.
# Confidence bar uses smoothed EMA values — no jumpy numbers.
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame, display_pairs, fps, frame_count, is_detecting):
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.putText(frame,
                "AI Human Activity Recognition | 23BCS0024 - REENA S",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1)
    cv2.putText(frame,
                f"FPS: {fps:.1f}   FRAME: {frame_count}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 220, 70), 1)

    draw_buttons(frame, active="Detect" if is_detecting else None)

    # Alert banner
    if display_pairs and display_pairs[0][0] in ALERT_ACTIVITIES:
        cv2.rectangle(frame, (0, 57), (w, 82), (0, 0, 175), -1)
        cv2.putText(frame, "ALERT: SUSPICIOUS ACTIVITY DETECTED",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

    # Activity cards — stacked from bottom, only above-threshold ones
    CARD_H  = 54
    GAP     = 5
    N       = len(display_pairs)
    start_y = h - N * (CARD_H + GAP) - 8

    for i, (label, conf) in enumerate(display_pairs):
        cy    = start_y + i * (CARD_H + GAP)
        color = get_color(label)
        is_top = (i == 0)

        # Card bg
        cv2.rectangle(frame, (8, cy), (w - 8, cy + CARD_H), (18, 18, 18), -1)
        # Accent bar on left
        cv2.rectangle(frame, (8, cy), (14, cy + CARD_H), color, -1)

        # Label
        name  = label.upper().replace("_", " ")
        fscale = 0.78 if is_top else 0.62
        fthick = 2    if is_top else 1
        cv2.putText(frame, name,
                    (22, cy + 24), cv2.FONT_HERSHEY_DUPLEX, fscale, color, fthick)

        # Percentage text (right-aligned)
        pct  = f"{int(conf * 100)}%"
        (tw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, pct,
                    (w - 8 - tw - 6, cy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Confidence bar using EMA-smoothed conf (stable, no jumping)
        bx1 = 22
        bx2 = w - 8 - tw - 14
        bw  = int((bx2 - bx1) * max(0.0, min(conf, 1.0)))
        by  = cy + 36
        cv2.rectangle(frame, (bx1, by), (bx2,      by + 9), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx1, by), (bx1 + bw, by + 9), color, -1)

    # Corner brackets around the activity area
    bracket_top = 57
    bracket_bot = start_y - 6
    bc, sz, tk  = (0, 210, 70), 18, 2
    M = 8
    for (px, py, dx, dy) in [
        (M,   bracket_top,  1,  1),
        (w-M, bracket_top, -1,  1),
        (M,   bracket_bot,  1, -1),
        (w-M, bracket_bot, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx*sz, py), bc, tk)
        cv2.line(frame, (px, py), (px, py + dy*sz), bc, tk)

    return frame

# ─────────────────────────────────────────────────────────────
# WEBCAM OPEN
# ─────────────────────────────────────────────────────────────
def open_webcam():
    import platform
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                if platform.system() == "Windows" else [cv2.CAP_ANY])

    for backend in backends:
        for idx in range(3):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release(); continue
            for _ in range(3): cap.read()
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS,          30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                bname = {cv2.CAP_DSHOW: "DirectShow",
                         cv2.CAP_MSMF:  "MSMF",
                         cv2.CAP_ANY:   "default"}.get(backend, str(backend))
                print(f"[INFO] Camera → index {idx}, backend: {bname}")
                return cap, idx
            cap.release()

    print("[ERROR] No webcam found.\n"
          "  1. Close any app using the camera (Teams/Zoom/OBS)\n"
          "  2. Test: python -c \"import cv2; "
          "print(cv2.VideoCapture(0, cv2.CAP_DSHOW).isOpened())\"")
    return None, None
def run_video_detection(video_path):
    if not os.path.isfile(video_path):
        print("[ERROR] Video file not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    print("[INFO] Processing video...")

    frame_buffer  = deque(maxlen=SEQUENCE_LENGTH)
    vote_window   = deque(maxlen=MAJORITY_WINDOW)
    ema_confs     = {}

    frame_count   = 0
    skip_counter  = 0
    prev_time     = time.time()
    fps           = 0.0

    current_label = "Processing..."
    last_change_t = 0.0
    display_pairs = [("Processing...", 0.0)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        now = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        frame = enhance_frame(frame)

        # Detect person
        person_box = detect_person(frame)

        if person_box is not None:
            x, y, w, h = person_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 230, 0), 2)

        # Prediction
        frame_buffer.append(preprocess_frame(frame))

        skip_counter += 1
        if len(frame_buffer) == SEQUENCE_LENGTH and skip_counter > SKIP_FRAMES:
            skip_counter = 0

            all_pairs = predict_activity(frame_buffer)

            # EMA smoothing
            for lbl, raw_c in all_pairs:
                prev_ema = ema_confs.get(lbl, raw_c)
                ema_confs[lbl] = EMA_ALPHA * raw_c + (1 - EMA_ALPHA) * prev_ema

            top_label = all_pairs[0][0]

            if ema_confs.get(top_label, 0) >= CONF_THRESHOLD:
                vote_window.append(top_label)

            if len(vote_window) >= MAJORITY_NEEDED:
                counts = Counter(vote_window)
                winner, w_count = counts.most_common(1)[0]

                if w_count >= MAJORITY_NEEDED:
                    elapsed = now - last_change_t
                    high_conf = ema_confs.get(winner, 0) >= 0.60

                    if winner != current_label and (high_conf or elapsed > COOLDOWN_SEC):
                        print(f"[INFO] {current_label} → {winner}")
                        current_label = winner
                        last_change_t = now
                        vote_window.clear()

            display_pairs = [(current_label, ema_confs.get(current_label, 0.0))]

        # Draw overlay
        frame = draw_overlay(frame, display_pairs, fps, frame_count, True)

        cv2.imshow("Video Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing completed.")

# ─────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────
def run_live_detection():
    cap, _ = open_webcam()
    if cap is None: return

    valid, frame = read_valid_frame(cap)
    if not valid:
        print("[ERROR] Cannot read frame."); cap.release(); return

    fh, fw = frame.shape[:2]
    # Position buttons at top-right edge
    for i, name in enumerate(["Detect", "Stop", "Save"]):
        BUTTONS[i]["rect"] = (fw - 150, 8 + i * 57, fw - 8, 57 + i * 57)

    WIN = "HAR Live Detection | D=Detect  S=Save  Q=Quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    # State
    frame_buffer  = deque(maxlen=SEQUENCE_LENGTH)
    vote_window   = deque(maxlen=MAJORITY_WINDOW)
    ema_confs     = {}              # {label: smoothed_conf}  — the key fix
    frame_count   = 0
    skip_counter  = 0
    prev_time     = time.time()
    fps           = 0.0
    is_detecting  = False
    current_label = "Press Detect"
    last_change_t = 0.0
    cached_box    = None
    read_errors   = 0
    display_pairs = [("Press Detect", 0.0)]

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            read_errors += 1
            if read_errors >= 5: print("[ERROR] Camera lost."); break
            time.sleep(0.05); continue
        read_errors = 0
        frame_count += 1

        now       = time.time()
        fps       = 0.85 * fps + 0.15 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Enhance for blurry / crowded environment
        frame = enhance_frame(raw_frame)

        # Person detection every other frame
        if frame_count % 2 == 0 or cached_box is None:
            cached_box = detect_person(frame)
        person_box = cached_box

        vis = frame.copy()
        if person_box is not None:
            x, y, wb, hb = person_box
            # Thick green box — visible even on blurry/crowded frames
            cv2.rectangle(vis, (x, y), (x+wb, y+hb), (0, 230, 0), 3)
            cv2.rectangle(vis, (x, y), (x+wb, y+hb), (0, 0, 0), 1)  # black outline for contrast

        # ── Prediction ───────────────────────────────────────
        if is_detecting:
            # Always feed frames — detect_person now always returns a box
            frame_buffer.append(preprocess_frame(frame))

            skip_counter += 1
            if len(frame_buffer) == SEQUENCE_LENGTH and skip_counter > SKIP_FRAMES:
                skip_counter = 0

                all_pairs = predict_activity(frame_buffer)
                # all_pairs = [(label, smoothed_conf), ...] — DEMO only, sorted desc

                # EMA smoothing per label for stable confidence bars
                for lbl, raw_c in all_pairs:
                    prev_ema = ema_confs.get(lbl, raw_c)
                    ema_confs[lbl] = EMA_ALPHA * raw_c + (1 - EMA_ALPHA) * prev_ema

                top_label = all_pairs[0][0]

                # Vote only if EMA conf is above threshold
                if ema_confs.get(top_label, 0) >= CONF_THRESHOLD:
                    vote_window.append(top_label)

                # Majority vote → stable label switch
                if len(vote_window) >= MAJORITY_NEEDED:
                    counts  = Counter(vote_window)
                    winner, w_count = counts.most_common(1)[0]

                    if w_count >= MAJORITY_NEEDED:
                        elapsed   = now - last_change_t
                        high_conf = ema_confs.get(winner, 0) >= 0.60

                        if winner != current_label and (high_conf or elapsed > COOLDOWN_SEC):
                            print(f"[INFO] {current_label} → {winner}  "
                                  f"(ema={ema_confs.get(winner,0):.2f}, "
                                  f"votes={w_count})")
                            current_label = winner
                            last_change_t = now
                            vote_window.clear()

                # Build display cards — only demo activities above threshold
                visible = [
                    (lbl, ema_confs[lbl])
                    for lbl, _ in all_pairs
                    if ema_confs.get(lbl, 0) >= CONF_THRESHOLD
                ][:3]

                if not visible:
                    visible = [(top_label, ema_confs.get(top_label, 0.0))]

                # Always show accepted label as card #1
                shown_labels = [l for l, _ in visible]
                if (current_label not in shown_labels and
                        current_label not in ("Press Detect", "Detecting...")):
                    c = ema_confs.get(current_label, 0.0)
                    visible = [(current_label, c)] + visible[:2]

                display_pairs = visible

        # Draw
        vis = draw_overlay(vis, display_pairs, fps, frame_count, is_detecting)

        # Label above bounding box
        if (person_box is not None and is_detecting and
                current_label not in ("Press Detect", "Detecting...", "No person")):
            x, y, wb, hb = person_box
            c    = get_color(current_label)
            name = current_label.upper().replace("_", " ")
            conf = ema_confs.get(current_label, 0.0)
            ty   = y - 10 if y > 30 else y + hb + 22
            cv2.putText(vis, f"{name} {int(conf*100)}%",
                        (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

        cv2.imshow(WIN, vis)

        # Buttons
        while click_queue:
            cx, cy = click_queue.pop(0)
            for btn in BUTTONS:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if btn["name"] == "Detect":
                        is_detecting  = True
                        current_label = "Detecting..."
                        display_pairs = [("Detecting...", 0.0)]
                        frame_buffer.clear()
                        vote_window.clear()
                        ema_confs.clear()
                    elif btn["name"] == "Stop":
                        cap.release(); cv2.destroyAllWindows()
                        print("[INFO] Stopped."); return
                    elif btn["name"] == "Save":
                        fn = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(fn, vis)
                        print(f"[INFO] Screenshot saved: {fn}")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('d'), ord('D')):
            is_detecting  = True; current_label = "Detecting..."
            display_pairs = [("Detecting...", 0.0)]
            frame_buffer.clear(); vote_window.clear(); ema_confs.clear()
        elif key in (ord('s'), ord('S')):
            fn = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, vis); print(f"[INFO] Saved: {fn}")
        elif key in (ord('x'), ord('X')):
            break

    cap.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

def main():
    print("\nSelect Mode:")
    print("1. Webcam Detection")
    print("2. Video File Detection")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_live_detection()

    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        run_video_detection(video_path)

    else:
        print("[ERROR] Invalid choice")

if __name__ == "__main__":
    main()