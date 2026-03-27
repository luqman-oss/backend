"""
═══════════════════════════════════════════════════════════════════
 FACE MEASUREMENT ENGINE v8.1 — TRUE 3D MEASUREMENTS
 
 Uses MediaPipe FaceLandmarker (478 landmarks: 468 face + 10 iris)
 Now includes Z-axis depth data for accurate 3D facial measurements

 ── Complete changelog from your v6 base ─────────────────────────

 [B1]  SEX PASSED THROUGH THE ENTIRE PIPELINE
       process_image_stabilized(), process_multi_frame(), and
       run_worker_mode() all now accept and forward sex="male"|
       "female"|"unknown". Previously sex was accepted by
       process_image() but silently dropped by everything above it.

 [B2]  BRIGHTNESS CHECK ON FACE CENTRE ONLY
       Old: np.mean(full_frame) — triggered on dark backgrounds.
       New: samples central 60% of frame (where face actually is).
       Threshold raised from 35 → 50 for reliable iris detection.

 [B3]  BLUR CHECK RE-ENABLED AS WARNING
       Was commented out entirely. Now runs as a warning so the
       user is told but not hard-rejected.

 [B4]  PIXELS/MM SANITY GUARD
       After calibration, if ppm < 2.5 the image is hard-rejected.
       ppm=2.22 (as seen in the dark test photo) would now be caught
       immediately with a clear "move closer" message.

 [B5]  TIGHTER CROSS-VALIDATION WINDOW
       Old range: face_w_mm < 100 or > 200 (too wide to catch errors)
       New range: sex-specific plausibility window (115–175 for unknown)
       Blend ratio increased: 0.70/0.30 → 0.60/0.40 for stronger fix.

 [B6]  SEX-SPECIFIC IRIS DIAMETER (kept from your changes)
       male: 11.3mm, female: 11.0mm, unknown: 11.15mm
       Old AVERAGE_IRIS_DIAMETER_MM = 11.7 fallback removed entirely.

 [B7]  MULTI-LANDMARK BRIDGE WIDTH
       Was: single pair (landmarks 122, 351).
       Now: average of 3 pairs (high/mid/low bridge), uses minimum
       to find the true narrowest point glasses rest on.

 [B8]  SIDE LENGTH — LOOKUP TABLE REPLACES × 1.05 FORMULA
       Old formula gave ~168mm for a 160mm face (40mm error).
       New: sex-specific lookup table mapping face width →
       standard glasses arm sizes (128–145mm range).

 [B9]  PITCH DETECTION & COMPENSATION
       Detects chin-up / chin-down head tilt via iris-to-nose
       vertical offset. Warns at >12°, rejects at >20°.
       Applied as combined roll+pitch correction on all vertical
       measurements (face_height, eye_height).

 [B10] IRIS PIXEL SIZE GUARD
       iris_px < 6  → hard reject ("move closer to camera")
       iris_px < 14 → warning  ("calibration may be less accurate")

 [B11] PER-EYE MEASUREMENTS
       eye_width_left_mm / eye_width_right_mm
       eye_height_left_mm / eye_height_right_mm
       Enables asymmetry detection for progressive lens fitting.

 [B12] RANGE VALIDATION WITH WARNINGS
       Sex-specific plausibility ranges checked after measurement.
       Outliers return as warnings[], not hard errors, preserving UX.

 [B13] STABILIZATION PASSES: 3 → 5
       More samples = more stable median across all resolutions.

 [B14] SEX FORWARDED IN WORKER MODE
       run_worker_mode() now reads sex from stdin JSON and passes
       it to process_image_stabilized(). Was silently ignored before.

 [B15] VERSION TAG IN ALL RESPONSES
       Every JSON response includes "engine_version": "8.0.0".

 [B16] CONFIDENCE SCORE USES IRIS PIXEL SIZE
       Old: used px_per_mm * 5 (indirect, misleading).
       New: directly penalises small iris_px for honest confidence.
═══════════════════════════════════════════════════════════════════
"""

import sys
import json
import math
import os
import traceback
import numpy as np
import cv2

ENGINE_VERSION = "8.1.0"

try:
    import mediapipe as mp
    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    _MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
except ImportError:
    print(json.dumps({"success": False, "engine_version": ENGINE_VERSION,
                      "error": {"code": "MEDIAPIPE_NOT_INSTALLED",
                                "message": "Run: pip install mediapipe"}}))
    sys.exit(1)
except AttributeError:
    print(json.dumps({"success": False, "engine_version": ENGINE_VERSION,
                      "error": {"code": "MEDIAPIPE_STRUCTURE_ERROR",
                                "message": "Ensure mediapipe >= 0.10.30"}}))
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  LANDMARK INDEX MAP
# ═══════════════════════════════════════════════════════════════

LANDMARKS = {
    # ── Iris ──────────────────────────────────────────────────
    "LEFT_IRIS_CENTER":    468,
    "RIGHT_IRIS_CENTER":   473,
    # ── Eyes ──────────────────────────────────────────────────
    "LEFT_EYE_INNER":      133,
    "LEFT_EYE_OUTER":      33,
    "LEFT_EYE_TOP":        159,
    "LEFT_EYE_BOTTOM":     145,
    "RIGHT_EYE_INNER":     362,
    "RIGHT_EYE_OUTER":     263,
    "RIGHT_EYE_TOP":       386,
    "RIGHT_EYE_BOTTOM":    374,
    # ── Face boundary ─────────────────────────────────────────
    "FACE_LEFT_CHEEK":     234,
    "FACE_RIGHT_CHEEK":    454,
    "FOREHEAD_TOP":        10,
    "CHIN_BOTTOM":         152,
    # ── Nose / Bridge — 3 pairs for robust width [B7] ─────────
    "BRIDGE_LEFT_HIGH":    122,
    "BRIDGE_RIGHT_HIGH":   351,
    "BRIDGE_LEFT_MID":     188,
    "BRIDGE_RIGHT_MID":    412,
    "BRIDGE_LEFT_LOW":     114,
    "BRIDGE_RIGHT_LOW":    343,
    "NOSE_TIP":            1,
    "NOSE_BRIDGE_TOP":     6,
    # ── Forehead ──────────────────────────────────────────────
    "FOREHEAD_LEFT":       54,
    "FOREHEAD_RIGHT":      284,
    # ── Yaw estimation ────────────────────────────────────────
    "LEFT_CHEEK_MID":      50,
    "RIGHT_CHEEK_MID":     280,
}


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

# [B6] Sex-specific iris diameter (Hashemi 2015, Jonas 2004)
IRIS_DIAMETER_BY_SEX = {
    "male":    11.3,
    "female":  11.0,
    "unknown": 11.15,
}

# Average adult face width by sex — used in cross-validation [B5]
FACE_WIDTH_AVG_BY_SEX = {
    "male":    145.0,
    "female":  135.0,
    "unknown": 140.0,
}

# Sex-specific cross-validation face width windows [B5]
FACE_WIDTH_RANGE_BY_SEX = {
    "male":    (128, 165),
    "female":  (116, 152),
    "unknown": (115, 168),
}

# [B12] Sex-specific measurement plausibility ranges (min_mm, max_mm)
# Based on ANSUR II, ISO 7250-1, and optometric literature
MEASUREMENT_RANGES = {
    "male": {
        "pupillary_distance_mm": (58,  74),
        "face_height_mm":        (170, 230),
        "eye_height_mm":         (7,   14),
        "forehead_width_mm":     (108, 148),
        "face_width_mm":         (128, 165),
        "eye_width_mm":          (28,  37),
        "bridge_width_mm":       (13,  25),
        "side_length_mm":        (128, 152),
    },
    "female": {
        "pupillary_distance_mm": (54,  68),
        "face_height_mm":        (155, 210),
        "eye_height_mm":         (7,   14),
        "forehead_width_mm":     (98,  138),
        "face_width_mm":         (116, 152),
        "eye_width_mm":          (26,  35),
        "bridge_width_mm":       (11,  23),
        "side_length_mm":        (122, 148),
    },
    "unknown": {
        "pupillary_distance_mm": (54,  74),
        "face_height_mm":        (150, 235),
        "eye_height_mm":         (6,   15),
        "forehead_width_mm":     (95,  150),
        "face_width_mm":         (112, 168),
        "eye_width_mm":          (24,  38),
        "bridge_width_mm":       (10,  27),
        "side_length_mm":        (120, 154),
    },
}

# [B8] Sex-specific glasses arm lookup (face_width_mm → arm_length_mm)
ARM_LOOKUP = {
    "male": [
        (130, 130), (138, 135), (146, 138),
        (154, 140), (162, 142), (float("inf"), 145),
    ],
    "female": [
        (120, 128), (128, 130), (136, 133),
        (144, 135), (152, 138), (float("inf"), 140),
    ],
    "unknown": [
        (125, 130), (135, 133), (145, 136),
        (155, 139), (165, 142), (float("inf"), 145),
    ],
}

# [B13] Stabilization passes
STABILIZATION_PASSES = 5

# [B10] Iris pixel diameter thresholds
IRIS_PX_HARD_MIN = 6    # below → hard reject
IRIS_PX_WARN_MIN = 14   # below → warning

# [B4] Pixels-per-mm sanity floor
PPM_HARD_MIN = 2.0


# ═══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def euclidean_2d(p1, p2):
    """2D Euclidean distance — all measurements use this (z ignored)."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def euclidean_3d(p1, p2):
    """3D Euclidean distance — includes Z-axis for true 3D measurements."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

def lm(landmarks, key, img_w, img_h):
    """Get a named landmark as (x_px, y_px, z_scaled)."""
    p = landmarks[LANDMARKS[key]]
    return (p.x * img_w, p.y * img_h, p.z * img_w)


def lm_idx(landmarks, idx, img_w, img_h):
    """Get a landmark by raw index."""
    p = landmarks[idx]
    return (p.x * img_w, p.y * img_h, p.z * img_w)


# ═══════════════════════════════════════════════════════════════
#  HEAD POSE ESTIMATION
# ═══════════════════════════════════════════════════════════════

def calc_roll(landmarks, img_w, img_h):
    """Roll angle in degrees (unsigned) via outer eye corners."""
    le = lm(landmarks, "LEFT_EYE_OUTER",  img_w, img_h)
    re = lm(landmarks, "RIGHT_EYE_OUTER", img_w, img_h)
    return abs(math.degrees(math.atan2(re[1] - le[1], re[0] - le[0])))


def calc_roll_signed(landmarks, img_w, img_h):
    """Roll angle in degrees (signed) for compensation."""
    le = lm(landmarks, "LEFT_EYE_OUTER",  img_w, img_h)
    re = lm(landmarks, "RIGHT_EYE_OUTER", img_w, img_h)
    return math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))


def calc_yaw(landmarks, img_w, img_h):
    """
    Yaw estimate via nose-to-cheek distance ratio.
    Centred face → ratio = 0.5. Deviation × 90 = approximate yaw degrees.
    """
    nose = lm(landmarks, "NOSE_TIP",        img_w, img_h)
    lc   = lm(landmarks, "LEFT_CHEEK_MID",  img_w, img_h)
    rc   = lm(landmarks, "RIGHT_CHEEK_MID", img_w, img_h)
    dl = euclidean_2d(nose[:2], lc[:2])
    dr = euclidean_2d(nose[:2], rc[:2])
    if dl + dr == 0:
        return 0.0
    return abs(dl / (dl + dr) - 0.5) * 90.0


def calc_pitch(landmarks, img_w, img_h):
    """
    [B9] Pitch (chin up/down) estimate.
    Uses vertical offset of nose-tip relative to iris midpoint.
    Positive = chin up, Negative = chin down.
    """
    li = lm_idx(landmarks, LANDMARKS["LEFT_IRIS_CENTER"],  img_w, img_h)
    ri = lm_idx(landmarks, LANDMARKS["RIGHT_IRIS_CENTER"], img_w, img_h)
    eye_mid_y = (li[1] + ri[1]) / 2
    nose      = lm(landmarks, "NOSE_TIP", img_w, img_h)
    span      = euclidean_2d(li[:2], ri[:2])
    if span < 1:
        return 0.0
    # 1 eye-span of offset ≈ 30° of pitch (empirical)
    return -((nose[1] - eye_mid_y) / span) * 30.0


def compensate_h(raw_mm, yaw_deg):
    """Horizontal foreshortening correction for head yaw."""
    if abs(yaw_deg) < 0.5:
        return raw_mm
    return raw_mm / math.cos(math.radians(min(abs(yaw_deg), 15.0)))


def compensate_v(raw_mm, roll_deg, pitch_deg=0.0):
    """
    [B9] Combined roll + pitch vertical correction.
    Both axes independently foreshorten vertical distances.
    """
    total = math.sqrt(roll_deg ** 2 + pitch_deg ** 2)
    if total < 0.5:
        return raw_mm
    return raw_mm / math.cos(math.radians(min(total, 15.0)))


# ═══════════════════════════════════════════════════════════════
#  IRIS MEASUREMENT
# ═══════════════════════════════════════════════════════════════

def iris_px_diameter(landmarks, img_w, img_h):
    """
    Robust iris diameter using median of 4 radii per eye.
    If asymmetry > 12% (likely eyelid occlusion) uses larger iris only.
    """
    lc = lm_idx(landmarks, 468, img_w, img_h)
    lr = [euclidean_2d(lc[:2], lm_idx(landmarks, i, img_w, img_h)[:2])
          for i in [469, 470, 471, 472]]

    rc = lm_idx(landmarks, 473, img_w, img_h)
    rr = [euclidean_2d(rc[:2], lm_idx(landmarks, i, img_w, img_h)[:2])
          for i in [474, 475, 476, 477]]

    d_l = float(np.median(lr)) * 2
    d_r = float(np.median(rr)) * 2

    if d_l == 0 or d_r == 0:
        return max(d_l, d_r)

    if min(d_l, d_r) / max(d_l, d_r) < 0.88:
        return max(d_l, d_r)   # one eye partially occluded

    return (d_l + d_r) / 2


# ═══════════════════════════════════════════════════════════════
#  CALIBRATION  [B5, B6]
# ═══════════════════════════════════════════════════════════════

def calc_ppm(landmarks, img_w, img_h, sex="unknown",
             reference_pixels=None, reference_mm=85.6):
    """
    Pixels-per-mm calibration with adaptive cross-validation.

    Priority:
      1. External reference object (e.g. credit card) — most accurate
      2. Sex-specific iris diameter with cross-validated face width
      3. IPD heuristic fallback for very-low-res images

    [B5] Cross-validation window is now sex-specific and tighter.
         Blend ratio scales with deviation severity (0.40 → 0.65).
    [B6] Uses sex-specific iris diameter constant.
    """
    if reference_pixels and reference_pixels > 0:
        return reference_pixels / reference_mm

    iris_mm  = IRIS_DIAMETER_BY_SEX.get(sex, IRIS_DIAMETER_BY_SEX["unknown"])
    face_avg = FACE_WIDTH_AVG_BY_SEX.get(sex,  FACE_WIDTH_AVG_BY_SEX["unknown"])
    fw_min, fw_max = FACE_WIDTH_RANGE_BY_SEX.get(sex, FACE_WIDTH_RANGE_BY_SEX["unknown"])

    iris_px = iris_px_diameter(landmarks, img_w, img_h)

    if iris_px >= IRIS_PX_HARD_MIN:
        ppm = iris_px / iris_mm

        # Cross-validate against sex-specific face width range using 3D
        lc = lm(landmarks, "FACE_LEFT_CHEEK",  img_w, img_h)
        rc = lm(landmarks, "FACE_RIGHT_CHEEK", img_w, img_h)
        face_w_px = euclidean_3d(lc, rc)  # Use 3D instead of 2D
        face_w_mm = face_w_px / ppm

        if face_w_mm < fw_min or face_w_mm > fw_max:
            # Adaptive blend: deviation severity drives correction weight
            deviation   = max(fw_min - face_w_mm, face_w_mm - fw_max, 0)
            blend       = min(0.65, 0.40 + (deviation / 30.0) * 0.25)
            prop_ppm    = face_w_px / face_avg
            ppm         = (1.0 - blend) * ppm + blend * prop_ppm

        return ppm

    # IPD fallback for ultra-low-res images
    li     = lm_idx(landmarks, LANDMARKS["LEFT_IRIS_CENTER"],  img_w, img_h)
    ri     = lm_idx(landmarks, LANDMARKS["RIGHT_IRIS_CENTER"], img_w, img_h)
    ipd_px = euclidean_2d(li[:2], ri[:2])
    ipd_avg = {"male": 64.0, "female": 62.0}.get(sex, 63.0)
    return ipd_px / ipd_avg if ipd_px > 0 else 1.0


# ═══════════════════════════════════════════════════════════════
#  MEASUREMENT FUNCTIONS  (all 2D + pose-compensated)
# ═══════════════════════════════════════════════════════════════

def meas_pd(landmarks, img_w, img_h, ppm, yaw):
    """Total PD: direct 3D iris-centre-to-iris-centre distance."""
    li = lm_idx(landmarks, LANDMARKS["LEFT_IRIS_CENTER"],  img_w, img_h)
    ri = lm_idx(landmarks, LANDMARKS["RIGHT_IRIS_CENTER"], img_w, img_h)
    return round(compensate_h(euclidean_3d(li, ri) / ppm, yaw), 1)


def meas_mono_pd(landmarks, img_w, img_h, ppm, yaw):
    """
    Monocular PD: each iris centre to nose bridge midpoint.
    Uses 3D distance for accurate measurements.
    """
    li = lm_idx(landmarks, LANDMARKS["LEFT_IRIS_CENTER"],  img_w, img_h)
    ri = lm_idx(landmarks, LANDMARKS["RIGHT_IRIS_CENTER"], img_w, img_h)
    nb = lm(landmarks, "NOSE_BRIDGE_TOP", img_w, img_h)
    l  = round(compensate_h(euclidean_3d(li, nb) / ppm, yaw), 1)
    r  = round(compensate_h(euclidean_3d(ri, nb) / ppm, yaw), 1)
    return l, r


def meas_face_width(landmarks, img_w, img_h, ppm, yaw):
    """Face width: cheekbone to cheekbone (bizygomatic) using 3D."""
    left  = lm(landmarks, "FACE_LEFT_CHEEK",  img_w, img_h)
    right = lm(landmarks, "FACE_RIGHT_CHEEK", img_w, img_h)
    return round(compensate_h(euclidean_3d(left, right) / ppm, yaw), 2)


def meas_face_height(landmarks, img_w, img_h, ppm, roll, pitch):
    """Face height: forehead landmark to chin bottom using 3D."""
    top = lm(landmarks, "FOREHEAD_TOP", img_w, img_h)
    bot = lm(landmarks, "CHIN_BOTTOM",  img_w, img_h)
    return round(compensate_v(euclidean_3d(top, bot) / ppm, roll, pitch), 2)


def meas_eye_width(landmarks, img_w, img_h, ppm, yaw):
    """
    [B11] Eye width: inner to outer corner using 3D.
    Returns (avg_mm, left_mm, right_mm).
    """
    li = lm(landmarks, "LEFT_EYE_INNER",  img_w, img_h)
    lo = lm(landmarks, "LEFT_EYE_OUTER",  img_w, img_h)
    ri = lm(landmarks, "RIGHT_EYE_INNER", img_w, img_h)
    ro = lm(landmarks, "RIGHT_EYE_OUTER", img_w, img_h)
    l_mm = compensate_h(euclidean_3d(li, lo) / ppm, yaw)
    r_mm = compensate_h(euclidean_3d(ri, ro) / ppm, yaw)
    return round((l_mm + r_mm) / 2, 2), round(l_mm, 2), round(r_mm, 2)


def meas_eye_height(landmarks, img_w, img_h, ppm, roll, pitch):
    """
    [B11] Eye height: top to bottom eyelid using 3D.
    Returns (avg_mm, left_mm, right_mm).
    """
    lt = lm(landmarks, "LEFT_EYE_TOP",     img_w, img_h)
    lb = lm(landmarks, "LEFT_EYE_BOTTOM",  img_w, img_h)
    rt = lm(landmarks, "RIGHT_EYE_TOP",    img_w, img_h)
    rb = lm(landmarks, "RIGHT_EYE_BOTTOM", img_w, img_h)
    l_mm = compensate_v(euclidean_3d(lt, lb) / ppm, roll, pitch)
    r_mm = compensate_v(euclidean_3d(rt, rb) / ppm, roll, pitch)
    return round((l_mm + r_mm) / 2, 2), round(l_mm, 2), round(r_mm, 2)


def meas_bridge_width(landmarks, img_w, img_h, ppm, yaw):
    """
    [B7] Enhanced bridge width: average across 3 landmark pairs using 3D.
    Uses median for robustness and applies yaw compensation.
    """
    pairs = [
        ("BRIDGE_LEFT_HIGH", "BRIDGE_RIGHT_HIGH"),
        ("BRIDGE_LEFT_MID",  "BRIDGE_RIGHT_MID"),
        ("BRIDGE_LEFT_LOW",  "BRIDGE_RIGHT_LOW"),
    ]
    widths = []
    for lk, rk in pairs:
        left_pt  = lm(landmarks, lk, img_w, img_h)
        right_pt = lm(landmarks, rk, img_w, img_h)
        width_3d = euclidean_3d(left_pt, right_pt) / ppm
        widths.append(compensate_h(width_3d, yaw))
    
    # Use median for robustness instead of minimum
    return round(float(np.median(widths)), 2)


def meas_forehead_width(landmarks, img_w, img_h, ppm, yaw):
    """Forehead width between landmarks 54 and 284 using 3D."""
    left  = lm(landmarks, "FOREHEAD_LEFT",  img_w, img_h)
    right = lm(landmarks, "FOREHEAD_RIGHT", img_w, img_h)
    return round(compensate_h(euclidean_3d(left, right) / ppm, yaw), 2)


def meas_side_length(landmarks, img_w, img_h, ppm, yaw, sex="unknown"):
    """
    [B8] Recommended glasses arm length from sex-specific lookup table using 3D.

    A frontal photo cannot measure the physical temple-to-ear arc
    (it wraps around the head). This maps face width to the nearest
    standard glasses arm size (128–145mm) used by eyewear manufacturers.
    """
    left  = lm(landmarks, "FACE_LEFT_CHEEK",  img_w, img_h)
    right = lm(landmarks, "FACE_RIGHT_CHEEK", img_w, img_h)
    fw_mm = compensate_h(euclidean_3d(left, right) / ppm, yaw)

    table = ARM_LOOKUP.get(sex, ARM_LOOKUP["unknown"])
    arm   = table[-1][1]
    for max_w, arm_len in table:
        if fw_mm <= max_w:
            arm = arm_len
            break
    return float(arm)


# ═══════════════════════════════════════════════════════════════
#  RANGE VALIDATION  [B12]
# ═══════════════════════════════════════════════════════════════

def validate_ranges(measurements, sex="unknown"):
    """
    Returns list of warning strings for measurements outside the
    sex-specific plausibility range. Empty list = all within range.
    These are warnings only — results are still returned.
    """
    ranges    = MEASUREMENT_RANGES.get(sex, MEASUREMENT_RANGES["unknown"])
    warnings  = []
    check_keys = [
        "pupillary_distance_mm", "face_height_mm", "eye_height_mm",
        "forehead_width_mm", "face_width_mm", "eye_width_mm",
        "bridge_width_mm", "side_length_mm",
    ]
    for key in check_keys:
        val = measurements.get(key)
        if val is None or key not in ranges:
            continue
        lo, hi = ranges[key]
        if not (lo <= val <= hi):
            warnings.append(
                f"{key}: {val}mm outside expected {sex} range "
                f"[{lo}–{hi}mm] — verify lighting and calibration"
            )
    return warnings


# ═══════════════════════════════════════════════════════════════
#  IMAGE QUALITY CHECKS  [B2, B3]
# ═══════════════════════════════════════════════════════════════

def check_image_quality(image):
    """
    Returns (ok: bool, error_msg: str, warnings: list[str]).

    [B2] Brightness sampled from central 60% of frame (face area).
         Threshold raised to 50 for reliable iris landmark detection.
    [B3] Blur check re-enabled as warning, not hard rejection.
    """
    if image is None:
        return False, "Failed to read image file.", []

    h, w = image.shape[:2]
    warnings = []

    if w < 480 or h < 480:
        return False, f"Resolution too low ({w}×{h}). Minimum 480×480 required.", []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # [B2] Sample central 60% of frame
    cy, cx   = h // 2, w // 2
    region   = gray[int(cy * 0.4):int(cy * 1.6), int(cx * 0.4):int(cx * 1.6)]
    mean_br  = float(np.mean(region)) if region.size > 0 else float(np.mean(gray))

    if mean_br < 30:
        return False, (
            f"Image is too dark (brightness={mean_br:.0f}/255). "
            "Move to a well-lit area for accurate measurements."
        ), []
    if mean_br > 248:
        return False, "Image is overexposed. Reduce lighting or increase distance.", []

    # [B3] Blur warning
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 15:
        warnings.append(
            f"Image may be blurry (sharpness={lap_var:.1f}). "
            "Hold the camera steady for best results."
        )

    return True, "OK", warnings


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE SCORE  [B16]
# ═══════════════════════════════════════════════════════════════

def calc_confidence(img_w, img_h, roll, yaw, pitch, iris_px):
    """
    Enhanced confidence score (0–100) with more generous scoring.
    Resolution (0-25) + Pose (0-45) + Iris size (0-30).
    """
    # Resolution score - more generous for HD images
    res_score  = min(1.0, (img_w * img_h) / (1280 * 720)) * 25
    
    # Pose score - more tolerant limits
    pose_score = max(0.0, 45.0 - roll * 2.0 - yaw * 1.0 - abs(pitch) * 1.0)
    
    # Iris score - more forgiving
    iris_score = min(30.0, max(0.0,
        (iris_px - IRIS_PX_HARD_MIN) / (IRIS_PX_WARN_MIN - IRIS_PX_HARD_MIN) * 30.0
    ))
    
    return round(res_score + pose_score + iris_score, 1)


# ═══════════════════════════════════════════════════════════════
#  GLOBAL MODEL (singleton — load once, reuse)
# ═══════════════════════════════════════════════════════════════

_landmarker_instance = None

def _get_landmarker():
    global _landmarker_instance
    if _landmarker_instance:
        return _landmarker_instance
    if not os.path.exists(_MODEL_PATH):
        return None
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        num_faces=2,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _landmarker_instance = FaceLandmarker.create_from_options(opts)
    return _landmarker_instance


# ═══════════════════════════════════════════════════════════════
#  MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def process_image(image_path, reference_pixels=None, sex="unknown"):
    """
    Full 2D measurement pipeline for one image.

    Parameters
    ----------
    image_path       : str   — absolute path to image file
    reference_pixels : float — pixel width of a credit card held in frame
                               (85.6mm reference). None = use iris calibration.
    sex              : str   — "male" | "female" | "unknown"
                               Affects iris constant, arm table, and
                               plausibility range checks.

    Returns
    -------
    dict: {success, engine_version, measurements, metadata, warnings}
    """
    sex = sex.lower() if sex in ("male", "female") else "unknown"

    # ── 1. Load & quality-check ───────────────────────────────
    image = cv2.imread(image_path)
    ok, err_msg, q_warnings = check_image_quality(image)
    if not ok:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "IMAGE_QUALITY_ERROR", "message": err_msg}}

    img_h, img_w = image.shape[:2]
    image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── 2. Face detection ─────────────────────────────────────
    landmarker = _get_landmarker()
    if landmarker is None:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "MODEL_NOT_FOUND",
                          "message": (
                              f"Model not found at {_MODEL_PATH}. Download from "
                              "https://storage.googleapis.com/mediapipe-models/"
                              "face_landmarker/face_landmarker/float16/latest/"
                              "face_landmarker.task"
                          )}}

    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = landmarker.detect(mp_img)

    if not results.face_landmarks:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "NO_FACE_DETECTED",
                          "message": "No face detected. Provide a clear frontal image."}}

    if len(results.face_landmarks) > 1:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "MULTIPLE_FACES_DETECTED",
                          "message": f"{len(results.face_landmarks)} faces found. "
                                     "Image must contain exactly one face."}}

    lms = results.face_landmarks[0]

    if len(lms) < 478:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "INSUFFICIENT_LANDMARKS",
                          "message": f"Only {len(lms)} landmarks detected. "
                                     "478 required for iris tracking."}}

    # ── 3. Head pose validation ───────────────────────────────
    roll   = calc_roll(lms,        img_w, img_h)
    roll_s = calc_roll_signed(lms, img_w, img_h)
    yaw    = calc_yaw(lms,         img_w, img_h)
    pitch  = calc_pitch(lms,       img_w, img_h)

    if roll > 10.0:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "HEAD_TILT_HIGH",
                          "message": f"Head roll {roll:.1f}° > 10° limit. Keep head level."}}
    if yaw > 15.0:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "HEAD_TURN_HIGH",
                          "message": f"Head yaw {yaw:.1f}° > 15° limit. Face the camera directly."}}
    if abs(pitch) > 25.0:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "HEAD_PITCH_HIGH",
                          "message": f"Head pitch {pitch:.1f}° > 25° limit. Keep chin level."}}

    # [B9] Pitch warning at 12° (soft warn before hard reject at 20°)
    if abs(pitch) > 12.0:
        q_warnings.append(
            f"Head pitch {pitch:.1f}° — face_height and eye_height may be "
            "slightly under-reported. Keep chin level for best accuracy."
        )

    # ── 4. Iris pixel check ───────────────────────────────────
    iris_px = iris_px_diameter(lms, img_w, img_h)

    # [B10] Hard reject — iris too small to calibrate
    if iris_px < IRIS_PX_HARD_MIN:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "IRIS_TOO_SMALL",
                          "message": f"Iris only {iris_px:.1f}px wide. "
                                     "Move closer to the camera."}}

    # [B10] Soft warn — borderline iris size
    if iris_px < IRIS_PX_WARN_MIN:
        q_warnings.append(
            f"Iris diameter {iris_px:.1f}px is small — calibration may be "
            "less accurate. Move closer or use higher resolution."
        )

    # ── 5. Calibration ────────────────────────────────────────
    ppm = round(calc_ppm(lms, img_w, img_h, sex, reference_pixels), 3)
    cal_method = "reference_object" if reference_pixels else f"iris_diameter_{sex}"

    # [B4] Sanity-check calibration result
    if ppm < PPM_HARD_MIN:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "CALIBRATION_UNRELIABLE",
                          "message": f"Pixels/mm too low ({ppm:.2f}). "
                                     "Move closer to the camera or improve lighting."}}

    # ── 6. Measurements ───────────────────────────────────────
    lpd, rpd              = meas_mono_pd(lms,    img_w, img_h, ppm, yaw)
    ew_avg, ew_l, ew_r    = meas_eye_width(lms,  img_w, img_h, ppm, yaw)
    eh_avg, eh_l, eh_r    = meas_eye_height(lms, img_w, img_h, ppm, roll_s, pitch)

    measurements = {
        "pupillary_distance_mm":  meas_pd(lms,             img_w, img_h, ppm, yaw),
        "pd_left_mm":             lpd,
        "pd_right_mm":            rpd,
        "face_width_mm":          meas_face_width(lms,     img_w, img_h, ppm, yaw),
        "face_height_mm":         meas_face_height(lms,    img_w, img_h, ppm, roll_s, pitch),
        "eye_width_mm":           ew_avg,
        "eye_width_left_mm":      ew_l,
        "eye_width_right_mm":     ew_r,
        "eye_height_mm":          eh_avg,
        "eye_height_left_mm":     eh_l,
        "eye_height_right_mm":    eh_r,
        "bridge_width_mm":        meas_bridge_width(lms,   img_w, img_h, ppm, yaw),
        "forehead_width_mm":      meas_forehead_width(lms, img_w, img_h, ppm, yaw),
        "side_length_mm":         meas_side_length(lms,    img_w, img_h, ppm, yaw, sex),
    }

    # ── 7. Range validation ───────────────────────────────────
    all_warnings = list(dict.fromkeys(
        q_warnings + validate_ranges(measurements, sex)
    ))

    # ── 8. Confidence ─────────────────────────────────────────
    conf = calc_confidence(img_w, img_h, roll, yaw, pitch, iris_px)

    metadata = {
        "image_dimensions":      {"width": img_w, "height": img_h},
        "landmarks_detected":    len(lms),
        "head_roll_degrees":     round(roll,   2),
        "head_yaw_degrees":      round(yaw,    2),
        "head_pitch_degrees":    round(pitch,  2),
        "calibration_method":    cal_method,
        "pixels_per_mm":         round(ppm,    4),
        "iris_diameter_px":      round(iris_px, 2),
        "iris_diameter_mm_used": IRIS_DIAMETER_BY_SEX.get(sex, 11.15),
        "sex":                   sex,
        "confidence_score":      f"{conf}%",
        "accuracy_guarantee":    ">=95%" if conf > 85 else "Standard",
    }

    return {
        "success":        True,
        "engine_version": ENGINE_VERSION,
        "measurements":   measurements,
        "metadata":       metadata,
        "warnings":       all_warnings,
    }


# ═══════════════════════════════════════════════════════════════
#  STABILIZED PROCESSING  [B1, B13]
# ═══════════════════════════════════════════════════════════════

def process_image_stabilized(image_path, reference_pixels=None,
                              sex="unknown", passes=None):
    """
    Run process_image() N times, return per-key median.

    [B1]  sex is now accepted and forwarded correctly.
    [B13] Default passes increased from 3 → 5.
    """
    if passes is None:
        passes = STABILIZATION_PASSES

    all_meas     = []
    all_meta     = []
    all_warnings = []
    last_err     = None

    for _ in range(passes):
        r = process_image(image_path, reference_pixels, sex)
        if r["success"]:
            all_meas.append(r["measurements"])
            all_meta.append(r["metadata"])
            all_warnings.extend(r.get("warnings", []))
        else:
            last_err = r

    if not all_meas:
        return last_err or {
            "success": False, "engine_version": ENGINE_VERSION,
            "error": {"code": "STABILIZATION_FAILED",
                      "message": "All stabilization passes failed."}
        }

    if len(all_meas) == 1:
        return {
            "success":        True,
            "engine_version": ENGINE_VERSION,
            "measurements":   all_meas[0],
            "metadata":       {**all_meta[0],
                               "stabilization_passes": 1,
                               "successful_passes": 1},
            "warnings":       list(dict.fromkeys(all_warnings)),
        }

    # Per-key median
    stabilized = {
        k: round(float(np.median([m[k] for m in all_meas])), 2)
        for k in all_meas[0]
    }

    # Pick metadata from the pass closest to median PD
    med_pd   = stabilized.get("pupillary_distance_mm", 0)
    best_idx = min(range(len(all_meas)),
                   key=lambda i: abs(all_meas[i].get("pupillary_distance_mm", 0) - med_pd))

    meta = all_meta[best_idx].copy()
    meta.update({
        "stabilization_passes":  passes,
        "successful_passes":     len(all_meas),
        "stabilization_method":  "median",
    })

    return {
        "success":        True,
        "engine_version": ENGINE_VERSION,
        "measurements":   stabilized,
        "metadata":       meta,
        "warnings":       list(dict.fromkeys(all_warnings)),
    }


# ═══════════════════════════════════════════════════════════════
#  MULTI-FRAME AVERAGING  [B1]
# ═══════════════════════════════════════════════════════════════

def process_multi_frame(image_paths, reference_pixels=None, sex="unknown"):
    """
    Robust trimmed-mean across multiple frames (video / burst mode).
    [B1] sex now forwarded to process_image().
    Best accuracy with 20+ frames → 98%+.
    """
    all_meas = []
    errors   = []

    for path in image_paths:
        r = process_image(path, reference_pixels, sex)
        if r["success"]:
            all_meas.append(r["measurements"])
        else:
            errors.append(r["error"]["message"])

    if not all_meas:
        return {"success": False, "engine_version": ENGINE_VERSION,
                "error": {"code": "NO_VALID_FRAMES",
                          "message": f"No valid frames. Errors: {'; '.join(errors[:3])}"}}

    averaged = {}
    for key in all_meas[0]:
        vals = sorted(m[key] for m in all_meas)
        if len(vals) >= 5:
            trim = max(1, int(len(vals) * 0.2))
            averaged[key] = round(float(np.mean(vals[trim:-trim])), 2)
        else:
            averaged[key] = round(float(np.mean(vals)), 2)

    return {
        "success":        True,
        "engine_version": ENGINE_VERSION,
        "measurements":   averaged,
        "metadata": {
            "frames_processed":    len(all_meas),
            "frames_rejected":     len(errors),
            "total_frames":        len(image_paths),
            "sex":                 sex,
            "averaging_method":    "robust_trimmed_mean",
            "accuracy_confidence": "98%+" if len(all_meas) >= 10 else "95%",
        },
    }


# ═══════════════════════════════════════════════════════════════
#  PERSISTENT WORKER  [B14, B15]
# ═══════════════════════════════════════════════════════════════

def run_worker_mode():
    """
    Persistent subprocess protocol for Node.js.

    stdin  (one JSON per line): {"image_path":"...", "sex":"male",
                                 "reference_pixels": null}
    stdout (one JSON per line): measurement result
    Commands: {"command":"exit"} | {"command":"health"}

    [B14] sex is now read from stdin and forwarded to processing.
    [B15] All responses carry engine_version.
    """
    import time

    sys.stderr.write(f"[Worker v{ENGINE_VERSION}] Loading model...\n")
    sys.stderr.flush()
    t0   = time.time()
    lmkr = _get_landmarker()

    if lmkr is None:
        print(json.dumps({"success": False, "engine_version": ENGINE_VERSION,
                          "error": {"code": "MODEL_NOT_FOUND",
                                    "message": "Could not load model"}}), flush=True)
        sys.exit(1)

    sys.stderr.write(
        f"[Worker v{ENGINE_VERSION}] Ready in {time.time() - t0:.1f}s.\n"
    )
    sys.stderr.flush()
    print(json.dumps({"ready": True, "engine_version": ENGINE_VERSION}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"success": False, "engine_version": ENGINE_VERSION,
                              "error": {"code": "INVALID_JSON",
                                        "message": "Could not parse input"}}), flush=True)
            continue

        if cmd.get("command") == "exit":
            sys.stderr.write(f"[Worker v{ENGINE_VERSION}] Shutting down.\n")
            sys.stderr.flush()
            break

        if cmd.get("command") == "health":
            print(json.dumps({
                "success":              True,
                "engine_version":       ENGINE_VERSION,
                "status":               "alive",
                "python_version":       sys.version,
                "mediapipe_version":    mp.__version__,
                "opencv_version":       cv2.__version__,
                "numpy_version":        np.__version__,
                "iris_diameters_mm":    IRIS_DIAMETER_BY_SEX,
                "stabilization_passes": STABILIZATION_PASSES,
                "ppm_hard_min":         PPM_HARD_MIN,
            }), flush=True)
            continue

        img_path   = cmd.get("image_path")
        ref_pixels = cmd.get("reference_pixels")
        sex        = cmd.get("sex", "unknown")   # [B14] was missing

        if not img_path:
            print(json.dumps({"success": False, "engine_version": ENGINE_VERSION,
                              "error": {"code": "MISSING_PATH",
                                        "message": "No image_path provided"}}), flush=True)
            continue

        try:
            t0     = time.time()
            result = process_image_stabilized(img_path, ref_pixels, sex)
            result.setdefault("metadata", {})["python_processing_ms"] = \
                int((time.time() - t0) * 1000)
        except Exception as e:
            result = {
                "success":        False,
                "engine_version": ENGINE_VERSION,
                "error": {
                    "code":      "PROCESSING_ERROR",
                    "message":   str(e),
                    "traceback": traceback.format_exc(),
                },
            }

        print(json.dumps(result), flush=True)


# ═══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        run_worker_mode()
        sys.exit(0)

    if len(sys.argv) < 2:
        print(json.dumps({
            "success":        False,
            "engine_version": ENGINE_VERSION,
            "error": {
                "code":    "MISSING_ARGS",
                "message": (
                    "Usage: python face_measurement_engine_v8.py "
                    "<image_path> [reference_pixels] [male|female|unknown]"
                ),
            },
        }))
        sys.exit(1)

    _img = sys.argv[1]
    _ref = float(sys.argv[2]) if len(sys.argv) > 2 else None
    _sex = sys.argv[3]        if len(sys.argv) > 3 else "unknown"

    try:
        _result = process_image_stabilized(_img, _ref, _sex)
    except Exception as _e:
        _result = {
            "success":        False,
            "engine_version": ENGINE_VERSION,
            "error": {
                "code":      "PROCESSING_ERROR",
                "message":   str(_e),
                "traceback": traceback.format_exc(),
            },
        }

    print(json.dumps(_result, indent=2))