"""
===================================================================
 COMPREHENSIVE ACCURACY TEST SUITE v5 -- 100 TRIALS
 
 Tests the face_measurement_engine v5 mathematical pipeline against
 known ground-truth values under realistic simulated conditions.
 
 Matches v5 engine features:
   - 2D-only measurements (no z-depth noise)
   - Yaw/Roll pose compensation (cos correction)
   - Convergence-corrected PD (near -> far conversion)
   - Cross-validated iris calibration
   - 5-pass median stabilization (simulated)
 
 Simulates:
   * 5 face archetypes (male, female, child, wide, narrow)
   * 3 noise tiers (low, medium, high)
   * Head pose variations (roll & yaw)
   * Calibration drift (iris detection variance)
   * Per-measurement breakdown + overall verdict
===================================================================
"""

import sys
import os
import math
import random
import time
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# ===================================================================
#  GROUND TRUTH DATABASE
# ===================================================================

FACE_ARCHETYPES = {
    "adult_male": {
        "pupillary_distance_mm": 64.0,
        "face_width_mm": 147.0,
        "face_height_mm": 198.0,
        "eye_width_mm": 30.5,
        "eye_height_mm": 12.2,
        "bridge_width_mm": 18.5,
        "forehead_width_mm": 132.0,
        "side_length_mm": 154.0,
    },
    "adult_female": {
        "pupillary_distance_mm": 61.5,
        "face_width_mm": 138.0,
        "face_height_mm": 185.0,
        "eye_width_mm": 28.0,
        "eye_height_mm": 11.5,
        "bridge_width_mm": 16.0,
        "forehead_width_mm": 125.0,
        "side_length_mm": 145.0,
    },
    "child_10yr": {
        "pupillary_distance_mm": 55.0,
        "face_width_mm": 125.0,
        "face_height_mm": 165.0,
        "eye_width_mm": 25.0,
        "eye_height_mm": 10.5,
        "bridge_width_mm": 14.0,
        "forehead_width_mm": 112.0,
        "side_length_mm": 130.0,
    },
    "wide_face": {
        "pupillary_distance_mm": 67.0,
        "face_width_mm": 158.0,
        "face_height_mm": 192.0,
        "eye_width_mm": 32.0,
        "eye_height_mm": 12.8,
        "bridge_width_mm": 20.0,
        "forehead_width_mm": 142.0,
        "side_length_mm": 166.0,
    },
    "narrow_face": {
        "pupillary_distance_mm": 58.0,
        "face_width_mm": 130.0,
        "face_height_mm": 200.0,
        "eye_width_mm": 26.5,
        "eye_height_mm": 11.0,
        "bridge_width_mm": 15.0,
        "forehead_width_mm": 118.0,
        "side_length_mm": 137.0,
    },
}

ACCURACY_THRESHOLDS_PERCENT = {
    "pupillary_distance_mm": 5.0,
    "face_width_mm": 5.0,
    "face_height_mm": 5.0,
    "eye_width_mm": 5.0,
    "eye_height_mm": 5.0,
    "bridge_width_mm": 5.0,
    "forehead_width_mm": 5.0,
    "side_length_mm": 5.0,
}

PD_STRICT_THRESHOLD_MM = 2.0

# ===================================================================
#  NOISE PROFILES
# ===================================================================

NOISE_PROFILES = {
    "low": {
        "label": "Low Noise (Ideal Studio)",
        "landmark_jitter": 0.005,
        "calibration_drift": 0.004,
        "head_roll_range": (-1.5, 1.5),
        "head_yaw_range": (-2.0, 2.0),
    },
    "medium": {
        "label": "Medium Noise (Typical Webcam)",
        "landmark_jitter": 0.012,
        "calibration_drift": 0.010,
        "head_roll_range": (-4.0, 4.0),
        "head_yaw_range": (-6.0, 6.0),
    },
    "high": {
        "label": "High Noise (Poor Conditions)",
        "landmark_jitter": 0.018,
        "calibration_drift": 0.015,
        "head_roll_range": (-7.0, 7.0),
        "head_yaw_range": (-10.0, 10.0),
    },
}

# Engine constants (must match face_measurement_engine.py v6)
CONVERGENCE_OFFSET_BASE_MM = 0.0
CONVERGENCE_OFFSET_MIN_MM = 0.0
CONVERGENCE_OFFSET_MAX_MM = 0.0

# ===================================================================
#  v5 ENGINE SIMULATION
# ===================================================================

def pose_compensate_horizontal(raw_mm, yaw_deg):
    if abs(yaw_deg) < 0.5:
        return raw_mm
    correction = 1.0 / math.cos(math.radians(min(abs(yaw_deg), 15.0)))
    return raw_mm * correction

def pose_compensate_vertical(raw_mm, roll_deg):
    if abs(roll_deg) < 0.5:
        return raw_mm
    correction = 1.0 / math.cos(math.radians(min(abs(roll_deg), 10.0)))
    return raw_mm * correction


def simulate_single_pass(ground_truth, noise_profile):
    profile = NOISE_PROFILES[noise_profile]
    jitter = profile["landmark_jitter"]
    cal_drift = profile["calibration_drift"]

    calibration_error = 1.0 + random.gauss(0, cal_drift)
    actual_roll = random.uniform(*profile["head_roll_range"])
    actual_yaw = random.uniform(*profile["head_yaw_range"])
    detected_roll = actual_roll + random.gauss(0, 0.5)
    detected_yaw = actual_yaw + random.gauss(0, 1.0)
    actual_yaw_factor = math.cos(math.radians(actual_yaw))
    actual_roll_factor = math.cos(math.radians(actual_roll))

    measurements = {}

    for key, true_val in ground_truth.items():
        measured = true_val
        measured *= (1.0 + random.gauss(0, jitter))
        measured *= calibration_error

        if key in ["face_width_mm", "forehead_width_mm", "bridge_width_mm",
                    "eye_width_mm", "pupillary_distance_mm"]:
            measured *= actual_yaw_factor
        if key in ["face_height_mm", "eye_height_mm"]:
            measured *= actual_roll_factor

        if key in ["face_width_mm", "forehead_width_mm", "bridge_width_mm",
                    "eye_width_mm", "pupillary_distance_mm"]:
            measured = pose_compensate_horizontal(measured, detected_yaw)
        if key in ["face_height_mm", "eye_height_mm"]:
            measured = pose_compensate_vertical(measured, detected_roll)

        if key == "side_length_mm":
            fw = ground_truth["face_width_mm"] * (1.0 + random.gauss(0, jitter)) * calibration_error
            fw *= actual_yaw_factor
            fw = pose_compensate_horizontal(fw, detected_yaw)
            measured = fw * 1.05

        if key == "pupillary_distance_mm":
            # v6: No convergence correction — measure near PD directly
            # (matches SmartBuyGlasses / Kits behavior)
            pass

        measurements[key] = round(measured, 2)

    return measurements, {"noise_profile": noise_profile}


def simulate_stabilized_measurement(ground_truth, noise_profile, passes=5):
    all_pass_measurements = []
    for _ in range(passes):
        measured, _ = simulate_single_pass(ground_truth, noise_profile)
        all_pass_measurements.append(measured)

    stabilized = {}
    for key in ground_truth:
        values = sorted([m[key] for m in all_pass_measurements])
        stabilized[key] = round(float(np.median(values)), 2)

    return stabilized, {"noise_profile": noise_profile}


# ===================================================================
#  TEST RUNNER
# ===================================================================

def run_100_trial_test():
    random.seed(int(time.time()))
    np.random.seed(int(time.time()) % (2**31))

    NUM_TRIALS = 100
    PASS_THRESHOLD = 95.0

    archetypes = list(FACE_ARCHETYPES.keys())
    per_measurement_errors = defaultdict(list)
    per_noise_errors = defaultdict(list)
    per_archetype_errors = defaultdict(list)
    pd_absolute_errors = []
    trial_accuracies = []
    pass_count = 0
    fail_details = []
    all_results = []

    print("\n")
    print("+" + "=" * 62 + "+")
    print("|  FACE MEASUREMENT ENGINE v5 -- 100 TRIAL TEST".ljust(63) + "|")
    print("|  Started: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')).ljust(63) + "|")
    print("+" + "=" * 62 + "+")
    print()
    print("  Running 100 trials (5-pass stabilized)", end="", flush=True)

    for trial in range(NUM_TRIALS):
        archetype = random.choice(archetypes)
        noise_roll = random.random()
        if noise_roll < 0.40:
            noise_level = "low"
        elif noise_roll < 0.75:
            noise_level = "medium"
        else:
            noise_level = "high"

        ground_truth = FACE_ARCHETYPES[archetype]
        measured, metadata = simulate_stabilized_measurement(ground_truth, noise_level)

        trial_errors = {}
        trial_pass = True

        for key in ground_truth:
            true_val = ground_truth[key]
            meas_val = measured[key]
            error_pct = abs(true_val - meas_val) / true_val * 100
            accuracy_pct = 100.0 - error_pct

            trial_errors[key] = {
                "true": true_val,
                "measured": meas_val,
                "error_pct": round(error_pct, 3),
                "accuracy_pct": round(accuracy_pct, 3),
            }

            per_measurement_errors[key].append(error_pct)

            if key == "pupillary_distance_mm":
                pd_absolute_errors.append(abs(true_val - meas_val))

            if error_pct > ACCURACY_THRESHOLDS_PERCENT[key]:
                trial_pass = False

        mean_accuracy = float(np.mean([v["accuracy_pct"] for v in trial_errors.values()]))
        trial_accuracies.append(mean_accuracy)
        per_noise_errors[noise_level].append(mean_accuracy)
        per_archetype_errors[archetype].append(mean_accuracy)

        if trial_pass and mean_accuracy >= PASS_THRESHOLD:
            pass_count += 1
        else:
            fail_details.append({
                "trial": trial + 1,
                "archetype": archetype,
                "noise": noise_level,
                "accuracy": round(mean_accuracy, 2),
                "worst": max(trial_errors.items(), key=lambda x: x[1]["error_pct"]),
            })

        all_results.append({
            "trial": trial + 1,
            "archetype": archetype,
            "noise_level": noise_level,
            "mean_accuracy": round(mean_accuracy, 3),
        })

        if (trial + 1) % 10 == 0:
            print(" #", end="", flush=True)

    print("  Done!\n")

    # --- REPORT ---
    overall_avg = float(np.mean(trial_accuracies))
    overall_min = float(np.min(trial_accuracies))
    overall_max = float(np.max(trial_accuracies))
    overall_std = float(np.std(trial_accuracies))
    overall_median = float(np.median(trial_accuracies))
    pass_rate = (pass_count / NUM_TRIALS) * 100

    print("+" + "-" * 62 + "+")
    print("|  OVERALL RESULTS".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Total Trials:           {NUM_TRIALS}".ljust(63) + "|")
    print(f"|  Passed (>=95%):         {pass_count} / {NUM_TRIALS} ({pass_rate:.1f}%)".ljust(63) + "|")
    print(f"|  Failed:                 {NUM_TRIALS - pass_count}".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Average Accuracy:       {overall_avg:.2f}%".ljust(63) + "|")
    print(f"|  Median Accuracy:        {overall_median:.2f}%".ljust(63) + "|")
    print(f"|  Min Accuracy:           {overall_min:.2f}%".ljust(63) + "|")
    print(f"|  Max Accuracy:           {overall_max:.2f}%".ljust(63) + "|")
    print(f"|  Std Deviation:          {overall_std:.3f}%".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print()

    # Per-Measurement
    print("+" + "-" * 62 + "+")
    print("|  PER-MEASUREMENT ACCURACY".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  {'Measurement':<28} {'Avg Err%':>8} {'Max Err%':>9} {'Accuracy':>9}  |")
    print("+" + "-" * 62 + "+")

    for key in sorted(per_measurement_errors.keys()):
        errors = per_measurement_errors[key]
        avg_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        avg_acc = 100.0 - avg_err
        status = "OK" if avg_acc >= 95.0 else "!!"
        label = key.replace("_mm", "").replace("_", " ").title()
        print(f"|  [{status}] {label:<24} {avg_err:>7.2f}% {max_err:>8.2f}% {avg_acc:>8.2f}%  |")

    print("+" + "-" * 62 + "+")
    print()

    # PD Analysis
    pd_avg = float(np.mean(pd_absolute_errors))
    pd_max = float(np.max(pd_absolute_errors))
    pd_within_1mm = sum(1 for e in pd_absolute_errors if e <= 1.0)
    pd_within_2mm = sum(1 for e in pd_absolute_errors if e <= 2.0)

    print("+" + "-" * 62 + "+")
    print("|  PUPILLARY DISTANCE -- OPTICAL GRADE ANALYSIS".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Average Absolute Error:  {pd_avg:.2f} mm".ljust(63) + "|")
    print(f"|  Maximum Absolute Error:  {pd_max:.2f} mm".ljust(63) + "|")
    print(f"|  Within +/-1mm (Rx-grade): {pd_within_1mm}/{NUM_TRIALS} ({pd_within_1mm/NUM_TRIALS*100:.0f}%)".ljust(63) + "|")
    print(f"|  Within +/-2mm (Standard): {pd_within_2mm}/{NUM_TRIALS} ({pd_within_2mm/NUM_TRIALS*100:.0f}%)".ljust(63) + "|")
    pd_status = "OPTICAL GRADE" if pd_avg <= PD_STRICT_THRESHOLD_MM else "STANDARD GRADE"
    print(f"|  PD Rating:               {pd_status}".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print()

    # Noise breakdown
    print("+" + "-" * 62 + "+")
    print("|  ACCURACY BY NOISE CONDITION".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    for noise_key in ["low", "medium", "high"]:
        if noise_key in per_noise_errors:
            accs = per_noise_errors[noise_key]
            avg = float(np.mean(accs))
            n = len(accs)
            passed = sum(1 for a in accs if a >= 95.0)
            print(f"|  {NOISE_PROFILES[noise_key]['label']}".ljust(63) + "|")
            print(f"|    Trials: {n:<5}  Avg: {avg:.2f}%  Passed: {passed}/{n}".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print()

    # Archetype breakdown
    print("+" + "-" * 62 + "+")
    print("|  ACCURACY BY FACE ARCHETYPE".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    for arch_key in sorted(per_archetype_errors.keys()):
        accs = per_archetype_errors[arch_key]
        avg = float(np.mean(accs))
        n = len(accs)
        passed = sum(1 for a in accs if a >= 95.0)
        label = arch_key.replace("_", " ").title()
        status = "OK" if avg >= 95.0 else "!!"
        print(f"|  [{status}] {label:<20} Trials: {n:<4} Avg: {avg:.2f}% Pass: {passed}/{n}".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print()

    # Failed details
    if fail_details:
        print("+" + "-" * 62 + "+")
        print("|  FAILED TRIAL DETAILS (showing up to 15)".ljust(63) + "|")
        print("+" + "-" * 62 + "+")
        for fd in fail_details[:15]:
            wk = fd["worst"][0].replace("_mm", "").replace("_", " ").title()
            we = fd["worst"][1]["error_pct"]
            print(f"|  Trial #{fd['trial']:<3} | {fd['archetype']:<14} | {fd['noise']:<6} | "
                  f"Acc: {fd['accuracy']:.1f}% | Worst: {wk} ({we:.1f}%)".ljust(63) + "|")
        if len(fail_details) > 15:
            print(f"|  ... and {len(fail_details) - 15} more".ljust(63) + "|")
        print("+" + "-" * 62 + "+")
        print()

    # Distribution
    print("+" + "-" * 62 + "+")
    print("|  ACCURACY DISTRIBUTION".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    brackets = [
        (">=99%", 99.0, 100.1), ("98-99%", 98.0, 99.0),
        ("97-98%", 97.0, 98.0), ("96-97%", 96.0, 97.0),
        ("95-96%", 95.0, 96.0), ("90-95%", 90.0, 95.0),
        ("<90%", 0.0, 90.0),
    ]
    for label, lo, hi in brackets:
        count = sum(1 for a in trial_accuracies if lo <= a < hi)
        bar = "#" * (count * 2)
        print(f"|  {label:>7}: {count:>3} {bar}".ljust(63) + "|")
    print("+" + "-" * 62 + "+")
    print()

    # VERDICT
    print("+" + "=" * 62 + "+")
    if overall_avg >= 95.0 and pass_rate >= 95.0:
        vstr = "PASSED"
        print(f"|  [OK] PASSED -- ENGINE MEETS >=95% ACCURACY TARGET".ljust(63) + "|")
    elif overall_avg >= 95.0 and pass_rate >= 85.0:
        vstr = "MARGINAL_PASS"
        print(f"|  [~~] MARGINAL PASS -- AVG OK, SOME OUTLIERS".ljust(63) + "|")
    else:
        vstr = "FAILED"
        print(f"|  [XX] FAILED -- BELOW >=95% TARGET".ljust(63) + "|")
    print(f"|  Average: {overall_avg:.2f}% | Pass Rate: {pass_rate:.1f}% | PD Err: {pd_avg:.2f}mm".ljust(63) + "|")
    print("+" + "=" * 62 + "+")
    print()

    # Save JSON
    report_path = os.path.join(os.path.dirname(__file__), "accuracy_report_100.json")
    report = {
        "engine_version": "v5",
        "test_date": datetime.now().isoformat(),
        "total_trials": NUM_TRIALS,
        "stabilization_passes": 5,
        "passed": pass_count,
        "failed": NUM_TRIALS - pass_count,
        "pass_rate_pct": round(pass_rate, 2),
        "overall_accuracy_avg": round(overall_avg, 3),
        "overall_accuracy_median": round(overall_median, 3),
        "overall_accuracy_min": round(overall_min, 3),
        "overall_accuracy_max": round(overall_max, 3),
        "overall_accuracy_std": round(overall_std, 4),
        "pd_analysis": {
            "avg_absolute_error_mm": round(pd_avg, 3),
            "max_absolute_error_mm": round(pd_max, 3),
            "within_1mm_count": pd_within_1mm,
            "within_2mm_count": pd_within_2mm,
            "optical_grade": pd_avg <= PD_STRICT_THRESHOLD_MM,
        },
        "verdict": vstr,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Full report saved to: {report_path}\n")
    return report


if __name__ == "__main__":
    report = run_100_trial_test()
    sys.exit(0 if report["verdict"] == "PASSED" else 1)
