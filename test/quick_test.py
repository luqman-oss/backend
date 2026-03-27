"""Quick test of the new mediapipe tasks API with face landmarker."""
import mediapipe as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(ROOT, 'python', 'face_landmarker.task')
img_path = os.path.join(ROOT, 'uploads', 'test_face.jpg')

print(f"Model: {model_path} (exists: {os.path.exists(model_path)})")
print(f"Image: {img_path} (exists: {os.path.exists(img_path)})")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_faces=2,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file(img_path)
    result = landmarker.detect(mp_image)
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        print(f"Faces detected: {len(result.face_landmarks)}")
        print(f"Landmarks per face: {len(lms)}")
        print(f"Landmark 0: x={lms[0].x:.4f}, y={lms[0].y:.4f}, z={lms[0].z:.4f}")
        if len(lms) > 473:
            print(f"Landmark 468 (L iris): x={lms[468].x:.4f}, y={lms[468].y:.4f}")
            print(f"Landmark 473 (R iris): x={lms[473].x:.4f}, y={lms[473].y:.4f}")
            print("Iris landmarks available!")
        else:
            print(f"Only {len(lms)} landmarks - no iris landmarks")
    else:
        print("No faces detected")
