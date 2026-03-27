"""Download the MediaPipe face_landmarker.task model."""
import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python", "face_landmarker.task")

print(f"Downloading to: {dest}")
urllib.request.urlretrieve(url, dest)
size = os.path.getsize(dest)
print(f"Downloaded: {size} bytes ({size/1024/1024:.1f} MB)")
