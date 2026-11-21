import os
from PIL import Image
import numpy as np
import pandas as pd
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

def extract_landmarks(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
    except Exception:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y, lm.z])
        return features
    else:
        return None

DATASET_PATH = r"C:\Users\sdivy\OneDrive\Documents\VS CODE\SML Project\asl_alphabet_train\asl_alphabet_train"

print("Checking dataset path...")
print(os.listdir(DATASET_PATH)[:5])

data = []
labels = []
count = 0

print("Starting extraction. This will take 45-90 minutes...")

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing letter: {label}")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        features = extract_landmarks(img_path)

        if features:
            data.append(features)
            labels.append(label)
            count += 1

            if count % 1000 == 0:
                print(f"  Processed {count} images...")

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('asl_landmarks.csv', index=False)

print("Extraction complete!")
print(f"Total samples saved: {len(df)}")
print(f"Dataset shape: {df.shape}")