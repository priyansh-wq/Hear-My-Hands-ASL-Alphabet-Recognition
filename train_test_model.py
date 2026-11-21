import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("asl_final_dataset.csv") 

counts = df['label'].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df['label'].isin(valid_classes)]

def normalize_features_2d(features):
    features = np.array(features).reshape(21, 3)
    wrist = features[0]
    features = features - wrist
    scale = np.linalg.norm(features[9]) 
    if scale > 0:
        features = features / scale
        
    features_2d = features[:, :2] # Only X and Y
    flattened = features_2d.flatten()
    
    # 🚨 FEATURE 1: U vs V distance
    dist_u_v = np.linalg.norm(features_2d[8] - features_2d[12])
    
    # 🚨 FEATURE 2: B vs Hello Thumb distance
    dist_thumb = np.linalg.norm(features_2d[4] - features_2d[5])
    
    # Total features = 42 + 1 + 1 = 44
    return np.append(flattened, [dist_u_v, dist_thumb])

X_raw = df.drop('label', axis=1).values
y = df['label'].values

X_norm = np.array([normalize_features_2d(row) for row in X_raw])

# Data Augmentation
X_flipped = X_norm.copy()
X_flipped[:, 0:-2:2] *= -1  # Flip X coordinates (ignoring the last 2 custom features)

X_final = np.vstack((X_norm, X_flipped))
y_final = np.concatenate((y, y)) 

le = LabelEncoder()
y_encoded = le.fit_transform(y_final)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

joblib.dump(lr, 'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained successfully!")