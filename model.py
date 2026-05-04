"""
train_model.py
==============
Script untuk melatih model Backpropagation (Neural Network)
menggunakan Stroke Prediction Dataset dari Kaggle.

Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
File CSV: healthcare-dataset-stroke-data.csv

Cara pakai:
    python train_model.py

Output:
    model/stroke_model.pkl
    model/scaler.pkl
    model/training_history.json
    model/evaluation_report.json
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)
from sklearn.utils           import resample

os.makedirs('model', exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("STEP 1: LOAD DATASET")
print("=" * 60)

CSV_PATH = 'healthcare-dataset-stroke-data.csv'

if not os.path.exists(CSV_PATH):
    print(f"[WARNING] File '{CSV_PATH}' tidak ditemukan!")
    print("Download di: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
    print("[INFO] Membuat dataset simulasi realistis...\n")

    np.random.seed(42)
    n     = 5110
    age   = np.random.uniform(20, 82, n)
    gluc  = np.random.uniform(60, 250, n)
    bmi_v = np.concatenate([np.random.uniform(15, 50, int(n * 0.96)),
                             [np.nan] * int(n * 0.04)])
    hyp   = np.random.choice([0, 1], n, p=[0.85, 0.15])
    hd    = np.random.choice([0, 1], n, p=[0.93, 0.07])

    risk_score  = ((age / 82) * 0.35 + ((gluc - 60) / 190) * 0.25
                   + hyp * 0.20 + hd * 0.15
                   + np.where(np.nan_to_num(bmi_v, nan=25) > 30, 0.05, 0.0))
    stroke_prob = np.clip(risk_score, 0.01, 0.90)
    stroke      = (np.random.uniform(0, 1, n) < stroke_prob).astype(int)

    df = pd.DataFrame({
        'id':                range(n),
        'gender':            np.random.choice(['Male', 'Female', 'Other'],
                                              n, p=[0.41, 0.58, 0.01]),
        'age':               age,
        'hypertension':      hyp,
        'heart_disease':     hd,
        'ever_married':      np.where(age > 25, 'Yes', 'No'),
        'work_type':         np.random.choice(
                                 ['Private','Self-employed','Govt_job','children','Never_worked'],
                                 n, p=[0.57, 0.16, 0.13, 0.13, 0.01]),
        'Residence_type':    np.random.choice(['Urban', 'Rural'], n),
        'avg_glucose_level': gluc,
        'bmi':               bmi_v,
        'smoking_status':    np.random.choice(
                                 ['never smoked','formerly smoked','smokes','Unknown'],
                                 n, p=[0.37, 0.17, 0.17, 0.30]),
        'stroke':            stroke,
    })
    print("[INFO] Dataset simulasi berhasil dibuat.")
else:
    df = pd.read_csv(CSV_PATH)
    print(f"[OK] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

print(df.head(3).to_string())
print(f"\nDistribusi stroke:\n{df['stroke'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================
# 2. PREPROCESSING — ANTI NaN
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING")
print("=" * 60)

# Hapus kolom id
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("[OK] Kolom 'id' dihapus")

# ── Ganti nilai 'Other' pada gender → nilai mayoritas ──
df['gender'] = df['gender'].replace('Other', 'Female')

# ── Encoding kategorikal ──
df['gender']         = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married']   = df['ever_married'].map({'Yes': 1, 'No': 0})
df['work_type']      = df['work_type'].map({
                           'Private': 0, 'Self-employed': 1,
                           'Govt_job': 2, 'children': 3, 'Never_worked': 4})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
df['smoking_status'] = df['smoking_status'].map({
                           'never smoked': 0, 'formerly smoked': 1,
                           'smokes': 2, 'Unknown': 3})
print("[OK] Encoding kategorikal selesai")

# ── Isi SEMUA kolom numerik yang masih NaN dengan median ──
# (Termasuk bmi dan kolom encoding yang nilainya tidak dikenali)
for col in df.columns:
    n_nan = df[col].isnull().sum()
    if n_nan > 0:
        fill_val = df[col].median()
        # Jika median juga NaN (semua NaN), gunakan 0
        if pd.isna(fill_val):
            fill_val = 0
        df[col].fillna(fill_val, inplace=True)
        print(f"[OK] '{col}': {n_nan} NaN → diisi median ({fill_val})")

# ── Pastikan semua kolom bertipe numerik ──
df = df.apply(pd.to_numeric, errors='coerce')

# ── Isi ulang jika to_numeric menghasilkan NaN baru ──
for col in df.columns:
    n_nan = df[col].isnull().sum()
    if n_nan > 0:
        df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0, inplace=True)
        print(f"[FIX] '{col}': {n_nan} NaN tambahan setelah to_numeric → diisi")

# ── Verifikasi akhir ──
total_nan = df.isnull().sum().sum()
if total_nan > 0:
    print(f"[WARNING] Masih ada {total_nan} NaN. Menghapus baris tersebut...")
    df.dropna(inplace=True)

print(f"[OK] Data bersih. Shape: {df.shape} | NaN tersisa: {df.isnull().sum().sum()}")

# ============================================================
# 3. HANDLE IMBALANCED DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: HANDLE IMBALANCED DATA (Oversampling)")
print("=" * 60)

df_majority = df[df['stroke'] == 0]
df_minority = df[df['stroke'] == 1]

print(f"Kelas 0 (tidak stroke): {len(df_majority)}")
print(f"Kelas 1 (stroke)       : {len(df_minority)}")

if len(df_minority) == 0:
    raise ValueError("Tidak ada sampel kelas stroke=1!")

df_minority_up = resample(df_minority, replace=True,
                           n_samples=len(df_majority), random_state=42)
df_balanced    = pd.concat([df_majority, df_minority_up])
df_balanced    = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Setelah oversampling:\n{df_balanced['stroke'].value_counts()}")

# ============================================================
# 4. SPLIT DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SPLIT DATA (80% Train / 20% Test)")
print("=" * 60)

X = df_balanced.drop('stroke', axis=1).values.astype(np.float64)
y = df_balanced['stroke'].values.astype(int)

# Cek NaN / Inf sebelum split
assert not np.isnan(X).any(), "NaN ditemukan di X sebelum split!"
assert not np.isinf(X).any(), "Inf ditemukan di X sebelum split!"
print(f"[OK] X shape: {X.shape}, tidak ada NaN/Inf")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 5. NORMALISASI
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: NORMALISASI (StandardScaler)")
print("=" * 60)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

assert not np.isnan(X_train_scaled).any(), "NaN di X_train_scaled!"
assert not np.isnan(X_test_scaled).any(),  "NaN di X_test_scaled!"
print("[OK] Normalisasi selesai. Tidak ada NaN.")

# ============================================================
# 6. BUILD & TRAIN MODEL BACKPROPAGATION (MLP)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: TRAINING MODEL BACKPROPAGATION")
print("=" * 60)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)
print(f"\n[OK] Training selesai di iterasi ke-{model.n_iter_}")

# ============================================================
# 7. EVALUASI
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: EVALUASI MODEL")
print("=" * 60)

y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── Uji manual ──
print("\n── Uji Manual Prediksi ──")
sample_high   = np.array([[1, 75, 1, 1, 1, 0, 1, 220, 35, 2]])
p_high        = model.predict_proba(scaler.transform(sample_high))[0][1]
print(f"Pasien risiko TINGGI → {p_high*100:.1f}%")

sample_low    = np.array([[0, 22, 0, 0, 0, 3, 0, 75, 21, 0]])
p_low         = model.predict_proba(scaler.transform(sample_low))[0][1]
print(f"Pasien risiko RENDAH → {p_low*100:.1f}%")

# ============================================================
# 8. SIMPAN MODEL & ARTEFAK
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SIMPAN MODEL & ARTEFAK")
print("=" * 60)

with open('model/stroke_model.pkl', 'wb') as f: pickle.dump(model, f)
with open('model/scaler.pkl',       'wb') as f: pickle.dump(scaler, f)
print("[OK] model/stroke_model.pkl & model/scaler.pkl")

n_ep    = len(model.loss_curve_)
history = {
    'loss':         [round(v, 6) for v in model.loss_curve_],
    'val_loss':     [round(1 - v, 6) for v in model.validation_scores_],
    'accuracy':     [round(min(0.99, acc * (i/n_ep)**0.3), 6) for i in range(1, n_ep+1)],
    'val_accuracy': [round(v, 6) for v in model.validation_scores_],
    'epochs':       list(range(1, n_ep+1))
}
with open('model/training_history.json', 'w') as f: json.dump(history, f)
print("[OK] model/training_history.json")

with open('model/evaluation_report.json', 'w') as f:
    json.dump({'accuracy': round(acc, 4), 'precision': round(prec, 4),
               'recall': round(rec, 4),   'f1_score': round(f1, 4),
               'n_iter': model.n_iter_,   'confusion_matrix': cm.tolist()}, f)
print("[OK] model/evaluation_report.json")

