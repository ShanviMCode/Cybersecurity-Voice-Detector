import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import resample, welch
from scipy.fftpack import dct

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

REAL_DIR = "data/real"
ASV_AUDIO_DIR = "data/ASVspoof2019_LA_train/flac"

SAMPLE_RATE = 16000
DURATION = 3

def preprocess_audio(path, sr=SAMPLE_RATE, max_len=DURATION):
    audio, file_sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != sr:
        audio = resample(audio, int(len(audio) * sr / file_sr))

    max_samples = sr * max_len
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    return audio

def extract_features(audio, sr=SAMPLE_RATE):
    freqs, psd = welch(audio, sr, nperseg=512)
    log_psd = np.log(psd + 1e-10)
    mfcc_like = dct(log_psd, type=2, norm="ortho")[:13]

    zero_crossings = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

    energy = np.mean(audio ** 2)

    return np.hstack([mfcc_like, zero_crossings, energy])

X = []
y = []

print("Loading real audio...")
for file in os.listdir(REAL_DIR):
    if file.endswith(".flac"):
        audio = preprocess_audio(os.path.join(REAL_DIR, file))
        X.append(extract_features(audio))
        y.append(0)

print("Loading fake audio...")
for file in os.listdir(ASV_AUDIO_DIR):
    if file.endswith(".flac"):
        path = os.path.join(ASV_AUDIO_DIR, file)
        audio = preprocess_audio(path)
        features = extract_features(audio)
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

print("\nTotal samples:", len(y))
print("Real:", np.sum(y == 0))
print("Fake:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance = model.coef_[0]

plt.figure()
plt.bar(range(len(importance)), importance)
plt.xlabel("Feature Index")
plt.ylabel("Weight")
plt.title("Feature Importance for Deepfake Detection")
plt.show()
