import os
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def extract_features(file_path, target_length=50000):
   	try:
      	data, sr = sf.read(file_path)

       	if len(data.shape) > 1:
       		data = np.mean(data, axis=1)

       	data = data / (np.max(np.abs(data)) + 1e-6)

       	if len(data) > target_length:
       		data = data[:target_length]
       	else:
           		data = np.pad(data, (0, target_length - len(data)))

       	fft = np.fft.fft(data)
       	fft_magnitude = np.abs(fft)[:len(fft)//2]

       	feature = np.mean(fft_magnitude.reshape(-1, 500), axis=1)

       	return feature

   	except Exception as e:
       	print(f"Error processing {file_path}: {e}")
       	return None

def load_data(directory, label, max_files=None):
   		features = []
   		labels = []

   		count = 0

   		for root, _, files in os.walk(directory):
       		for file in files:
           			if file.endswith(".wav") or file.endswith(".flac"):
               			path = os.path.join(root, file)

               			feature = extract_features(path)

               		if feature is not None:
                   		features.append(feature)
                   		labels.append(label)
                   		count += 1

               		if max_files and count >= max_files:
                   		return features, labels

   		return features, labels

#Data 1 real1fake1
real_path = "data/real"
fake_path = "data/ASVspoof2019_LA_train/flac"

#Data 2 real2fake2
'''real_path = "data/NaturalSpeech3"
fake_path = "data/real_samples"'''

#Data 3 real2fake1
'''real_path = "data/real"
fake_path = "data/real_samples"'''

#Data 4 real1fake2
'''real_path = "data/NaturalSpeech3"
fake_path = "data/ASVspoof2019_LA_train/flac"'''

print("Loading real samples...")
real_features, real_labels = load_data(real_path, 0, max_files=250)
	
print("Loading fake samples...")
fake_features, fake_labels = load_data(fake_path, 1, max_files=250)

X = np.array(real_features + fake_features)
y = np.array(real_labels + fake_labels)

print("Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
   	X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle = '--')
plt.xlabel = ("False Positive Rate")
plt.ylabel = ("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc: .2f})")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
