import cv2
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Function to detect face and extract features
def detect_and_extract_features(image, face_cascade, feature_extractor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    features = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        features.append(feature_extractor(face_roi))
    return features

# Function to record and store facial images with labels
def record_faces(face_cascade, n_samples, save_dir):
    for i in range(n_samples):
        name = input(f"Enter name for person {i+1}: ")
        os.makedirs(os.path.join(save_dir, name), exist_ok=True)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            faces = detect_and_extract_features(frame, face_cascade, lbph_feature_extractor)
            if len(faces) > 0:
                for j, face in enumerate(faces):
                    cv2.imwrite(os.path.join(save_dir, name, f"{i+1}_{j}.jpg"), face)
                break
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

def lbph_feature_extractor(image):
    
    if len(image.shape) > 2 and image.shape[2] == 3:  # Check for color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale or invalid image
        gray = image
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create LBP image
    lbp = cv2.createLBP(image=gray, radius=1, neighbors=8)

    # Calculate LBP histogram
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])

    return hist.flatten()  # Flatten the histogram for feature vector

# Record facial images (modify parameters as needed)
record_faces(face_cascade=cv2.CascadeClassifier("resource/haarcascade_frontalface_default.xml"), n_samples=10, save_dir="face_data")

# Extract features and labels from recorded images
features = []
labels = np.array([1, 2, 3, 4], dtype=np.int32)  # Correct label format
for folder in os.listdir("face_data"):
    for filename in os.listdir(os.path.join("face_data", folder)):
        image = cv2.imread(os.path.join("face_data", folder, filename))
        feature = lbph_feature_extractor(image)
        features.append(feature)
        labels.append(folder)

# Preprocess features (if needed)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(features, labels)

# Save the trained KNN model in HDF5 format
h5_model_path = "knn_model.h5"
joblib.dump(knn_clf, h5_model_path)

print("Face recording and training completed!")
