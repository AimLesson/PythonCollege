import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function for feature extraction (replace with your preferred method)
def extract_features(face_roi):
    # Implement your feature extraction technique here, e.g., LBPH
    # ...
    return features

# Load training data (replace with your actual data)
features = np.load('features.npy')
labels = np.load('labels.npy')

# Train KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors as needed
knn_clf.fit(features, labels)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract features from the detected face
        face_roi = gray[y:y+h, x:x+w]
        face_features = extract_features(face_roi)

        # Predict the label using KNN
        predicted_label = knn_clf.predict([face_features])[0]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display predicted label
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
