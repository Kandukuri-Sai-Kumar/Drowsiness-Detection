import cv2
import numpy as np
import dlib
import time
import serial
from twilio.rest import Client
from imutils import face_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Function to send SMS using Twilio
def send_sms(message):
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )


# Serial communication with hardware
s = serial.Serial('COM5', 9600)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector and landmark detector
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Placeholder lists for features (EAR values) and labels
features = []
labels = []
status = ""
color = (0, 0, 0)
EYE_AR_THRESH = 0.25  # Threshold for eye aspect ratio
BLINKING_TIME_THRESHOLD = 0.3  # Time in seconds to consider blinking
SLEEPY_TIME_THRESHOLD = 1  # Time in seconds to consider drowsiness

# Track the start time of potential drowsiness
ear_below_thresh_start_time = None
is_drowsy = False
is_blinking = False

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to obtain ground truth label for each frame
def get_ground_truth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_ratio = eye_aspect_ratio(landmarks[36:42])
        right_eye_ratio = eye_aspect_ratio(landmarks[42:48])

        # Compute the average eye aspect ratio
        eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # If the average eye aspect ratio is below the threshold, consider the driver drowsy
        if eye_avg_ratio < EYE_AR_THRESH:
            return 1  # Ground truth is 1 if drowsy
        else:
            return 0  # Ground truth is 0 if not drowsy
    
    return 0

# Inside the main loop where frames are processed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get ground truth label for the current frame
    ground_truth = get_ground_truth(frame)

    faces = hog_face_detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        left_eye_ear = eye_aspect_ratio(landmarks[36:42])
        right_eye_ear = eye_aspect_ratio(landmarks[42:48])
        avg_ear = (left_eye_ear + right_eye_ear) / 2

        # Append eye aspect ratio (EAR) to features list
        features.append(avg_ear)
        # Append ground truth label to labels list
        labels.append(ground_truth)

        if avg_ear < EYE_AR_THRESH:
            if ear_below_thresh_start_time is None:
                ear_below_thresh_start_time = time.time()
            else:
                elapsed_time = time.time() - ear_below_thresh_start_time
                if elapsed_time >= SLEEPY_TIME_THRESHOLD:
                    if not is_drowsy:
                        s.write(b'a')  # Buzzer ON
                        send_sms("Driver is sleeping")
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    is_drowsy = True
                    is_blinking = False
                elif elapsed_time >= BLINKING_TIME_THRESHOLD:
                    if not is_blinking:
                        s.write(b'b')  # Buzzer ON
                        send_sms("Driver is blinking")
                    status = "BLINKING !!!"
                    color = (0, 255, 255)
                    is_blinking = True
                    is_drowsy = False
        else:
            ear_below_thresh_start_time = None
            if is_drowsy or is_blinking:
                s.write(b'c')  # Buzzer OFF
            status = "Active :)"
            color = (0, 255, 0)
            is_drowsy = False
            is_blinking = False

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays for training
features = np.array(features).reshape(-1, 1)  # Reshape to a single feature column
labels = np.array(labels)

# Check if features and labels have the same length
print(f"Number of features: {len(features)}")
print(f"Number of labels: {len(labels)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
