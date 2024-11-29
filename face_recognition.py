import cv2
import dlib
from fer import FER
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_detector = FER(mtcnn=True)

def analyze_micro_expressions(landmarks):
    key_features = {
        "eyebrows": [20, 23, 24, 27],
        "eyes": [36, 39, 42, 45],
        "mouth": [48, 54, 57]
    }
    movements = {}
    descriptions = []

    for feature, points in key_features.items():
        coords = np.array([[landmarks.part(p).x, landmarks.part(p).y] for p in points])
        movement_range = np.ptp(coords, axis=0)  
        movements[feature] = movement_range

        if movement_range[1] > 5:
            descriptions.append(f"{feature.capitalize()} movement detected")

    return ", ".join(descriptions) if descriptions else "No significant micro-expressions"

video_path = "emotions.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        micro_expression_desc = analyze_micro_expressions(landmarks)
        emotion, score = emotion_detector.top_emotion(frame)
        
        text = f"Emotion: {emotion} ({score*100:.2f}%)" if emotion else "Emotion: Neutral"
        cv2.putText(frame, text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, micro_expression_desc, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imshow("Micro-Expression Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






