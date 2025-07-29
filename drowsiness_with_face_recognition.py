import cv2
import numpy as np
import dlib
import face_recognition
import os
import csv
from datetime import datetime
from imutils import face_utils
from threading import Thread
import playsound

# ========== Step 1: Load known employee faces ==========
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join("known_faces", filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].replace("_", " ")
            known_face_names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}")

# ========== Step 2: Setup logging ==========
def sound_alarm(alarm_file):
    if os.path.exists(alarm_file):
        playsound.playsound(alarm_file)
    else:
        print(f"Alarm file not found: {alarm_file}")

def log_event_with_screenshot(frame, status, name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    filename_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = f"logs/{name.replace(' ', '_')}"
    os.makedirs(log_folder, exist_ok=True)

    filename = f"{log_folder}/screenshot_{filename_time}_{status.replace(' ', '_').replace('!', '')}.png"
    cv2.imwrite(filename, frame)

    log_path = f"{log_folder}/log.csv"
    write_header = not os.path.exists(log_path)

    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Time", "Status", "Screenshot"])
        writer.writerow([timestamp, status, filename])

# ========== Step 3: Drowsiness Detection Setup ==========
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
ALARM_ON = False
MAR_THRESH = 0.6
employee_name = "Unknown"

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.2 < ratio <= 0.25:
        return 1
    else:
        return 0

def mouth_aspect_ratio(mouth):
    A = compute(mouth[1], mouth[7])
    B = compute(mouth[2], mouth[6])
    C = compute(mouth[3], mouth[5])
    D = compute(mouth[0], mouth[4])
    mar = (A + B + C) / (2.0 * D)
    return mar

# ========== Step 4: Main loop ==========
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            employee_name = known_face_names[best_match_index]
            break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        mouth = landmarks[60:68]
        mar = mouth_aspect_ratio(mouth)

        if mar > MAR_THRESH:
            drowsy += 1
            sleep = 0
            active = 0
            status = "Yawning (Drowsy)"
            color = (0, 0, 255)
            if not ALARM_ON:
                ALARM_ON = True
                Thread(target=sound_alarm, args=('/Users/rohit/Desktop/Real_time_drosiness detection_final_project/alarm-clock-short-6402.mp3',), daemon=True).start()
            if employee_name != "Unknown":
                log_event_with_screenshot(frame, status, employee_name)

        elif left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 10:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if not ALARM_ON:
                    ALARM_ON = True
                    Thread(target=sound_alarm, args=('/Users/rohit/Desktop/Real_time_drosiness detection_final_project/alarm-clock-short-6402.mp3',), daemon=True).start()
                if employee_name != "Unknown":
                    log_event_with_screenshot(frame, status, employee_name)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                if not ALARM_ON:
                    ALARM_ON = True
                    Thread(target=sound_alarm, args=('/Users/rohit/Desktop/Real_time_drosiness detection_final_project/alarm-clock-short-6402.mp3',), daemon=True).start()
                if employee_name != "Unknown":
                    log_event_with_screenshot(frame, status, employee_name)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
            ALARM_ON = False

        cv2.putText(frame, f"{employee_name}: {status}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break
