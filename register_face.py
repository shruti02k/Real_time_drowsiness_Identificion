
import cv2
import os

# Step 1: Ask user name
name = input("Enter your full name: ").strip().replace(" ", "_")
save_path = f"known_faces/{name}.jpg"

# Step 2: Create directory if not exists
os.makedirs("known_faces", exist_ok=True)

# Step 3: Capture image from webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 's' to save the image or 'q' to quit without saving.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        print("[ERROR] Failed to capture image.")
        break

    cv2.putText(frame, "Press 's' to Save or 'q' to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Face image saved to {save_path}")
        break
    elif key == ord('q'):
        print("[INFO] Face registration cancelled.")
        break

cap.release()
cv2.destroyAllWindows()
