import cv2
import numpy as np
from tensorflow.keras.models import load_model

GESTURES = {
    '01_palm': 0, '02_l': 1, '03_fist': 2, '04_fist_moved': 3, '05_thumb': 4,
    '06_index': 5, '07_ok': 6, '08_palm_moved': 7, '09_c': 8, '10_down': 9
}
labels_map = {v: k for k, v in GESTURES.items()}

try:
    model = load_model('hand_gesture_model.h5')
    print("Model loaded successfully!")
except:
    print("Error: 'hand_gesture_model.h5' not found.")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Bounding Box Coordinates
    x1, y1 = int(width * 0.55), int(height * 0.2)
    x2, y2 = int(width * 0.95), int(height * 0.7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Crop the Hand Area
    roi = frame[y1:y2, x1:x2]
    
    # --- SKIN COLOR SEGMENTATION PIPELINE ---
    # Convert ROI from BGR color space to YCrCb
    ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    
    # Define exact ranges for human skin color in YCrCb
    min_YCrCb = np.array([0, 135, 85], np.uint8)
    max_YCrCb = np.array([255, 180, 135], np.uint8)
    
    # Filter the image to only show skin pixels (turns skin white, everything else black)
    skin_mask = cv2.inRange(ycrcb_roi, min_YCrCb, max_YCrCb)
    
    # Apply morphological operations to clean up small background specks
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    # Resize and Format for Model Prediction
    resized = cv2.resize(skin_mask, (64, 64))
    normalized = resized / 255.0
    final_input = np.reshape(normalized, (1, 64, 64, 1))
    
    # Model Predicts
    prediction = model.predict(final_input, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Display Label
    if confidence > 0.65:
        gesture_name = labels_map[class_idx].split('_')[-1].upper()
        display_text = f"{gesture_name} ({confidence*100:.1f}%)"
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Waiting for hand...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Windows
    cv2.imshow("Live Video Feed", frame)
    cv2.imshow("Skin Segmentation View", skin_mask) # Check if your hand is fully white here!
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()