import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# ------------------------------------------------------------
# 1. LOAD TRAINED MODELS
# ------------------------------------------------------------
print("Loading models...")
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


def speak_text(text):
    """Speak the completed message using the system voice."""
    text = str(text).strip()
    if not text:
        return

    print(f"SPEAKING: {text}")


# 2. INITIALIZE MEDIAPIPE HAND LANDMARKER

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)


BG_DARK = (31, 17, 7)          # #07111F
PANEL = (46, 29, 16)           # #101D2E
CYAN = (255, 217, 0)           # #00D9FF
VIOLET = (247, 85, 168)        # #A855F7
WHITE = (249, 245, 241)        # #F1F5F9
MUTED = (184, 163, 148)        # #94A3B8
GREEN = (153, 211, 52)         # #34D399
TRACK = (77, 53, 34)



background = cv2.imread("bg2.jpg")
background = cv2.resize(background, (1280, 720))

black_layer = np.zeros_like(background)
background = cv2.addWeighted(background, 0.62, black_layer, 0.38, 0)



# 5. CAMERA AND APPLICATION STATE

cap = cv2.VideoCapture(0)

frame_timestamp_ms = 0
prediction_history = deque(maxlen=10)
current_word = ""
current_letter = ""
confidence = 0.0


# 6. SMALL UI HELPERS

def draw_panel(image, top_left, bottom_right, color=PANEL, alpha=0.82):
    """Draw a translucent rectangular panel."""
    layer = image.copy()
    cv2.rectangle(layer, top_left, bottom_right, color, -1)
    cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)


def draw_corner_frame(image, x1, y1, x2, y2):
    """Draw a cyan frame with small violet corner accents."""
    cv2.rectangle(image, (x1, y1), (x2, y2), CYAN, 2, cv2.LINE_AA)
    corner = 24


def remove_last_letter(word):
    """Remove the last added letter, including its separator space."""
    word = word.rstrip()
    if word:
        word = word[:-1]
    return word.rstrip()



# 7. MAIN APPLICATION LOOP

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        canvas = background.copy()

        # Convert the webcam frame for MediaPipe.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        hand_detected = False

        if result.hand_landmarks:
            hand_detected = True
            hand_landmarks = result.hand_landmarks[0]

            features = []
            for lm in hand_landmarks:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(21, 3)
            wrist = features[0]
            features = features - wrist

            scale = np.linalg.norm(features[9])
            if scale > 0:
                features = features / scale

            features_2d = features[:, :2]
            flattened = features_2d.flatten()

            dist_u_v = np.linalg.norm(features_2d[8] - features_2d[12])
            dist_thumb = np.linalg.norm(features_2d[4] - features_2d[5])
            features_normalized = np.append(flattened, [dist_u_v, dist_thumb])

            features_scaled = scaler.transform([features_normalized])

            probabilities = model.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilities)
            prediction = model.predict(features_scaled)[0]
            predicted_letter = le.inverse_transform([prediction])[0]

            prediction_history.append(predicted_letter)
            current_letter = Counter(prediction_history).most_common(1)[0][0]
            confidence = float(max_prob * 100)

            # Draw the 21 hand landmarks in cyan.
            for lm in hand_landmarks:
                cx = int(lm.x * frame.shape[1])
                cy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, CYAN, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 8, WHITE, 1, cv2.LINE_AA)
        else:
            current_letter = ""
            confidence = 0.0
            prediction_history.clear()

        # ----------------------------------------------------
        # 8. HEADER
        # ----------------------------------------------------
        cv2.putText(
            canvas, "HEAR MY HANDS", (40, 42),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "ASL ALPHABET RECOGNITION", (350, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1, cv2.LINE_AA
        )
        cv2.line(canvas, (40, 51), (1240, 51), (75, 63, 43), 1, cv2.LINE_AA)

        # ----------------------------------------------------
        # 9. WEBCAM PANEL
        # ----------------------------------------------------
        canvas[60:540, 40:680] = frame
        draw_corner_frame(canvas, 36, 56, 684, 544)

        # ----------------------------------------------------
        # 10. RECOGNITION PANEL
        # ----------------------------------------------------
        draw_panel(canvas, (720, 60), (1240, 544), PANEL, 0.88)
        draw_corner_frame(canvas, 720, 60, 1240, 544)

        cv2.putText(
            canvas, "RECOGNITION", (745, 103),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, CYAN, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "PREDICTED SIGN", (745, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, MUTED, 1, cv2.LINE_AA
        )

        display_letter = current_letter if current_letter else "-"
        font_scale = 3.2 if len(display_letter) <= 2 else 1.5
        cv2.putText(
            canvas, display_letter, (745, 245),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, VIOLET, 4, cv2.LINE_AA
        )

        cv2.putText(
            canvas, "MODEL CONFIDENCE", (745, 310),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, MUTED, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"{confidence:.1f}%", (745, 355),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, CYAN, 2, cv2.LINE_AA
        )

        # Confidence bar; clamp its width so it never exceeds the panel.
        bar_x1, bar_y1 = 745, 380
        bar_x2, bar_y2 = 1215, 393
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), TRACK, -1)
        fill_width = int((bar_x2 - bar_x1) * np.clip(confidence, 0, 100) / 100)
        if fill_width > 0:
            cv2.rectangle(
                canvas, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2),
                CYAN, -1
            )

        cv2.putText(
            canvas, "TRACKING STATUS", (745, 445),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, MUTED, 1, cv2.LINE_AA
        )

        if hand_detected and confidence > 40:
            status_text = f"STABLE: {current_letter}"
            status_color = GREEN
        elif hand_detected:
            status_text = "LOW CONFIDENCE"
            status_color = (44, 191, 251)  # Amber in BGR
        else:
            status_text = "SHOW YOUR HAND"
            status_color = MUTED

        cv2.putText(
            canvas, status_text, (745, 490),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2, cv2.LINE_AA
        )

        # ----------------------------------------------------
        # 11. MESSAGE BUILDER PANEL
        # ----------------------------------------------------
        draw_panel(canvas, (40, 570), (1240, 685), PANEL, 0.90)
        draw_corner_frame(canvas, 40, 570, 1240, 685)

        cv2.putText(
            canvas, "YOUR MESSAGE", (62, 610),
            cv2.FONT_HERSHEY_DUPLEX, 0.55, CYAN, 1, cv2.LINE_AA
        )

        # Reduce the message font slightly if the message becomes long.
        message_font_scale = 1.15
        while current_word and cv2.getTextSize(
            current_word, cv2.FONT_HERSHEY_DUPLEX, message_font_scale, 2
        )[0][0] > 1120:
            message_font_scale -= 0.05

        cv2.putText(
            canvas, current_word, (62, 660),
            cv2.FONT_HERSHEY_DUPLEX, message_font_scale, WHITE, 2, cv2.LINE_AA
        )

        instructions = "ENTER: ADD   |   SPACE: BACKSPACE   |   V: SPEAK   |   C: CLEAR   |   Q: QUIT"
        cv2.putText(
            canvas, instructions, (390, 710),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA
        )

        cv2.imshow("Hear My Hands - ASL Recognition", canvas)

        # ----------------------------------------------------
        # 12. KEYBOARD CONTROLS
        # ----------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == 13:  # ENTER: add the current prediction
            if current_letter and confidence > 50:
                current_word += current_letter + " "
                print(f"ADDED: {current_letter} | Current Message: {current_word.strip()}")

        elif key == 32:  # SPACE: remove the last added letter
            current_word = remove_last_letter(current_word)
            print(f"BACKSPACE | Current Message: {current_word}")

        elif key == ord("c"):  # CLEAR: clear the full message
            current_word = ""
            print("CLEARED MESSAGE")

        elif key == ord("v"):  # SPEAK: speak the completed message
            if current_word.strip():
                speak_text(current_word)
            else:
                print("Cannot speak: Message is empty.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
