import cv2
import tensorflow as tf
import numpy as np
import random
import time

# ============================================================
# 1. Load Model
# ============================================================

MODEL_PATH = "rock_paper_scissors_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["rock", "paper", "scissors"]

print("✅ Model loaded successfully!")


# ============================================================
# 2. Game Settings
# ============================================================

MOVES = ["rock", "paper", "scissors"]

score_you = 0
score_ai = 0
draws = 0

round_number = 0


# ============================================================
# 3. Game Logic
# ============================================================

def determine_winner(player, ai):

    if player == ai:
        return "draw"

    if (
        (player == "rock" and ai == "scissors") or
        (player == "paper" and ai == "rock") or
        (player == "scissors" and ai == "paper")
    ):
        return "you"

    return "ai"


# ============================================================
# 4. Open DroidCam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open DroidCam")
    exit()

print("✅ DroidCam connected!")
print("Press Q to quit.")


# ============================================================
# 5. Game Variables
# ============================================================

game_state = "waiting"

countdown_start = None
countdown_duration = 3

ai_move = None
player_move = None
winner = None

result_start_time = None
result_duration = 3


# ============================================================
# 6. Main Loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Center ROI
    # --------------------------------------------------------

    box_size = min(height, width) // 2

    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    img = cv2.resize(roi, (128, 128))

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    confidence = predictions[predicted_index]

    detected_move = class_names[predicted_index]


    # --------------------------------------------------------
    # Draw ROI
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # ========================================================
    # WAITING STATE
    # ========================================================

    if game_state == "waiting":

        cv2.putText(
            frame,
            "GET READY!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to start",
            (30, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # COUNTDOWN STATE
    # ========================================================

    elif game_state == "countdown":

        elapsed = time.time() - countdown_start

        remaining = countdown_duration - int(elapsed)

        if remaining > 0:

            cv2.putText(
                frame,
                str(remaining),
                (width // 2 - 30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 255),
                5
            )

            cv2.putText(
                frame,
                "Show your move!",
                (width // 2 - 130, height // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:

            # Take player's final move
            player_move = detected_move

            # AI chooses move
            ai_move = random.choice(MOVES)

            # Determine winner
            winner = determine_winner(
                player_move,
                ai_move
            )

            # Update score
            if winner == "you":
                score_you += 1

            elif winner == "ai":
                score_ai += 1

            else:
                draws += 1

            round_number += 1

            result_start_time = time.time()

            game_state = "result"


    # ========================================================
    # RESULT STATE
    # ========================================================

    elif game_state == "result":

        # ----------------------------------------------------
        # Display Player Move
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"YOU: {player_move.upper()}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Display AI Move
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"AI: {ai_move.upper()}",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Winner
        # ----------------------------------------------------

        if winner == "you":

            result_text = "YOU WIN!"

        elif winner == "ai":

            result_text = "AI WINS!"

        else:

            result_text = "DRAW!"


        cv2.putText(
            frame,
            result_text,
            (width // 2 - 150, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3
        )


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"You: {score_you}",
            (30, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"AI: {score_ai}",
            (180, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Draw: {draws}",
            (320, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Start Next Round
        # ----------------------------------------------------

        if time.time() - result_start_time >= result_duration:

            game_state = "waiting"


    # ========================================================
    # Always Display Prediction Confidence
    # ========================================================

    confidence_text = (
        f"Detected: {detected_move.upper()} "
        f"({confidence * 100:.1f}%)"
    )

    cv2.putText(
        frame,
        confidence_text,
        (30, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ========================================================
    # Display Round
    # ========================================================

    cv2.putText(
        frame,
        f"Round: {round_number}",
        (width - 180, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # Show Frame
    # ========================================================

    cv2.imshow(
        "Rock Paper Scissors AI Game",
        frame
    )


    # ========================================================
    # Keyboard Controls
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # SPACE → Start Round
    if key == ord(" ") and game_state == "waiting":

        countdown_start = time.time()

        game_state = "countdown"


    # Q → Quit
    elif key == ord("q"):

        break


# ============================================================
# 7. Release Resources
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n==============================")
print("GAME OVER")
print("==============================")
print(f"You:   {score_you}")
print(f"AI:    {score_ai}")
print(f"Draws: {draws}")