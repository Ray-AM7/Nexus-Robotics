import numpy as np
import tensorflow as tf
import time
import os
import random
import argparse
import serial


# === CONFIGURATION ===
MODEL_PATH = "sEMG1DCNNlstm7.4.h5"
SAMPLE_RATE = 1024
BUFFER_TIME = 0.2
NUM_CHANNELS = 8
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_TIME)
PREMADE_DIR = "saved_samples/"  # directory with labeled .npy gesture files

# Open serial connection to Arduino (adjust COM port as needed)
arduino = serial.Serial('COM5', 9600)
time.sleep(2)


gesture_map = {
    0: "Fist/Lateral Prehension", 1: "3 Fingers/Thumb and Little Finger Opposition", 2: "Thumb and Little Finger Extension", 3: "Little Finger Extension",
    4: "Index Extension", 5: "Thumb Extension", 6: "Wrist Flexion", 7: "Wrist Extension",
    8: "Forearm Supination", 9: "Forearm Pronation", 10: "Palm All fingers Flexed Out", 11: "Rest"
}

# === LOAD MODEL ===
print("[INFO] Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# === INFERENCE FUNCTION ===
def predict_gesture(input_buffer):
    input_buffer = np.expand_dims(input_buffer, axis=0)  # shape (1, time, channels)
    start = time.time()
    predictions = model.predict(input_buffer, verbose=0)[0]
    end = time.time()

    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])
    gesture = gesture_map.get(predicted_class, f"Unknown({predicted_class})")

    return {
        "gesture": gesture,
        "confidence": confidence,
        "inference_time_ms": round((end - start) * 1000, 2),
        "number": predicted_class
    }

# === PREMADE SAMPLE HANDLING ===
def list_premade_samples():
    return [f for f in os.listdir(PREMADE_DIR) if f.endswith(".npy")]

def load_random_premade_sample():
    files = list_premade_samples()
    if not files:
        raise FileNotFoundError("No premade gesture samples found.")
    chosen = random.choice(files)
    data = np.load(os.path.join(PREMADE_DIR, chosen))
    return data, chosen

# === SIMULATED LIVE DATA ===
def simulate_live_data():
    return np.random.normal(0, 0.1, size=(BUFFER_SIZE, NUM_CHANNELS))

# === MAIN LOOP ===
def run_cli(mode):
    print("[INFO] EMG Gesture Recognition CLI - Press Ctrl+C to stop.")
    print(f"[INFO] Mode: {'Live Samples' if mode == 'live_samples' else 'Live Simulated'}\n")

    while True:
        try:
            if mode == "live_samples":
                sample, file = load_random_premade_sample()
            else:
                sample = simulate_live_data()
                file = None

            result = predict_gesture(sample)
            print(f"{'[LIVE SAMPLES]' if file else '[LIVE SENSORS]'} {file if file else ''}")
            print(f"  → Gesture: {result['gesture']}")
            print(f"  → Confidence: {result['confidence']:.2f}")
            print(f"  → Inference Time: {result['inference_time_ms']} ms\n")
            print(f"  → Arduino Gesture Sequence #: {(result['number']+1)}\n")

            gesture_id = result["number"]

            # Convert gesture_id (0–11) to char: 0–9 → '0'–'9', 10 → 'A', 11 → 'B'
            if gesture_id < 10:
                command_char = chr(ord('0') + gesture_id)
            elif gesture_id == 10:
                command_char = 'A'
            elif gesture_id == 11:
                command_char = 'B'
            else:
                command_char = None

            if command_char:
                arduino.write(command_char.encode())
                print(f"[SERIAL] Sent gesture ID {gesture_id} → '{command_char}' to Arduino")
            else:
                print("[SERIAL] Invalid gesture ID.")


            time.sleep(1)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            break

# === CLI ARGUMENTS ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EMG Gesture Recognition CLI")
    parser.add_argument("--live_samples", action="store_true", help="Use sampled .npy gesture files")
    parser.add_argument("--live_sensors", action="store_true", help="Use live sensor data")

    args = parser.parse_args()

    if args.live_samples:
        run_cli("live_samples")
    elif args.live_sensors:
        run_cli("live_sensors")
    else:
        print("[ERROR] Please use --live_samples or --live_sensors")
