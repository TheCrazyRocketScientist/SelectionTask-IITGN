import asyncio
import websockets
import pickle
import numpy as np
from datetime import datetime
from collections import deque
from feature_extraction import extract_features  # Import feature extraction functions

ESP32_WS_URL = "ws://192.168.119.219/ws"  # Change this to your ESP32 WebSocket URL
MODEL_FILE = "saved_models\\XGBoost_best_model.pkl"
WINDOW_SIZE = 50  # Number of observations in each feature extraction window
STEP = 25  # Step size for overlapping windows
BUFFER_SIZE = 2 * WINDOW_SIZE  # Maintain 2x window size in the buffer

# Load the pre-trained model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Initialize a circular buffer for sensor readings
data_buffer = deque(maxlen=BUFFER_SIZE)


async def websocket_client():
    global data_buffer
    while True:
        try:
            print("Connecting to ESP32 WebSocket...")
            async with websockets.connect(ESP32_WS_URL) as ws:
                print("Connected to ESP32 WebSocket!")

                while True:
                    try:
                        message = await ws.recv()
                        data = message.strip().split(",")

                        if len(data) == 5:
                            try:
                                # Parse sensor readings
                                clean_data = [int(val.replace(",", "")) for val in data]
                                x, y, z, tap, double_tap = clean_data  # Ignore tap & double_tap

                                # Store only X, Y, Z in the buffer
                                data_buffer.append([x, y, z])

                                # Ensure buffer has at least 2 * WINDOW_SIZE before processing
                                if len(data_buffer) >= 2 * WINDOW_SIZE:
                                    # Extract the last WINDOW_SIZE samples
                                    window_data = np.array(list(data_buffer)[-WINDOW_SIZE:])

                                    # Extract features
                                    feature_vector = extract_features(window_data)
                                    feature_array = np.array(list(feature_vector.values())).reshape(1, -1)

                                    # Predict using the model
                                    prediction = model.predict(feature_array)[0]
                                    print(prediction)

                                    # Send prediction result to WebSocket
                                    await ws.send(str(prediction))
                                    print(f"Prediction sent: {prediction}")

                                    # Maintain buffer history (remove only STEP elements)
                                    for _ in range(STEP):
                                        if data_buffer:
                                            data_buffer.popleft()

                                # Send acknowledgment to ESP32
                                await ws.send("ACK")

                            except ValueError as e:
                                print(f"Invalid data format! Error: {e}, Message: {message}")

                        else:
                            print(f"Unexpected data length! Message: {message}")

                        # Small delay to prevent overwhelming the WebSocket
                        await asyncio.sleep(0.01)

                    except websockets.ConnectionClosed:
                        print("WebSocket connection closed. Reconnecting...")
                        break
                    except Exception as e:
                        print(f"Error receiving data: {e}")

        except (websockets.exceptions.WebSocketException, OSError) as e:
            print(f"WebSocket error: {e}. Retrying in 3 seconds...")
            await asyncio.sleep(3)


if __name__ == "__main__":
    try:
        asyncio.run(websocket_client())
    except KeyboardInterrupt:
        print("Script interrupted. Exiting...")
