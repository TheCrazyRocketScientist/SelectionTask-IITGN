import asyncio
import websockets
import csv
import os
from datetime import datetime

ESP32_WS_URL = "ws://192.168.28.219/ws"  # Change this to your ESP32 WebSocket URL
CSV_FILE = "imu_data.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "X", "Y", "Z", "Tap", "Double_Tap"])

async def websocket_client():
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
                                x, y, z, tap, double_tap = map(int, data)

                                # Log data to CSV
                                with open(CSV_FILE, mode="a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow([datetime.now().isoformat(), x, y, z, tap, double_tap])

                                print(f"X: {x}, Y: {y}, Z: {z}, Tap: {tap}, Double Tap: {double_tap}")
                            except ValueError:
                                print(f"Invalid data format! Message: {message}")
                        else:
                            print(f"Unexpected data length! Message: {message}")

                        await ws.send("0")  # Send acknowledgment
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
