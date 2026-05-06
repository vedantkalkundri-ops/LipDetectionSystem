import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from lip_tracker import LipTracker
from speech_model import SpeechModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = LipTracker()
model = SpeechModel()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            
            # Remove header if present (e.g. data:image/jpeg;base64,...)
            if "," in data:
                data = data.split(",")[1]
                
            # Decode image
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Process frame
            lip_data = tracker.process_frame(frame)
            
            # Predict gesture and text
            gesture, predicted_text = model.predict(lip_data)
            
            # Send results back
            response = {
                "gesture": gesture,
                "predicted_text": predicted_text,
                "landmarks": lip_data["landmarks"],
                "mar": lip_data["mar"]
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
