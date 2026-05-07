import cv2
import numpy as np
import base64
import json
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


@app.get("/api/languages")
def list_languages():
    return {
        "languages": [
            {"code": "en", "label": "English"},
            {"code": "hi", "label": "Hindi"},
            {"code": "es", "label": "Spanish"},
        ]
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            language = "en"
            data = raw

            # Supports either plain base64 string or JSON message
            if raw.startswith("{"):
                parsed = json.loads(raw)
                data = parsed.get("frame", "")
                language = parsed.get("language", "en")
            
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
            prediction = model.predict(lip_data, language=language, frame=frame)
            
            # Send results back
            response = {
                "gesture": prediction["gesture"],
                "hand_gesture": lip_data.get("hand_gesture", "None"),
                "facial_gesture": lip_data.get("facial_gesture", "Neutral"),
                "predicted_text": prediction["predicted_text"],
                "partial_text": prediction["partial_text"],
                "confidence": prediction["confidence"],
                "phrase_id": prediction["phrase_id"],
                "is_final": prediction["is_final"],
                "new_word": prediction["new_word"],
                "landmarks": lip_data["landmarks"],
                "mar": lip_data["mar"],
                "spread": lip_data["spread"],
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
