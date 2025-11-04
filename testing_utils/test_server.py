from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import random
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/inference")
async def inference(audio: UploadFile = File(...),
                    chunkNumber:int = Form(...),
                    timestamp:int = Form(...),):
    try:
        audio_bytes = await audio.read()

        print(f"Received audio chunk #{chunkNumber}")
        print(f"Filename: {audio.filename}")
        print(f"Content-Type: {audio.content_type}")
        print(f"Size: {len(audio_bytes)} bytes")
        print(f"Timestamp: {timestamp}")

        is_ai = random.random() > 0.5,
        return {
            "decision": "AI" if is_ai else "Not AI",
            "confidence": random.random(),
            "timestamp": timestamp,
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)