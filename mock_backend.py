import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

@app.post("/warmup")
async def warmup():
    return {"status": "warm"}

@app.post("/voice")
async def voice(data: dict):
    # Simulate a short delay for processing
    await asyncio.sleep(0.1)
    return {"response": "This is a mock response from Daraz AI. Electronics can be returned in 7 days."}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Simulate TTFT delay
            await asyncio.sleep(0.05)
            
            tokens = ["Hello", "!", " I", " am", " a", " mock", " bot", "."]
            for token in tokens:
                await websocket.send_json({"token": token})
                await asyncio.sleep(0.02) # Inter-token latency
            
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
