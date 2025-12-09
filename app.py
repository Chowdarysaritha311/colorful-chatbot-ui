from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import ollama
import pyttsx3
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "phi3.5:3.8b"

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico")
    return {}

@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    model = data.get("model", MODEL)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": user_message}]
    )
    bot_reply = response["message"]["content"]

    engine = pyttsx3.init()
    engine.say(bot_reply)
    engine.runAndWait()

    return {"reply": bot_reply}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
