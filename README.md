# 🀄 普通話 TTS + STT Web App

Free Mandarin Chinese Text-to-Speech and Speech-to-Text web app.

## Features

- 🔊 **TTS** — 7 Mandarin neural voices (大陸 + 台灣) via Microsoft Edge TTS
- 🎙️ **STT** — Record or upload audio, transcribed locally via OpenAI Whisper
- 📥 Download generated MP3
- 📋 Copy transcript
- 🌙 Dark UI, no login, no API keys required

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5055

## Tech Stack

| Component | Tool |
|-----------|------|
| TTS | `edge-tts` (Microsoft Neural, free) |
| STT | `openai-whisper` (local, free) |
| Server | Flask |
| Frontend | Vanilla JS, no dependencies |
