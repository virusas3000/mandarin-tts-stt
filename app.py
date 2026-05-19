#!/usr/bin/env python3
"""
Mandarin TTS + STT Web App
- TTS: Edge TTS (free, neural voices)
- STT: OpenAI Whisper (local, free)
"""

import asyncio, os, uuid, threading, re
from flask import Flask, request, jsonify, send_file, render_template_string
import edge_tts
import whisper
from gtts import gTTS
import azure.cognitiveservices.speech as speechsdk

AZURE_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastasia")

app = Flask(__name__)
AUDIO_DIR = "/tmp/tts_app/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Job queue for async TTS
jobs = {}  # job_id -> {"status": "pending|done|error", "url": ..., "error": ...}

# Load Whisper model once at startup (base is fast, use "small" for better accuracy)
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper ready.")

VOICES = [
    {"id": "edge:zh-CN-XiaoxiaoNeural",  "label": "Edge · 晓晓 Xiaoxiao (女)"},
    {"id": "edge:zh-CN-YunxiNeural",     "label": "Edge · 云希 Yunxi (男)"},
    {"id": "edge:zh-CN-XiaoyiNeural",    "label": "Edge · 晓伊 Xiaoyi (女)"},
    {"id": "edge:zh-CN-YunjianNeural",   "label": "Edge · 云健 Yunjian (男)"},
    {"id": "edge:zh-CN-XiaochenNeural",  "label": "Edge · 晓辰 Xiaochen (女)"},
    {"id": "edge:zh-TW-HsiaoChenNeural", "label": "Edge · 曉臻 台灣 (女)"},
    {"id": "edge:zh-TW-YunJheNeural",    "label": "Edge · 雲哲 台灣 (男)"},
    {"id": "gtts:zh-CN", "label": "Google TTS · 普通話 (女)"},
    {"id": "gtts:zh-TW", "label": "Google TTS · 台灣普通話 (女)"},
    {"id": "azure:zh-CN-YunxiNeural",     "label": "Azure · 云希 Yunxi (男) ⭐"},
    {"id": "azure:zh-CN-XiaoxiaoNeural",  "label": "Azure · 晓晓 Xiaoxiao (女) ⭐"},
    {"id": "azure:zh-CN-YunjianNeural",   "label": "Azure · 云健 Yunjian (男)"},
    {"id": "azure:zh-CN-XiaoyiNeural",    "label": "Azure · 晓伊 Xiaoyi (女)"},
    {"id": "azure:zh-TW-YunJheNeural",    "label": "Azure · 雲哲 YunJhe (男 台灣)"},
    {"id": "azure:zh-TW-HsiaoChenNeural", "label": "Azure · 曉臻 HsiaoChen (女 台灣)"},
]

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>普通話 TTS + STT</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #0f0f14; color: #e8e8e8; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 32px 20px; }
  h1 { text-align: center; font-size: 1.8rem; margin-bottom: 4px; color: #fff; }
  .subtitle { text-align: center; color: #888; margin-bottom: 32px; font-size: 0.9rem; }
  .card { background: #1a1a24; border: 1px solid #2a2a38; border-radius: 16px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 1rem; color: #a78bfa; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  select, textarea { width: 100%; background: #0f0f14; border: 1px solid #2a2a38; border-radius: 10px; color: #e8e8e8; padding: 12px; font-size: 1rem; font-family: inherit; outline: none; transition: border 0.2s; }
  select:focus, textarea:focus { border-color: #a78bfa; }
  textarea { resize: vertical; min-height: 120px; line-height: 1.6; }
  .row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
  button { flex: 1; min-width: 120px; padding: 12px 20px; border: none; border-radius: 10px; font-size: 1rem; cursor: pointer; font-family: inherit; font-weight: 600; transition: opacity 0.2s, transform 0.1s; }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: linear-gradient(135deg, #7c3aed, #a78bfa); color: #fff; }
  .btn-secondary { background: #2a2a38; color: #e8e8e8; }
  .btn-danger { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; }
  .btn-success { background: linear-gradient(135deg, #059669, #34d399); color: #fff; }
  audio { width: 100%; margin-top: 14px; border-radius: 8px; }
  .status { margin-top: 12px; font-size: 0.88rem; color: #888; min-height: 20px; }
  .status.ok { color: #34d399; }
  .status.err { color: #f87171; }
  .status.loading { color: #a78bfa; }
  .transcript-box { background: #0f0f14; border: 1px solid #2a2a38; border-radius: 10px; padding: 14px; margin-top: 14px; min-height: 60px; line-height: 1.7; font-size: 1rem; white-space: pre-wrap; color: #e8e8e8; }
  .download-link { display: inline-block; margin-top: 10px; color: #a78bfa; font-size: 0.88rem; text-decoration: none; }
  .download-link:hover { text-decoration: underline; }
  .rec-dot { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; display: inline-block; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .char-count { text-align: right; font-size: 0.78rem; color: #555; margin-top: 4px; }
</style>
</head>
<body>
<div class="container">
  <h1>🀄 普通話 TTS + STT</h1>
  <p class="subtitle">文字轉語音 &amp; 語音轉文字 · Mandarin Chinese</p>

  <!-- TTS Card -->
  <div class="card">
    <h2>🔊 文字轉語音 Text → Speech</h2>
    <select id="voice">
      {% for v in voices %}
      <option value="{{ v.id }}">{{ v.label }}</option>
      {% endfor %}
    </select>
    <div style="margin:10px 0 4px; display:flex; align-items:center; gap:12px">
      <label style="color:#aaa; font-size:0.88rem; white-space:nowrap">語速 Speed</label>
      <input type="range" id="speed" min="0.5" max="2.0" step="0.05" value="1.0" style="flex:1; accent-color:#c9a227">
      <span id="speed-val" style="color:#c9a227; font-size:0.88rem; width:36px; text-align:right">1.0×</span>
    </div>
    <textarea id="tts-text" placeholder="在這裡輸入普通話文字…&#10;Type Mandarin text here..."></textarea>
    <div class="row">
      <button class="btn-primary" onclick="doTTS()">🔊 生成語音</button>
      <button class="btn-secondary" onclick="document.getElementById('tts-text').value=''">清除</button>
    </div>
    <div class="status" id="tts-status"></div>
    <audio id="tts-audio" controls style="display:none"></audio>
    <a id="tts-download" class="download-link" style="display:none" download>⬇ 下載 MP3</a>
  </div>

  <!-- STT Card -->
  <div class="card">
    <h2>🎙️ 語音轉文字 Speech → Text</h2>
    <p style="color:#888; font-size:0.88rem; margin-bottom:14px">上傳音頻文件，或直接錄音。模型：Whisper（本地免費）</p>

    <div style="margin-bottom:12px">
      <input type="file" id="audio-file" accept="audio/*" style="color:#aaa; font-size:0.88rem" onchange="fileSelected()">
    </div>

    <div class="row">
      <button class="btn-danger" id="rec-btn" onclick="toggleRecord()">🎙️ 開始錄音</button>
      <button class="btn-success" onclick="doSTT()" id="stt-btn">📝 轉錄文字</button>
    </div>
    <div class="status" id="stt-status"></div>
    <div class="transcript-box" id="transcript" style="display:none"></div>
    <button class="btn-secondary" id="copy-btn" style="display:none; margin-top:10px; flex:none; width:auto" onclick="copyTranscript()">📋 複製文字</button>
  </div>
</div>

<script>
let mediaRecorder, audioChunks = [], recordedBlob = null, isRecording = false;

// Speed slider
document.getElementById('speed').addEventListener('input', function() {
  document.getElementById('speed-val').textContent = parseFloat(this.value).toFixed(2).replace(/\.?0+$/, '') + '×';
});

async function doTTS() {
  const text = document.getElementById('tts-text').value.trim();
  if (!text) { setStatus('tts', '請輸入文字', 'err'); return; }
  const voice = document.getElementById('voice').value;
  const speed = parseFloat(document.getElementById('speed').value);
  setStatus('tts', '⏳ 生成中，長文本需要稍等...', 'loading');
  document.querySelector('.btn-primary').disabled = true;
  try {
    // Submit job
    const r = await fetch('/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, voice, speed})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    const jobId = d.job_id;

    // Poll until done
    let url = null;
    for (let i = 0; i < 300; i++) {
      await new Promise(res => setTimeout(res, 2000));
      const sr = await fetch('/tts/status/' + jobId);
      const sd = await sr.json();
      if (sd.status === 'done') { url = sd.url; break; }
      if (sd.status === 'error') throw new Error(sd.error);
      const elapsed = Math.round((i+1)*2);
      setStatus('tts', `⏳ 生成中... ${elapsed}s`, 'loading');
    }
    if (!url) throw new Error('Timeout waiting for audio');

    const audio = document.getElementById('tts-audio');
    audio.src = url + '?t=' + Date.now();
    audio.style.display = 'block';
    audio.play();
    const dl = document.getElementById('tts-download');
    dl.href = url;
    dl.style.display = 'inline-block';
    setStatus('tts', '✅ 完成！', 'ok');
  } catch(e) { setStatus('tts', '❌ ' + e.message, 'err'); }
  document.querySelector('.btn-primary').disabled = false;
}

async function toggleRecord() {
  if (!isRecording) {
    audioChunks = []; recordedBlob = null;
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(audioChunks, {type: 'audio/webm'});
      stream.getTracks().forEach(t => t.stop());
      setStatus('stt', '✅ 錄音完成，點擊「轉錄文字」', 'ok');
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('rec-btn').innerHTML = '<span class="rec-dot"></span> 停止錄音';
    document.getElementById('rec-btn').className = 'btn-secondary';
    setStatus('stt', '🔴 錄音中...', 'loading');
  } else {
    mediaRecorder.stop();
    isRecording = false;
    document.getElementById('rec-btn').innerHTML = '🎙️ 開始錄音';
    document.getElementById('rec-btn').className = 'btn-danger';
  }
}

function fileSelected() {
  recordedBlob = null;
  setStatus('stt', '📂 文件已選擇，點擊「轉錄文字」', 'ok');
}

async function doSTT() {
  const fileInput = document.getElementById('audio-file');
  let blob = recordedBlob || (fileInput.files[0] || null);
  if (!blob) { setStatus('stt', '請先錄音或選擇文件', 'err'); return; }
  setStatus('stt', '⏳ 轉錄中（Whisper）...', 'loading');
  document.getElementById('stt-btn').disabled = true;
  const fd = new FormData();
  fd.append('audio', blob, 'audio.webm');
  try {
    const r = await fetch('/stt', {method: 'POST', body: fd});
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    const box = document.getElementById('transcript');
    box.textContent = d.text;
    box.style.display = 'block';
    document.getElementById('copy-btn').style.display = 'inline-block';
    setStatus('stt', '✅ 轉錄完成', 'ok');
  } catch(e) { setStatus('stt', '❌ ' + e.message, 'err'); }
  document.getElementById('stt-btn').disabled = false;
}

function copyTranscript() {
  navigator.clipboard.writeText(document.getElementById('transcript').textContent);
  document.getElementById('copy-btn').textContent = '✅ 已複製';
  setTimeout(() => document.getElementById('copy-btn').textContent = '📋 複製文字', 1500);
}

function setStatus(id, msg, type) {
  const el = document.getElementById(id + '-status');
  el.textContent = msg;
  el.className = 'status ' + (type || '');
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, voices=VOICES)

def chunk_text(text, size=500):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[。！？\.\!\?])', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > size and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text]

async def save_chunk(chunk, voice, path, retries=3, rate="+0%"):
    for attempt in range(retries):
        try:
            await asyncio.wait_for(
                edge_tts.Communicate(chunk, voice, rate=rate).save(path),
                timeout=30
            )
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2)

async def edge_synthesize(text, voice, path, speed=1.0):
    rate = f"{int((speed - 1.0) * 100):+d}%"
    chunks = chunk_text(text)
    if len(chunks) == 1:
        await save_chunk(text, voice, path, rate=rate)
        return
    parts = []
    for i, chunk in enumerate(chunks):
        p = path + f".part{i}.mp3"
        await save_chunk(chunk, voice, p, rate=rate)
        parts.append(p)
    with open(path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
            os.remove(p)

def gtts_synthesize(text, lang_code, path):
    tld = "com.tw" if lang_code == "zh-TW" else "com"
    gTTS(text=text, lang="zh", tld=tld).save(path)

def azure_synthesize(text, voice, path, speed=1.0):
    import urllib.request, html as html_mod
    rate_pct = f"{int((speed - 1.0) * 100):+d}%"
    token_url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    # token fetched inside _call per chunk

    def _call(chunk):
        import urllib.request
        ssml = f"""<speak version='1.0' xml:lang='zh-CN'>
<voice name='{voice}'><prosody rate='{rate_pct}'>{html_mod.escape(chunk)}</prosody></voice>
</speak>"""
        token_req = urllib.request.Request(
            f"https://{AZURE_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
            method="POST",
            headers={"Ocp-Apim-Subscription-Key": AZURE_KEY, "Content-Length": "0"}
        )
        with urllib.request.urlopen(token_req) as resp:
            token = resp.read().decode()

        tts_req = urllib.request.Request(
            f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            data=ssml.encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3",
                "User-Agent": "mandarin-tts-app"
            }
        )
        with urllib.request.urlopen(tts_req, timeout=180) as resp:
            chunks_data = []
            while True:
                try:
                    buf = resp.read(65536)
                except Exception:
                    break
                if not buf:
                    break
                chunks_data.append(buf)
            return b"".join(chunks_data)

    # Split at paragraph breaks every ~3000 chars max
    chunks = chunk_text(text, size=3000)
    with open(path, "wb") as f:
        for chunk in chunks:
            f.write(_call(chunk))

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "").strip()
    voice = data.get("voice", "edge:zh-CN-XiaoxiaoNeural")
    speed = float(data.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))
    if not text:
        return jsonify({"error": "No text provided"})

    job_id = uuid.uuid4().hex
    fname = f"{job_id}.mp3"
    path = os.path.join(AUDIO_DIR, fname)
    jobs[job_id] = {"status": "pending"}

    engine, voice_id = voice.split(":", 1) if ":" in voice else ("edge", voice)

    def run():
        try:
            if engine == "gtts":
                gtts_synthesize(text, voice_id, path)
            elif engine == "azure":
                azure_synthesize(text, voice_id, path, speed=speed)
            else:
                try:
                    asyncio.run(edge_synthesize(text, voice_id, path, speed=speed))
                except Exception:
                    lang = "zh-TW" if "TW" in voice_id else "zh-CN"
                    gtts_synthesize(text, lang, path)
            jobs[job_id] = {"status": "done", "url": f"/audio/{fname}"}
        except Exception as e:
            jobs[job_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/tts/status/<job_id>")
def tts_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)

@app.route("/audio/<fname>")
def serve_audio(fname):
    path = os.path.join(AUDIO_DIR, fname)
    return send_file(path, mimetype="audio/mpeg")

@app.route("/stt", methods=["POST"])
def stt():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "No audio file"})
    fname = f"{uuid.uuid4().hex}.webm"
    path = os.path.join(AUDIO_DIR, fname)
    f.save(path)
    result = whisper_model.transcribe(path, language="zh")
    os.remove(path)
    return jsonify({"text": result["text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
