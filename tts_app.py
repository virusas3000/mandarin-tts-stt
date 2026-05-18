#!/usr/bin/env python3
"""
Free TTS App — powered by Microsoft Edge TTS (free, no API key needed)
Usage: python3 tts_app.py
"""

import asyncio
import os
import sys
import edge_tts

# Available voices (a curated selection)
VOICES = {
    "1": ("en-US-AriaNeural",    "English (US) - Aria (Female)"),
    "2": ("en-US-GuyNeural",     "English (US) - Guy (Male)"),
    "3": ("en-GB-SoniaNeural",   "English (UK) - Sonia (Female)"),
    "4": ("en-AU-NatashaNeural", "English (AU) - Natasha (Female)"),
    "5": ("zh-HK-HiuGaaiNeural", "Cantonese (HK) - HiuGaai (Female)"),
    "6": ("zh-HK-WanLungNeural", "Cantonese (HK) - WanLung (Male)"),
    "7": ("zh-CN-XiaoxiaoNeural","Mandarin (CN) - Xiaoxiao (Female)"),
    "8": ("ja-JP-NanamiNeural",  "Japanese - Nanami (Female)"),
    "9": ("ko-KR-SunHiNeural",   "Korean - SunHi (Female)"),
}

async def synthesize(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def main():
    print("=" * 50)
    print("       🎙️  Free TTS App (Edge TTS)")
    print("=" * 50)

    # Pick voice
    print("\nAvailable voices:")
    for k, (_, label) in VOICES.items():
        print(f"  [{k}] {label}")
    print("  [l] List ALL voices")

    choice = input("\nSelect voice (1-9) or press Enter for default [1]: ").strip()

    if choice == "l":
        asyncio.run(list_all_voices())
        return

    voice_id, voice_label = VOICES.get(choice, VOICES["1"])
    print(f"✅ Voice: {voice_label}")

    # Input text
    print("\nEnter text to convert (type END on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    text = "\n".join(lines).strip()

    if not text:
        print("❌ No text entered. Exiting.")
        sys.exit(1)

    # Output file
    default_out = "output.mp3"
    out = input(f"\nOutput filename (default: {default_out}): ").strip()
    if not out:
        out = default_out
    if not out.endswith(".mp3"):
        out += ".mp3"

    output_path = os.path.abspath(out)

    print(f"\n⏳ Generating audio...")
    asyncio.run(synthesize(text, voice_id, output_path))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Done! Saved to: {output_path} ({size_kb:.1f} KB)")
    print(f"\n▶️  Play with: open \"{output_path}\"  (macOS)")

async def list_all_voices():
    print("\nFetching all available voices...")
    voices = await edge_tts.list_voices()
    for v in voices:
        print(f"  {v['ShortName']:40s}  {v['Gender']:7s}  {v['Locale']}")

if __name__ == "__main__":
    main()
