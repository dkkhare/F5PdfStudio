"""Start the single Colab media stack.

This script keeps all remote models in the same Colab runtime. It launches
Fish Speech on localhost:8080 and the F5PdfStudio wrapper on localhost:7861.
Use a Cloudflare/ngrok tunnel for port 7861 only.
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path
import requests

ROOT = Path(os.getenv("FISH_ROOT", "/content/fish-speech"))
FISH_PYTHON = os.getenv("FISH_PYTHON", str(ROOT / ".venv/bin/python"))
FISH_LLAMA = os.getenv("FISH_LLAMA_CHECKPOINT", "/content/fish-speech/checkpoints/s2-pro")
FISH_DECODER = os.getenv("FISH_DECODER_CHECKPOINT", "/content/fish-speech/checkpoints/s2-pro/codec.pth")
FISH_PORT = int(os.getenv("FISH_PORT", "8080"))
WRAPPER_PORT = int(os.getenv("F5PDFSTUDIO_PORT", "7861"))

fish_cmd = [
    FISH_PYTHON if Path(FISH_PYTHON).is_file() else "python",
    str(ROOT / "tools/api_server.py"),
    "--llama-checkpoint-path", FISH_LLAMA,
    "--decoder-checkpoint-path", FISH_DECODER,
    "--listen", f"127.0.0.1:{FISH_PORT}",
    "--half",
]
if os.getenv("FISH_LAZY_LOAD", "1") == "1":
    fish_cmd.append("--lazy-load")

fish = subprocess.Popen(fish_cmd, cwd=str(ROOT))
os.environ["FISH_API_URL"] = f"http://127.0.0.1:{FISH_PORT}"
os.environ["FISH_PID"] = str(fish.pid)

for _ in range(180):
    try:
        if requests.get(f"http://127.0.0.1:{FISH_PORT}/v1/health", timeout=2).ok:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    fish.terminate()
    raise RuntimeError("Fish Speech server did not become healthy.")

wrapper_cmd = ["uvicorn", "colab_media_server:app", "--host", "0.0.0.0", "--port", str(WRAPPER_PORT)]
print("Fish Speech:", fish_cmd)
print("F5PdfStudio media wrapper:", wrapper_cmd)
subprocess.run(wrapper_cmd, cwd=str(Path(__file__).resolve().parent), check=True)
