# ---------------------------------------------------------------------------
# One Colab cell: copy models to local disk (in the background), start the
# F5PdfStudio bridge, open the Cloudflare tunnel, then launch ComfyUI.
#
# Before the first run:
#   * put colab_ltx23_comfyui_bridge.py somewhere on Drive and set BRIDGE_SCRIPT below
#   * (optional, fixed URLs) Colab secrets: CF_TUNNEL_TOKEN (+ CF_BRIDGE_URL, CF_COMFY_URL for display) - see README
# ---------------------------------------------------------------------------
import os, shutil, subprocess, threading, time, socket
from concurrent.futures import ThreadPoolExecutor

BRIDGE_SCRIPT = "/content/drive/MyDrive/ComfyUI-cloud/colab_ltx23_comfyui_bridge.py"
SRC = "/content/drive/MyDrive/SharedFolder/models"          # models on Drive
WORKSPACE = os.environ.get("WORKSPACE", "/content/ComfyUI")  # where ComfyUI lives
DST = f"{WORKSPACE}/models"
BRIDGE_PORT, COMFY_PORT = 8190, 8188

os.environ["MPLBACKEND"] = "Agg"   # notebook's inline backend is not in the ComfyUI venv

# ---- 1. dependencies for the bridge (system python) and cloudflared ----
subprocess.run("pip install -q fastapi uvicorn python-multipart requests", shell=True, check=False)
if shutil.which("cloudflared") is None:
    subprocess.run("wget -q -O /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
                   " && dpkg -i /tmp/cloudflared.deb > /dev/null", shell=True, check=False)

# ---- 2. models: symlinks to Drive now, real local copies as they finish ----
# (folders to look in, filename) - biggest first. Edit to match your workflows.
MODELS = [
    (["diffusion_models", "unet"], "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    (["text_encoders"],            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    (["latent_upscale_models"],    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    (["loras"],                    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
    (["vae"],                      "ltx-2.5-video-vae-bf16.safetensors"),
    (["vae"],                      "ltx-2.5-audio-vae-bf16.safetensors"),
]
jobs = []
for folders, fname in MODELS:
    for f in folders:
        if os.path.exists(f"{SRC}/{f}/{fname}"):
            jobs.append((f, fname)); break
    else:
        print("WARNING: not found on Drive:", fname)
local_folders = {f for f, _ in jobs}

if os.path.islink(DST):
    os.unlink(DST)                      # removes only the link, never the Drive files
os.makedirs(DST, exist_ok=True)
for name in os.listdir(SRC):
    d = f"{DST}/{name}"
    if name in local_folders:
        os.makedirs(d, exist_ok=True)
        for item in os.listdir(f"{SRC}/{name}"):
            if not os.path.lexists(f"{d}/{item}"):
                os.symlink(f"{SRC}/{name}/{item}", f"{d}/{item}")
    elif not os.path.lexists(d):
        os.symlink(f"{SRC}/{name}", d)

def copy_one(job):
    folder, fname = job
    src, dst = f"{SRC}/{folder}/{fname}", f"{DST}/{folder}/{fname}"
    if os.path.lexists(dst) and not os.path.islink(dst):
        return
    t = time.time()
    shutil.copyfile(src, dst + ".part")
    os.replace(dst + ".part", dst)      # atomic swap: symlink -> real local file
    print(f"copied {fname} ({time.time()-t:.0f}s)")

def copy_all():
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(copy_one, jobs))
    print("\n=== ALL MODELS COPIED TO LOCAL DISK: generation will now load fast ===\n")

threading.Thread(target=copy_all, daemon=True).start()

# ---- 3. bridge (talks to ComfyUI on localhost) ----
bridge_env = dict(os.environ, COMFYUI_DIR=WORKSPACE, COMFYUI_URL=f"http://127.0.0.1:{COMFY_PORT}",
                  BRIDGE_PORT=str(BRIDGE_PORT))
bridge_log = open("/content/ltx_bridge.log", "a")
bridge = subprocess.Popen(["python", "-u", BRIDGE_SCRIPT], env=bridge_env, stdout=bridge_log,
                          stderr=subprocess.STDOUT, start_new_session=True)
print("Bridge PID:", bridge.pid, "(log: /content/ltx_bridge.log)")

# ---- 4. tunnels: one URL for the bridge (F5PdfStudio), one for the ComfyUI browser UI ----
import re

TUNNELS = {"bridge": BRIDGE_PORT, "comfy": COMFY_PORT}
URLS = {}

def _print_urls():
    if len(URLS) == len(TUNNELS):
        print("\n" + "=" * 64)
        print("  F5PdfStudio  ->  LTX bridge URL :", URLS["bridge"])
        print("  ComfyUI web UI                   :", URLS["comfy"])
        print("  (ComfyUI URL shows a 502 page until ComfyUI has finished loading)")
        print("=" * 64 + "\n")

def quick_tunnel(name, port):
    """Temporary trycloudflare.com tunnel for one local port; retries if Cloudflare fails once."""
    for attempt in range(1, 4):
        p = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        found = False
        for raw in p.stderr:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", raw.decode(errors="ignore"))
            if m:
                URLS[name] = m.group(0)
                found = True
                _print_urls()
                break
        if found:
            for _ in p.stderr:      # keep draining so the process never blocks
                pass
            return
        p.kill()
        print(f"[{name}] tunnel attempt {attempt} failed, retrying...")
        time.sleep(3)
    print(f"[{name}] could not create a tunnel. Re-run the cell.")

token = ""
try:
    from google.colab import userdata
    token = userdata.get("CF_TUNNEL_TOKEN") or ""
except Exception:
    pass

if token:
    # Named tunnel: fixed URLs. Dashboard routes (Zero Trust -> Networks -> Tunnels -> your tunnel):
    #   bridge.yourdomain.com -> http://localhost:8190   (F5PdfStudio)
    #   comfy.yourdomain.com  -> http://localhost:8188   (ComfyUI browser UI)
    subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", token],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for name, secret in (("bridge", "CF_BRIDGE_URL"), ("comfy", "CF_COMFY_URL")):  # optional secrets, just for display
        try:
            URLS[name] = userdata.get(secret)
        except Exception:
            URLS[name] = f"(your {name} hostname from the Cloudflare dashboard)"
    _print_urls()
else:
    print("No CF_TUNNEL_TOKEN secret: creating temporary tunnels (URLs change every session)...")
    for name, port in TUNNELS.items():
        threading.Thread(target=quick_tunnel, args=(name, port), daemon=True).start()

# ---- 5. ComfyUI (blocks; keep this cell running) ----
os.chdir(WORKSPACE)
get_ipython().system(  # noqa: F821  (available inside Colab/Jupyter)
    f"venv.sh python main.py --cache-none --mmap-torch-files --disable-pinned-memory --dont-print-server "
    f"--enable-dynamic-vram --listen 0.0.0.0 --port {COMFY_PORT} --enable-cors-header 'http://127.0.0.1:{COMFY_PORT}'"
)
