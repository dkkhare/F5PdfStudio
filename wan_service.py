from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

WAN_DEFAULT_ROOT = os.getenv("WAN22_ROOT", r"D:\Wan2.2")
WAN_DEFAULT_GENERATE = str(Path(WAN_DEFAULT_ROOT) / "generate.py")
WAN_DEFAULT_T2V_CKPT = os.getenv("WAN22_T2V_CKPT", str(Path(WAN_DEFAULT_ROOT) / "Wan2.2-T2V-A14B"))
WAN_DEFAULT_TI2V_CKPT = os.getenv("WAN22_TI2V_CKPT", str(Path(WAN_DEFAULT_ROOT) / "Wan2.2-TI2V-5B"))
WAN_DEFAULT_S2V_CKPT = os.getenv("WAN22_S2V_CKPT", str(Path(WAN_DEFAULT_ROOT) / "Wan2.2-S2V-14B"))


def _run_python_check(python_exe: str, code: str, timeout: int = 30):
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except Exception as exc:
        return -1, str(exc)


def check_wan_installation(
    root: str = WAN_DEFAULT_ROOT,
    generate_py: str = "",
    python_exe: str = "",
) -> str:
    root_path = Path(root or WAN_DEFAULT_ROOT)
    script = Path(generate_py or (root_path / "generate.py"))
    python_path = Path(python_exe or find_wan_python(str(root_path)))
    lines = [
        f"Wan2.2 root: {root_path}",
        f"generate.py: {script}",
        f"Wan2.2 Python: {python_path}",
    ]
    if not root_path.is_dir():
        lines.append("ERROR: Wan2.2 root folder was not found.")
        return "\n".join(lines)
    if not script.is_file():
        lines.append("ERROR: generate.py was not found.")
        return "\n".join(lines)
    lines.append("OK: generate.py found.")

    if not python_path.is_file():
        lines.append(
            "ERROR: Dedicated Wan2.2 Python was not found. "
            f"Create/install it at: {root_path / '.venv' / 'Scripts' / 'python.exe'}"
        )
        lines.append(
            "IMPORTANT: F5-TTS Python is deliberately NOT used as a fallback for Wan2.2."
        )
        return "\n".join(lines)

    code = """import sys
mods = ['easydict','diffusers','torch','transformers','safetensors','einops','PIL','yaml']
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
print('PYTHON=' + sys.executable)
print('VERSION=' + sys.version.split()[0])
print('MISSING=' + ','.join(missing))
try:
    import torch
    print('TORCH=' + torch.__version__)
    print('CUDA=' + str(torch.cuda.is_available()))
except Exception as e:
    print('TORCH_ERROR=' + str(e))
"""
    rc, output = _run_python_check(str(python_path), code)
    if rc != 0:
        lines.append(f"ERROR: Could not run Wan2.2 Python.\n{output.strip()}")
        return "\n".join(lines)
    parsed = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    lines.append(f"Python version: {parsed.get('VERSION', 'unknown')}")
    missing = [x for x in parsed.get("MISSING", "").split(",") if x]
    if missing:
        lines.append("Missing Python packages: " + ", ".join(missing))
    else:
        lines.append("Core Python packages: OK")
    lines.append(f"PyTorch: {parsed.get('TORCH', 'not available')}")
    lines.append(f"Wan CUDA available: {parsed.get('CUDA', 'unknown')}")

    import_code = "import wan; print('WAN_IMPORT_OK')"
    rc2, output2 = _run_python_check(str(python_path), import_code, timeout=60)
    if rc2 == 0 and "WAN_IMPORT_OK" in output2:
        lines.append("Wan package import: OK")
    else:
        tail = output2.strip()[-3000:]
        lines.append("Wan package import: ERROR")
        if tail:
            lines.append(tail)
    if not missing and rc2 == 0:
        lines.append("Wan2.2 environment is ready for launch.")
    else:
        lines.append("Install the missing Wan2.2 dependencies in this dedicated environment before generating video.")
    return "\n".join(lines)


def _resolve_ckpt(task: str, checkpoint_dir: str) -> str:
    """Resolve the checkpoint without accidentally mixing TI2V and S2V weights."""
    supplied = Path(checkpoint_dir) if checkpoint_dir else None
    if task == "s2v-14B":
        # Never accept a TI2V checkpoint for S2V just because the directory exists.
        if supplied and supplied.is_dir() and "S2V-14B" in supplied.name.upper():
            return str(supplied)
        return WAN_DEFAULT_S2V_CKPT
    if task == "t2v-A14B":
        if supplied and supplied.is_dir() and "T2V-A14B" in supplied.name.upper():
            return str(supplied)
        return WAN_DEFAULT_T2V_CKPT
    if task == "ti2v-5B":
        if supplied and supplied.is_dir() and "TI2V-5B" in supplied.name.upper():
            return str(supplied)
        return WAN_DEFAULT_TI2V_CKPT
    return str(supplied) if supplied and supplied.is_dir() else (checkpoint_dir or WAN_DEFAULT_T2V_CKPT)


def find_wan_python(root: str = WAN_DEFAULT_ROOT) -> str:
    root_path = Path(root or WAN_DEFAULT_ROOT)
    candidates = [
        os.getenv("WAN22_PYTHON", ""),
        str(root_path / ".venv" / "Scripts" / "python.exe"),
        str(root_path / "venv" / "Scripts" / "python.exe"),
        str(root_path / "env" / "Scripts" / "python.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    # Return the expected dedicated environment path, not the F5-TTS Python.
    return str(root_path / ".venv" / "Scripts" / "python.exe")


def generate_wan_video(
    prompt: str,
    image_path: str,
    output_path: str,
    task: str = "ti2v-5B",
    wan_root: str = WAN_DEFAULT_ROOT,
    generate_py: str = "",
    checkpoint_dir: str = "",
    size: str = "1280*720",
    frame_num: int = 81,
    sample_steps: Optional[int] = None,
    seed: int = -1,
    offload_model: bool = True,
    t5_cpu: bool = True,
    convert_model_dtype: bool = True,
    audio_path: str = "",
    start_from_ref: bool = True,
    python_exe: str = "",
    timeout: int = 3600,
) -> str:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Wan2.2 prompt is empty.")
    root = Path(wan_root or WAN_DEFAULT_ROOT)
    script = Path(generate_py or (root / "generate.py"))
    if not script.is_file():
        raise FileNotFoundError(f"Wan2.2 generate.py not found: {script}")
    if image_path and not Path(image_path).is_file():
        raise FileNotFoundError(f"Wan2.2 input image not found: {image_path}")
    if task == "s2v-14B" and (not audio_path or not Path(audio_path).is_file()):
        raise FileNotFoundError("S2V requires an existing WAV/MP3 audio file.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = _resolve_ckpt(task, checkpoint_dir)
    if not Path(ckpt).is_dir():
        raise FileNotFoundError(
            f"Wan2.2 {task} checkpoint directory was not found: {ckpt}"
            "\nDownload/install the matching Wan2.2 model weights first."
        )
    exe = python_exe or find_wan_python(str(root))
    if not Path(exe).is_file():
        raise RuntimeError(
            "Dedicated Wan2.2 Python was not found: " + str(exe) +
            "\nCreate the environment at D:\\Wan2.2\\.venv and install the Wan2.2 requirements. "
            "F5-TTS Python is not used for Wan2.2."
        )

    cmd = [exe, str(script), "--task", task, "--ckpt_dir", ckpt,
           "--size", size, "--frame_num", str(int(frame_num)),
           "--prompt", prompt, "--save_file", str(out),
           "--offload_model", "True"]
    if image_path:
        cmd += ["--image", str(image_path)]
    if t5_cpu:
        cmd.append("--t5_cpu")
    if convert_model_dtype:
        cmd.append("--convert_model_dtype")
    if sample_steps is not None and int(sample_steps) > 0:
        cmd += ["--sample_steps", str(int(sample_steps))]
    if int(seed) >= 0:
        cmd += ["--base_seed", str(int(seed))]
    if task == "s2v-14B":
        cmd += ["--audio", str(audio_path)]
        if start_from_ref:
            cmd.append("--start_from_ref")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            cmd, cwd=str(root), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=max(60, int(timeout))
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Wan2.2 timed out after {timeout} seconds.\n{exc}") from exc
    log = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        tail = log[-12000:] if log else "No Wan2.2 output was captured."
        raise RuntimeError(f"Wan2.2 failed (exit code {completed.returncode}).\n\n{tail}")
    if not out.is_file():
        raise RuntimeError(f"Wan2.2 finished without creating the expected video: {out}\n\n{log[-8000:]}")
    return str(out)
