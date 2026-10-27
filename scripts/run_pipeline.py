"""One-command Blender -> stable-diffusion.cpp CPU pipeline.

Examples:
    python scripts/run_pipeline.py --tag A --res 512 --steps 4 --cfg 1.2
    python scripts/run_pipeline.py --tag B --res 768 --steps 4 --cfg 1.2
    python scripts/run_pipeline.py --tag D --reuse-raw A --control-strength 0.5
    python scripts/run_pipeline.py --tag E --res 512 --mist-start 4.0 --mist-end 6.0

All timings are parsed from the real sd-cli log; failed runs keep their log and
a run JSON with a non-zero exit code.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BLENDER = ROOT / "env" / "blender" / "blender"
BLEND = ROOT / "assets" / "demo_white_model.blend"
EXPORT_SCRIPT = ROOT / "blender" / "export_control_maps.py"
SDCLI = ROOT / "vendor" / "sdcpp" / "build" / "bin" / "sd-cli"
MODEL = Path(os.environ.get("SD_MODEL", ROOT / "models" / "sd_turbo_q4_0.gguf"))
CONTROLNET = Path(
    os.environ.get("SD_CONTROLNET", ROOT / "models" / "control_depth_sd21-fp16.gguf")
)

PROMPT = (
    "a matte white sphere, a matte white torus, a matte white beveled cube, "
    "a matte white tall cylinder, geometric still life, photorealistic product "
    "render, studio soft lighting, PBR material, preserve original 3d geometry, sharp"
)
NEGATIVE = (
    "deformed geometry, perspective changed, cup, bowl, canister, phone, "
    "smartphone, metallic, noisy background"
)
SAMPLER = "euler_a"


def utc_now():
    return datetime.now(timezone.utc)


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def ensure_rgb(src, dst):
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        rgb.save(dst)
        return rgb.size


def sha256_cached(path, cache_path):
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    key = str(Path(path).resolve())
    st = Path(path).stat()
    entry = cache.get(key)
    if entry and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
        return entry["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return digest


def command_version(cmd, timeout=30):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as exc:
        return "unavailable: %s" % exc


def parse_timings(log_text):
    def grab(pattern):
        m = re.search(pattern, log_text)
        return float(m.group(1)) if m else None

    return {
        "sample_s": grab(r"sampling completed, taking ([0-9.]+)s"),
        "vae_s": grab(r"decode_first_stage completed, taking ([0-9.]+)s"),
        "sd_total_s": grab(r"generate_image completed in ([0-9.]+)s"),
    }


def run_logged(cmd, log_path, timeout, cwd=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "$ " + " ".join(str(c) for c in cmd) + "\n"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        wall = time.perf_counter() - t0
        body = header + (proc.stdout or "") + (proc.stderr or "")
        log_path.write_text(body, encoding="utf-8", errors="replace")
        return proc.returncode, wall, body
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        body = header + "TIMEOUT after %ss\n" % timeout
        body += (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        body += (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        log_path.write_text(body, encoding="utf-8", errors="replace")
        return 124, wall, body


def copy_raw(src_tag, tag, raw_dir):
    src_dir = RESULTS / "raw" / src_tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("depth", "normal", "clay", "meta"):
        ext = "json" if suffix == "meta" else "png"
        src = src_dir / f"{src_tag}_{suffix}.{ext}"
        dst = raw_dir / f"{tag}_{suffix}.{ext}"
        if not src.exists():
            raise FileNotFoundError("missing reused raw file: %s" % src)
        shutil.copyfile(src, dst)


def write_env(args):
    env = {
        "timestamp_utc": utc_now().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL_VERSION,
        "os_cpu_count": os.cpu_count(),
        "blender": command_version([BLENDER, "--version"]),
        "sd_cli": command_version([SDCLI, "--version"]),
        "sd_cli_path": rel(SDCLI),
        "blender_path": rel(BLENDER),
        "model": rel(MODEL),
        "controlnet": rel(CONTROLNET),
        "model_size": MODEL.stat().st_size if MODEL.exists() else None,
        "controlnet_size": CONTROLNET.stat().st_size if CONTROLNET.exists() else None,
    }
    path = RESULTS / "metrics" / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--res", type=int, default=512, choices=(512, 768))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.2)
    parser.add_argument("--control-strength", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--control-image", default=None)
    parser.add_argument("--reuse-raw", default=None)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--mist-start", type=float, default=None)
    parser.add_argument("--mist-end", type=float, default=None)
    parser.add_argument("--no-floor-depth", action="store_true")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tag = args.tag or "r%d_s%d_cfg%s_cn%s_seed%d" % (
        args.res,
        args.steps,
        str(args.cfg).replace(".", "p"),
        str(args.control_strength).replace(".", "p"),
        args.seed,
    )
    for path, label in ((MODEL, "model"), (CONTROLNET, "controlnet")):
        if not path.exists():
            raise SystemExit("missing %s: %s" % (label, path))
    if not SDCLI.exists():
        raise SystemExit("missing sd-cli: %s" % SDCLI)
    if not BLENDER.exists() and not args.control_image and not args.reuse_raw:
        raise SystemExit("missing Blender: %s" % BLENDER)

    raw_dir = RESULTS / "raw" / tag
    image_dir = RESULTS / "images"
    log_dir = RESULTS / "logs"
    metric_dir = RESULTS / "metrics"
    for d in (raw_dir, image_dir, log_dir, metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    out_image = image_dir / f"{tag}_photoreal.png"
    run_json = metric_dir / f"run_{tag}.json"
    if out_image.exists() and not args.overwrite:
        raise SystemExit("output already exists (use --overwrite): %s" % out_image)

    env = write_env(args)
    started = utc_now()
    run = {
        "tag": tag,
        "status": "running",
        "started_utc": started.isoformat(),
        "config": {
            "res": args.res,
            "steps": args.steps,
            "cfg": args.cfg,
            "control_strength": args.control_strength,
            "seed": args.seed,
            "threads": args.threads,
            "samples": args.samples,
            "sampler": SAMPLER,
            "prompt": args.prompt,
            "negative": args.negative,
            "no_control": args.no_control,
            "mist_start": args.mist_start,
            "mist_end": args.mist_end,
        },
        "env": env,
        "model_sha256": sha256_cached(MODEL, metric_dir / "model_hashes.json"),
        "controlnet_sha256": sha256_cached(CONTROLNET, metric_dir / "model_hashes.json"),
        "paths": {"raw": rel(raw_dir), "log": rel(log_dir), "image": rel(out_image)},
    }

    blender_s = 0.0
    depth_path = raw_dir / f"{tag}_depth.png"
    if args.reuse_raw:
        copy_raw(args.reuse_raw, tag, raw_dir)
    elif args.control_image:
        shutil.copyfile(args.control_image, depth_path)
        meta = {"source": "external", "control_image": str(Path(args.control_image).resolve())}
        (raw_dir / f"{tag}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        if not BLEND.exists():
            raise SystemExit("missing scene file: %s (run blender/make_demo_scene.py)" % BLEND)
        cmd = [
            BLENDER,
            "-b",
            BLEND,
            "-P",
            EXPORT_SCRIPT,
            "--",
            "--outdir",
            raw_dir,
            "--name",
            tag,
            "--res",
            str(args.res),
            "--samples",
            str(args.samples),
        ]
        if args.mist_start is not None:
            cmd += ["--mist-start", str(args.mist_start)]
        if args.mist_end is not None:
            cmd += ["--mist-end", str(args.mist_end)]
        if args.no_floor_depth:
            cmd += ["--no-floor-depth"]
        rc, blender_s, body = run_logged(
            cmd, log_dir / f"{tag}_blender.log", timeout=900
        )
        run["blender_exit_code"] = rc
        run["blender_s"] = round(blender_s, 3)
        if rc != 0:
            run["status"] = "failed"
            run["blender_log_tail"] = body[-4000:]
            run["finished_utc"] = utc_now().isoformat()
            run_json.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit("Blender export failed, see %s" % (log_dir / f"{tag}_blender.log"))

    rgb_depth = raw_dir / f"{tag}_depth_rgb.png"
    depth_size = ensure_rgb(depth_path, rgb_depth)
    if depth_size != (args.res, args.res):
        raise SystemExit(
            "depth image is %s but --res is %d; refusing mismatched control map"
            % (depth_size, args.res)
        )
    run["depth_size"] = list(depth_size)

    cmd = [
        SDCLI,
        "-m",
        MODEL,
        "-p",
        args.prompt,
        "-n",
        args.negative,
        "-W",
        str(args.res),
        "-H",
        str(args.res),
        "--steps",
        str(args.steps),
        "--cfg-scale",
        str(args.cfg),
        "--sampling-method",
        SAMPLER,
        "-s",
        str(args.seed),
        "-t",
        str(args.threads),
        "--backend",
        "cpu",
        "--log-level",
        "info",
        "-o",
        out_image,
    ]
    if not args.no_control:
        cmd += [
            "--control-net",
            CONTROLNET,
            "--control-image",
            rgb_depth,
            "--control-strength",
            str(args.control_strength),
        ]
    rc, sd_wall, body = run_logged(
        cmd, log_dir / f"{tag}_sdcli.log", timeout=args.timeout, cwd=SDCLI.parent
    )
    timings = parse_timings(body)
    finished = utc_now()
    run.update(
        {
            "finished_utc": finished.isoformat(),
            "sd_exit_code": rc,
            "sd_wall_s": round(sd_wall, 3),
            "blender_s": round(blender_s, 3),
            "total_s": round(blender_s + sd_wall, 3),
            "sample_s": timings["sample_s"],
            "vae_s": timings["vae_s"],
            "sd_total_s": timings["sd_total_s"],
            "model_load_s": round(sd_wall - timings["sd_total_s"], 3)
            if timings["sd_total_s"]
            else None,
            "command": " ".join(str(c) for c in cmd),
            "sdcli_log": rel(log_dir / f"{tag}_sdcli.log"),
            "blender_log": rel(log_dir / f"{tag}_blender.log") if not args.reuse_raw else None,
            "output_image": rel(out_image),
            "status": "ok" if rc == 0 and out_image.exists() else "failed",
        }
    )
    if rc != 0 or not out_image.exists():
        run["sd_log_tail"] = body[-6000:]
        run_json.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("sd-cli failed rc=%d, see %s" % (rc, log_dir / f"{tag}_sdcli.log"))

    run_json.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "RUN_OK tag=%s total=%.1fs blender=%.1fs sample=%s vae=%s image=%s"
        % (tag, run["total_s"], blender_s, run["sample_s"], run["vae_s"], out_image)
    )


if __name__ == "__main__":
    main()
