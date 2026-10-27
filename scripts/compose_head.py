"""Head/face compositing helper for the optional extension.

The chain is: locate a head box from the depth map, soft-key the white-background
portrait, erode/feather the alpha, resize in premultiplied space (avoiding white
fringes), match brightness, add a contact shadow, Poisson-blend the boundary and
match grain.  It is intentionally a local OpenCV/numpy implementation so every
step can be inspected.

Example:
    python scripts/compose_head.py --base results/images/headfig_photoreal.png \
        --head portrait.png --depth results/raw/headfig/headfig_depth.png \
        --out results/images/headfig_face.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_tuple(text, n, cast=float):
    values = [cast(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError("expected %d comma-separated values" % n)
    return values


def head_alpha(rgb, face, ellipse_soft=3.0):
    """White-background soft key, restricted to an ellipse around the face."""
    key = np.clip((255.0 - rgb.min(axis=2) - 6.0) / 18.0, 0.0, 1.0)
    left, top, right, chin = face
    cx, cy = (left + right) / 2.0, (top + chin) / 2.0 + (chin - top) * 0.03
    rx, ry = (right - left) / 2.0 * 1.05, (chin - top) / 2.0 * 1.06
    emask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.ellipse(
        emask,
        (int(cx), int(cy)),
        (int(rx), int(ry)),
        0,
        0,
        360,
        255,
        -1,
    )
    emask = cv2.GaussianBlur(emask, (0, 0), ellipse_soft).astype(np.float32) / 255.0
    alpha = np.clip(key * emask, 0.0, 1.0)
    hard = (alpha > 0.55).astype(np.uint8)
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    return np.clip(cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.2), 0.0, 1.0)


def resize_premul(rgb, alpha, size):
    """Multiply alpha before resizing so white background pixels cannot bleed."""
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    p = cv2.resize(
        rgb.astype(np.float32) * alpha[..., None],
        size,
        interpolation=cv2.INTER_AREA,
    )
    safe = np.maximum(a[..., None], 1e-3)
    un = np.where(a[..., None] > 1e-3, p / safe, 0.0)
    return np.clip(un, 0, 255).astype(np.uint8), a, p


def foreground_box(depth):
    depth = cv2.GaussianBlur(depth, (5, 5), 0)
    _, mask = cv2.threshold(depth, max(12, int(depth.max() * 0.15)), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("depth map has no foreground contour")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if h < 20 or w < 20:
        raise RuntimeError("foreground box too small: %s" % ((x, y, w, h),))
    # A portrait's head occupies the upper part of the foreground bbox.
    head = [
        int(x + w * 0.22),
        int(y),
        int(x + w * 0.78),
        int(y + h * 0.52),
    ]
    return head


def expand_box(box, grow):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return [
        int(round(x0 - w * grow[0])),
        int(round(y0 - h * grow[1])),
        int(round(x1 + w * grow[2])),
        int(round(y1 + h * grow[3])),
    ]


def grain_match(image, mask, strength, seed=7):
    if strength <= 0:
        return image
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 4.0 * strength, image.shape[:2]).astype(np.float32)
    out = image.astype(np.float32)
    out += noise[..., None] * (mask[..., None] / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--face", default="0.16,0.02,0.85,0.71")
    parser.add_argument("--box-grow", default="0.13,0.14,0.13,0.06")
    parser.add_argument("--clone", type=float, default=0.6)
    parser.add_argument("--shadow", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=1.0)
    parser.add_argument("--warp-fill", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base = cv2.imread(args.base, cv2.IMREAD_COLOR)
    head = cv2.imread(args.head, cv2.IMREAD_COLOR)
    depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
    if base is None or head is None or depth is None:
        raise SystemExit("cannot read base/head/depth")
    if depth.shape != base.shape[:2]:
        depth = cv2.resize(depth, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

    face = parse_tuple(args.face, 4)
    box_grow = parse_tuple(args.box_grow, 4)
    face_px = [
        int(face[0] * head.shape[1]),
        int(face[1] * head.shape[0]),
        int(face[2] * head.shape[1]),
        int(face[3] * head.shape[0]),
    ]
    alpha = head_alpha(head, face_px)
    target = expand_box(foreground_box(depth), box_grow)
    x0, y0, x1, y1 = target
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(base.shape[1], x1), min(base.shape[0], y1)
    tw, th = max(1, x1 - x0), max(1, y1 - y0)
    if args.warp_fill:
        size = (tw, th)
    else:
        scale = min(tw / head.shape[1], th / head.shape[0])
        size = (max(1, int(head.shape[1] * scale)), max(1, int(head.shape[0] * scale)))
    head_rgb, a, premul = resize_premul(head, alpha, size)

    # Brightness alignment inside the resized face ellipse.
    fy0 = int(face[1] * size[1])
    fy1 = int(face[3] * size[1])
    fx0 = int(face[0] * size[0])
    fx1 = int(face[2] * size[0])
    head_gray = cv2.cvtColor(head_rgb, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(base[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    dst_region = dst_gray[
        fy0 : min(fy1, dst_gray.shape[0]), fx0 : min(fx1, dst_gray.shape[1])
    ]
    src_region = head_gray[fy0:fy1, fx0:fx1]
    if src_region.size and dst_region.size:
        ratio = float(dst_region.mean() + 1e-6) / float(src_region.mean() + 1e-6)
        ratio = float(np.clip(ratio, 0.70, 1.30))
        head_rgb = np.clip(head_rgb.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
        premul = premul * ratio

    # Place the head in the target box, centered horizontally.
    canvas = base.copy()
    ox = x0 + (tw - size[0]) // 2
    oy = y0 + (th - size[1]) // 2
    canvas, a_full = _paste_premul(canvas, head_rgb, a, premul, ox, oy)

    if args.shadow > 0:
        _add_contact_shadow(canvas, target, args.shadow)

    if args.clone > 0:
        mask = (a_full > 0.55).astype(np.uint8) * 255
        if mask.sum() > 0:
            center = (x0 + tw // 2, y0 + th // 2)
            try:
                cloned = cv2.seamlessClone(canvas, base, mask, center, cv2.NORMAL_CLONE)
                canvas = cv2.addWeighted(cloned, args.clone, canvas, 1.0 - args.clone, 0)
            except cv2.error as exc:
                print("WARN seamlessClone failed: %s" % exc)

    canvas = grain_match(canvas, a_full * 255, args.grain, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    debug = np.concatenate([base, canvas], axis=1)
    cv2.imwrite(str(Path(args.out).with_name(Path(args.out).stem + "_debug.png")), debug)
    print(
        "COMPOSE_OK out=%s head_box=%s size=%s clone=%.2f shadow=%.2f"
        % (args.out, target, size, args.clone, args.shadow)
    )


def _paste_premul(base, rgb, alpha, premul, ox, oy):
    h, w = rgb.shape[:2]
    bh, bw = base.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(bw, ox + w), min(bh, oy + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("head box outside base image")
    sx0, sy0 = x0 - ox, y0 - oy
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    a = alpha[sy0:sy1, sx0:sx1, None]
    p = premul[sy0:sy1, sx0:sx1]
    region = base[y0:y1, x0:x1].astype(np.float32)
    out = p + region * (1.0 - a)
    base[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    full = np.zeros(base.shape[:2], np.float32)
    full[y0:y1, x0:x1] = a[..., 0]
    return base, full


def _add_contact_shadow(image, box, strength):
    x0, y0, x1, y1 = box
    h, w = image.shape[:2]
    cx = (x0 + x1) // 2
    cy = int(y0 + (y1 - y0) * 0.94)
    rx = max(8, int((x1 - x0) * 0.42))
    ry = max(5, int((y1 - y0) * 0.10))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(3, ry / 2.0)).astype(np.float32) / 255.0
    shadow = 1.0 - strength * 0.45 * mask
    image[:] = np.clip(image.astype(np.float32) * shadow[..., None], 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
