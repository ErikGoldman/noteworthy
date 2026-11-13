"""Training-time augmentation of (image, masks) crops.

Geometric transforms are applied identically to the image and the masks;
photometric degradations touch the image only. Two regimes:

* capture: what a phone photo / screenshot / scan does to an engraved page
  (perspective, blur, JPEG, uneven lighting, dark page borders, faded lines).
* handwriting: approximates why manuscripts are harder than engravings --
  wobbly lines (smooth elastic warp), uneven stroke weight, symbols that drift
  off their nominal positions, and broken/faint strokes. Real handwriting
  (MUSCIMA++) is mixed in separately; this regime makes rendered pages
  (especially the Petaluma font) look less mechanically perfect.
"""
from __future__ import annotations

import cv2
import numpy as np

from .channels import CHANNEL_INDEX, map_packed, unpack_masks


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], sigma: float, amplitude: float) -> np.ndarray:
    h, w = shape
    # Generate at low resolution and upsample: equivalent to a large-sigma blur, much cheaper.
    step = max(1, int(sigma / 2))
    small = rng.standard_normal((h // step + 2, w // step + 2)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    small /= small.std() + 1e-6
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amplitude


def _remap(image: np.ndarray, packed_masks: np.ndarray, map_x: np.ndarray, map_y: np.ndarray):
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return image, map_packed(packed_masks, lambda m: cv2.remap(m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def elastic(rng: np.random.Generator, image: np.ndarray, masks: np.ndarray, interline: float):
    h, w = image.shape
    sigma = interline * rng.uniform(2.0, 5.0)
    amp = interline * rng.uniform(0.1, 0.35)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _remap(image, masks, gx + _smooth_field(rng, (h, w), sigma, amp), gy + _smooth_field(rng, (h, w), sigma, amp))


def perspective(rng: np.random.Generator, image: np.ndarray, masks: np.ndarray, max_shift_frac: float = 0.03):
    h, w = image.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-max_shift_frac, max_shift_frac, src.shape).astype(np.float32) * np.float32([w, h])
    m = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    denom = m[2, 0] * gx + m[2, 1] * gy + m[2, 2]
    map_x = ((m[0, 0] * gx + m[0, 1] * gy + m[0, 2]) / denom).astype(np.float32)
    map_y = ((m[1, 0] * gx + m[1, 1] * gy + m[1, 2]) / denom).astype(np.float32)
    return _remap(image, masks, map_x, map_y)


def stroke_weight(rng: np.random.Generator, image: np.ndarray, interline: float, thin: bool = True) -> np.ndarray:
    """Thicken or (when `thin`) thin ink, optionally varying smoothly across the crop (like pen
    pressure). Thinning erases strokes no thicker than twice the kernel's radius outright."""
    radius = max(1, int(round(interline * 0.06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    heavier = cv2.erode(image, kernel)
    lighter = cv2.dilate(image, kernel)
    t = _smooth_field(rng, image.shape, interline * 6, 1.0)
    t = np.clip(t + rng.uniform(-0.8, 0.8), -1 if thin else 0, 1)
    out = np.where(t > 0, image * (1 - t) + heavier * t, image * (1 + t) - lighter * t)
    return out.astype(np.uint8)


def fade_staff_lines(rng: np.random.Generator, image: np.ndarray, packed_masks: np.ndarray) -> np.ndarray:
    """Lighten or break staff lines (where nothing else is drawn); common in scans and photos."""
    masks = unpack_masks(packed_masks)
    staff = CHANNEL_INDEX["staff_line"]
    lines = masks[..., staff] & ~np.delete(masks, staff, axis=-1).any(-1)
    strength = np.clip(_smooth_field(rng, image.shape, 30, 1.0) + rng.uniform(-0.5, 0.8), 0, 0.9)
    out = image.astype(np.float32)
    out[lines] = out[lines] + (255 - out[lines]) * strength[lines]
    return out.astype(np.uint8)


def _bezier(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)


def pen_annotations(rng: np.random.Generator, image: np.ndarray, interline: float) -> np.ndarray:
    """Draw hand-made markup (arrows, loops, zigzags) of the kind people add on tablets.

    Labels are deliberately left unchanged: markup is not music, and the network must
    learn to see through it (Night Album.pdf has such iPad annotations).
    """
    out = image.copy()
    h, w = out.shape
    for _ in range(rng.integers(1, 4)):
        tone = int(rng.uniform(60, 180))
        thickness = max(1, int(round(interline * rng.uniform(0.05, 0.2))))
        p0 = rng.uniform([0, 0], [w, h])
        length = interline * rng.uniform(3, 15)
        angle = rng.normal(0, 0.3) + (np.pi if rng.random() < 0.5 else 0)
        p2 = p0 + length * np.array([np.cos(angle), np.sin(angle)])
        p1 = (p0 + p2) / 2 + rng.normal(0, length * 0.15, 2)
        kind = rng.integers(3)
        if kind == 2:  # zigzag / chevrons
            n = rng.integers(3, 7)
            xs = np.linspace(p0[0], p0[0] + interline * rng.uniform(1, 2), n)
            ys = p0[1] + np.arange(n) * interline * 0.45
            for y in ys:
                pts = np.array([[xs[0], y + interline * 0.3], [xs.mean(), y], [xs[-1], y + interline * 0.3]], np.int32)
                cv2.polylines(out, [pts], False, tone, thickness, cv2.LINE_AA)
            continue
        curve = _bezier(p0, p1, p2)
        cv2.polylines(out, [curve], False, tone, thickness, cv2.LINE_AA)
        if kind == 1:  # arrow head
            d = (curve[-1] - curve[-5]).astype(float)
            d /= np.linalg.norm(d) + 1e-6
            for sign in (-1, 1):
                rot = np.array([[np.cos(sign * 2.6), -np.sin(sign * 2.6)], [np.sin(sign * 2.6), np.cos(sign * 2.6)]])
                tip = curve[-1] + (rot @ d) * interline * 1.2
                cv2.line(out, tuple(int(v) for v in curve[-1]), tuple(int(v) for v in tip), tone, thickness, cv2.LINE_AA)
    return out


def capture_degradation(rng: np.random.Generator, image: np.ndarray, interline: float) -> np.ndarray:
    img = image.astype(np.float32)
    h, w = img.shape
    # Ink and paper tones, with low-frequency illumination variation.
    ink, paper = rng.uniform(0, 70), rng.uniform(170, 255)
    img = ink + (img / 255.0) * (paper - ink)
    img += _smooth_field(rng, (h, w), max(h, w) / 3, rng.uniform(0, 25))
    if rng.random() < 0.3:  # dark scan/screenshot border like in Night Album.pdf
        side = rng.integers(4)
        width = int(rng.uniform(2, interline * 2))
        sl = [np.s_[:width, :], np.s_[-width:, :], np.s_[:, :width], np.s_[:, -width:]][side]
        img[sl] = rng.uniform(0, 60)
    blur = rng.uniform(0, interline * 0.07)
    if blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0, 10)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 95))])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def augment(rng: np.random.Generator, image: np.ndarray, masks: np.ndarray, interline: float, handwritten_font: bool):
    """image: (H, W) uint8; masks: packed (H, W, P) uint8. Returns augmented copies."""
    handwriting = handwritten_font or rng.random() < 0.25
    if handwriting:
        image, masks = elastic(rng, image, masks, interline)
        image = stroke_weight(rng, image, interline)
    elif rng.random() < 0.3:
        image = stroke_weight(rng, image, interline)
    if rng.random() < 0.5:
        image, masks = perspective(rng, image, masks)
    if rng.random() < 0.3:
        image = fade_staff_lines(rng, image, masks)
    if rng.random() < 0.3:
        image = pen_annotations(rng, image, interline)
    if rng.random() < 0.9:
        image = capture_degradation(rng, image, interline)
    return image, masks
