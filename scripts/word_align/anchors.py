"""Line segmentation and hard-anchor detection (Ayah medallions, Rub' stars, Sajdah rosettes)."""
import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .config import EditionConfig


def load_ink_masks(img_path: str, cfg: EditionConfig) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (black_mask, warm_marker_mask, W, H) boolean arrays from a page PNG."""
    im = np.array(Image.open(img_path).convert('RGBA'))
    H, W = im.shape[:2]
    alpha = im[:, :, 3] >= cfg.alpha_min
    r = im[:, :, 0].astype(np.int16)
    g = im[:, :, 1].astype(np.int16)
    b = im[:, :, 2].astype(np.int16)
    t = cfg.black_max_rgb
    black = alpha & (r < t) & (g < t) & (b < t)
    # Remove full-page decorative frame borders (very tall vertical lines or wide horizontal lines)
    col_frac = black.mean(axis=0)
    row_frac = black.mean(axis=1)
    if (col_frac > 0.55).any():
        black[:, col_frac > 0.55] = False
    if (row_frac > 0.65).any():
        black[row_frac > 0.65, :] = False
    # Red or warm reddish-brown marker ink (Sajdah rosettes on all 15 pages including p515)
    red = alpha & (r > 125) & ((r - b) > 55) & (g < 130)
    return black, red, W, H


def detect_lines(black: np.ndarray, cfg: EditionConfig) -> List[Dict]:
    """Detect text lines via smoothed horizontal projection and valley splitting."""
    H, W = black.shape
    scale_y = H / float(cfg.ref_height)
    min_dist = max(40, int(round(cfg.min_line_dist_px * scale_y)))

    row_sum = black.sum(axis=1).astype(np.float32)
    k = max(5, int(round(7 * scale_y)) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(row_sum, kernel, mode='same')

    thresh = max(25.0, 45.0 * (W / float(cfg.ref_width)))
    peaks: List[int] = []
    margin = int(round(18 * scale_y))
    for y in range(margin, H - margin):
        if smooth[y] > thresh and smooth[y] >= smooth[y - 1] and smooth[y] >= smooth[y + 1]:
            if peaks and y - peaks[-1] < min_dist:
                if smooth[y] > smooth[peaks[-1]]:
                    peaks[-1] = y
            else:
                peaks.append(y)

    lines: List[Dict] = []
    win = int(round(20 * scale_y))
    for p in peaks:
        y0 = max(0, p - win)
        y1 = min(H, p + win)
        y_base = y0 + int(np.argmax(row_sum[y0:y1]))
        lines.append({'y_base': y_base})

    # Estimate typical inter-line step for clean first/last line caps
    steps = [lines[i]['y_base'] - lines[i - 1]['y_base'] for i in range(1, len(lines)) if 90 * scale_y <= (lines[i]['y_base'] - lines[i - 1]['y_base']) <= 240 * scale_y]
    med_step = float(np.median(steps)) if steps else 142.0 * scale_y
    max_ascent = int(round(med_step * 0.72))
    max_descent = int(round(med_step * 0.38))

    for i, ln in enumerate(lines):
        if i == 0:
            lo = max(0, ln['y_base'] - max_ascent)
            ys = np.where(row_sum[lo:ln['y_base']] > 12)[0]
            ln['y_top'] = lo + int(ys[0]) - 2 if len(ys) else lo
        else:
            prev_b = lines[i - 1]['y_base']
            if ln['y_base'] - prev_b > med_step * 1.45:
                # Gap (e.g., Surah header or Basmalah between lines)
                hi_prev = min(H, prev_b + max_descent)
                ys_p = np.where(row_sum[prev_b:hi_prev] > 12)[0]
                lines[i - 1]['y_bot'] = prev_b + int(ys_p[-1]) + 2 if len(ys_p) else hi_prev
                lo_cur = max(0, ln['y_base'] - max_ascent)
                ys_c = np.where(row_sum[lo_cur:ln['y_base']] > 12)[0]
                ln['y_top'] = lo_cur + int(ys_c[0]) - 2 if len(ys_c) else lo_cur
            else:
                v_start = prev_b + int(round(18 * scale_y))
                v_end = max(v_start + 1, ln['y_base'] - int(round(22 * scale_y)))
                valley = v_start + int(np.argmin(smooth[v_start:v_end]))
                ln['y_top'] = valley
                lines[i - 1]['y_bot'] = valley
        if i == len(lines) - 1:
            hi = min(H, ln['y_base'] + max_descent)
            ys = np.where(row_sum[ln['y_base']:hi] > 12)[0]
            ln['y_bot'] = ln['y_base'] + int(ys[-1]) + 2 if len(ys) else hi

    return lines


def detect_medallions(black: np.ndarray, lines: List[Dict], cfg: EditionConfig) -> List[Dict]:
    """Detect Ayah-end ornamental rings (wordsno=999) via CC geometry + enclosed hole."""
    H, W = black.shape
    scale = H / float(cfg.ref_height)
    min_sz = int(round(cfg.medallion_min_px * scale))
    max_sz = int(round(125 * scale))  # Allow larger Fatiha/Baqara medallions on p2/p3
    min_hole = int(round(cfg.medallion_min_hole_px * scale * scale))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(black.astype(np.uint8), connectivity=8)
    medallions = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if not (min_sz <= w <= max_sz and min_sz <= h <= max_sz):
            continue
        ar = w / float(h)
        if not (0.76 <= ar <= 1.28):
            continue
        if not (int(900 * scale * scale) <= area <= int(8500 * scale * scale)):
            continue
        crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = crop
        inv = 1 - padded
        n_inv, inv_lbl, inv_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        bg_lbl = inv_lbl[0, 0]
        inner_hole_area = 0
        for j in range(1, n_inv):
            if j != bg_lbl:
                inner_hole_area += int(inv_stats[j, cv2.CC_STAT_AREA])
        if inner_hole_area < min_hole:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        if not lines:
            continue
        best_li = min(range(len(lines)), key=lambda li: abs(lines[li]['y_base'] - cy))
        if abs(lines[best_li]['y_base'] - cy) > 120 * scale:
            continue
        medallions.append({
            'x0': int(x), 'y0': int(y), 'x1': int(x + w), 'y1': int(y + h),
            'cx': cx, 'cy': cy, 'line_idx': best_li, 'area': int(area),
            'hole_area': inner_hole_area
        })
    return medallions


def find_medallion_by_template(
    black: np.ndarray,
    line: Dict,
    expected_x: float,
    med_tmpl: Optional[np.ndarray],
    existing_meds: List[Dict],
) -> Optional[Dict]:
    """Fallback template search when a medallion touches an adjacent letter or diacritic."""
    if med_tmpl is None:
        return None
    H, W = black.shape
    th, tw = med_tmpl.shape[:2]
    y0 = max(0, line['y_top'] - 25)
    y1 = min(H, line['y_bot'] + 15)
    x0 = max(0, int(expected_x) - 190)
    x1 = min(W, int(expected_x) + 190)
    if (y1 - y0) < th or (x1 - x0) < tw:
        return None
    roi = black[y0:y1, x0:x1].astype(np.uint8) * 255
    res = cv2.matchTemplate(roi, med_tmpl, cv2.TM_CCOEFF_NORMED)
    # Suppress already-matched medallions
    for em in existing_meds:
        mx0 = max(0, em['x0'] - x0 - tw // 2)
        mx1 = min(res.shape[1], em['x1'] - x0 + tw // 2)
        if mx1 > mx0:
            res[:, mx0:mx1] = -1.0
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= 0.36:
        rx0 = x0 + max_loc[0]
        ry0 = y0 + max_loc[1]
        rx1 = rx0 + tw
        ry1 = ry0 + th
        return {
            'x0': int(rx0), 'y0': int(ry0), 'x1': int(rx1), 'y1': int(ry1),
            'cx': 0.5 * (rx0 + rx1), 'cy': 0.5 * (ry0 + ry1),
            'score': float(max_val),
        }
    return None


def build_marker_templates(cfg: EditionConfig, conn) -> Dict[str, Optional[np.ndarray]]:
    """Extract canonical binary templates for Medallion (999), Rub' star (1001), and Sajdah rosette (1000)."""
    templates: Dict[str, Optional[np.ndarray]] = {'medallion': None, 'rub': None, 'sajdah': None}
    c = conn.cursor()

    # 0. Medallion outer-ring template (from page 4)
    p4_path = os.path.join(cfg.pages_dir, '4.png')
    if os.path.exists(p4_path):
        b4, _, _, _ = load_ink_masks(p4_path, cfg)
        lines4 = detect_lines(b4, cfg)
        meds4 = detect_medallions(b4, lines4, cfg)
        if meds4:
            # Average normalized crops of all detected medallions on page 4 (ring stays sharp, digits average out!)
            tw = int(round(np.median([m['x1'] - m['x0'] for m in meds4])))
            th = int(round(np.median([m['y1'] - m['y0'] for m in meds4])))
            acc = np.zeros((th, tw), dtype=np.float32)
            for m in meds4:
                crop = b4[m['y0']:m['y1'], m['x0']:m['x1']].astype(np.float32)
                acc += cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            acc /= float(len(meds4))
            templates['medallion'] = (acc * 255.0).astype(np.uint8)

    # 1. Rub' star template
    if cfg.rub_template_page is not None:
        p = cfg.rub_template_page
        img_path = os.path.join(cfg.pages_dir, f'{p}.png')
        row = c.execute(
            f"SELECT {cfg.x_col}, {cfg.y_col}, {cfg.w_col} FROM wordsall WHERE {cfg.page_col}=? AND wordsno=1001 LIMIT 1",
            (p,)
        ).fetchone() if cfg.x_col else None
        if os.path.exists(img_path) and row:
            black, _, W, H = load_ink_masks(img_path, cfg)
            ex, ey = (row[0] + row[2] * 0.5) * W, (row[1] - 0.025) * H
            num, lbl, stats, cents = cv2.connectedComponentsWithStats(black.astype(np.uint8), connectivity=8)
            best = None
            best_d = 1e9
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if 20 <= w <= 42 and 20 <= h <= 42 and 280 <= area <= 900:
                    d = math.hypot(cents[i][0] - ex, cents[i][1] - ey)
                    if d < best_d:
                        best_d = d
                        best = (lbl[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            templates['rub'] = best

    # 2. Sajdah rosette template (compact red CC, excluding overline)
    if cfg.sajdah_template_page is not None:
        p = cfg.sajdah_template_page
        img_path = os.path.join(cfg.pages_dir, f'{p}.png')
        if os.path.exists(img_path):
            _, red, W, H = load_ink_masks(img_path, cfg)
            num, lbl, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if 30 <= w <= 65 and 50 <= h <= 95 and area >= 600:
                    templates['sajdah'] = (lbl[y:y + h, x:x + w] == i).astype(np.uint8) * 255
                    break

    return templates


def detect_rub_stars(
    black: np.ndarray,
    lines: List[Dict],
    expected_rubs: List[Dict],
    rub_tmpl: Optional[np.ndarray],
    cfg: EditionConfig,
) -> Dict[int, Dict]:
    """Detect Rub' stars (wordsno=1001) using normalized cross-correlation template matching."""
    found: Dict[int, Dict] = {}
    if not expected_rubs or rub_tmpl is None:
        return found
    H, W = black.shape
    th, tw = rub_tmpl.shape[:2]
    b_u8 = black.astype(np.uint8) * 255

    for er in expected_rubs:
        if er['old_x'] is not None and er['old_y'] is not None:
            ex = int((er['old_x'] + (er['old_w'] or 0.03) * 0.5) * W)
            ey = int((er['old_y'] - 0.025) * H)
            y0, y1 = max(0, ey - 120), min(H, ey + 120)
            x0, x1 = max(0, ex - 220), min(W, ex + 220)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        if (y1 - y0) < th or (x1 - x0) < tw:
            continue
        res = cv2.matchTemplate(b_u8[y0:y1, x0:x1], rub_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.42:
            rx0 = x0 + max_loc[0]
            ry0 = y0 + max_loc[1]
            rx1 = rx0 + tw
            ry1 = ry0 + th
            cx, cy = 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)
            li = min(range(len(lines)), key=lambda idx: abs(lines[idx]['y_base'] - cy))
            found[er['row_idx']] = {
                'x0': int(rx0), 'y0': int(ry0), 'x1': int(rx1), 'y1': int(ry1),
                'cx': float(cx), 'cy': float(cy), 'line_idx': li, 'score': float(max_val)
            }
    return found


def detect_sajdahs(
    red: np.ndarray,
    black: np.ndarray,
    lines: List[Dict],
    expected_sajdahs: List[Dict],
    sajdah_tmpl: Optional[np.ndarray],
    cfg: EditionConfig,
) -> Dict[int, Dict]:
    """Detect Sajdah rosettes (wordsno=1000): compact red CC first, black template fallback."""
    found: Dict[int, Dict] = {}
    if not expected_sajdahs:
        return found
    H, W = black.shape
    num_r, lbl_r, stats_r, cents_r = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)

    for es in expected_sajdahs:
        ex = (es['old_x'] + (es['old_w'] or 0.03) * 0.5) * W if es['old_x'] is not None else W * 0.5
        ey = (es['old_y'] - 0.025) * H if es['old_y'] is not None else H * 0.5

        # 1. Look for compact red rosette CC (excludes wide horizontal overline!)
        best = None
        best_d = 1e9
        for i in range(1, num_r):
            x, y, w, h, area = stats_r[i]
            if not (25 <= w <= 75 and 40 <= h <= 105 and area >= 400):
                continue
            cx, cy = float(cents_r[i][0]), float(cents_r[i][1])
            d = math.hypot(cx - ex, (cy - ey) * 1.5)
            if d < 260 and d < best_d:
                best_d = d
                li = min(range(len(lines)), key=lambda idx: abs(lines[idx]['y_base'] - cy))
                best = {
                    'x0': int(x), 'y0': int(y), 'x1': int(x + w), 'y1': int(y + h),
                    'cx': cx, 'cy': cy, 'line_idx': li, 'score': 1.0
                }

        # 2. Fallback: look for compact rosette in black ink (e.g., page 515)
        if best is None and sajdah_tmpl is not None:
            th, tw = sajdah_tmpl.shape[:2]
            b_u8 = black.astype(np.uint8) * 255
            y0, y1 = max(0, int(ey) - 130), min(H, int(ey) + 130)
            x0, x1 = max(0, int(ex) - 220), min(W, int(ex) + 220)
            if (y1 - y0) >= th and (x1 - x0) >= tw:
                res = cv2.matchTemplate(b_u8[y0:y1, x0:x1], sajdah_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.28:
                    rx0, ry0 = x0 + max_loc[0], y0 + max_loc[1]
                    rx1, ry1 = rx0 + tw, ry0 + th
                    cx, cy = 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)
                    li = min(range(len(lines)), key=lambda idx: abs(lines[idx]['y_base'] - cy))
                    best = {
                        'x0': int(rx0), 'y0': int(ry0), 'x1': int(rx1), 'y1': int(ry1),
                        'cx': cx, 'cy': cy, 'line_idx': li, 'score': float(max_val)
                    }

        if best is not None:
            found[es['row_idx']] = best
    return found
