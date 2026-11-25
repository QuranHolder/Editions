"""Core PAW-based Dynamic Programming Forced Aligner for Arabic Mushaf pages."""
import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .config import EditionConfig
from .arabic_features import get_uthmani_words_for_ayah, get_word_features
from .anchors import (
    load_ink_masks,
    detect_lines,
    detect_medallions,
    find_medallion_by_template,
    detect_rub_stars,
    detect_sajdahs,
)


def extract_line_paws(
    black: np.ndarray,
    line: Dict,
    exclude_boxes: List[Dict],
    scale_y: float = 1.0,
) -> Tuple[List[Dict], np.ndarray]:
    """Extract main-body PAWs (connected components) on a single line, sorted RTL."""
    H, W = black.shape
    y0, y1 = line['y_top'], line['y_bot']
    y_base = line['y_base'] - y0
    strip = black[y0:y1, :].copy()

    for b in exclude_boxes:
        bx0 = max(0, b['x0'] - 2)
        bx1 = min(W, b['x1'] + 2)
        by0 = max(0, b['y0'] - y0 - 2)
        by1 = min(y1 - y0, b['y1'] - y0 + 2)
        if by1 > by0 and bx1 > bx0:
            strip[by0:by1, bx0:bx1] = 0

    num, lbl, stats, centroids = cv2.connectedComponentsWithStats(strip.astype(np.uint8), connectivity=8)
    paws = []
    sh = y1 - y0
    core_lo = max(0, y_base - int(round(30 * scale_y)))
    core_hi = min(sh, y_base + int(round(18 * scale_y)))
    tall_thresh = int(round(48 * scale_y))
    min_core_px = int(round(88 * scale_y * scale_y))
    min_thin_px = int(round(42 * scale_y * scale_y))
    min_thin_h = int(round(27 * scale_y))
    min_large_area = int(round(440 * scale_y * scale_y))

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 15:
            continue
        comp_mask = (lbl[y:y + h, x:x + w] == i)
        oy0 = max(y, core_lo)
        oy1 = min(y + h, core_hi)
        core_px = int(comp_mask[oy0 - y:oy1 - y, :].sum()) if oy1 > oy0 else 0

        is_paw = (
            (core_px >= min_core_px)
            or (core_px >= min_thin_px and h >= min_thin_h)
            or (area >= min_large_area and oy1 > oy0)
        )
        if not is_paw:
            continue

        # Count tall vertical strokes in this PAW
        any_col = comp_mask.any(axis=0)
        top_y = np.where(any_col, np.argmax(comp_mask, axis=0) + y, y_base)
        tall_cols = (y_base - top_y) >= tall_thresh
        tall_cnt = 0
        in_run = False
        for tc in tall_cols:
            if tc and not in_run:
                tall_cnt += 1
                in_run = True
            elif not tc:
                in_run = False

        # Is this PAW a standalone Alif-like vertical stroke?
        is_alif_like = (tall_cnt == 1 and w <= int(round(38 * scale_y)) and h >= int(round(45 * scale_y)))

        if oy1 > oy0 and core_px > 0:
            sub = comp_mask[oy0 - y:oy1 - y, :]
            xs = np.where(sub)[1] + x
            core_cx = float(xs.mean())
        else:
            core_cx = float(centroids[i][0])

        paws.append({
            'x0': int(x),
            'x1': int(x + w),
            'y0': int(y + y0),
            'y1': int(y + h + y0),
            'cx': core_cx,
            'w': int(w),
            'h': int(h),
            'area': int(area),
            'tall': tall_cnt,
            'is_alif_like': is_alif_like,
        })

    paws.sort(key=lambda p: -p['cx'])
    return paws, strip


def _split_paw_at_valley(paw: Dict, strip: np.ndarray) -> Tuple[Dict, Dict]:
    """Split a touching PAW at its narrowest vertical ink projection valley."""
    x0, x1 = paw['x0'], paw['x1']
    w = x1 - x0
    if w < 10:
        mid = (x0 + x1) // 2
    else:
        col_proj = strip[:, x0:x1].sum(axis=0)
        lo = max(2, int(w * 0.22))
        hi = min(w - 2, int(w * 0.78))
        mid = x0 + lo + int(np.argmin(col_proj[lo:hi]))
    p_right = dict(
        paw,
        x0=mid,
        w=x1 - mid,
        cx=0.5 * (mid + x1),
        tall=max(0, paw['tall'] // 2),
        is_alif_like=False,
    )
    p_left = dict(
        paw,
        x1=mid,
        w=mid - x0,
        cx=0.5 * (x0 + mid),
        tall=paw['tall'] - max(0, paw['tall'] // 2),
        is_alif_like=False,
    )
    return p_right, p_left


def align_segment_dp(
    words: List[Dict],
    paws: List[Dict],
    strip: np.ndarray,
    seg_x_left: float,
    seg_x_right: float,
    W_img: int,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Align M words (RTL) to K PAWs (RTL) using Dynamic Programming.
    Returns (list of (x0, x1) spans, average per-word DP cost).
    """
    M = len(words)
    if M == 0:
        return [], 0.0

    paws = list(paws)
    K = len(paws)
    if K == 0:
        tot_w = sum(w['weight'] for w in words)
        cur_r = seg_x_right
        res = []
        for w in words:
            ww = (seg_x_right - seg_x_left) * (w['weight'] / tot_w)
            res.append((int(round(max(seg_x_left, cur_r - ww))), int(round(cur_r))))
            cur_r -= ww
        return res, 15.0

    split_penalty = 0.0
    while len(paws) < M:
        idx = max(range(len(paws)), key=lambda i: paws[i]['w'])
        p_r, p_l = _split_paw_at_valley(paws[idx], strip)
        paws[idx:idx + 1] = [p_r, p_l]
        split_penalty += 2.5
    K = len(paws)

    tot_weight = sum(w['weight'] for w in words)
    seg_span = max(1.0, seg_x_right - seg_x_left)
    gaps = [paws[k]['x0'] - paws[k + 1]['x1'] for k in range(K - 1)]

    INF = 1e18
    dp = [[INF] * (K + 1) for _ in range(M + 1)]
    parent = [[-1] * (K + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for m in range(1, M + 1):
        w = words[m - 1]
        exp_p = w['exp_paws']
        exp_tall = w['exp_tall']
        starts_alif = w['starts_alif']
        starts_tall = w['starts_tall']
        exp_w_px = seg_span * (w['weight'] / tot_weight)
        prior_cx = (w['old_x'] + 0.5 * w['old_w']) * W_img if (w['old_x'] is not None and w['old_w'] is not None) else None

        k_min = m
        k_max = K - (M - m)
        for k in range(k_min, k_max + 1):
            a_min = m - 1
            a_max = k - 1
            for a in range(a_min, a_max + 1):
                if dp[m - 1][a] >= INF:
                    continue
                cnt = k - a
                span_x0 = min(paws[p]['x0'] for p in range(a, k))
                span_x1 = max(paws[p]['x1'] for p in range(a, k))
                span_w = max(1.0, span_x1 - span_x0)
                span_cx = 0.5 * (span_x0 + span_x1)
                span_tall = sum(paws[p]['tall'] for p in range(a, k))

                # 1. PAW count cost
                diff_p = abs(cnt - exp_p)
                if diff_p == 0:
                    cost_paw = 0.0
                elif diff_p == 1:
                    cost_paw = 3.2
                else:
                    cost_paw = 3.2 + (diff_p - 1) * 8.0

                # 2. Kashida-tolerant width cost
                ratio = span_w / max(15.0, exp_w_px)
                if ratio < 1.0:
                    cost_w = ((1.0 - ratio) ** 2) * 5.0
                else:
                    cost_w = min(2.0, (ratio - 1.0) * 1.15)

                # 3. Tall stroke count + initial Alif/tall alignment
                cost_tall = abs(span_tall - exp_tall) * 0.85
                first_paw = paws[a]
                if starts_alif:
                    if first_paw['is_alif_like']:
                        cost_tall -= 1.2  # Strong bonus for matching leading Alif!
                    elif first_paw['tall'] == 0:
                        cost_tall += 1.4
                elif not starts_tall and first_paw['is_alif_like']:
                    cost_tall += 1.3  # Penalize stealing a leading Alif from the next/prev word!

                # 4. Boundary gap bonus/penalty
                cost_gap = 0.0
                if m > 1:
                    g = gaps[a - 1]
                    if g < -8:
                        cost_gap = 3.5
                    elif g < 0:
                        cost_gap = 1.0
                    else:
                        cost_gap = -min(g, 18.0) * 0.14

                # 5. Weak prior from old x (optional, zero when no prior exists)
                cost_prior = 0.0
                if prior_cx is not None:
                    cost_prior = min(2.0, abs(span_cx - prior_cx) / 250.0) * 0.55

                total = dp[m - 1][a] + cost_paw + cost_w + cost_tall + cost_gap + cost_prior
                if total < dp[m][k]:
                    dp[m][k] = total
                    parent[m][k] = a

    if dp[M][K] >= INF or parent[M][K] < 0:
        tot_w = sum(w['weight'] for w in words)
        cur_r = seg_x_right
        res = []
        for w in words:
            ww = (seg_x_right - seg_x_left) * (w['weight'] / tot_w)
            res.append((int(round(max(seg_x_left, cur_r - ww))), int(round(cur_r))))
            cur_r -= ww
        return res, 15.0

    spans = []
    cur_k = K
    for m in range(M, 0, -1):
        a = parent[m][cur_k]
        span_x0 = min(paws[p]['x0'] for p in range(a, cur_k))
        span_x1 = max(paws[p]['x1'] for p in range(a, cur_k))
        spans.append((int(span_x0), int(span_x1)))
        cur_k = a
    spans.reverse()

    avg_cost = (dp[M][K] + split_penalty) / float(M)
    return spans, avg_cost


def align_page(
    pnum: int,
    rows: List[Tuple],
    quran_text: Dict[str, str],
    templates: Dict[str, Optional[np.ndarray]],
    cfg: EditionConfig,
) -> Dict:
    """
    Align all tokens on page `pnum`.
    `rows`: list of (wordindex, surah, ayah, wordsno, aya_index, word, rawword, old_x, old_y, old_w)
    Returns dict with aligned word entries + QA metrics.
    """
    img_path = os.path.join(cfg.pages_dir, f'{pnum}.png')
    black, red, W, H = load_ink_masks(img_path, cfg)
    scale_y = H / float(cfg.ref_height)

    lines = detect_lines(black, cfg)
    medallions = detect_medallions(black, lines, cfg)

    uthmani_cache: Dict[int, List[str]] = {}
    tokens = []
    expected_rubs = []
    expected_sajdahs = []
    expected_meds = 0

    for idx, row in enumerate(rows):
        wi, s, a, wn, aidx, word, rawword, ox, oy, ow = row
        tok = {
            'row_idx': idx, 'id': wi, 's': s, 'a': a, 'n': wn, 'ayaIdx': aidx,
            'word': word, 'rawword': rawword,
            'old_x': ox, 'old_y': oy, 'old_w': ow,
        }
        if wn == 1001:
            expected_rubs.append(tok)
        elif wn == 1000:
            expected_sajdahs.append(tok)
        elif wn == 999:
            expected_meds += 1
        else:
            if aidx not in uthmani_cache:
                uthmani_cache[aidx] = get_uthmani_words_for_ayah(quran_text, aidx)
            u_list = uthmani_cache[aidx]
            u_word = u_list[wn - 1] if (1 <= wn <= len(u_list)) else (word or rawword or '')
            feats = get_word_features(u_word)
            tok.update(feats)
            tok['uthmani'] = u_word
        tokens.append(tok)

    rub_by_row = detect_rub_stars(black, lines, expected_rubs, templates.get('rub'), cfg)
    saj_by_row = detect_sajdahs(red, black, lines, expected_sajdahs, templates.get('sajdah'), cfg)

    # Group tokens into DB lines by old_y (or fallback if no old_y)
    db_lines: List[Dict] = []
    for tok in tokens:
        ty = tok['old_y'] if tok['old_y'] is not None else 0.0
        if not db_lines or abs(ty - db_lines[-1]['y']) > 0.022:
            db_lines.append({'y': ty, 'tokens': [tok]})
        else:
            db_lines[-1]['tokens'].append(tok)

    # Map each db_line to closest detected image line
    for dbl in db_lines:
        target_y = (dbl['y'] - 0.022) * H
        best_li = min(range(len(lines)), key=lambda li: abs(lines[li]['y_base'] - target_y))
        dbl['img_line_idx'] = best_li

    meds_by_line: Dict[int, List[Dict]] = {}
    for m in medallions:
        meds_by_line.setdefault(m['line_idx'], []).append(m)
    for li in meds_by_line:
        meds_by_line[li].sort(key=lambda m: -m['cx'])

    new_boxes: Dict[int, Tuple[int, int, int, int, float]] = {}  # row_idx -> (x0, y0, x1, y1, conf)
    meds_matched = 0
    seg_costs: List[float] = []

    for dbl in db_lines:
        li = dbl['img_line_idx']
        ln = lines[li]
        ltoks = dbl['tokens']
        lmeds = list(meds_by_line.get(li, []))

        tok_999 = [t for t in ltoks if t['n'] == 999]
        anchor_map: Dict[int, Dict] = {}
        if len(tok_999) == len(lmeds):
            for t, m in zip(tok_999, lmeds):
                anchor_map[t['row_idx']] = m
                meds_matched += 1
        else:
            used_m = set()
            matched_med_boxes = []
            for t in tok_999:
                tx = (t['old_x'] + 0.5 * (t['old_w'] or 0.05)) * W if t['old_x'] is not None else W * 0.5
                best_j = None
                best_d = 1e9
                for j, m in enumerate(lmeds):
                    if j in used_m:
                        continue
                    d = abs(m['cx'] - tx)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is not None and best_d < 220:
                    used_m.add(best_j)
                    anchor_map[t['row_idx']] = lmeds[best_j]
                    matched_med_boxes.append(lmeds[best_j])
                    meds_matched += 1
                else:
                    # Fallback: template match on this line for touching medallions!
                    tm = find_medallion_by_template(black, ln, tx, templates.get('medallion'), matched_med_boxes)
                    if tm is not None:
                        anchor_map[t['row_idx']] = tm
                        matched_med_boxes.append(tm)
                        meds_matched += 1

        for t in ltoks:
            if t['row_idx'] in rub_by_row:
                anchor_map[t['row_idx']] = rub_by_row[t['row_idx']]
            elif t['row_idx'] in saj_by_row:
                anchor_map[t['row_idx']] = saj_by_row[t['row_idx']]

        for ridx, ab in anchor_map.items():
            new_boxes[ridx] = (ab['x0'], ln['y_top'], ab['x1'], ln['y_bot'], 1.0)

        paws, strip = extract_line_paws(black, ln, list(anchor_map.values()), scale_y)
        all_x0 = [p['x0'] for p in paws] + [ab['x0'] for ab in anchor_map.values()]
        all_x1 = [p['x1'] for p in paws] + [ab['x1'] for ab in anchor_map.values()]
        line_x_left = min(all_x0) if all_x0 else int(0.02 * W)
        line_x_right = max(all_x1) if all_x1 else int(0.98 * W)

        seg_words: List[Dict] = []
        seg_right = float(line_x_right)
        for t in ltoks:
            if t['row_idx'] in anchor_map:
                ab = anchor_map[t['row_idx']]
                seg_left = float(ab['x1'])
                if seg_words:
                    seg_paws = [p for p in paws if seg_left - 8 <= p['cx'] <= seg_right + 8]
                    spans, avg_cost = align_segment_dp(seg_words, seg_paws, strip, seg_left, seg_right, W)
                    seg_costs.append(avg_cost)
                    conf = max(0.0, min(1.0, 1.0 - avg_cost / 15.0))
                    for sw, (sx0, sx1) in zip(seg_words, spans):
                        new_boxes[sw['row_idx']] = (sx0, ln['y_top'], sx1, ln['y_bot'], conf)
                    seg_words = []
                seg_right = float(ab['x0'])
            elif t['n'] not in (999, 1000, 1001):
                seg_words.append(t)
            else:
                # Unmatched special marker fallback to old_x
                ox = int(round((t['old_x'] or 0.0) * W))
                ow = max(20, int(round((t['old_w'] or 0.03) * W)))
                new_boxes[t['row_idx']] = (ox, ln['y_top'], ox + ow, ln['y_bot'], 0.2)

        if seg_words:
            seg_left = float(line_x_left)
            seg_paws = [p for p in paws if seg_left - 8 <= p['cx'] <= seg_right + 8]
            spans, avg_cost = align_segment_dp(seg_words, seg_paws, strip, seg_left, seg_right, W)
            seg_costs.append(avg_cost)
            conf = max(0.0, min(1.0, 1.0 - avg_cost / 15.0))
            for sw, (sx0, sx1) in zip(seg_words, spans):
                new_boxes[sw['row_idx']] = (sx0, ln['y_top'], sx1, ln['y_bot'], conf)

    # Convert to normalized output records (x, y_top, width, height)
    aligned_items = []
    for tok in tokens:
        ridx = tok['row_idx']
        if ridx in new_boxes:
            x0, y0, x1, y1, conf = new_boxes[ridx]
        else:
            ox = (tok['old_x'] or 0.0) * W
            ow = (tok['old_w'] or 0.05) * W
            oy = ((tok['old_y'] or 0.06) - 0.055) * H
            x0, y0, x1, y1, conf = int(ox), int(oy), int(ox + ow), int(oy + 0.055 * H), 0.1

        x_rat = round(max(0.0, min(1.0, x0 / float(W))), 4)
        y_rat = round(max(0.0, min(1.0, y0 / float(H))), 4)
        w_rat = round(max(0.002, min(1.0 - x_rat, (x1 - x0) / float(W))), 4)
        h_rat = round(max(0.005, min(1.0 - y_rat, (y1 - y0) / float(H))), 4)

        aligned_items.append({
            'id': tok['id'],
            's': tok['s'],
            'a': tok['a'],
            'n': tok['n'],
            'ayaIdx': tok['ayaIdx'],
            'x': x_rat,
            'y': y_rat,
            'w': w_rat,
            'h': h_rat,
            'conf': round(conf, 3),
        })

    mean_cost = float(np.mean(seg_costs)) if seg_costs else 0.0
    max_cost = float(np.max(seg_costs)) if seg_costs else 0.0
    return {
        'page': pnum,
        'width': W,
        'height': H,
        'items': aligned_items,
        'qa': {
            'lines_detected': len(lines),
            'db_lines': len(db_lines),
            'meds_expected': expected_meds,
            'meds_matched': meds_matched,
            'rubs_expected': len(expected_rubs),
            'rubs_matched': len(rub_by_row),
            'sajdahs_expected': len(expected_sajdahs),
            'sajdahs_matched': len(saj_by_row),
            'mean_dp_cost': round(mean_cost, 3),
            'max_dp_cost': round(max_cost, 3),
        }
    }
