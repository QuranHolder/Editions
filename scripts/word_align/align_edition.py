"""
CLI Runner for Multi-Edition Word-Coordinate Forced Alignment.

Usage:
  python -m word_align.align_edition --edition shmrly --workers 8
  python -m word_align.align_edition --edition shmrly --pages 2,3,4,24,144,522
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Allow running both as `python -m word_align.align_edition` and directly
if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word_align.config import EDITIONS, EditionConfig
from word_align.arabic_features import load_quran_text
from word_align.anchors import build_marker_templates
from word_align.aligner import align_page

_WORKER_CTX: Dict = {}


def _init_worker(cfg: EditionConfig, quran_text: Dict[str, str], templates: Dict):
    _WORKER_CTX['cfg'] = cfg
    _WORKER_CTX['quran_text'] = quran_text
    _WORKER_CTX['templates'] = templates


def _run_page_worker(args: Tuple[int, List[Tuple]]) -> Dict:
    pnum, rows = args
    return align_page(
        pnum,
        rows,
        _WORKER_CTX['quran_text'],
        _WORKER_CTX['templates'],
        _WORKER_CTX['cfg'],
    )


def write_outputs(cfg: EditionConfig, results: List[Dict]):
    os.makedirs(cfg.output_dir, exist_ok=True)
    pages_out = os.path.join(cfg.output_dir, 'pages')
    os.makedirs(pages_out, exist_ok=True)

    # 1. Per-page JSON files (compact: id, s, a, n, ayaIdx, x, y, w, h)
    for res in results:
        pnum = res['page']
        compact = [
            {
                'id': it['id'], 's': it['s'], 'a': it['a'], 'n': it['n'],
                'ayaIdx': it['ayaIdx'],
                'x': it['x'], 'y': it['y'], 'w': it['w'], 'h': it['h'],
            }
            for it in res['items']
        ]
        with open(os.path.join(pages_out, f'{pnum}.json'), 'w', encoding='utf-8') as f:
            json.dump(compact, f, ensure_ascii=False, separators=(',', ':'))

    # 2. Unified SQLite database (<edition>_words.db)
    db_out = os.path.join(cfg.output_dir, f'{cfg.edition_id}_words.db')
    if os.path.exists(db_out):
        os.remove(db_out)
    conn = sqlite3.connect(db_out)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE words (
            page INTEGER NOT NULL,
            wordindex INTEGER NOT NULL,
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            wordsno INTEGER NOT NULL,
            aya_index INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    batch = []
    for res in results:
        pnum = res['page']
        for it in res['items']:
            batch.append((
                pnum, it['id'], it['s'], it['a'], it['n'], it['ayaIdx'],
                it['x'], it['y'], it['w'], it['h'], it['conf']
            ))
    c.executemany(
        "INSERT INTO words VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch
    )
    c.execute("CREATE INDEX idx_words_page ON words(page)")
    c.execute("CREATE INDEX idx_words_surah_ayah ON words(surah, ayah)")
    conn.commit()
    conn.close()

    # 3. Manifest + QA summary report
    tot_med_exp = sum(r['qa']['meds_expected'] for r in results)
    tot_med_mat = sum(r['qa']['meds_matched'] for r in results)
    tot_rub_exp = sum(r['qa']['rubs_expected'] for r in results)
    tot_rub_mat = sum(r['qa']['rubs_matched'] for r in results)
    tot_saj_exp = sum(r['qa']['sajdahs_expected'] for r in results)
    tot_saj_mat = sum(r['qa']['sajdahs_matched'] for r in results)
    tot_items = sum(len(r['items']) for r in results)

    flagged = []
    for r in results:
        qa = r['qa']
        reasons = []
        if qa['meds_matched'] < qa['meds_expected']:
            reasons.append(f"meds {qa['meds_matched']}/{qa['meds_expected']}")
        if qa['rubs_matched'] < qa['rubs_expected']:
            reasons.append(f"rubs {qa['rubs_matched']}/{qa['rubs_expected']}")
        if qa['sajdahs_matched'] < qa['sajdahs_expected']:
            reasons.append(f"sajdah {qa['sajdahs_matched']}/{qa['sajdahs_expected']}")
        if qa['mean_dp_cost'] > 2.2:
            reasons.append(f"mean_cost={qa['mean_dp_cost']}")
        if reasons:
            flagged.append({'page': r['page'], 'reasons': reasons, 'qa': qa})

    flagged.sort(key=lambda x: (-len(x['reasons']), -x['qa']['mean_dp_cost']))

    manifest = {
        'schemaVersion': 2,
        'edition': cfg.edition_id,
        'name': cfg.name,
        'yOrigin': 'top',
        'coordinates': {
            'x': 'left edge ratio (0..1)',
            'y': 'top edge ratio (0..1)',
            'w': 'width ratio (0..1)',
            'h': 'height ratio (0..1)',
        },
        'specialWordsNo': {
            '999': 'ayah_end_medallion',
            '1000': 'sajdah_sign',
            '1001': 'rub_hizb_sign',
        },
        'pagesFrom': min(r['page'] for r in results),
        'pagesTo': max(r['page'] for r in results),
        'totalPages': len(results),
        'totalTokens': tot_items,
        'qaSummary': {
            'medallions': f'{tot_med_mat}/{tot_med_exp}',
            'rubs': f'{tot_rub_mat}/{tot_rub_exp}',
            'sajdahs': f'{tot_saj_mat}/{tot_saj_exp}',
            'flaggedPagesCount': len(flagged),
        },
        'flaggedPages': flagged[:50],
    }
    with open(os.path.join(cfg.output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE: {cfg.edition_id} ===")
    print(f"Output dir : {cfg.output_dir}")
    print(f"Total pages: {len(results)}, Total tokens: {tot_items}")
    print(f"Medallions : {tot_med_mat}/{tot_med_exp} ({100.0*tot_med_mat/max(1,tot_med_exp):.2f}%)")
    print(f"Rub' stars : {tot_rub_mat}/{tot_rub_exp} ({100.0*tot_rub_mat/max(1,tot_rub_exp):.2f}%)")
    print(f"Sajdahs    : {tot_saj_mat}/{tot_saj_exp} ({100.0*tot_saj_mat/max(1,tot_saj_exp):.2f}%)")
    print(f"Flagged    : {len(flagged)} pages")
    for fl in flagged[:15]:
        print(f"  Flagged p{fl['page']}: {', '.join(fl['reasons'])}")


def main():
    parser = argparse.ArgumentParser(description='Align Quran word coordinates for an edition.')
    parser.add_argument('--edition', required=True, choices=list(EDITIONS.keys()))
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument('--pages', type=str, default='', help='Optional comma-separated page list')
    args = parser.parse_args()

    cfg = EDITIONS[args.edition]
    t0 = time.time()

    quran_text = load_quran_text(cfg.quran_text_path)
    conn = sqlite3.connect(cfg.words_db_path)
    templates = build_marker_templates(cfg, conn)
    print(f"Templates built: rub={templates['rub'].shape if templates['rub'] is not None else None}, "
          f"sajdah={templates['sajdah'].shape if templates['sajdah'] is not None else None}")

    if args.pages:
        target_pages = [int(p.strip()) for p in args.pages.split(',') if p.strip()]
    else:
        target_pages = list(range(cfg.pages_from, cfg.pages_to + 1))

    c = conn.cursor()
    x_sel = cfg.x_col or 'NULL'
    y_sel = cfg.y_col or 'NULL'
    w_sel = cfg.w_col or 'NULL'

    tasks = []
    for p in target_pages:
        rows = c.execute(
            f"SELECT wordindex, surah, ayah, wordsno, aya_index, word, rawword, {x_sel}, {y_sel}, {w_sel} "
            f"FROM wordsall WHERE {cfg.page_col}=? ORDER BY wordindex, wordsno",
            (p,)
        ).fetchall()
        if rows:
            tasks.append((p, rows))
    conn.close()

    print(f"Aligning {len(tasks)} pages using {args.workers} workers...")
    results = []
    if args.workers <= 1 or len(tasks) <= 4:
        _init_worker(cfg, quran_text, templates)
        for t in tasks:
            results.append(_run_page_worker(t))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(cfg, quran_text, templates),
        ) as ex:
            futs = [ex.submit(_run_page_worker, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    print(f"  Progress: {done}/{len(tasks)} ({time.time()-t0:.1f}s)")

    results.sort(key=lambda r: r['page'])
    write_outputs(cfg, results)
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
