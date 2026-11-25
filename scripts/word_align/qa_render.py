"""
Visual QA renderer for verifying aligned word coordinates (x, y, w, h).

Usage:
  python -m word_align.qa_render --edition shmrly --pages 2,3,4,24,144,515,522 --out <dir>
"""
import argparse
import json
import os
import sys
from PIL import Image, ImageDraw

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word_align.config import EDITIONS


def render_page_qa(edition_id: str, pnum: int, out_dir: str, scale: float = 0.5):
    cfg = EDITIONS[edition_id]
    img_path = os.path.join(cfg.pages_dir, f'{pnum}.png')
    json_path = os.path.join(cfg.output_dir, 'pages', f'{pnum}.json')
    if not os.path.exists(img_path) or not os.path.exists(json_path):
        print(f"Skipping p{pnum}: missing image or json")
        return

    with open(json_path, encoding='utf-8') as f:
        items = json.load(f)

    im = Image.open(img_path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    draw = ImageDraw.Draw(bg)
    W, H = im.size

    for it in items:
        x0 = it['x'] * W
        y0 = it['y'] * H
        x1 = (it['x'] + it['w']) * W
        y1 = (it['y'] + it['h']) * H
        wn = it['n']
        if wn == 999:
            color = (220, 0, 0, 230)
        elif wn == 1001:
            color = (0, 170, 0, 230)
        elif wn == 1000:
            color = (190, 0, 200, 230)
        else:
            color = (0, 90, 220, 200)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'qa_{edition_id}_p{pnum}.png')
    out_im = bg.convert('RGB')
    if scale != 1.0:
        out_im = out_im.resize((int(W * scale), int(H * scale)))
    out_im.save(out_path)
    print(f"Saved {out_path} ({len(items)} items)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', required=True, choices=list(EDITIONS.keys()))
    parser.add_argument('--pages', required=True, help='Comma-separated page numbers')
    parser.add_argument('--out', required=True, help='Output directory for QA PNGs')
    parser.add_argument('--scale', type=float, default=0.5)
    args = parser.parse_args()

    for p in args.pages.split(','):
        p = p.strip()
        if p:
            render_page_qa(args.edition, int(p), args.out, args.scale)


if __name__ == '__main__':
    main()
