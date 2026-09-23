#!/usr/bin/env python3
r"""
Convert Mushaf PDF to high-quality transparent PNG pages with minimal file size.
Specifically tuned for Arabic Mushaf editions with diacritics, colored qiraat marks, and ornate borders.
Supports both digital white-background PDFs and scanned parchment/cream-background PDFs.

Features:
- Automatic paper background color detection (cream/parchment vs digital white)
- De-fringing of residual paper background halo around text for clean dark mode
- Text contrast enhancement (deepens Arabic ink, removes scan noise, improves compressibility)
- Scanner margin gutter cleanup on standard text pages
- Adaptive palette quantization (36 colors for text pages, 128 colors for frontispiece)

Usage:
    python convert_editions.py --pdf "D:\editionsbuffer\madinaold1.pdf" --edition-symbol madinaold1 --dpi 120
    python convert_editions.py --pdf "D:\editionsbuffer\mushaf3shemerlyHQ.pdf" --edition-symbol shmrly_qalon
    python convert_editions.py --pdf "D:\editionsbuffer\Sho3baShamarlyDOC1.pdf" --edition-symbol shmrly_shoba
    python convert_editions.py --all
"""

import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Ensure UTF-8 console output on Windows
sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_EDITIONS_REPO = r"D:\Editions"
DEFAULT_BUFFER_DIR = r"D:\editionsbuffer"


def detect_mushaf_bg_color(pdf_path: str) -> tuple:
    """
    Probes multiple standard text pages to automatically detect the background paper color
    and whether the Mushaf has a cream/parchment scanned background vs pure white.
    
    Returns (bg_color_rgb_array, is_cream)
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    # Sample 5 standard text pages (skipping frontispiece pages 0 and 1)
    sample_indices = [idx for idx in [3, 9, 19, 49, 99] if idx < total_pages]
    if not sample_indices:
        sample_indices = [0]
    
    corner_samples = []
    for idx in sample_indices:
        p = doc[idx]
        pix = p.get_pixmap(dpi=72, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        # Extract 5x5 corner patches
        c1 = arr[:5, :5].reshape(-1, 3)
        c2 = arr[:5, -5:].reshape(-1, 3)
        c3 = arr[-5:, :5].reshape(-1, 3)
        c4 = arr[-5:, -5:].reshape(-1, 3)
        corner_samples.extend([c1, c2, c3, c4])
    doc.close()
    
    all_corners = np.vstack(corner_samples)
    bg_color = np.median(all_corners, axis=0)
    
    # If all components are >= 238, it's a white/near-white digital background
    is_cream = not (bg_color[0] >= 238 and bg_color[1] >= 238 and bg_color[2] >= 238)
    return bg_color, is_cream


def process_single_page(
    pdf_path: str,
    page_idx: int,
    output_file: str,
    dpi: int = 120,
    bg_color: tuple = None,
    is_cream: bool = False,
    thresh: int = None,
    max_colors: int = 128,
    clean_margins: bool = True,
    enhance_contrast: bool = True,
    text_colors: int = 36
) -> tuple:
    """
    Renders a single PDF page, removes background with de-fringing, enhances ink contrast,
    quantizes to optimized palette PNG with transparency, and saves to file.
    
    Returns (page_idx, file_size_bytes, success)
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        
        # Render page
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_rgb = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        arr = np.array(img_rgb)
        
        if is_cream and bg_color is not None:
            # Parchment / cream scanned background detection
            dist_thresh = thresh if thresh is not None else 30
            paper_gray = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
            diff = np.max(np.abs(arr.astype(float) - np.array(bg_color, dtype=float)), axis=-1)
            sat = np.max(arr, axis=-1) - np.min(arr, axis=-1)
            
            if page_idx < 2:
                # Decorative frontispiece pages 1 & 2
                bg_mask = (diff <= 26) & (gray >= (paper_gray - 20))
            else:
                # Standard text pages:
                # De-fringe: remove residual cream paper halo (gray >= 200)
                bg_mask = ((diff <= dist_thresh) & (gray >= (paper_gray - 28))) | (gray >= 200)
                
                # Clear outer scanner gutter/spine dark strip
                if clean_margins:
                    margin_w = max(1, int(12 * dpi / 150))
                    bg_mask[:, :margin_w] = True
                    bg_mask[:, -margin_w:] = True
                
                # Enhance text contrast: deepen dark calligraphy ink (gray < 140, sat < 40)
                if enhance_contrast:
                    ink_mask = (gray < 140) & (sat < 40) & (~bg_mask)
                    arr[ink_mask] = (arr[ink_mask].astype(float) * 0.65).astype(np.uint8)
        else:
            # Standard white / near-white background
            white_thresh = thresh if thresh is not None else 240
            bg_mask = (arr[:, :, 0] >= white_thresh) & (arr[:, :, 1] >= white_thresh) & (arr[:, :, 2] >= white_thresh)
        
        # Create alpha channel: 0 for background, 255 for content
        alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
        rgba_arr = np.dstack([arr, alpha])
        img_rgba = Image.fromarray(rgba_arr, 'RGBA')
        
        # Adaptive palette budget
        if page_idx < 2:
            colors = max_colors
        else:
            colors = min(text_colors, max_colors)
        
        # Quantize to palette mode with transparency
        img_p = img_rgba.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
        
        # Save optimized PNG
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        img_p.save(output_file, format='PNG', optimize=True)
        
        file_size = os.path.getsize(output_file)
        return (page_idx, file_size, True)
    except Exception as e:
        print(f"Error processing page {page_idx}: {e}", file=sys.stderr)
        return (page_idx, 0, False)


def convert_edition(
    pdf_path: str,
    edition_symbol: str,
    editions_repo_dir: str = DEFAULT_EDITIONS_REPO,
    dpi: int = 120,
    workers: int = None,
    bg_mode: str = "auto",
    thresh: int = None,
    clean_margins: bool = True,
    enhance_contrast: bool = True,
    text_colors: int = 36
):
    r"""
    Converts all pages in a PDF to the editions directory structure:
    D:\Editions\{edition_symbol}\assets\edition\pages\{1..N}.png
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    pages_dir = os.path.join(editions_repo_dir, edition_symbol, "assets", "edition", "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    # Detect or configure background
    if bg_mode == "auto":
        detected_bg, is_cream = detect_mushaf_bg_color(pdf_path)
    elif bg_mode == "white":
        detected_bg, is_cream = np.array([255., 255., 255.]), False
    elif bg_mode == "cream":
        detected_bg, is_cream = detect_mushaf_bg_color(pdf_path)
        is_cream = True
    else:
        # Custom RGB string, e.g. "249,242,215"
        try:
            parts = [float(x.strip()) for x in bg_mode.split(",")]
            detected_bg = np.array(parts)
            is_cream = not (detected_bg[0] >= 238 and detected_bg[1] >= 238 and detected_bg[2] >= 238)
        except Exception:
            detected_bg, is_cream = detect_mushaf_bg_color(pdf_path)
    
    bg_desc = f"Parchment/Cream [R={detected_bg[0]:.0f}, G={detected_bg[1]:.0f}, B={detected_bg[2]:.0f}]" if is_cream else f"White/Digital (thresh={thresh or 240})"
    
    print(f"==================================================")
    print(f"Starting conversion for edition: {edition_symbol}")
    print(f"Source PDF:         {pdf_path}")
    print(f"Total pages:        {total_pages}")
    print(f"Output directory:   {pages_dir}")
    print(f"Rendering DPI:      {dpi}")
    print(f"Background mode:    {bg_desc}")
    print(f"Clean margins:      {clean_margins}")
    print(f"Enhance contrast:   {enhance_contrast}")
    print(f"Text palette colors:{text_colors}")
    print(f"==================================================")
    
    start_time = time.time()
    total_bytes = 0
    success_count = 0
    
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    
    tasks = []
    bg_tuple = tuple(detected_bg)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for p_idx in range(total_pages):
            page_num = p_idx + 1  # 1-indexed (1.png, 2.png, ..., 604.png)
            output_file = os.path.join(pages_dir, f"{page_num}.png")
            tasks.append(
                executor.submit(
                    process_single_page,
                    pdf_path,
                    p_idx,
                    output_file,
                    dpi,
                    bg_tuple,
                    is_cream,
                    thresh,
                    128,
                    clean_margins,
                    enhance_contrast,
                    text_colors
                )
            )
        
        for future in as_completed(tasks):
            p_idx, sz, success = future.result()
            if success:
                success_count += 1
                total_bytes += sz
            
            if success_count % 50 == 0 or success_count == total_pages:
                elapsed = time.time() - start_time
                pct = (success_count / total_pages) * 100
                print(f"Progress: {success_count}/{total_pages} ({pct:.1f}%) | "
                      f"Current Total: {total_bytes / (1024 * 1024):.2f} MB | "
                      f"Elapsed: {elapsed:.1f}s")
    
    total_mb = total_bytes / (1024 * 1024)
    elapsed_total = time.time() - start_time
    print(f"==================================================")
    print(f"Finished {edition_symbol} successfully!")
    print(f"Converted:  {success_count}/{total_pages} pages")
    print(f"Total Size: {total_mb:.2f} MB (Average: {total_mb * 1024 / max(1, success_count):.1f} KB/page)")
    print(f"Time Taken: {elapsed_total:.2f} seconds ({elapsed_total / max(1, success_count):.2f} s/page)")
    print(f"==================================================\n")
    return success_count, total_mb


def main():
    parser = argparse.ArgumentParser(description="Convert Quran PDF editions to transparent PNGs.")
    parser.add_argument("--pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--edition-symbol", type=str, help="Target edition symbol (e.g. madinaold1, shmrly_qalon)")
    parser.add_argument("--repo-dir", type=str, default=DEFAULT_EDITIONS_REPO, help="Editions repository directory")
    parser.add_argument("--dpi", type=int, default=120, help="Rendering DPI (default 120, use 150 for high-dpi)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--bg-mode", type=str, default="auto", help="Background mode: 'auto', 'white', 'cream', or 'R,G,B' (default: auto)")
    parser.add_argument("--thresh", type=int, default=None, help="Threshold override (color distance for cream or min brightness for white)")
    parser.add_argument("--no-clean-margins", action="store_true", help="Disable scanner margin gutter cleanup on standard pages")
    parser.add_argument("--no-contrast", action="store_true", help="Disable calligraphy ink contrast deepening")
    parser.add_argument("--colors", type=int, default=36, help="Palette colors for text pages (default 36)")
    parser.add_argument("--all", action="store_true", help="Process all default buffer PDFs")
    
    args = parser.parse_args()
    
    if args.all or (not args.pdf and not args.edition_symbol):
        jobs = [
            (os.path.join(DEFAULT_BUFFER_DIR, "warshkf.pdf"), "warshkf"),
            (os.path.join(DEFAULT_BUFFER_DIR, "madinaold1.pdf"), "madinaold1"),
        ]
        for pdf_path, symbol in jobs:
            if os.path.exists(pdf_path):
                convert_edition(
                    pdf_path,
                    symbol,
                    editions_repo_dir=args.repo_dir,
                    dpi=args.dpi,
                    workers=args.workers,
                    bg_mode=args.bg_mode,
                    thresh=args.thresh,
                    clean_margins=not args.no_clean_margins,
                    enhance_contrast=not args.no_contrast,
                    text_colors=args.colors
                )
            else:
                print(f"File not found: {pdf_path}", file=sys.stderr)
    else:
        if not args.pdf or not args.edition_symbol:
            parser.error("Both --pdf and --edition-symbol are required when not using --all")
        convert_edition(
            args.pdf,
            args.edition_symbol,
            editions_repo_dir=args.repo_dir,
            dpi=args.dpi,
            workers=args.workers,
            bg_mode=args.bg_mode,
            thresh=args.thresh,
            clean_margins=not args.no_clean_margins,
            enhance_contrast=not args.no_contrast,
            text_colors=args.colors
        )


if __name__ == "__main__":
    main()
