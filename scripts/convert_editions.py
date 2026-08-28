#!/usr/bin/env python3
r"""
Convert Mushaf PDF to high-quality transparent PNG pages with minimal file size.
Specifically tuned for Arabic Mushaf editions with diacritics, colored qiraat marks, and ornate borders.

Usage:
    python convert_editions.py --pdf "D:\editionsbuffer\QaloonShamarlyDoc1.pdf" --edition-symbol shmrly_qalon
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

def process_single_page(pdf_path: str, page_idx: int, output_file: str, dpi: int = 150, thresh: int = 242, max_colors: int = 128) -> tuple:
    """
    Renders a single PDF page, makes white background transparent,
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
        
        # Detect white/near-white background
        # Background is where R, G, B are all >= threshold
        bg_mask = (arr[:, :, 0] >= thresh) & (arr[:, :, 1] >= thresh) & (arr[:, :, 2] >= thresh)
        
        # Create alpha channel: 0 for background, 255 for content
        alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
        rgba_arr = np.dstack([arr, alpha])
        img_rgba = Image.fromarray(rgba_arr, 'RGBA')
        
        # Check colored pixels to adaptively choose palette size
        # Pages with heavy borders/decorations use full palette (128 colors)
        # Standard text pages use 64 colors for even smaller footprint
        fg_mask = ~bg_mask
        fg_pixels = arr[fg_mask]
        if len(fg_pixels) > 0:
            colored_mask = (
                (np.abs(fg_pixels[:, 0].astype(int) - fg_pixels[:, 1].astype(int)) > 25) |
                (np.abs(fg_pixels[:, 1].astype(int) - fg_pixels[:, 2].astype(int)) > 25) |
                (np.abs(fg_pixels[:, 0].astype(int) - fg_pixels[:, 2].astype(int)) > 25)
            )
            is_decorative = np.sum(colored_mask) > 10000
            colors = max_colors if is_decorative else min(64, max_colors)
        else:
            colors = 32
        
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


def convert_edition(pdf_path: str, edition_symbol: str, editions_repo_dir: str = DEFAULT_EDITIONS_REPO, dpi: int = 150, workers: int = None):
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
    
    print(f"==================================================")
    print(f"Starting conversion for edition: {edition_symbol}")
    print(f"Source PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Output directory: {pages_dir}")
    print(f"Rendering DPI: {dpi}")
    print(f"==================================================")
    
    start_time = time.time()
    total_bytes = 0
    success_count = 0
    
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    
    tasks = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for p_idx in range(total_pages):
            page_num = p_idx + 1  # 1-indexed (1.png, 2.png, ..., 522.png)
            output_file = os.path.join(pages_dir, f"{page_num}.png")
            tasks.append(
                executor.submit(process_single_page, pdf_path, p_idx, output_file, dpi)
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
    print(f"Converted: {success_count}/{total_pages} pages")
    print(f"Total Size: {total_mb:.2f} MB (Average: {total_mb * 1024 / max(1, success_count):.1f} KB/page)")
    print(f"Time Taken: {elapsed_total:.2f} seconds ({elapsed_total / max(1, success_count):.2f} s/page)")
    print(f"==================================================\n")
    return success_count, total_mb


def main():
    parser = argparse.ArgumentParser(description="Convert Quran PDF editions to transparent PNGs.")
    parser.add_argument("--pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--edition-symbol", type=str, help="Target edition symbol (e.g. shmrly_qalon)")
    parser.add_argument("--repo-dir", type=str, default=DEFAULT_EDITIONS_REPO, help="Editions repository directory")
    parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI (default 150)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--all", action="store_true", help="Process all default buffer PDFs (Qaloon and Sho3ba)")
    
    args = parser.parse_args()
    
    if args.all or (not args.pdf and not args.edition_symbol):
        # Default batch run for the buffer folder
        jobs = [
            (os.path.join(DEFAULT_BUFFER_DIR, "QaloonShamarlyDoc1.pdf"), "shmrly_qalon"),
            (os.path.join(DEFAULT_BUFFER_DIR, "Sho3baShamarlyDOC1.pdf"), "shmrly_shoba"),
        ]
        for pdf_path, symbol in jobs:
            if os.path.exists(pdf_path):
                convert_edition(pdf_path, symbol, editions_repo_dir=args.repo_dir, dpi=args.dpi, workers=args.workers)
            else:
                print(f"File not found: {pdf_path}", file=sys.stderr)
    else:
        if not args.pdf or not args.edition_symbol:
            parser.error("Both --pdf and --edition-symbol are required when not using --all")
        convert_edition(args.pdf, args.edition_symbol, editions_repo_dir=args.repo_dir, dpi=args.dpi, workers=args.workers)


if __name__ == "__main__":
    main()
