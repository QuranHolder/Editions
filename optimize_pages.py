import os
import glob
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import PIL.Image as Image
import numpy as np

PAGES_DIR = r'D:\Editions\madinaold1\assets\edition\pages'
OPTIPNG = r'f:\optipng-7.9.1-win32\optipng.exe'
TARGET_SIZE = (1260, 2038)
THRESHOLDS = [200, 140, 70]
PAL_VALS = [255, 175, 95, 0]
PALETTE = []
for v in PAL_VALS:
    PALETTE.extend([v, v, v])
PALETTE.extend([0] * (768 - len(PALETTE)))


def process_one(path):
    orig_size = os.path.getsize(path)
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, im).convert('L')
    if comp.size != TARGET_SIZE:
        comp = comp.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    gray = np.array(comp)
    idx = np.zeros_like(gray, dtype=np.uint8)
    for i, t in enumerate(THRESHOLDS):
        idx[gray < t] = i + 1
    im_out = Image.fromarray(idx, mode='P')
    im_out.putpalette(PALETTE)
    tmp_path = path + '.tmp.png'
    im_out.save(tmp_path, optimize=True, transparency=0, bits=2)
    subprocess.run([OPTIPNG, '-o2', '-quiet', tmp_path], check=False)
    os.replace(tmp_path, path)
    new_size = os.path.getsize(path)
    return orig_size, new_size


if __name__ == '__main__':
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(PAGES_DIR, '*.png')))
    print(f'Processing {len(files)} files with {os.cpu_count()} workers...', flush=True)
    total_orig = 0
    total_new = 0
    done = 0
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(process_one, f) for f in files]
        for fut in as_completed(futs):
            o, n = fut.result()
            total_orig += o
            total_new += n
            done += 1
            if done % 50 == 0 or done == len(files):
                print(
                    f'  [{done}/{len(files)}] {total_orig/1024/1024:.2f} MB -> {total_new/1024/1024:.2f} MB ({time.time()-t0:.1f}s)',
                    flush=True,
                )
    print(
        f'Done! {total_orig/1024/1024:.2f} MB -> {total_new/1024/1024:.2f} MB '
        f'(Saved {(total_orig-total_new)/1024/1024:.2f} MB, {(1 - total_new/total_orig)*100:.1f}%)',
        flush=True,
    )
