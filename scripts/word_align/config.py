from dataclasses import dataclass
from typing import Optional
import os

EDITIONS_ROOT = r'D:\editions'
DEFAULT_WORDS_DB = r'D:\Qeraat\QuranData\words.db'
DEFAULT_QURAN_TEXT = r'D:\Qeraat\wathkro\public\data\quran_text.json'


@dataclass
class EditionConfig:
    edition_id: str
    name: str
    pages_dir: str
    pages_from: int
    pages_to: int
    output_dir: str
    words_db_path: str = DEFAULT_WORDS_DB
    quran_text_path: str = DEFAULT_QURAN_TEXT
    # Columns in words.db (set prior cols to None for editions without prior coords)
    page_col: str = 'page_number2'
    x_col: Optional[str] = 'x'
    y_col: Optional[str] = 'y'
    w_col: Optional[str] = 'width'
    # Ink classification thresholds (RGBA)
    black_max_rgb: int = 95
    alpha_min: int = 128
    # Reference pages to auto-extract canonical marker templates
    rub_template_page: Optional[int] = 6
    sajdah_template_page: Optional[int] = 206
    # Line detection geometry (at reference height ~2189px; auto-scaled by H/2189)
    ref_height: int = 2189
    ref_width: int = 1409
    min_line_dist_px: int = 85
    # Medallion (999) expected size range at reference scale
    medallion_min_px: int = 48
    medallion_max_px: int = 105
    medallion_min_hole_px: int = 650


EDITIONS = {
    'shmrly': EditionConfig(
        edition_id='shmrly',
        name='مصحف الشمرلي',
        pages_dir=os.path.join(EDITIONS_ROOT, 'shmrly', 'assets', 'edition', 'pages'),
        pages_from=2,
        pages_to=522,
        output_dir=os.path.join(EDITIONS_ROOT, 'words', 'shmrly'),
        page_col='page_number2',
        x_col='x',
        y_col='y',
        w_col='width',
        rub_template_page=6,
        sajdah_template_page=206,
    ),
    # Future editions can be registered here with their own profiles, e.g.:
    # 'madina': EditionConfig(
    #     edition_id='madina',
    #     name='مصحف المدينة المنورة',
    #     pages_dir=os.path.join(EDITIONS_ROOT, 'madina', 'assets', 'edition', 'pages'),
    #     pages_from=1,
    #     pages_to=604,
    #     output_dir=os.path.join(EDITIONS_ROOT, 'words', 'madina'),
    #     page_col='page_number1',
    #     x_col='x1', y_col='y1', w_col='width1',
    # ),
}
