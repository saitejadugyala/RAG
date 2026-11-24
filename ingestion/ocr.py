"""OCR helper (Tesseract), used by step 2 for images and scanned pages.

Install the engine once:  brew install tesseract   (or apt install tesseract-ocr)
If it isn't installed, OCR is skipped with a warning instead of crashing.
"""
from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass

from PIL import Image, ImageOps

from .errors import OCRError
from .settings import Settings

log = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


@dataclass
class OCRResult:
    text: str
    confidence: float  # average word confidence, 0-100
    words: int


class OCR:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.available = bool(settings.ocr_enabled and pytesseract and shutil.which("tesseract"))
        if settings.ocr_enabled and not self.available:
            log.warning("Tesseract not found - images and scanned pages will be skipped. "
                        "Install it with `brew install tesseract`.")

    def is_good(self, result: OCRResult) -> bool:
        return result.words >= 3 and result.confidence >= self.settings.ocr_min_confidence

    def read_image_bytes(self, data: bytes, *, is_table: bool = False) -> OCRResult:
        """OCR an image file's bytes (an image embedded in a PDF or DOCX)."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as e:
            raise OCRError(f"Unreadable image: {e}") from e
        if min(image.size) < self.settings.min_image_px:
            return OCRResult("", 0.0, 0)  # an icon or bullet, not content
        return self.read_image(image, is_table=is_table)

    def read_image(self, image: Image.Image, *, is_table: bool = False) -> OCRResult:
        """OCR a PIL image, trying a few Tesseract layouts until one reads well.

        Page segmentation modes (psm):
          3  = automatic layout (normal pages of prose)
          4  = one column of lines (keeps table rows together)
          11 = scattered text (charts, labels) - last resort
        """
        if not self.available:
            raise OCRError("OCR is not available")
        try:
            gray = _to_grayscale(image)
            enlarged = _enlarge_if_small(gray)
            first, second = (4, 3) if is_table else (3, 4)
            attempts = [(enlarged, first), (gray, first), (enlarged, second), (enlarged, 11)]

            best = OCRResult("", 0.0, 0)
            for img, psm in attempts:
                result = self._run_tesseract(img, psm)
                if self.is_good(result):
                    return result
                if result.words * result.confidence > best.words * best.confidence:
                    best = result
            return best
        except Exception as e:
            raise OCRError(f"Tesseract failed: {e}") from e

    def _run_tesseract(self, image: Image.Image, psm: int) -> OCRResult:
        data = pytesseract.image_to_data(
            image, lang=self.settings.ocr_language,
            config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
        # Rebuild the text line by line, dropping words Tesseract is unsure about
        lines: dict[tuple, list[str]] = {}
        confidences: list[float] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            conf = _to_float(data["conf"][i])
            if not word or conf < 0:
                continue
            confidences.append(conf)
            if conf < self.settings.ocr_min_confidence / 2:
                continue
            line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(line_id, []).append(word)

        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        average = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text, average, len(confidences))


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _to_grayscale(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):  # put transparent images on white
        background = Image.new("RGB", image.size, "white")
        image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1])
        image = background
    return ImageOps.autocontrast(ImageOps.grayscale(image))


def _enlarge_if_small(image: Image.Image) -> Image.Image:
    """Low-resolution images have tiny letters; doubling them helps Tesseract.
    Large images are left alone - oversized letters hurt accuracy too."""
    if min(image.size) < 400:
        return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    return image
