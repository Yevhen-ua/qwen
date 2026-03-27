from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLOR = (255, 0, 0)
DEFAULT_RADIUS = 16
DEFAULT_LINE_WIDTH = 4
DEFAULT_TEXT_PADDING = 6
DEFAULT_ARROW_SIZE = 14


def _get_output_path(image_path: str | Path, output_path: str | Path | None) -> Path:
    image_path = Path(image_path)
    if output_path is not None:
        return Path(output_path)
    return image_path.parent / f"marked_{image_path.name}"


def _extract_point(obj: Any) -> tuple[int | None, int | None]:
    if not isinstance(obj, dict):
        return None, None
    x = obj.get("x")
    y = obj.get("y")
    if isinstance(x, int) and isinstance(y, int):
        return x, y
    return None, None


def _extract_drag(obj: Any) -> tuple[int | None, int | None, int | None, int | None]:
    if not isinstance(obj, dict):
        return None, None, None, None
    x = obj.get("x")
    y = obj.get("y")
    x2 = obj.get("x2")
    y2 = obj.get("y2")
    if all(isinstance(v, int) for v in (x, y, x2, y2)):
        return x, y, x2, y2
    return None, None, None, None


def _clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def _draw_cross(draw_obj: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    cross_len = DEFAULT_RADIUS + 10
    draw_obj.ellipse(
        (x - DEFAULT_RADIUS, y - DEFAULT_RADIUS, x + DEFAULT_RADIUS, y + DEFAULT_RADIUS),
        outline=color,
        width=DEFAULT_LINE_WIDTH,
    )
    draw_obj.line((x - cross_len, y, x + cross_len, y), fill=color, width=DEFAULT_LINE_WIDTH)
    draw_obj.line((x, y - cross_len, x, y + cross_len), fill=color, width=DEFAULT_LINE_WIDTH)
    dot_r = 3
    draw_obj.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r), fill=color)


def _load_font() -> ImageFont.ImageFont | None:
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_label(
    draw_obj: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    font = _load_font()
    bbox = draw_obj.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = min(max(8, x + 18), max(8, width - text_w - 2 * DEFAULT_TEXT_PADDING - 8))
    text_y = min(max(8, y - text_h - 18), max(8, height - text_h - 2 * DEFAULT_TEXT_PADDING - 8))

    rect = (
        text_x,
        text_y,
        text_x + text_w + 2 * DEFAULT_TEXT_PADDING,
        text_y + text_h + 2 * DEFAULT_TEXT_PADDING,
    )
    draw_obj.rounded_rectangle(rect, radius=6, fill=(255, 255, 255), outline=color, width=2)
    draw_obj.text(
        (text_x + DEFAULT_TEXT_PADDING, text_y + DEFAULT_TEXT_PADDING),
        text,
        fill=color,
        font=font,
    )


def _draw_arrow(
    draw_obj: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    import math

    draw_obj.line((x1, y1, x2, y2), fill=color, width=DEFAULT_LINE_WIDTH)

    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - DEFAULT_ARROW_SIZE * math.cos(angle - math.pi / 6),
        y2 - DEFAULT_ARROW_SIZE * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - DEFAULT_ARROW_SIZE * math.cos(angle + math.pi / 6),
        y2 - DEFAULT_ARROW_SIZE * math.sin(angle + math.pi / 6),
    )
    draw_obj.polygon([(x2, y2), left, right], fill=color)


def draw(
    answer: dict[str, Any],
    image_path: str | Path,
    output_path: str | Path | None = None,
    label: str | None = None,
) -> Path | None:
    """
    Draw point or drag result on the image.

    Supported formats:
      {"x": 123, "y": 456}
      {"mode": "point", "answer": {"x": 123, "y": 456}, "comment": "..."}
      {"mode": "drag", "answer": {"x": 1, "y": 2, "x2": 3, "y2": 4}, "comment": "..."}

    Returns the saved image path, or None if there are no valid coordinates.
    """
    payload = answer.get("answer") if isinstance(answer, dict) and isinstance(answer.get("answer"), dict) else answer
    mode = answer.get("mode") if isinstance(answer, dict) else None

    image_path = Path(image_path)
    out_path = _get_output_path(image_path, output_path)

    img = Image.open(image_path).convert("RGB")
    draw_obj = ImageDraw.Draw(img)
    width, height = img.size

    x, y, x2, y2 = _extract_drag(payload)
    if mode == "drag" or (x is not None and y is not None and x2 is not None and y2 is not None):
        x, y = _clamp_point(x, y, width, height)
        x2, y2 = _clamp_point(x2, y2, width, height)

        _draw_cross(draw_obj, x, y, DEFAULT_COLOR)
        _draw_cross(draw_obj, x2, y2, DEFAULT_COLOR)
        _draw_arrow(draw_obj, x, y, x2, y2, DEFAULT_COLOR)

        text = label or f"({x}, {y}) -> ({x2}, {y2})"
        _draw_label(draw_obj, text, x2, y2, width, height, DEFAULT_COLOR)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path

    x, y = _extract_point(payload)
    if x is None or y is None:
        return None

    x, y = _clamp_point(x, y, width, height)
    _draw_cross(draw_obj, x, y, DEFAULT_COLOR)

    text = label or f"({x}, {y})"
    _draw_label(draw_obj, text, x, y, width, height, DEFAULT_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
