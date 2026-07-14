"""Generate the PNG/ICO icon set expected by the Tauri Windows bundle."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src-tauri" / "icons"


def draw_icon(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), "#091229")
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            mix = (x + y) / (2 * size)
            pixels[x, y] = (
                round(23 - 14 * mix),
                round(42 - 24 * mix),
                round(82 - 41 * mix),
                255,
            )

    draw = ImageDraw.Draw(image)
    radius = size * 0.22
    # Reapply a transparent rounded mask so the Windows icon has clean corners.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    image.putalpha(mask)

    board = (size * 0.23, size * 0.26, size * 0.77, size * 0.75)
    draw.rounded_rectangle(board, radius=size * 0.035, fill="#152f61", outline="#5b7fc3", width=max(2, size // 45))
    xs = (size * 0.34, size * 0.50, size * 0.66)
    ys = (size * 0.37, size * 0.51, size * 0.65)
    cell_radius = size * 0.055
    colors = {(0, 2): "#ff5269", (1, 1): "#ffc857", (2, 0): "#54d6ff"}
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            color = colors.get((column, row), "#091229")
            box = (x - cell_radius, y - cell_radius, x + cell_radius, y + cell_radius)
            draw.ellipse(box, fill=color, outline="#4268ad", width=max(2, size // 100))
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = draw_icon()
    resampling = Image.Resampling.LANCZOS
    for filename, size in (
        ("32x32.png", 32),
        ("128x128.png", 128),
        ("128x128@2x.png", 256),
        ("icon.png", 512),
    ):
        source.resize((size, size), resampling).save(OUTPUT / filename)
    source.save(
        OUTPUT / "icon.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
