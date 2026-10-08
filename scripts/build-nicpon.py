"""Generuje warstwowy, pixel-artowy SVG Nicponia z oryginalnego PNG.

    python scripts/build-nicpon.py

Wejście:  src/assets/nicpon.png  (obrazek "pixel art" z nieregularną siatką ~4,6 px)
Wyjście: src/assets/nicpon/nicpon-cat.svg

Kroki: próbkowanie siatki pikseli, wycięcie tła (flood fill od krawędzi), kwantyzacja palety,
podział na ruchome części (oczy, ogon, łapki, uszy, klawisze) i zapis jako ścieżki SVG.
Wymaga: numpy, Pillow.
"""

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "assets" / "nicpon.png"
OUT = ROOT / "src" / "assets" / "nicpon" / "nicpon-cat.svg"

PX = 4.603          # rozmiar "piksela" w oryginale (wyznaczony FFT z krawędzi)
ORIGIN = (304.34, 151.35)
GRID_W, GRID_H = 159, 119


# ---------- 1. Siatka pikseli ----------
def sample_grid(img):
    a = np.asarray(img.convert("RGB")).astype(float)
    ox, oy = ORIGIN
    out = np.zeros((GRID_H, GRID_W, 3), int)
    for r in range(GRID_H):
        for c in range(GRID_W):
            x0 = int(ox + c * PX + PX * 0.3); x1 = int(ox + (c + 1) * PX - PX * 0.3) + 1
            y0 = int(oy + r * PX + PX * 0.3); y1 = int(oy + (r + 1) * PX - PX * 0.3) + 1
            out[r, c] = np.median(a[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return out


def flood(mask_ok, seeds):
    H, W = mask_ok.shape
    seen = np.zeros((H, W), bool)
    q = deque(seeds)
    while q:
        y, x = q.popleft()
        if not (0 <= y < H and 0 <= x < W) or seen[y, x] or not mask_ok[y, x]:
            continue
        seen[y, x] = True
        q.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    return seen


def components(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not lab[y, x]:
                n += 1
                lab[y, x] = n
                q = deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = n
                            q.append((ny, nx))
    return lab, n


# ---------- 2. Ścieżki SVG z pikseli ----------
def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def runs_path(mask, dx=0, dy=0):
    """Piksele z maski jako jedna ścieżka z poziomych odcinków."""
    parts = []
    H, W = mask.shape
    for y in range(H):
        x = 0
        while x < W:
            if mask[y, x]:
                s = x
                while x < W and mask[y, x]:
                    x += 1
                parts.append(f"M{s + dx} {y + dy}h{x - s}v1h-{x - s}z")
            else:
                x += 1
    return "".join(parts)


def layer(colors, mask, pal, cls=None, dx=0, dy=0):
    """Grupa ścieżek, po jednej na kolor."""
    paths = []
    for ci in sorted(set(colors[mask].tolist())):
        if ci < 0:
            continue
        d = runs_path(mask & (colors == ci), dx, dy)
        if d:
            paths.append(f'<path fill="{hexc(pal[ci])}" d="{d}"/>')
    attr = f' class="{cls}"' if cls else ""
    return f"<g{attr}>{''.join(paths)}</g>"


def main():
    grid = sample_grid(Image.open(SRC))
    H, W, _ = grid.shape
    r, g, b = grid[..., 0], grid[..., 1], grid[..., 2]
    lum = (r * 299 + g * 587 + b * 114) / 1000
    Y = np.arange(H)[:, None]
    X = np.arange(W)[None, :]

    # Wycięcie granatowego tła: zalewanie od krawędzi po pikselach "tła"
    bgish = ((g - r) >= 22) & ((b - g) >= 18)
    seeds = [(y, x) for y in range(H) for x in (0, W - 1)] + [(y, x) for x in range(W) for y in (0, H - 1)]
    fg = ~flood(bgish, seeds) & (Y >= 2)
    # Pod blatem zostają tylko nogi biurka (bez napisu "Nicpoń" i tła)
    legs = ((X >= 8) & (X <= 26)) | ((X >= 130) & (X <= 150))
    fg = np.where(Y >= 104, fg & legs & ((b - r) < 15) & (lum < 200), fg)
    lab, n = components(fg)
    keep = np.zeros_like(fg)
    for i in range(1, n + 1):
        comp = lab == i
        ys = np.where(comp)[0]
        if comp.sum() >= 40 or (ys.min() >= 104 and comp.sum() >= 6):
            keep |= comp

    # Paleta (kwantyzacja tylko pikseli sceny)
    pix = Image.fromarray(grid[keep].astype(np.uint8)[None, :, :])
    q = pix.quantize(colors=40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[: 40 * 3]).reshape(-1, 3)
    colors = np.full((H, W), -1, int)
    colors[keep] = np.asarray(q)[0]

    def nearest(rgb):
        return int(np.argmin(((pal - np.array(rgb)) ** 2).sum(axis=1)))

    # ---------- 3. Części ----------
    EYES = {"l": (64, 28), "r": (85, 28)}          # lewy górny róg pola 11 x 9
    ring = nearest(grid[27, 69])                     # jasna obwódka wokół oczu
    for ex, ey in EYES.values():
        box = (Y >= ey) & (Y < ey + 9) & (X >= ex) & (X < ex + 11)
        colors[box & keep] = ring

    tail = keep & (X >= 111) & (Y >= 36) & (Y <= 72)
    paw_l = keep & (X >= 55) & (X <= 66) & (Y >= 71) & (Y <= 78)
    paw_r = keep & (X >= 93) & (X <= 104) & (Y >= 71) & (Y <= 78)
    ear_l = keep & (X >= 52) & (X <= 63) & (Y <= 16)
    ear_r = keep & (X >= 96) & (X <= 107) & (Y <= 16)
    base = keep & ~tail

    # Klawisze: jasne "czapki" klawiszy na klawiaturze
    key_px = keep & (Y >= 76) & (Y <= 92) & (X >= 28) & (X <= 132) & (lum > 70) & (lum < 135) & ~paw_l & ~paw_r
    klab, kn = components(key_px)
    keys = []
    for i in range(1, kn + 1):
        comp = klab == i
        if 2 <= comp.sum() <= 14:
            ys, xs = np.where(comp)
            keys.append((float(xs.mean()), float(ys.mean()), runs_path(comp)))

    # Ogon: 5 klatek, każdy wiersz przesunięty tym mocniej, im bliżej czubka
    tail_frames = []
    tys = np.where(tail.any(axis=1))[0]
    top, bottom = tys.min(), tys.max()
    for amp in (-3, -1.5, 0, 1.5, 3):
        m = np.zeros_like(tail)
        c = np.full((H, W), -1, int)
        for y in range(top, bottom + 1):
            t = (bottom - y) / max(1, bottom - top)
            s = int(round(amp * t ** 1.6))
            xs = np.where(tail[y])[0]
            for x in xs:
                if 0 <= x + s < W:
                    m[y, x + s] = True
                    c[y, x + s] = colors[y, x]
        tail_frames.append(layer(c, m, pal))

    # ---------- 4. Oczy ----------
    eye_rows = [
        "..#######..",
        ".#iiiiiii#.",
        "#iiiiiiiii#",
        "#iiiiiiiii#",
        "#iiiiiiiii#",
        "#iiiiiiiii#",
        "#iiiiiiiii#",
        ".#iiiiiii#.",
        "..#######..",
    ]
    outline_c = "#0c0f22"
    iris = ["#d9c7a0", "#cbb68d", "#bca57d", "#a8926e", "#937d5e", "#7c6a52", "#655744"]
    pupil_c = "#0a0c1b"

    def rect_path(cells):
        return "".join(f"M{x} {y}h1v1h-1z" for x, y in cells)

    def eye_svg(name, ex, ey):
        out_cells, inner = [], []
        for yy, row in enumerate(eye_rows):
            for xx, ch in enumerate(row):
                if ch == "#":
                    out_cells.append((ex + xx, ey + yy))
                elif ch == "i":
                    inner.append((ex + xx, ey + yy))
        iris_paths = []
        for k in range(1, 8):
            cells = [(x, y) for x, y in inner if y == ey + k]
            iris_paths.append(f'<path fill="{iris[k - 1]}" d="{rect_path(cells)}"/>')
        clip = f"nc-clip-{name}"
        # Źrenica 5 x 5 ze ściętymi rogami i błyskiem, wyśrodkowana w polu 9 x 7
        pupil = (
            f'<path fill="{pupil_c}" d="M{ex + 4} {ey + 2}h3v1h1v3h-1v1h-3v-1h-1v-3h1z"/>'
            f'<path fill="#f4f6f0" d="M{ex + 4} {ey + 3}h1v1h-1z"/>'
        )
        cells = inner + out_cells
        ring_c = hexc(pal[ring])
        # Powieka do połowy: górna część oka w kolorze futra, krawędź powieki jako ciemna linia
        lid_half = rect_path([(x, y) for x, y in cells if y <= ey + 3])
        lash_half = "".join(f"M{x} {ey + 4}h1v1h-1z" for x in range(ex, ex + 11))
        # Zamknięte oko: całe pole w kolorze futra i jedna linia rzęs
        lid_full = rect_path(cells)
        lash_closed = "".join(f"M{ex + x} {ey + y}h1v1h-1z" for x, y in [(1, 4), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 4)])
        happy = [(1, 5), (2, 4), (3, 3), (4, 3), (5, 2), (6, 3), (7, 3), (8, 4), (9, 5)]
        happy_d = "".join(f"M{ex + x} {ey + y}h1v1h-1z" for x, y in happy)
        return (
            f'<g class="nc-eye nc-eye-{name}" data-cx="{ex + 5.5}" data-cy="{ey + 4.5}">'
            f'<clipPath id="{clip}"><path d="{rect_path(inner)}"/></clipPath>'
            f'<path fill="{outline_c}" d="{rect_path(out_cells)}"/>'
            f"{''.join(iris_paths)}"
            f'<g clip-path="url(#{clip})"><g class="nc-pupil">{pupil}</g></g>'
            f'<g class="nc-lid nc-lid-half"><path fill="{ring_c}" d="{lid_half}"/><path fill="{outline_c}" d="{lash_half}"/></g>'
            f'<g class="nc-lid nc-lid-closed"><path fill="{ring_c}" d="{lid_full}"/><path fill="{outline_c}" d="{lash_closed}"/></g>'
            f'<g class="nc-lid nc-lid-happy"><path fill="{ring_c}" d="{lid_full}"/><path fill="{outline_c}" d="{happy_d}"/></g>'
            "</g>"
        )

    # ---------- 5. Złożenie SVG ----------
    sil = runs_path(keep)
    key_glow = "".join(
        f'<path class="nc-key" data-x="{x:.1f}" data-y="{y:.1f}" d="{d}"/>' for x, y, d in keys
    )
    svg = f"""<svg class="nc-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges" role="img" aria-label="Kot Nicpoń przy klawiaturze">
<defs>
<radialGradient id="nc-backdrop" cx="50%" cy="45%" r="55%"><stop offset="0" stop-color="#123a6b" stop-opacity=".9"/><stop offset=".55" stop-color="#0b2448" stop-opacity=".55"/><stop offset="1" stop-color="#0b1120" stop-opacity="0"/></radialGradient>
<radialGradient id="nc-screen" cx="50%" cy="78%" r="65%"><stop offset="0" stop-color="#34d399" stop-opacity=".55"/><stop offset=".5" stop-color="#22d3ee" stop-opacity=".18"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/></radialGradient>
<clipPath id="nc-sil"><path d="{sil}"/></clipPath>
</defs>
<g class="nc-back"><ellipse cx="{W / 2}" cy="{H * 0.46}" rx="{W * 0.48}" ry="{H * 0.52}" fill="url(#nc-backdrop)" shape-rendering="geometricPrecision"/></g>
<g class="nc-floor"><ellipse cx="{W / 2}" cy="{H - 2}" rx="{W * 0.46}" ry="3.5" fill="#000" opacity=".45" shape-rendering="geometricPrecision"/></g>
<g class="nc-tail">{''.join(f'<g class="nc-tail-f nc-tail-f{i}">{f}</g>' for i, f in enumerate(tail_frames))}</g>
{layer(colors, base, pal, "nc-base")}
<g class="nc-keys">{key_glow}</g>
<path class="nc-strip" d="M31 89h98v1h-98z"/>
{layer(colors, ear_l, pal, "nc-ear nc-ear-l")}
{layer(colors, ear_r, pal, "nc-ear nc-ear-r")}
{eye_svg("l", *EYES["l"])}
{eye_svg("r", *EYES["r"])}
{layer(colors, paw_l, pal, "nc-paw nc-paw-l")}
{layer(colors, paw_r, pal, "nc-paw nc-paw-r")}
<g class="nc-light" clip-path="url(#nc-sil)"><rect x="0" y="0" width="{W}" height="{H}" fill="url(#nc-screen)" shape-rendering="geometricPrecision"/></g>
</svg>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(svg) / 1024:.1f} kB, kolorów {len(set(colors[keep].tolist()))}, klawiszy {len(keys)}")


# ---------- Mały kotek, który czasem przebiega przez stronę ----------
KITTEN_OUT = ROOT / "src" / "assets" / "nicpon" / "kitten.svg"
KITTEN_COLORS = {
    "O": "#14161f",  # obrys
    "G": "#8e9096",  # futro
    "L": "#b9bbbe",  # jasne futro
    "D": "#4f5159",  # paski
    "W": "#e6e8e4",  # łapki i pyszczek
    "P": "#e8907f",  # uszy i nos
    "E": "#0c0f22",  # oko
}
HEAD = [
    ".O...O..",
    "OPO.OPO.",
    "OGGOGGGO",
    "OGGGGGGO",
    "OGGGGEGO",
    "OLGGGWWP",
    ".OLWWWO.",
    "..OOOO..",
]
BODY = [
    ".OOOOOOOOOOO.",
    "OGGDGGDGGDGGO",
    "OGGDGGDGGDGGO",
    "OGGGGGGGGGGLO",
    "OLLLLLLLLLWWO",
    ".OOOOOOOOOOO.",
]
TAIL = [
    ".OOO.",
    "OGDGO",
    "OGOO.",
    "ODO..",
    "OGO..",
    ".ODO.",
    "..OGO",
]
LEG = ["OGO", "OGO", "OWO"]
BALL = [
    "..O..O....",
    ".OPOOPOO..",
    ".OGGGGGGO.",
    "OGDGGDGGGO",
    "OGGGGGGEGO",
    "OGGGGWWWPO",
    "OGLLWWWLGO",
    ".OLLLLLLO.",
    "..OGDGGO..",
    "...OOOO...",
]
STAR = [".Y.", "YYY", ".Y."]
# Klatki biegu: x nóg (tylne, tylne, przednie, przednie) i podskok tułowia
RUN_FRAMES = [((3, 6, 14, 17), 0), ((5, 8, 12, 15), 1), ((7, 9, 11, 13), 1), ((5, 8, 12, 15), 0)]
KW, KH = 22, 16
Y0 = 3  # miejsce nad głową na gwiazdki


def stamp(canvas, art, ox, oy):
    for y, row in enumerate(art):
        for x, ch in enumerate(row):
            if ch != "." and 0 <= oy + y < len(canvas) and 0 <= ox + x < len(canvas[0]):
                canvas[oy + y][ox + x] = ch


def art_paths(canvas):
    by_color = {}
    for y, row in enumerate(canvas):
        x = 0
        while x < len(row):
            ch = row[x]
            if ch == ".":
                x += 1
                continue
            s = x
            while x < len(row) and row[x] == ch:
                x += 1
            by_color.setdefault(ch, []).append(f"M{s} {y}h{x - s}v1h-{x - s}z")
    colors = dict(KITTEN_COLORS, Y="#fde68a")
    return "".join(f'<path fill="{colors[c]}" d="{"".join(d)}"/>' for c, d in by_color.items())


def kitten():
    frames = []
    for i, (legs, bob) in enumerate(RUN_FRAMES):
        cv = [["."] * KW for _ in range(KH)]
        for lx in legs:
            stamp(cv, LEG, lx, Y0 + 10)
        stamp(cv, TAIL, 0, Y0 - bob)
        stamp(cv, BODY, 3, Y0 + 5 - bob)
        stamp(cv, HEAD, 13, Y0 + 1 - bob)
        frames.append(f'<g class="kf kf-run{i}">{art_paths(cv)}</g>')
    ball = [["."] * KW for _ in range(KH)]
    stamp(ball, BALL, 6, Y0 + 4)
    frames.append(f'<g class="kf kf-ball">{art_paths(ball)}</g>')
    stars = [["."] * KW for _ in range(KH)]
    stamp(stars, STAR, 11, 0)
    stamp(stars, STAR, 17, 1)
    frames.append(f'<g class="kf-stars">{art_paths(stars)}</g>')
    svg = (
        f'<svg class="kitten-svg" viewBox="0 0 {KW} {KH}" xmlns="http://www.w3.org/2000/svg" '
        f'shape-rendering="crispEdges" aria-hidden="true">{"".join(frames)}</svg>\n'
    )
    KITTEN_OUT.write_text(svg, encoding="utf-8")
    print(f"{KITTEN_OUT.relative_to(ROOT)}: {len(svg) / 1024:.1f} kB")


if __name__ == "__main__":
    main()
    kitten()
