"""Generates the move animations of the battle viewer (images/move_animations).

Original pixel art drawn by code in the style of the GBA games: 64x64 cells, short
palettes, hard pixels (no anti-aliasing) and a dark 1 px outline. One animation per
elemental type of generation 3. The result is deterministic (fixed seeds).

Usage:  python tools/generate_move_effects.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "move_animations"
S = 64  # cell size
C = S // 2  # centre


# --------------------------------------------------------------------------- toolkit


def canvas() -> Image.Image:
    return Image.new("RGBA", (S, S), (0, 0, 0, 0))


def rgba(c, a=255):
    return (c[0], c[1], c[2], a)


def outline(img: Image.Image, color) -> Image.Image:
    """Dark 1 px border around every opaque pixel (4-neighbourhood), like GBA sprites."""
    a = np.array(img)
    mask = a[..., 3] > 0
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    a[grown & ~mask] = rgba(color)
    return Image.fromarray(a)


def bitmap(rows: list, palette: dict) -> Image.Image:
    """Pixel icon from an ASCII map ('.' is transparent)."""
    h, w = len(rows), max(len(r) for r in rows)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                px[x, y] = rgba(palette[ch])
    return img


def paste(dst: Image.Image, icon: Image.Image, cx: float, cy: float, scale: int = 1):
    if scale != 1:
        icon = icon.resize((icon.width * scale, icon.height * scale), Image.NEAREST)
    dst.alpha_composite(icon, (int(round(cx - icon.width / 2)), int(round(cy - icon.height / 2))))


def turn(icon: Image.Image, quarter: int, flip: bool = False) -> Image.Image:
    """Rotations by 90 degrees and mirror keep the pixel art crisp."""
    out = icon.transpose(Image.FLIP_LEFT_RIGHT) if flip else icon
    for _ in range(quarter % 4):
        out = out.transpose(Image.ROTATE_90)
    return out


def star_points(cx, cy, r_out, r_in, n, rot=0.0, jitter=None):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        if jitter is not None and i % 2 == 0:
            r *= jitter[i // 2]
        ang = rot + math.pi * i / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def disc(d: ImageDraw.ImageDraw, cx, cy, r, fill):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba(fill))


def ring(d: ImageDraw.ImageDraw, cx, cy, r, color, width=2):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=rgba(color), width=width)


def sparkle(d: ImageDraw.ImageDraw, x, y, size, color, core=None):
    """Four-point pixel sparkle."""
    x, y = int(x), int(y)
    for i in range(-size, size + 1):
        d.point((x + i, y), fill=rgba(color))
        d.point((x, y + i), fill=rgba(color))
    if core is not None:
        d.point((x, y), fill=rgba(core))


def flame_shape(d, cx, base, h, w, colors, lean=0.0):
    """Layered flame (outer -> inner colours)."""
    for k, col in enumerate(colors):
        f = 1 - k * 0.26
        hh, ww = h * f, w * f
        tip = (cx + lean * hh * 0.4, base - hh)
        pts = [(cx - ww / 2, base), (cx - ww * 0.55, base - hh * 0.45), (cx - ww * 0.2, base - hh * 0.7), tip,
               (cx + ww * 0.25, base - hh * 0.62), (cx + ww * 0.55, base - hh * 0.4), (cx + ww / 2, base)]
        d.polygon(pts, fill=rgba(col))
        d.ellipse((cx - ww / 2, base - ww * 0.45, cx + ww / 2, base + ww * 0.25), fill=rgba(col))


def zigzag(d, x0, y0, x1, y1, segments, amp, color, width, rng):
    pts = [(x0, y0)]
    for i in range(1, segments):
        t = i / segments
        x = x0 + (x1 - x0) * t + rng.uniform(-amp, amp)
        y = y0 + (y1 - y0) * t
        pts.append((x, y))
    pts.append((x1, y1))
    d.line(pts, fill=rgba(color), width=width)


# --------------------------------------------------------------------------- icons

LEAF = ["........kk",
        "......kkgk",
        "....kkgghk",
        "...kggghgk",
        "..kgggdgk.",
        ".kggdggk..",
        ".kgdggk...",
        "kgdkkk....",
        "kdk.......",
        "k........."]
LEAF_PAL = {"k": (24, 80, 32), "g": (104, 200, 72), "h": (184, 240, 136), "d": (48, 136, 48)}

ROCK = ["...kkkkk....",
        "..khhgggkk..",
        ".khgggggggk.",
        "khgggdggggdk",
        "kggggddgggdk",
        "kgggggggggdk",
        "kdggggggdddk",
        ".kdgggggddk.",
        "..kkddddkk..",
        "....kkkk...."]
ROCK_PAL = {"k": (56, 40, 24), "g": (168, 144, 104), "h": (224, 204, 160), "d": (112, 88, 56)}

PEBBLE = [".kk.",
          "kgdk",
          "kddk",
          ".kk."]

CRYSTAL = ["....k....",
           "...kwk...",
           "..kwbbk..",
           "..kwbbk..",
           ".kwbbbck.",
           ".kwbbbck.",
           "kwbbbbcck",
           ".kwbbbck.",
           ".kwbbbck.",
           "..kwbck..",
           "..kwbck..",
           "...kck...",
           "....k...."]
CRYSTAL_PAL = {"k": (40, 88, 160), "w": (248, 252, 255), "b": (168, 224, 248), "c": (96, 168, 232)}

FEATHER = ["......kk",
           ".....kwk",
           "....kwwk",
           "...kwwck",
           "..kwwck.",
           ".kwcck..",
           "kcck....",
           "kk......"]
FEATHER_PAL = {"k": (72, 96, 128), "w": (248, 248, 255), "c": (184, 216, 240)}

TEETH_UP = ["kkkkkkkkkkkkkkkkkkkkkkkkkkkkkk",
            "kmmmmmmmmmmmmmmmmmmmmmmmmmmmmk",
            "kmmmmmmmmmmmmmmmmmmmmmmmmmmmmk",
            ".kwwkwwwkwwwkwwwkwwwkwwwkwwkk.",
            "..kwk.kwk.kwk.kwk.kwk.kwk.kk..",
            "...k...k...k...k...k...k......"]
TEETH_PAL = {"k": (16, 8, 24), "m": (72, 48, 104), "w": (248, 248, 248)}


# --------------------------------------------------------------------------- effects


def normal():
    pal = {"w": (255, 255, 255), "y": (255, 236, 120), "o": (248, 168, 40), "k": (104, 56, 16)}
    frames = []
    for r_out, r_in, sp in ((7, 3, 0), (14, 6, 0), (21, 9, 0), (25, 10, 27), (0, 0, 30), (0, 0, 32)):
        img = canvas()
        d = ImageDraw.Draw(img)
        if r_out:
            d.polygon(star_points(C, C, r_out, r_in, 8, rot=math.pi / 8), fill=rgba(pal["o"]))
            d.polygon(star_points(C, C, r_out * 0.72, r_in * 0.72, 8, rot=math.pi / 8), fill=rgba(pal["y"]))
            disc(d, C, C, max(2, r_in * 0.6), pal["w"])
        if sp:
            for i in range(8):
                a = i * math.pi / 4
                sparkle(d, C + sp * math.cos(a), C + sp * math.sin(a), 2 if sp < 31 else 1, pal["y"], pal["w"])
        frames.append(outline(img, pal["k"]))
    return frames


def fighting():
    pal = {"w": (255, 255, 255), "y": (255, 224, 72), "o": (248, 120, 24), "r": (216, 40, 24), "k": (72, 16, 8)}
    rng = random.Random(2)
    jit = [rng.uniform(0.7, 1.15) for _ in range(12)]
    frames = []
    for i in range(7):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 2:  # chop motion lines
            off = 18 - i * 10
            for k in range(3):
                x0, y0 = 46 + off + k * 4, 4 + off - k * 3
                d.line((x0, y0, x0 - 18, y0 + 18), fill=rgba(pal["w"]), width=2)
        if 1 <= i <= 4:
            r = (12, 22, 27, 22)[i - 1]
            d.polygon(star_points(C, C, r, r * 0.45, 12, rot=0.2, jitter=jit), fill=rgba(pal["r"]))
            d.polygon(star_points(C, C, r * 0.75, r * 0.34, 12, rot=0.2, jitter=jit), fill=rgba(pal["o"]))
            d.polygon(star_points(C, C, r * 0.48, r * 0.22, 12, rot=0.2, jitter=jit), fill=rgba(pal["y"]))
            if i == 3:
                disc(d, C, C, 5, pal["w"])
        if i >= 4:
            rr = 20 + (i - 4) * 6
            for k in range(10):
                a = k * math.pi / 5 + 0.3
                x, y = C + rr * math.cos(a), C + rr * math.sin(a)
                d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=rgba(pal["o"] if k % 2 else pal["y"]))
        frames.append(outline(img, pal["k"]))
    return frames


def flying():
    pal = {"w": (255, 255, 255), "c": (176, 224, 248), "b": (104, 168, 224), "k": (48, 80, 128)}
    feather = bitmap(FEATHER, FEATHER_PAL)
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        for k in range(3):
            r = 8 + k * 7 + i * 2
            start = i * 50 + k * 120
            col = (pal["w"], pal["c"], pal["b"])[k]
            if r < 31:
                d.arc((C - r, C - r * 0.8, C + r, C + r * 0.8), start, start + 150, fill=rgba(col), width=2)
        for k in range(2):
            fx = C - 18 + k * 30 + int(6 * math.sin(i + k))
            fy = 8 + i * 5 + k * 6
            if fy < 60:
                paste(img, turn(feather, (i + k) // 2 % 2, flip=k == 1), fx, fy, 2)
        frames.append(outline(img, pal["k"]))
    return frames


def poison():
    pal = {"p": (168, 72, 200), "l": (216, 144, 240), "w": (248, 224, 255), "d": (104, 32, 136), "k": (48, 8, 64)}
    rng = random.Random(4)
    bubbles = [(rng.uniform(12, 52), rng.uniform(40, 62), rng.uniform(3, 7), k % 4) for k in range(9)]
    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        for (x, y, r, delay) in bubbles:
            t = i - delay
            if t < 0:
                continue
            yy = y - t * 6
            if t <= 3 and yy > 4:
                rr = r * (0.5 + t * 0.2)
                disc(d, x, yy, rr, pal["p"])
                disc(d, x - rr * 0.3, yy - rr * 0.3, max(1, rr * 0.35), pal["l"])
                d.point((int(x - rr * 0.4), int(yy - rr * 0.45)), fill=rgba(pal["w"]))
            elif t == 4 and yy > 0:  # pop
                for k in range(4):
                    a = k * math.pi / 2 + math.pi / 4
                    d.rectangle((x + 5 * math.cos(a) - 1, yy + 5 * math.sin(a) - 1,
                                 x + 5 * math.cos(a) + 1, yy + 5 * math.sin(a) + 1), fill=rgba(pal["l"]))
        frames.append(outline(img, pal["k"]))
    return frames


def ground():
    pal = {"b": (176, 128, 64), "l": (224, 184, 112), "d": (112, 72, 32), "k": (56, 32, 8)}
    pebble = bitmap(PEBBLE, ROCK_PAL)
    rng = random.Random(5)
    chunks = [(rng.uniform(-1.2, 1.2), rng.uniform(6, 10), rng.choice((2, 2, 3))) for _ in range(8)]
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 3:  # crack
            w = 6 + i * 8
            d.line([(C - w, 56), (C - w // 3, 52), (C, 57), (C + w // 3, 51), (C + w, 56)], fill=rgba(pal["d"]),
                   width=2)
        if i >= 1:  # dust cloud at the base
            t = i - 1
            for k in range(5):
                x = C - 22 + k * 11
                r = 3 + min(t, 3) * 1.2 - max(0, t - 4) * 1.5
                if r > 0:
                    disc(d, x, 56 - (k % 2) * 3, r, pal["l"] if k % 2 else pal["b"])
        if i >= 2:
            t = i - 2
            for vx, vy, sc in chunks:
                x = C + vx * 8 * t
                y = 54 - vy * t + 1.2 * t * t
                if 2 < y < 62:
                    paste(img, pebble, x, y, sc)
        frames.append(outline(img, pal["k"]))
    return frames


def rock():
    pal = {"l": (232, 216, 184), "k": (56, 40, 24)}
    boulder = bitmap(ROCK, ROCK_PAL)
    pebble = bitmap(PEBBLE, ROCK_PAL)
    drops = [(-14, 0, 2), (12, 2, 2), (0, 4, 2)]
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        for dx, delay, sc in drops:
            t = i - delay // 2
            if t < 0:
                continue
            y = -8 + t * 16
            if y < 44:
                paste(img, boulder, C + dx, y, sc)
            elif t <= 5:  # shatter
                u = t - 3
                for k in range(5):
                    a = math.pi + k * math.pi / 4
                    paste(img, pebble, C + dx + (6 + u * 5) * math.cos(a), 48 + (4 + u * 4) * math.sin(a))
                disc(d, C + dx, 52, 3 + u, pal["l"])
        frames.append(outline(img, pal["k"]))
    return frames


def bug():
    pal = {"w": (240, 255, 216), "g": (168, 216, 48), "d": (96, 152, 24), "k": (40, 64, 8)}
    angles = [math.radians(a) for a in (200, 240, 290, 330, 20, 150)]
    frames = []
    for i in range(7):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 4:
            r = 30 - i * 8
            for a in angles:
                x0, y0 = C + r * math.cos(a), C + r * math.sin(a)
                x1, y1 = C + (r + 10) * math.cos(a), C + (r + 10) * math.sin(a)
                d.line((x0, y0, x1, y1), fill=rgba(pal["g"]), width=2)
                d.point((int(x0), int(y0)), fill=rgba(pal["w"]))
        if i >= 3:
            t = i - 3
            for a in angles:
                sparkle(d, C + (4 + t * 5) * math.cos(a), C + (4 + t * 5) * math.sin(a), 2 if t < 3 else 1,
                        pal["g"], pal["w"])
            if t < 2:
                disc(d, C, C, 4 - t * 2, pal["w"])
        frames.append(outline(img, pal["k"]))
    return frames


def ghost():
    pal = {"c": (24, 8, 40), "p": (96, 48, 152), "l": (168, 112, 216), "w": (232, 208, 255), "k": (16, 0, 24)}
    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 6:
            r = 4 + i * 3
            for k in range(6):  # orbiting wisps
                a = i * 0.7 + k * math.pi / 3
                disc(d, C + (r + 6) * math.cos(a), C + (r + 6) * math.sin(a), 2, pal["l"])
            disc(d, C, C, r, pal["p"])
            disc(d, C, C, r * 0.7, pal["c"])
            d.point((C - r // 2, C - r // 2), fill=rgba(pal["w"]))
            if r > 8:
                d.arc((C - r + 2, C - r + 2, C + r - 2, C + r - 2), 200, 260, fill=rgba(pal["l"]), width=1)
        else:  # burst into shards
            t = i - 5
            for k in range(10):
                a = k * math.pi / 5
                x, y = C + (8 + t * 7) * math.cos(a), C + (8 + t * 7) * math.sin(a)
                d.polygon([(x, y - 3), (x + 2, y), (x, y + 3), (x - 2, y)], fill=rgba(pal["l"] if k % 2 else pal["p"]))
        frames.append(outline(img, pal["k"]))
    return frames


def steel():
    pal = {"w": (255, 255, 255), "s": (200, 208, 224), "g": (128, 136, 160), "k": (40, 44, 64)}
    frames = []
    for i in range(7):
        img = canvas()
        d = ImageDraw.Draw(img)
        for k in range(3):
            t = i - k
            if 0 <= t <= 3:
                off = (k - 1) * 10
                L = (10, 24, 32, 32)[t]
                x0, y0 = C + 16 + off, C - 16
                x1, y1 = x0 - L, y0 + L
                d.line((x0, y0, x1, y1), fill=rgba(pal["g"]), width=4)
                d.line((x0, y0, x1, y1), fill=rgba(pal["s"] if t == 3 else pal["w"]), width=2)
        if i >= 4:
            t = i - 4
            sparkle(d, C + 12, C - 10, 5 - t, pal["w"], pal["w"])
            sparkle(d, C - 14, C + 12, 3 - t, pal["s"], pal["w"]) if t < 3 else None
        frames.append(outline(img, pal["k"]))
    return frames


def fire():
    pal = {"r": (224, 56, 24), "o": (248, 144, 32), "y": (255, 224, 80), "w": (255, 252, 216), "k": (96, 16, 8)}
    rng = random.Random(10)
    flames = [(-14, 0.85, 1), (14, 0.85, 2), (0, 1.15, 0)]
    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        for dx, size, delay in flames:
            t = i - delay
            if t < 0:
                continue
            grow = min(1.0, 0.35 + t * 0.22) * (1.0 if t < 6 else max(0.0, 1 - (t - 5) * 0.35))
            if grow <= 0.05:
                continue
            h = 40 * size * grow * (1 + 0.08 * ((i + delay) % 2))
            flame_shape(d, C + dx, 58, h, h * 0.6, (pal["r"], pal["o"], pal["y"], pal["w"]),
                        lean=0.25 * math.sin(i * 1.7 + dx))
        if i >= 4:  # embers
            for k in range(5):
                x = C + rng.uniform(-24, 24)
                y = 40 - (i - 4) * 7 - rng.uniform(0, 10)
                if y > 2:
                    d.rectangle((x, y, x + 1, y + 1), fill=rgba(pal["y"] if k % 2 else pal["o"]))
        frames.append(outline(img, pal["k"]))
    return frames


def water():
    pal = {"w": (240, 252, 255), "l": (136, 200, 248), "b": (56, 128, 232), "d": (32, 80, 192), "k": (16, 32, 96)}
    rng = random.Random(11)
    drops = [(rng.uniform(-1.1, 1.1), rng.uniform(5, 9)) for _ in range(10)]

    def crown(d, h, w, peaks=5):
        pts = [(C - w, 60)]
        for k in range(peaks * 2 + 1):
            x = C - w + k * (2 * w) / (peaks * 2)
            edge = 1 - abs(k - peaks) / (peaks + 1)  # taller in the middle
            y = 60 - (h * (0.55 + 0.45 * edge) if k % 2 == 1 else h * 0.45 * (0.6 + 0.4 * edge))
            pts.append((x, y))
        pts.append((C + w, 60))
        for col, f in ((pal["d"], 1.0), (pal["b"], 0.78), (pal["l"], 0.5)):
            d.polygon([(C + (x - C) * f, 60 - (60 - y) * f) for x, y in pts], fill=rgba(col))
        for x, y in pts[2:-1:2]:
            d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=rgba(pal["w"]))

    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 6:
            h = (10, 22, 32, 38, 30, 18)[i]
            w = (10, 16, 21, 24, 26, 27)[i]
            crown(d, h, w)
        if i >= 3:  # droplets thrown out of the crown
            t = i - 3
            for vx, vy in drops:
                x = C + vx * 8 * (t + 1)
                y = 24 - vy * t + 2.0 * t * t
                if 2 < y < 60:
                    disc(d, x, y, 2, pal["b"])
                    d.point((int(x - 1), int(y - 1)), fill=rgba(pal["w"]))
        if i >= 5:  # ripples
            t = i - 5
            r = 14 + t * 6
            d.ellipse((C - r, 57 - r * 0.22, C + r, 57 + r * 0.22), outline=rgba(pal["l"]), width=1)
        frames.append(outline(img, pal["k"]))
    return frames


def grass():
    pal = {"w": (232, 255, 200), "g": (120, 208, 80), "k": (24, 80, 32)}
    leaf = bitmap(LEAF, LEAF_PAL)
    starts = [(-30, -20), (30, -24), (-32, 18), (30, 20), (0, -34)]
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 5:
            f = 1 - i / 4
            for k, (sx, sy) in enumerate(starts):
                paste(img, turn(leaf, i + k, flip=k % 2 == 1), C + sx * f, C + sy * f, 2 if i < 2 else 1)
        if i >= 4:
            t = i - 4
            for k in range(3):  # cut slashes
                y = C - 8 + k * 8
                L = 6 + t * 6 if t < 3 else 0
                if L:
                    d.line((C - L, y + L // 3, C + L, y - L // 3), fill=rgba(pal["w"]), width=2)
            for k in range(6):
                a = k * math.pi / 3
                sparkle(d, C + (6 + t * 6) * math.cos(a), C + (6 + t * 6) * math.sin(a), 1, pal["g"], pal["w"])
        frames.append(outline(img, pal["k"]))
    return frames


def electric():
    pal = {"w": (255, 255, 255), "y": (255, 232, 40), "o": (248, 176, 16), "k": (104, 72, 0)}
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        rng = random.Random(100 + i)
        if i in (0, 1, 3, 5):  # bolts striking the centre
            for sx in (-20, 0, 18):
                zigzag(d, C + sx, 0, C + sx * 0.2, C + 2, 5, 6, pal["o"], 4, rng)
                zigzag(d, C + sx, 0, C + sx * 0.2, C + 2, 5, 2, pal["y"], 2, random.Random(7 + i))
        if i >= 1:
            r = (6, 12, 9, 14, 10, 8, 4)[i - 1]
            disc(d, C, C + 2, r, pal["y"])
            disc(d, C, C + 2, r * 0.55, pal["w"])
        if i >= 2:  # sparks around the target
            for k in range(6):
                a = k * math.pi / 3 + i * 0.4
                rr = 16 + (i % 3) * 4
                x, y = C + rr * math.cos(a), C + 2 + rr * math.sin(a)
                d.line([(x - 3, y), (x, y - 2), (x + 1, y + 2), (x + 4, y)], fill=rgba(pal["y"]), width=1)
        frames.append(outline(img, pal["k"]))
    return frames


def psychic():
    pal = {"p": (248, 104, 168), "m": (200, 64, 200), "l": (255, 184, 224), "w": (255, 240, 248), "k": (96, 16, 88)}
    cols = [pal["p"], pal["m"], pal["l"]]
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        for k in range(4):
            r = 30 - ((i * 4 + k * 8) % 32)  # rings converge on the target
            if r > 2:
                d.ellipse((C - r, C - r * 0.85, C + r, C + r * 0.85), outline=rgba(cols[(k + i) % 3]), width=2)
        disc(d, C, C, 2 + (i % 3), pal["w"])
        frames.append(outline(img, pal["k"]))
    return frames


def ice():
    pal = {"w": (255, 255, 255), "c": (168, 224, 248), "k": (40, 88, 160)}
    crystal = bitmap(CRYSTAL, CRYSTAL_PAL)
    spots = [(0, 0, 0), (-16, 6, 1), (16, 6, 1), (-9, -14, 2), (10, -13, 2)]
    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 6:
            for dx, dy, delay in spots:
                t = i - delay
                if t < 0:
                    continue
                sc = 1 if t < 2 else 2
                ic = crystal if t != 1 else crystal.resize((crystal.width, crystal.height), Image.NEAREST)
                paste(img, turn(ic, 0, flip=dx < 0), C + dx, C + dy, sc)
            if i == 5:
                sparkle(d, C + 6, C - 18, 4, pal["w"], pal["w"])
        else:  # shatter
            t = i - 5
            for k in range(12):
                a = k * math.pi / 6
                x, y = C + (10 + t * 7) * math.cos(a), C + (10 + t * 7) * math.sin(a)
                d.polygon([(x, y - 2), (x + 2, y), (x, y + 2), (x - 2, y)], fill=rgba(pal["c"] if k % 2 else pal["w"]))
        frames.append(outline(img, pal["k"]))
    return frames


def dragon():
    pal = {"d": (72, 40, 200), "b": (112, 96, 248), "l": (184, 176, 255), "w": (240, 240, 255), "k": (24, 8, 88)}
    frames = []
    for i in range(9):
        img = canvas()
        d = ImageDraw.Draw(img)
        n = 5
        for k in range(n):
            a = i * 0.55 + k * 2 * math.pi / n
            r = 6 + min(i, 5) * 3.5 - max(0, i - 6) * 4
            if r <= 0:
                continue
            x, y = C + r * math.cos(a), C + 6 + r * 0.6 * math.sin(a)
            h = 14 + 6 * ((i + k) % 2)
            if i >= 7:
                h *= 0.6
            flame_shape(d, x, y + h * 0.4, h, h * 0.6, (pal["d"], pal["b"], pal["l"], pal["w"]),
                        lean=0.4 * math.sin(a))
        frames.append(outline(img, pal["k"]))
    return frames


def dark():
    pal = {"p": (72, 48, 104), "l": (136, 104, 176), "w": (232, 224, 240), "k": (16, 8, 24)}
    upper = bitmap(TEETH_UP, TEETH_PAL)
    lower = upper.transpose(Image.FLIP_TOP_BOTTOM)
    frames = []
    for i in range(8):
        img = canvas()
        d = ImageDraw.Draw(img)
        if i < 5:
            gap = (20, 13, 7, 2, 0)[i]
            paste(img, upper, C, C - gap - 6, 2)
            paste(img, lower, C, C + gap + 6, 2)
        if i >= 4:
            t = i - 4
            d.polygon(star_points(C, C, 10 + t * 6, 4 + t * 2, 8, rot=0.4), fill=rgba(pal["p"]))
            d.polygon(star_points(C, C, 6 + t * 4, 3 + t, 8, rot=0.4), fill=rgba(pal["l"]))
            if t < 2:
                disc(d, C, C, 2, pal["w"])
        frames.append(outline(img, pal["k"]))
    return frames


EFFECTS = {
    "normal": normal, "fighting": fighting, "flying": flying, "poison": poison, "ground": ground,
    "rock": rock, "bug": bug, "ghost": ghost, "steel": steel, "fire": fire, "water": water, "grass": grass,
    "electric": electric, "psychic": psychic, "ice": ice, "dragon": dragon, "dark": dark,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"cell_size": S, "types": {}}
    for name, fn in EFFECTS.items():
        frames = fn()
        sheet = Image.new("RGBA", (S * len(frames), S), (0, 0, 0, 0))
        for k, fr in enumerate(frames):
            sheet.alpha_composite(fr, (k * S, 0))
        sheet.save(OUT / f"{name}.png", optimize=True)
        index["types"][name] = {"file": f"{name}.png", "columns": len(frames), "cells": list(range(len(frames)))}
    (OUT / "animations.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(EFFECTS)} animaciones generadas en {OUT}")


if __name__ == "__main__":
    main()
