"""Render a 1200x1200 LinkedIn card: what the classifier does + what it scored."""
import glob, random
from PIL import Image, ImageDraw, ImageFont

W = H = 1200
PAPER   = (245, 246, 247)
CARD    = (255, 255, 255)
INK     = (20, 23, 26)
INK2    = (77, 85, 94)
INK3    = (140, 148, 157)
BLUE    = (42, 120, 214)
BLUE_MUT= (154, 182, 220)
RED     = (227, 73, 72)
RULE    = (219, 223, 228)
GRID    = (232, 235, 239)

F = "C:/Windows/Fonts/"
def font(name, size): return ImageFont.truetype(F + name, size)
BOLD, SEMI, REG = "segoeuib.ttf", "seguisb.ttf", "segoeui.ttf"
MONO, MONOB = "consola.ttf", "consolab.ttf"

img = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(img)
M = 76


def text(xy, s, f, fill, anchor="la", spacing=0):
    if spacing:
        x, y = xy
        for ch in s:
            d.text((x, y), ch, font=f, fill=fill, anchor=anchor)
            x += d.textlength(ch, font=f) + spacing
        return x
    d.text(xy, s, font=f, fill=fill, anchor=anchor)
    return xy[0] + d.textlength(s, font=f)


# ---------------------------------------------------- header (left block)
y = 68
text((M, y), "FACE EXPRESSION CLASSIFIER", font(MONOB, 21), BLUE, spacing=2.4)
y += 42
text((M, y), "Happy vs Sad", font(BOLD, 84), INK)
y += 102
text((M, y), "Binary CNN on 600 labelled face crops", font(REG, 29), INK2)
y += 39
text((M, y), "FER2013  ·  48 x 48 grayscale  ·  balanced 300 / 300", font(MONO, 22), INK3)
header_bottom = y + 34

# ---------------------------------------------------- header (hero number)
# Placed top-right, where a feed reader's eye lands first.
text((W - M, 108), "87.3%", font(BOLD, 112), BLUE, anchor="ra")
text((W - M, 242), "cross-validated accuracy", font(MONO, 22), INK3, anchor="ra")

# ---------------------------------------------------- sample strips
y = header_bottom + 24
rng = random.Random(7)
THUMB, GAP = 104, 10
for cls in ("Happy", "Sad"):
    files = sorted(glob.glob(f"data/{cls}/*.png"))
    picks = [files[i] for i in rng.sample(range(len(files)), 8)]
    text((M, y + THUMB // 2), cls.upper(), font(MONOB, 21), INK2, anchor="lm", spacing=1.6)
    x = M + 108
    for p in picks:
        im = Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.NEAREST)
        img.paste(im, (x, y))
        d.rectangle([x, y, x + THUMB - 1, y + THUMB - 1], outline=RULE)
        x += THUMB + GAP
    y += THUMB + 12
y += 4
text((M, y), "actual training images, shown unsmoothed at source resolution",
     font(REG, 21), INK3)

# ---------------------------------------------------- results panel
PANEL_T = y + 50
PANEL_B = PANEL_T + 452
d.rectangle([M, PANEL_T, W - M, PANEL_B], fill=CARD, outline=RULE)

PAD = 34
px = M + PAD
py = PANEL_T + 32
text((px, py), "5-FOLD CROSS-VALIDATED ACCURACY", font(MONOB, 21), INK3, spacing=2.0)

BAR_W, BAR_H = 700, 30
VAL_X = px + BAR_W + 20          # one clean value column, clear of every track
by = py + 62
ROWS = [
    ("CNN from scratch",  60.67, 3.70, False),
    ("ResNet18 @64px",    79.17, 2.04, False),
    ("ResNet18 @224px",   87.33, 0.62, True),
]
for name, val, sd, win in ROWS:
    text((px, by - 26), name, font(SEMI if win else REG, 25), INK if win else INK2)
    ty = by + 6
    d.rectangle([px, ty, px + BAR_W, ty + BAR_H], fill=GRID)
    d.rectangle([px, ty, px + int(BAR_W * val / 100), ty + BAR_H],
                fill=BLUE if win else BLUE_MUT)
    text((VAL_X, ty + BAR_H // 2), f"{val:.1f}%", font(MONOB, 25),
         BLUE if win else INK2, anchor="lm")
    text((VAL_X + 96, ty + BAR_H // 2), f"± {sd:.2f}", font(MONO, 22), INK3, anchor="lm")
    by += BAR_H + 72

fy = PANEL_B - 52
d.line([px, fy - 16, W - M - PAD, fy - 16], fill=RULE)
text((px, fy), "Every image predicted once, by a model that never trained on it.",
     font(REG, 23), INK2)

# ---------------------------------------------------- footer
fy = PANEL_B + 24
lead = "Our first result was 95.8%"
text((M, fy), lead, font(SEMI, 24), RED)
text((M + d.textlength(lead, font=font(SEMI, 24)), fy),
     "  —  it was data leakage. This is the number after the fix.",
     font(REG, 24), INK2)
fy += 36
text((M, fy), "Shalev Yosefashvili  ·  Dimitry Todoseyev", font(MONO, 21), INK3)

img.save("linkedin_card.png")
print("wrote linkedin_card.png", img.size)
