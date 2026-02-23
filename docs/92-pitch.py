#!/usr/bin/env python3
"""Generate a one-page investor pitch PDF — AwakeMind framing."""

from fpdf import FPDF

OUTPUT = "/Users/vbaranov/My-Apps/j/notes/92-pitch.pdf"
FONT_DIR = "/System/Library/Fonts/"

# Colors — warm, alert palette
BLACK = (18, 18, 18)
WHITE = (255, 255, 255)
ACCENT = (234, 170, 50)     # warm gold
DARK_ACCENT = (160, 100, 10)  # deep amber
GRAY = (120, 120, 130)
LIGHT_BG = (252, 248, 240)  # warm cream


pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=False)

pdf.add_font("HN", "", FONT_DIR + "HelveticaNeue.ttc")
pdf.add_font("HN", "B", FONT_DIR + "HelveticaNeue.ttc")
pdf.add_font("HN", "I", FONT_DIR + "HelveticaNeue.ttc")

pdf.add_page()
pdf.set_margins(16, 10, 16)

W = 210
CONTENT_W = W - 32

# ── Top bar ──
pdf.set_fill_color(*BLACK)
pdf.rect(0, 0, W, 48, "F")

pdf.set_xy(18, 12)
pdf.set_text_color(*WHITE)
pdf.set_font("HN", "B", 28)
pdf.cell(0, 10, "AwakeMind", new_x="LMARGIN", new_y="NEXT")

pdf.set_xy(18, 24)
pdf.set_font("HN", "I", 13)
pdf.set_text_color(*ACCENT)
pdf.cell(0, 8, "The mind that never stops learning.", new_x="LMARGIN", new_y="NEXT")

pdf.set_xy(18, 34)
pdf.set_font("HN", "", 9)
pdf.set_text_color(180, 180, 180)
pdf.cell(0, 6, "A personal AI that watches, listens, and grows with you \u2014 always on, always yours, always local.", new_x="LMARGIN", new_y="NEXT")

y = 54

# ── The Problem ──
pdf.set_xy(16, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(0, 7, "THE PROBLEM", new_x="LMARGIN", new_y="NEXT")
y += 8

pdf.set_font("HN", "", 9)
pdf.set_text_color(*BLACK)
problems = [
    "Today\u2019s AI is asleep at the wheel \u2014 it forgets you the moment the conversation ends.",
    "Cloud AI learns from everyone except you. Your data improves their product, not yours.",
    "The best knowledge lives in people\u2019s heads. No one has built a way to keep it alive.",
]
for p in problems:
    pdf.set_x(16)
    pdf.multi_cell(CONTENT_W, 4.5, "\u2022  " + p, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() + 1
    pdf.set_y(y)

y += 2

# ── The Solution ──
pdf.set_xy(16, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(0, 7, "THE SOLUTION", new_x="LMARGIN", new_y="NEXT")
y += 8

pdf.set_xy(16, y)
pdf.set_font("HN", "", 9)
pdf.set_text_color(*BLACK)
pdf.multi_cell(CONTENT_W, 4.5,
    "A device that stays awake with you. It sees your screen, hears your conversations, "
    "and absorbs how you think. Every night it consolidates what it learned into permanent memory "
    "using novel on-device training (MEMIT + LoRA). No cloud. No subscription. "
    "By morning, it\u2019s not just an assistant \u2014 it\u2019s an extension of your mind.",
    new_x="LMARGIN", new_y="NEXT"
)
y = pdf.get_y() + 4

# ── Three boxes ──
box_w = (CONTENT_W - 8) / 3
box_h = 30
titles = ["WATCH", "ABSORB", "BECOME"]
descs = [
    "Always aware.\nSees your screen,\nhears your voice.",
    "Consolidates nightly\ninto permanent\nweight-level memory.",
    "Wakes up knowing\nwhat you know.\nGrows as you grow.",
]

for i in range(3):
    bx = 16 + i * (box_w + 4)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(bx, y, box_w, box_h, "F")

    pdf.set_xy(bx, y + 2)
    pdf.set_font("HN", "B", 9)
    pdf.set_text_color(*DARK_ACCENT)
    pdf.cell(box_w, 5, titles[i], align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(bx, y + 8)
    pdf.set_font("HN", "", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(box_w, 3.5, descs[i], align="C", new_x="LMARGIN", new_y="NEXT")

y += box_h + 5

# ── Why Now ──
pdf.set_xy(16, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(0, 7, "WHY NOW", new_x="LMARGIN", new_y="NEXT")
y += 8

pdf.set_font("HN", "", 9)
pdf.set_text_color(*BLACK)
why_now = [
    "Apple Silicon makes on-device LLM training real \u2014 no cloud GPU required.",
    "Post-GDPR / HIPAA world: local-only isn\u2019t a limitation, it\u2019s a compliance moat.",
    "Microsoft Recall proved demand for ambient capture. The privacy backlash proved it must be local.",
    "Rewind.ai ($350M valuation) does dumb search over screenshots. We make the AI actually learn.",
]
for w in why_now:
    pdf.set_x(16)
    pdf.multi_cell(CONTENT_W, 4.5, "\u2022  " + w, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() + 0.5
    pdf.set_y(y)

y += 3

# ── Market + Business Model side by side ──
half_w = (CONTENT_W - 6) / 2

pdf.set_xy(16, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(half_w, 7, "MARKET", new_x="LMARGIN", new_y="NEXT")

market_y = y + 8
pdf.set_xy(16, market_y)
pdf.set_text_color(*BLACK)
markets = [
    ("Knowledge workers", "consultants, analysts, researchers"),
    ("Skilled trades", "mechanics, surgeons, engineers"),
    ("Small businesses", "owner-operators, solo experts"),
    ("Enterprise", "onboarding, expert scaling, continuity"),
]
for title, desc in markets:
    pdf.set_x(16)
    pdf.set_font("HN", "B", 8.5)
    pdf.cell(half_w, 4.2, "\u2022  " + title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(21)
    pdf.set_font("HN", "", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.cell(half_w, 3.8, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BLACK)
    pdf.set_y(pdf.get_y() + 0.5)

bm_x = 16 + half_w + 6
pdf.set_xy(bm_x, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(half_w, 7, "BUSINESS MODEL", new_x="LMARGIN", new_y="NEXT")

pdf.set_xy(bm_x, market_y)
pdf.set_text_color(*BLACK)
bm_items = [
    ("Device sale", "$500\u2013800 consumer / $1,500+ pro"),
    ("No subscription", "zero recurring cost = headline differentiator"),
    ("Expert cloning", "per-clone license to deploy trained models"),
    ("Enterprise fleet", "bulk pricing + support contracts"),
]
for title, desc in bm_items:
    pdf.set_x(bm_x)
    pdf.set_font("HN", "B", 8.5)
    pdf.cell(half_w, 4.2, "\u2022  " + title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(bm_x + 5)
    pdf.set_font("HN", "", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.cell(half_w, 3.8, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BLACK)
    pdf.set_y(pdf.get_y() + 0.5)

y = max(pdf.get_y(), market_y + 38) + 4

# ── The Moat ──
pdf.set_xy(16, y)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*DARK_ACCENT)
pdf.cell(0, 7, "THE MOAT", new_x="LMARGIN", new_y="NEXT")
y += 8

pdf.set_xy(16, y)
pdf.set_font("HN", "", 9)
pdf.set_text_color(*BLACK)
pdf.multi_cell(CONTENT_W, 4.5,
    "The longer it\u2019s awake with you, the more irreplaceable it becomes. "
    "Not through lock-in \u2014 through accumulated intelligence. "
    "Competitors can copy the hardware. They can\u2019t copy six months of learning your mind.",
    new_x="LMARGIN", new_y="NEXT"
)

# ── Bottom bar / Ask ──
pdf.set_fill_color(*BLACK)
pdf.rect(0, 272, W, 25, "F")

pdf.set_xy(18, 274)
pdf.set_font("HN", "B", 11)
pdf.set_text_color(*WHITE)
pdf.cell(0, 6, "THE ASK", new_x="LMARGIN", new_y="NEXT")

pdf.set_xy(18, 281)
pdf.set_font("HN", "", 9)
pdf.set_text_color(180, 180, 180)
pdf.cell(0, 5,
    "Seed round to fund hardware prototyping, expand the research team, and ship the first 1,000 devices.",
    new_x="LMARGIN", new_y="NEXT"
)

pdf.set_xy(W - 74, 274)
pdf.set_font("HN", "I", 9)
pdf.set_text_color(*ACCENT)
pdf.multi_cell(58, 4.5, "\u201cNot trained by us.\nTrained by you.\u201d", align="R")

pdf.output(OUTPUT)
print(f"PDF saved to {OUTPUT}")
