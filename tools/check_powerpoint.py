"""Render in installed PowerPoint and inspect text bounds."""
from pathlib import Path
import json
import win32com.client
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'artifacts' / 'powerpoint_preview'
PREVIEW.mkdir(parents=True, exist_ok=True)
app = win32com.client.DispatchEx('PowerPoint.Application')
deck = None
issues = []
try:
    deck = app.Presentations.Open(str(ROOT / 'SyFI_ML_Serving_refined.pptx'), True, False, False)
    slide_count = deck.Slides.Count
    hidden = []
    for slide in deck.Slides:
        if slide.SlideShowTransition.Hidden:
            hidden.append(slide.SlideIndex)
        slide.Export(str(PREVIEW / f'slide_{slide.SlideIndex:02}.png'), 'PNG', 1600, 900)
        for shape in slide.Shapes:
            if shape.HasTextFrame and shape.TextFrame.HasText:
                tf = shape.TextFrame2
                available = shape.Height - tf.MarginTop - tf.MarginBottom
                if tf.TextRange.BoundHeight > available + 3:
                    issues.append([slide.SlideIndex, 'textbox', shape.TextFrame.TextRange.Text[:70], round(tf.TextRange.BoundHeight, 1), round(available, 1)])
                available_width = shape.Width - tf.MarginLeft - tf.MarginRight
                if tf.TextRange.BoundWidth > available_width + 3:
                    issues.append([slide.SlideIndex, 'textbox width', tf.TextRange.Text[:70], round(tf.TextRange.BoundWidth, 1), round(available_width, 1)])
            if shape.HasTable:
                for r in range(1, shape.Table.Rows.Count + 1):
                    for c in range(1, shape.Table.Columns.Count + 1):
                        cell = shape.Table.Cell(r, c).Shape
                        tf = cell.TextFrame2
                        available = cell.Height - tf.MarginTop - tf.MarginBottom
                        if tf.TextRange.BoundHeight > available + 3:
                            issues.append([slide.SlideIndex, f'cell {r},{c}', tf.TextRange.Text[:70], round(tf.TextRange.BoundHeight, 1), round(available, 1)])
    result = {'slides': deck.Slides.Count, 'hidden_slides': hidden, 'text_overflow': issues}
    (PREVIEW / 'render_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
finally:
    if deck is not None:
        deck.Close()
    app.Quit()

sheet = Image.new('RGB', (1600, ((slide_count + 2) // 3) * 325), '#DCE2EA')
draw = ImageDraw.Draw(sheet)
for index in range(slide_count):
    im = Image.open(PREVIEW / f'slide_{index+1:02}.png').convert('RGB')
    im.thumbnail((510, 287))
    x, y = 15 + (index % 3) * 530, 12 + (index // 3) * 325
    sheet.paste(im, (x, y))
    draw.text((x+4, y+292), f'Slide {index+1}' + (' — backup' if index+1 in hidden else ''), fill='#18283B')
sheet.save(PREVIEW / 'contact_sheet.jpg', quality=90)
assert not issues, issues
