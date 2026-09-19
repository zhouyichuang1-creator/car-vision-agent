#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 report/实训报告.md 转成 report/实训报告.docx，并把训练曲线 + 混淆矩阵嵌入。

任务书允许 docx/md 任意一种，docx 更便于打分老师直接打开批注。
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
SRC_MD = ROOT / "report" / "实训报告.md"
OUT_DOCX = ROOT / "report" / "实训报告.docx"
CHART_TRAIN = ROOT / "checkpoints" / "training_curve.png"
CHART_CONF = ROOT / "checkpoints" / "confusion_matrix.png"
CHART_INFER = ROOT / "checkpoints" / "infer_strategy.png"
CHART_CONFDIST = ROOT / "checkpoints" / "conf_distribution.png"


# ---------- 工具：构建一个有正常中文字体的 doc ----------
def _set_cn_font(run, size_pt: float = 10.5, bold: bool = False, color: tuple[int,int,int] | None = None):
    run.font.name = "Microsoft YaHei"
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = rPr.makeelement(qn('w:rFonts'), {})
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), "Microsoft YaHei")
    rFonts.set(qn('w:ascii'), "Microsoft YaHei")
    rFonts.set(qn('w:hAnsi'), "Microsoft YaHei")
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def _add_paragraph(doc, text="", style="Normal", align=None, size=10.5, bold=False, color=None,
                   space_before=0, space_after=4, indent_left=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if indent_left is not None:
        pf.left_indent = Cm(indent_left)
    if text:
        run = p.add_run(text)
        _set_cn_font(run, size_pt=size, bold=bold, color=color)
    return p


def _add_heading(doc, text: str, level: int):
    sizes = {1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    _set_cn_font(run, size_pt=sizes[level], bold=True, color=(30, 58, 138) if level <= 2 else (55, 65, 81))
    return p


def _add_inline(p, text: str, bold: bool = False, color=None, size=10.5):
    run = p.add_run(text)
    _set_cn_font(run, size_pt=size, bold=bold, color=color)


# ---------- 表格 ----------
def _set_cell_borders(cell, color="C8D0E0", sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = tcPr.makeelement(qn('w:tcBorders'), {})
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right'):
        b = tcBorders.find(qn(f'w:{edge}'))
        if b is None:
            b = tcBorders.makeelement(qn(f'w:{edge}'), {})
            tcBorders.append(b)
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), sz)
        b.set(qn('w:color'), color)


def _set_cell_bg(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = tcPr.makeelement(qn('w:shd'), {})
        tcPr.append(shd)
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)


def _add_table(doc, rows: list[list[str]], header_bg="EEF2FF", header_color=(30, 58, 138),
               col_widths_cm: list[float] | None = None):
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.autofit = False
    if col_widths_cm:
        for i, w in enumerate(col_widths_cm):
            for cell in table.columns[i].cells:
                cell.width = Cm(w)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_borders(cell)
            if r_idx == 0:
                _set_cell_bg(cell, header_bg)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(str(val))
            _set_cn_font(
                run,
                size_pt=10,
                bold=(r_idx == 0),
                color=(header_color if r_idx == 0 else None),
            )
    # 表后留白
    _add_paragraph(doc, "", size=4)


def _add_image_centered(doc, img_path: Path, width_cm: float = 14.0, caption: str = ""):
    if not img_path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=Cm(width_cm))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        _set_cn_font(run, size_pt=9, color=(120, 130, 150))


# ---------- Markdown 行解析 ----------
def _flush_table(buf: list[str], doc):
    """buf 是连续多行 markdown 表格，先解析成二维 list，再写出。"""
    if not buf:
        return
    rows = []
    for ln in buf:
        ln = ln.strip()
        if not ln.startswith('|'):
            continue
        cells = [c.strip() for c in ln.strip('|').split('|')]
        # 跳过分隔行（包含 --- : 的）
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
        rows.append(cells)
    _add_table(doc, rows)
    buf.clear()


def _flush_list(buf: list[str], doc):
    if not buf:
        return
    for ln in buf:
        m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', ln)
        if not m:
            continue
        indent_spaces = len(m.group(1))
        depth = indent_spaces // 2
        text = m.group(3)
        is_ordered = bool(re.match(r'^\d+\.', ln.lstrip()))
        p = doc.add_paragraph(style="List Bullet" if not is_ordered else "List Number")
        p.paragraph_format.left_indent = Cm(0.6 + depth * 0.6)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        _render_inline(p, text)
    buf.clear()


def _render_inline(p, text: str):
    """处理 **bold**、*italic*、`code`、链接等行内元素。"""
    # 简单 token 切分
    tokens = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**'):
            _add_inline(p, tok[2:-2], bold=True, color=(30, 58, 138))
        elif tok.startswith('*') and tok.endswith('*'):
            _add_inline(p, tok[1:-1], bold=False)
        elif tok.startswith('`') and tok.endswith('`'):
            run = p.add_run(tok[1:-1])
            _set_cn_font(run, size_pt=9.5)
            run.font.name = "Consolas"
            run.font.color.rgb = RGBColor(199, 37, 78)
        else:
            _add_inline(p, tok)


def _flush_para(text: str, doc):
    text = text.strip()
    if not text:
        _add_paragraph(doc, "")
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    _render_inline(p, text)


def md_to_docx(md_path: Path, out_path: Path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    # 全局样式
    style = doc.styles['Normal']
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10.5)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = rpr.makeelement(qn('w:rFonts'), {})
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), "Microsoft YaHei")
    rfonts.set(qn('w:ascii'), "Microsoft YaHei")
    rfonts.set(qn('w:hAnsi'), "Microsoft YaHei")

    # 页面边距
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    table_buf: list[str] = []
    list_buf: list[str] = []

    def _flush_table_now():
        _flush_table(table_buf, doc)
    def _flush_list_now():
        _flush_list(list_buf, doc)

    for raw in lines:
        line = raw.rstrip()

        # 表格块
        if line.startswith('|'):
            _flush_list_now()
            table_buf.append(line)
            continue
        else:
            _flush_table_now()

        # 列表
        if re.match(r'^\s*([-*]|\d+\.)\s+', line):
            list_buf.append(line)
            continue
        else:
            _flush_list_now()

        # 分隔线
        if re.match(r'^\s*-{3,}\s*$', line):
            _add_paragraph(doc, "")
            continue

        # 标题
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            level = len(m.group(1))
            _add_heading(doc, m.group(2).strip(), level)
            continue

        # 引用
        if line.startswith('>'):
            text = line.lstrip('>').strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run('▎ ')
            _set_cn_font(run, size_pt=10.5, color=(120, 130, 150))
            _render_inline(p, text)
            continue

        # 普通段落
        if line.strip():
            _flush_para(line, doc)

    _flush_table_now()
    _flush_list_now()

    # ---------- 插入图表（按节） ----------
    # 找到 4.1 节后面插训练曲线 + 混淆矩阵
    body = doc.element.body
    # 先把已经写完的文档保存出来，再用 python-docx 重新打开后在指定段落后插图
    # 这里采取更简单的做法：在文档末尾追加图表节（先于提交清单），并给个明确标题
    doc.add_paragraph()
    _add_heading(doc, "附图：训练曲线、推理策略对比、置信度分布与混淆矩阵", 2)
    _add_paragraph(doc, "以下四张图均来自本组训练好的模型（checkpoints/best.pt）。")
    _add_image_centered(doc, CHART_TRAIN, width_cm=15.5, caption="图 A1 训练 Loss / 验证 Macro-F1 / 验证 Accuracy 随 epoch 变化（15 epoch）")
    _add_image_centered(doc, CHART_INFER, width_cm=15.5, caption="图 A2 推理策略对比：192px → 224px → +水平翻转 TTA 提升至 0.7723；5-crop 反而下降至 0.7592")
    _add_image_centered(doc, CHART_CONFDIST, width_cm=15.5, caption="图 A3 验证集置信度分布：预测正确样本 75% 的 conf ≥ 0.85，预测错误样本平均仅 0.57")
    _add_image_centered(doc, CHART_CONF, width_cm=15.5, caption="图 A4 验证集 51 类混淆矩阵（按 ID 升序，最后一列为 unknown）")

    doc.save(str(out_path))
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    md_to_docx(SRC_MD, OUT_DOCX)