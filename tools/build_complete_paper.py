"""Build one integrated Word paper from the Q1/Q2 builder and Q3/Q4 reports."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "数学建模F题完整论文.docx"


def clean(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("__", "").replace("`", "")


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", line))


def row(line: str) -> list[str]:
    return [re.sub(r"\s+", " ", x.strip()) for x in line.strip().strip("|").split("|")]


def add_body(doc: Document, text: str, style: str | None = None) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.12
    r = p.add_run(clean(text))
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(10.2)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    # Put genuinely wide tables on a landscape page so their headings remain readable.
    n = len(rows[0])
    landscape = n > 10
    if landscape:
        sec = doc.add_section(WD_SECTION.NEW_PAGE)
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
        sec.left_margin = Inches(0.45)
        sec.right_margin = Inches(0.45)
        sec.top_margin = Inches(0.55)
        sec.bottom_margin = Inches(0.55)
    table = doc.add_table(rows=len(rows), cols=n)
    table.style = "Table Grid"
    hdr = table.rows[0]._tr.get_or_add_trPr()
    rep = OxmlElement("w:tblHeader"); rep.set(qn("w:val"), "true"); hdr.append(rep)
    for i, values in enumerate(rows):
        for j in range(n):
            c = table.cell(i, j)
            c.text = values[j] if j < len(values) else ""
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.name = "Microsoft YaHei"
                    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
                    r.font.size = Pt(6.8 if landscape and i == 0 else (6.5 if landscape else (7.3 if i == 0 else 7.2)))
                    if i == 0:
                        r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
            if i == 0:
                tcpr = c._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), "315B78"); tcpr.append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    if landscape:
        doc.add_section(WD_SECTION.NEW_PAGE)


def append_markdown(doc: Document, source: Path, heading: str) -> None:
    doc.add_page_break()
    add_body(doc, heading, "Heading 1")
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip() or line.startswith("# "):
            i += 1; continue
        if line.startswith("## "):
            add_body(doc, line[3:], "Heading 2"); i += 1; continue
        if line.startswith("### "):
            add_body(doc, line[4:], "Heading 3"); i += 1; continue
        if line.startswith("$$"):
            add_body(doc, clean(line.strip("$"))); i += 1; continue
        if line.startswith("!["):
            m = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if m:
                image = source.parent / m.group(2)
                if image.exists():
                    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.add_run().add_picture(str(image), width=Inches(6.25))
                    c = doc.add_paragraph(m.group(1)); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in c.runs: r.italic = True; r.font.size = Pt(8.5)
                else: add_body(doc, m.group(1))
            i += 1; continue
        if line.startswith("|") and i + 1 < len(lines) and is_sep(lines[i+1]):
            rows = [row(line)]; i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(row(lines[i])); i += 1
            add_table(doc, rows); continue
        add_body(doc, line); i += 1


def remove_reference_tail(doc: Document) -> None:
    paras = doc.paragraphs
    start = next((idx for idx, p in enumerate(paras) if p.text.strip() == "参考文献"), None)
    if start is None:
        return
    body = doc._body._element
    for p in list(doc.paragraphs[start:]):
        body.remove(p._element)


def main() -> None:
    # Reuse the established Q1/Q2 builder so its tables and figures remain unchanged.
    import sys
    sys.path.insert(0, str(ROOT))
    import build_unified_docx
    build_unified_docx.main()
    base = ROOT / "outputs" / "数学建模F题统一论文.docx"
    doc = Document(base)
    remove_reference_tail(doc)
    for p in doc.paragraphs:
        if p.text.strip() == "数学建模 F 题统一建模论文":
            p.text = "算力约束下提升大语言模型能力的资源配置建模"
        elif p.text.strip() == "语料质量评分与大模型训练标度律":
            p.text = "数学建模 F 题完整论文（问题一至问题四）"
    # Replace the short two-question lead with a four-question abstract statement.
    for p in doc.paragraphs:
        if p.text.startswith("本文将问题一的语料质量评分"):
            p.text = ("本文围绕算力约束下的语料质量、数据配比、参数规模、训练数据量与模型能力配置，"
                      "建立质量评分与成分数据回归、逐域 Loss 标度律、资源约束优化以及规模—时间技术进步分解四层模型。"
                      "第一问以 A1 全局 ECDF、五组语义等权、零值乘性替换和 Helmert ILR 保持质量与配比口径；"
                      "第二问以 Pythia B1 网格及 B6–B8 质量实验建立具有物理方向的非对称标度律；"
                      "第三问在预算、上下文、质量成本和 17 域配比约束下求解资源配置；"
                      "第四问接入 C1/C3/C4/C6/C8，完成逐任务聚合、贡献占比和前沿情景分析。"
                      "所有跨数据源桥接、外推、损坏 JSON 与插补不确定性均单独标注。")
            break
    append_markdown(doc, ROOT / "outputs" / "q3" / "第三问完整论文.md", "第三问 算力约束下的多维资源联合优化")
    append_markdown(doc, ROOT / "outputs" / "q4" / "第四问完整论文.md", "第四问 模型效率前沿与规模技术进步分解")
    add_body(doc, "附录 A C8 JSON 文件排除清单", "Heading 1")
    add_body(doc, "C8 目录共发现 1,958 个 JSON 文件，其中 1,954 个可解析文件进入逐任务聚合。下列 4 个文件因 JSON 语法损坏被排除，不参与任何评分、回归或前沿统计；原始文件保留在数据目录，排除记录同步写入 q4_json_exclusion_manifest.csv。")
    add_table(doc, [
        ["文件标识", "排除原因"],
        ["DreadPoor_Winter_Dawn-8B-TIES", "字符串未闭合，第 1453 行"],
        ["FINGU-AI_Chocolatine-Fusion-14B", "字符串未闭合，第 1162 行"],
        ["Intel_neural-chat-7b-v3-3", "字符串未闭合，第 1194 行"],
        ["L-RAGE_3_PRYMMAL-ECE-7B-SLERP-V1", "对象属性名格式错误，第 1090 行"],
    ])
    add_body(doc, "参考文献", "Heading 1")
    refs = [
        "[1] Aitchison J. The Statistical Analysis of Compositional Data. Chapman and Hall, 1986.",
        "[2] Egozcue J J et al. Isometric logratio transformations for compositional data analysis. Mathematical Geology, 2003.",
        "[3] Kaplan J et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.",
        "[4] Hoffmann J et al. Training Compute-Optimal Large Language Models. arXiv:2203.15556, 2022.",
        "[5] Xia M et al. RegMix: Data Mixture as Regression for Language Model Pre-training. ICML, 2024.",
        "[6] Biderman S et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML, 2023.",
        "[7] Bahri Y et al. Explaining neural scaling laws. PNAS, 2024.",
    ]
    for ref in refs: add_body(doc, ref)
    doc.core_properties.title = "算力约束下提升大语言模型能力的资源配置建模"
    doc.core_properties.subject = "数学建模 F 题完整论文"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
