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
    return (text.replace("**", "").replace("__", "").replace("`", "")
            .replace("續表", "续表").replace("重複", "重复"))


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
    if landscape and n > 10:
        key_cols, part_width = 2, 8
        groups = [(start, min(start + part_width - key_cols, n))
                  for start in range(key_cols, n, part_width - key_cols)]
        table_rows = [[row[:key_cols] + row[start:end] for row in rows] for start, end in groups]
    else:
        table_rows = [rows]
    for part_index, part in enumerate(table_rows):
        part_n = len(part[0])
        if len(table_rows) > 1:
            add_body(doc, f"续表 {part_index + 1}/{len(table_rows)}（重复前两列作为记录标识）")
        table = doc.add_table(rows=len(part), cols=part_n)
        table.style = "Table Grid"
        hdr = table.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader"); rep.set(qn("w:val"), "true"); hdr.append(rep)
        for i, values in enumerate(part):
            for j in range(part_n):
                c = table.cell(i, j)
                value = values[j] if j < len(values) else ""
                if str(value).strip().lower() in {"nan", "none", "nat"}:
                    value = "—"
                c.text = clean(str(value))
                for p in c.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after = Pt(1)
                    for r in p.runs:
                        r.font.name = "Microsoft YaHei"
                        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
                        r.font.size = Pt(8.0 if landscape else (7.3 if i == 0 else 7.2))
                        if i == 0:
                            r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
                if i == 0:
                    tcpr = c._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), "315B78"); tcpr.append(shd)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    if landscape:
        doc.add_section(WD_SECTION.NEW_PAGE)


def append_markdown(doc: Document, source: Path, heading: str, chapter_num: int) -> None:
    doc.add_page_break()
    add_body(doc, heading, "Heading 1")
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    skip_section = False
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip() or line.startswith("# "):
            i += 1; continue
        if line.startswith("## "):
            title = line[3:].strip()
            if title in {"摘要", "参考文献"}:
                skip_section = True
                i += 1
                continue
            skip_section = False
            m = re.match(r"^(\d+)\s+(.+)$", title)
            if m:
                title = f"{chapter_num}.{m.group(1)} {m.group(2)}"
            elif title == "数据交付":
                title = f"{chapter_num}.8 数据交付"
            elif title == "附录 结果文件":
                title = f"{chapter_num}.8 附录 结果文件"
            add_body(doc, title, "Heading 2"); i += 1; continue
        if line.startswith("### "):
            if skip_section:
                i += 1; continue
            title = line[4:].strip()
            m = re.match(r"^(\d+\.\d+)\s+(.+)$", title)
            if m:
                title = f"{chapter_num}.{m.group(1)} {m.group(2)}"
            elif title == "C3 经验包络":
                title = f"{chapter_num}.5.1 {title}"
            add_body(doc, title, "Heading 3"); i += 1; continue
        if skip_section:
            i += 1; continue
        if line.startswith("$$"):
            add_body(doc, clean(line.strip("$"))); i += 1; continue
        if line.startswith("!["):
            m = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if m:
                image = source.parent / m.group(2)
                if not image.exists():
                    image = ROOT / "outputs" / "figures" / m.group(2)
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


def add_toc_before(doc: Document, anchor_text: str) -> None:
    """Insert a populated static outline before the first chapter."""
    anchor = next((p for p in doc.paragraphs if p.text.strip().startswith(anchor_text)), None)
    if anchor is None:
        return
    # Remove any cached outline left by an intermediate builder so the final
    # document contains one deterministic table of contents.
    body = doc._body._element
    for p in list(doc.paragraphs):
        # The intermediate builder's Chinese text may be decoded differently
        # by the Windows COM round trip. Remove the two outline blocks by
        # their style and position as well as by literal text.
        if p.text.strip() in {"目录", "目录将在 Word 中更新", "Ŀ¼"} or (
            p.style.name == "Heading 1" and p.text.strip() in {"Ŀ¼", "目录"}
        ):
            body.remove(p._element)
    heading = doc.add_paragraph("目录", style="Heading 1")
    entries = []
    for para in doc.paragraphs:
        if para._p is heading._p or para.text.strip() == "目录" or not para.style.name.startswith("Heading "):
            continue
        level = int(para.style.name.split()[-1])
        if level <= 2 and para.text.strip() and para.text.strip() != "摘要":
            entries.append((level, para.text.strip()))
    anchor._p.addprevious(heading._p)
    for level, title in entries:
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Inches(0.24 * (level - 1))
        p.paragraph_format.space_after = Pt(2)
        p.add_run(title)
        anchor._p.addprevious(p._p)


def add_page_numbers(doc: Document) -> None:
    """Add a centered PAGE / NUMPAGES footer to every section."""
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.text = "第 "
        for run in p.runs:
            run.font.name = "Microsoft YaHei"; run.font.size = Pt(9)
        def append_field(name: str):
            for kind, value in (("begin", None), ("instr", name), ("separate", None), ("text", "1"), ("end", None)):
                run = OxmlElement("w:r")
                if kind == "instr":
                    node = OxmlElement("w:instrText"); node.set(qn("xml:space"), "preserve"); node.text = value
                elif kind == "text":
                    node = OxmlElement("w:t"); node.text = value
                else:
                    node = OxmlElement("w:fldChar"); node.set(qn("w:fldCharType"), kind)
                run.append(node); p._p.append(run)
        append_field("PAGE")
        p.add_run(" 页 / ")
        append_field("NUMPAGES")
        p.add_run(" 页")


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
            p_pr = p._p.get_or_add_pPr()
            p_bdr = p_pr.find(qn("w:pBdr"))
            if p_bdr is not None:
                p_pr.remove(p_bdr)
        elif p.text.strip() == "语料质量评分与大模型训练标度律":
            p.text = "数学建模 F 题完整论文（问题一至问题四）"
    # Replace the short two-question lead with a four-question abstract statement.
    intro_text = None
    for p in doc.paragraphs:
        if p.text.startswith("本文将问题一的语料质量评分"):
            intro_text = ("本文围绕算力约束下的语料质量、数据配比、参数规模、训练数据量与模型能力配置，"
                      "建立质量评分与成分数据回归、逐域 Loss 标度律、资源约束优化以及规模—时间技术进步分解四层模型。"
                      "第一问以 A1 全局 ECDF、五组语义等权、零值乘性替换和 Helmert ILR 保持质量与配比口径；"
                      "第二问以 Pythia B1 网格及 B6–B8 质量实验建立具有物理方向的非对称标度律；"
                      "第三问在预算、上下文、质量成本和 17 域配比约束下求解资源配置；"
                      "第四问接入 C1/C3/C4/C6/C8，完成逐任务聚合、贡献占比和前沿情景分析。"
                      "所有跨数据源桥接、外推、损坏 JSON 与插补不确定性均单独标注。")
            # The lead belongs to the unified abstract, not to the title block.
            body = doc._body._element
            body.remove(p._p)
            break
    if intro_text:
        abstract_heading = next((p for p in doc.paragraphs if p.text.strip() == "摘要"), None)
        if abstract_heading is not None:
            intro = doc.add_paragraph(intro_text)
            abstract_heading._p.addnext(intro._p)
    append_markdown(
        doc,
        ROOT / "outputs" / "q3" / "第三问完整论文.md",
        "第三问 算力约束下的多维资源联合优化",
        chapter_num=3,
    )
    append_markdown(
        doc,
        ROOT / "outputs" / "q4" / "第四问完整论文.md",
        "第四问 模型效率前沿与规模技术进步分解",
        chapter_num=4,
    )
    add_body(doc, "附录 A C8 JSON 文件排除清单", "Heading 1")
    add_body(doc, "C8 目录共发现 1,958 个 JSON 文件，其中 1,954 个可解析文件进入逐任务聚合。下列 4 个文件因 JSON 语法损坏被排除，不参与任何评分、回归或前沿统计；原始文件保留在数据目录，排除记录同步写入 q4_json_exclusion_manifest.csv。")
    add_table(doc, [
        ["文件标识", "排除原因"],
        ["DreadPoor_Winter_Dawn-8B-TIES", "字符串未闭合，第 1453 行"],
        ["FINGU-AI_Chocolatine-Fusion-14B", "字符串未闭合，第 1162 行"],
        ["Intel_neural-chat-7b-v3-3", "字符串未闭合，第 1194 行"],
        ["L-RAGE_3_PRYMMAL-ECE-7B-SLERP-V1", "对象属性名格式错误，第 1090 行"],
    ])
    add_body(doc, "附录 B 数据利用清单", "Heading 1")
    add_table(doc, [
        ["附件", "用途", "主模型角色"],
        ["A1", "22 项质量指标的 ECDF 与文档评分", "主输入"],
        ["A2/A3", "域级质量与冲突扩展复核", "扩展验证"],
        ["A4–A11", "17 域配方训练和独立检验", "主输入及检验"],
        ["A12–A15", "估算尺度外推和生成机制诊断", "情景诊断"],
        ["A17/A18", "领域先验说明与脱敏文本定性核验", "辅助"],
        ["B1", "Pythia N-D 带 E 标度律", "主输入"],
        ["B6–B8", "Q_score 质量缺陷项", "条件校准"],
        ["B9/B10", "超百亿估算外推", "敏感性"],
        ["C1/C3/C4/C6/C8", "能力、年度包络、算力、桥接与逐任务校准", "分层使用"],
    ])
    add_body(doc, "附录 C 人工智能工具使用说明", "Heading 1")
    add_body(doc, "本项目使用人工智能工具辅助代码审阅、程序调试、文献检索建议、图表排版和文字润色。数据读取、模型设定、统计检验、结果计算和结论均由项目代码执行并经人工复核；人工智能工具未替代参赛者对数据口径、模型假设与结果的独立判断。")
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
    add_toc_before(doc, "1 问题重述与数据边界")
    add_page_numbers(doc)
    doc.core_properties.title = "算力约束下提升大语言模型能力的资源配置建模"
    doc.core_properties.subject = "数学建模 F 题完整论文"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    # The legacy Q1/Q2 builder is used only as an intermediate source. Keep
    # the repository delivery to the single integrated paper.
    if base.exists() and base.resolve() != OUT.resolve():
        base.unlink()
    from organize_outputs import organize_figures
    organize_figures(ROOT / "outputs")
    # Q3/Q4 Markdown files are generated solely to assemble this Word paper.
    for name in (
        "q1/第一问_完整建模报告.md",
        "q1/第一问_补充实证与解释.md",
        "q3/第三问完整论文.md",
        "q4/第四问完整论文.md",
    ):
        (ROOT / "outputs" / name).unlink(missing_ok=True)
    print(OUT)


if __name__ == "__main__":
    main()
