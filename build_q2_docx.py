"""Build the single complete Question 2 report."""
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "q2"
DOC = OUT / "第二问_完整报告.docx"


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = tcPr.find(qn("w:shd"))
    if shd is None: shd = OxmlElement("w:shd"); tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def borders(table_obj):
    tblPr = table_obj._tbl.tblPr; b = tblPr.first_child_found_in("w:tblBorders")
    if b is None: b = OxmlElement("w:tblBorders"); tblPr.append(b)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = b.find(qn("w:" + edge))
        if el is None: el = OxmlElement("w:" + edge); b.append(el)
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "D9D9D9")


def remove_paragraph_borders(paragraph):
    """Remove inherited Word title-rule borders from a paragraph."""
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is not None:
        ppr.remove(pbdr)


def table(doc, frame, digits=4, max_rows=18):
    if frame is None or len(frame) == 0: return
    f = frame.head(max_rows).copy()
    t = doc.add_table(rows=1, cols=len(f.columns)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = "Table Grid"; borders(t)
    for j, c in enumerate(f.columns):
        cell = t.rows[0].cells[j]; cell.text = str(c); shade(cell, "1F4E78"); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in cell.paragraphs[0].runs: r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); r.font.size = Pt(7.5)
    for _, row in f.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(f.columns):
            v = row[c]
            if isinstance(v, float): s = "" if pd.isna(v) else f"{v:.{digits}f}"
            else: s = str(v)
            cells[j].text = s; cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[j].paragraphs:
                for r in p.runs: r.font.size = Pt(7.5)
        if len(t.rows) % 2 == 0:
            for cell in cells: shade(cell, "F3F6F9")


def para(doc, text, lead=None):
    p = doc.add_paragraph()
    if lead and text.startswith(lead): p.add_run(lead).bold = True; p.add_run(text[len(lead):])
    else: p.add_run(text)
    p.paragraph_format.space_after = Pt(5); p.paragraph_format.line_spacing = 1.15
    return p


def fig(doc, name, caption, width=6.2):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(OUT / name), width=Inches(width))
    c = doc.add_paragraph(caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs: r.italic = True; r.font.size = Pt(9)


def main():
    d = Document(); sec = d.sections[0]
    sec.top_margin = Inches(.65); sec.bottom_margin = Inches(.65); sec.left_margin = Inches(.75); sec.right_margin = Inches(.75)
    styles = d.styles; styles["Normal"].font.name = "Microsoft YaHei"; styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei"); styles["Normal"].font.size = Pt(10)
    for s in ["Title", "Heading 1", "Heading 2"]:
        styles[s].font.name = "Microsoft YaHei"; styles[s]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei"); styles[s].font.color.rgb = RGBColor(0, 0, 0)
    # The default Word Title style may carry a blue bottom rule; remove it so
    # the title remains a clean report heading.
    title_style_ppr = styles["Title"]._element.get_or_add_pPr()
    title_style_pbdr = title_style_ppr.find(qn("w:pBdr"))
    if title_style_pbdr is not None:
        title_style_ppr.remove(title_style_pbdr)
    title = d.add_paragraph(style="Title"); title.alignment = WD_ALIGN_PARAGRAPH.CENTER; title.add_run("问题二 跨维度数据融合与广义标度律").bold = True
    remove_paragraph_borders(title)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run("物理纠偏、硬指标验证与工程化接口报告").italic = True
    para(d, "本问在 B1--B10 附件和问题一接口上建立同时包含参数规模 N、数据量 D、质量变量和 17 域配比的标度律。所有 B 表 val_loss 均保留其 Pythia/Cerebras 原生验证集口径；The Pile Loss 与 B 表不直接混算。本文先纠正 Q_score 的物理方向，再给出结构化验证、边际技术替代率和跨语料桥接的可识别性边界。")

    d.add_heading("1 结论摘要", 1)
    para(d, "B1 不是普通随机样本，而是 8 个 N 乘 147 个 D 的完整无重复解析网格。格点均值解释 Loss 方差 100%，留一 N 的结构化检验均保持极高 R²，因此应表述为“解析网格流形下的无偏参数精确辨识”，不能把 R²=1.0000 当作独立外部泛化证据。")
    para(d, "B6--B8 的 Q_score 是噪声缺陷率：原始相关 r=0.6067，扣除 B1 的 N-D 基线后与 Loss 残差的相关 r=0.6661；全部 2090 条观测的质量方向为正。Q_quality=1-Q_score 与 |Q-1| 在 [0,1] 上只是同一保号仿射变换，不能翻转偏导。因此主模型改为 L=c+A N^-alpha+B D^-beta+gamma Q_score^theta，其中 gamma、theta>0，等价地对 Q_quality 的偏导严格为负。选择该形式的首要依据是物理正确性，不追求 R² 人为提升。")
    para(d, "B2 的绝对 Loss 通过同域五折仿射重标定后再评价，B4/B5 主要报告排序保序性。跨语料桥接 L_Pile=u L_Pythia+v 已提出并检验，但 High 层只有 7/75 条真正同模型同验证集配对，区间很宽，结论是“假设可检验但当前定量识别不足”，而不是预设拒绝。")

    d.add_heading("2 数据边界与接口", 1)
    table(d, pd.read_csv(OUT / "q2_input_usage_boundary.csv"), max_rows=10)
    para(d, "B9/B10 明确定位为超百亿参数外推的情景敏感性工具，不作为独立实验验证。问题一接口中的 Q_star、Q_bar(p) 和 ILR 替代向量仅作为场景输入；A 与 B 的质量量表和验证集不同，不能直接把 A1 的 Q_star 当作 B6--B8 的 Q_score。")

    d.add_heading("3 B1 经典标度律与网格识别", 1)
    para(d, "经典模型为 L=c+A N^-alpha+B D^-beta。B1 含 1176 行，8 个参数规模和 147 个数据规模，恰好填满 8×147 网格且 (N,D) 无重复。按 (N,D) 分组计算格点均值得到 group-mean R²=1.000000；该结果说明生成结构被精确辨识，而不是新增数据上的泛化保证。")
    table(d, pd.read_csv(OUT / "q2_b1_fit_coefficients.csv"), digits=6, max_rows=8)
    table(d, pd.read_csv(OUT / "q2_b1_grid_structure.csv"), digits=6, max_rows=4)
    table(d, pd.read_csv(OUT / "q2_b1_group_holdout_metrics.csv"), digits=6, max_rows=4)
    table(d, pd.read_csv(OUT / "q2_b1_leave_one_n.csv"), digits=6, max_rows=10)
    fig(d, "fig_q2_b1_observed_predicted.png", "图 1 B1 观测值与经典标度律预测值")
    fig(d, "fig_q2_b1_residuals.png", "图 2 B1 残差诊断")

    d.add_heading("4 B2--B5 验证与重标定", 1)
    table(d, pd.read_csv(OUT / "q2_scaling_validation_summary.csv"), digits=5, max_rows=8)
    table(d, pd.read_csv(OUT / "q2_b2_recalibration.csv"), digits=5, max_rows=8)
    table(d, pd.read_csv(OUT / "q2_b3_trajectory_validation.csv"), digits=5, max_rows=10)
    fig(d, "fig_q2_b3_trajectory_validation.png", "图 3 B3 八条轨迹，实线为观测、虚线为预测", width=6.6)
    para(d, "B2 原始绝对 Loss 受模型族和验证集偏移影响，仿射重标定只作为同一外部数据集内的校准诊断；B4/B5 的排序相关性比绝对 R² 更稳定。因此可迁移的是相对排序和趋势，不能把 B1 截距直接当成所有模型族的共同标尺。")

    d.add_heading("5 Qscore 物理纠偏与低端饱和", 1)
    para(d, "线性中心化模型和其镜像形式的预测值完全相同，镜像并不能改变偏导符号。主模型采用 q_defect=Q_score、q_quality=1-q_defect：L=c+A N^-alpha+B D^-beta+gamma q_defect^theta。于是 ∂L/∂q_defect=gamma theta q_defect^(theta-1)>0，∂L/∂q_quality<0；当 q_quality→1 时缺陷项趋于零，退化为经典 N-D 标度律。alpha、beta 围绕 B1 的 0.339957、0.279878 采用惩罚项松弛，lambda 取 0、0.1、1、10。")
    qtab = pd.read_csv(OUT / "q2_quality_effect_coefficients.csv")
    table(d, qtab[["lambda", "c", "A", "B", "gamma", "alpha", "beta", "theta", "alpha_minus_B1", "beta_minus_B1", "r2", "mae"]], digits=6, max_rows=8)
    table(d, pd.read_csv(OUT / "q2_quality_direction_diagnostics.csv"), digits=6, max_rows=5)
    fig(d, "fig_q2_quality_effect.png", "图 4 原生 Qscore 缺陷率与验证 Loss")
    low = pd.read_csv(OUT / "q2_low_q_saturation_check.csv")
    para(d, f"低端敏感性检验共得到 {len(low)} 个 Qscore=0.05/0.10 的 (N,D) 配对，其中 {int(low.equal_loss.sum())} 对 Loss 完全相等（{low.equal_loss.mean():.1%}）。例如 N=0.07、D=5 时两点分别为 1.0401 和 1.1624；N=1.0、D≥100 时两点均为 0.5000。该结果说明低端存在与规模相关的截断/饱和机制，不能把局部反常解释为质量方向翻转。")
    table(d, low.head(12), digits=4, max_rows=12)

    d.add_heading("6 MRTS 与统一广义标度律", 1)
    para(d, "将问题一的 16 维 ILR 坐标记为 z(p)，把逐域或宏观质量项写入统一形式：L_d(N,D,q,p)=c_d+A_d N^-alpha_d+B_d D^-beta_d+gamma_d q^theta+t_d^T z(p)+Delta_d^(2)。其中 Delta^(2) 是固定单纯形约束下的二阶非加性增量，系数语义为相对替代，不是绝对增加。")
    para(d, "全微分满足 dL=(∂L/∂lnN)dlnN+(∂L/∂lnD)dlnD+(∂L/∂q)dq+sum_i(∂L/∂p_i)dp_i=0，且 sum_i dp_i=0。质量提升令 dq<0，保持 Loss 不变时 ΔlnN=-(∂L/∂q·Δq)/(∂L/∂lnN)<0，ΔN%=(exp(ΔlnN)-1)×100%，N_equiv=N_base exp(ΔlnN)。")
    table(d, pd.read_csv(OUT / "q2_mrts_summary.csv"), digits=6, max_rows=12)
    para(d, "表中为完整 N-D 网格的中位数和四分位区间，避免用单点替代整体波动。当前非对称拟合在 q_quality=0.65、质量提升 0.1 的基准下给出参数替代区间；这是一种局部情景计算，不应外推为跨验证集的因果结论。")
    fig(d, "fig_q2_elasticity_heatmap.png", "图 5 参数弹性、数据弹性与有效质量边际效用的 1×3 网格")

    d.add_heading("7 17 域配比与二阶非加性", 1)
    para(d, "宏观质量泛函为 Q_bar(p)=sum_i p_i Q_i_star。配比使用单纯形闭合和 16 维 ILR，提升某域必然从其他域替代。问题一接口传入的 65 条 Top-5 记录全部来自 independent test quadratic model，是二次模型的条件预测，不是实验观测。")
    table(d, pd.read_csv(OUT / "q2_domain_substitution_effects.csv"), digits=5, max_rows=20)
    table(d, pd.read_csv(OUT / "q2_second_order_nonadditivity_summary.csv"), digits=5, max_rows=4)
    para(d, "65 条非加性增量的正号占比为 55.4%，中位数约 0.025，符号近乎混合。因此本文使用“模型条件非加性增量”和“条件替代效应”，不拔高为因果协同或互补机制。")

    d.add_heading("8 跨语料仿射桥接", 1)
    para(d, "提出可检验的跨语料假设 L_Pile=u L_Pythia+v，并分别在全部、High 和 Medium 层估计。High 层只有 7 条，占 75 条的 9.3%，且均为同一模型同一验证集的终检点；因此其 u 的置信区间较宽，当前只能给出不可稳定识别的定量证据。")
    btab = pd.read_csv(OUT / "q2_affine_bridge_test_results.csv")
    table(d, btab[["tier", "n", "u", "v", "r2", "F", "p_value", "ci_width_u", "identifiability"]], digits=5, max_rows=6)

    d.add_heading("9 大规模外推与 A18 脱敏", 1)
    table(d, pd.read_csv(OUT / "q2_large_scale_metrics.csv"), digits=5, max_rows=4)
    fig(d, "fig_q2_large_scale_extrapolation.png", "图 6 B9/B10 超百亿规模情景外推")
    para(d, "A18 仅用于技术性定性例证，覆盖 17 个域；每域只保留域名、3 个关键词、样本量和平均字符数，不保留任何连续原文。该表不能计算 Q，也不能构造冲突标签。")
    atab = pd.read_csv(OUT / "q2_A18_deidentified_domain_examples.csv")
    table(d, atab[["domain", "keyword_1", "keyword_2", "keyword_3", "sample_rows", "mean_text_chars"]], digits=1, max_rows=17)

    d.add_heading("10 复现与结论", 1)
    para(d, "复现命令为 $env:PYTHONPATH='src'; .venv\\Scripts\\python.exe -m f_question.q2，再运行 bundled Python 的 build_q2_docx.py。结论：B1 网格被精确识别；Qscore 按缺陷率建模；B2 需重标定；跨语料桥接暂不可稳定识别；二阶配比项仅作条件非加性；B9/B10 仅支持情景分析。")
    d.save(DOC); print(DOC)


if __name__ == "__main__": main()
