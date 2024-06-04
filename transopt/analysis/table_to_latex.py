import numpy as np
from typing import Union, Dict


def matrix_to_latex(Data: Dict, col_names, row_names, caption, oder="min"):
    mean = Data["mean"]
    std = Data["std"]
    significance = Data["significance"]
    num_cols = len(mean.keys())
    num_rows = len(row_names)

    if len(col_names) != num_cols or len(row_names) != num_rows:
        raise ValueError(
            "Mismatch between matrix dimensions and provided row/column names."
        )

    latex_code = []
    # 添加文档类和宏包
    latex_code.append("\\documentclass{article}")
    latex_code.append("\\usepackage{geometry}")
    latex_code.append("\\geometry{a4paper, margin=1in}")
    latex_code.append("\\usepackage{graphicx}")
    latex_code.append("\\usepackage{colortbl}")
    latex_code.append("\\usepackage{booktabs}")
    latex_code.append("\\usepackage{threeparttable}")
    latex_code.append("\\usepackage{caption}")
    latex_code.append("\\usepackage{xcolor}")
    latex_code.append("\\pagestyle{empty}")

    # 开始文档
    latex_code.append("\\begin{document}")
    latex_code.append("")
    latex_code.append("\\begin{table*}[t!]")
    latex_code.append("    \\scriptsize")
    latex_code.append("    \\centering")
    latex_code.append(f"    \\caption{{{caption}}}")
    latex_code.append("    \\resizebox{1.0\\textwidth}{!}{")
    latex_code.append("    \\begin{tabular}{c|" + "".join(["c"] * (num_rows)) + "}")
    latex_code.append("        \\hline")

    # Adding column names
    col_header = " & ".join([""] + row_names) + " \\\\"
    latex_code.append("        " + col_header)
    latex_code.append("        \\hline")

    # Adding rows
    for i in range(num_cols):
        str_data = []
        for j in range(num_rows):
            str_format = ""
            if oder == "min":
                if mean[col_names[i]][j] == np.min(mean[col_names[i]]):
                    str_format += "\cellcolor[rgb]{ .682,  .667,  .667}\\textbf{"
                    str_format += "%.3E(%.3E)" % (
                        float(mean[col_names[i]][j]),
                        std[col_names[i]][j],
                    )
                    str_format += "}"
                    str_data.append(str_format)
                else:
                    if significance[col_names[i]][row_names[j]] == "+":
                        str_data.append(
                            "%.3E(%.3E)$^\dagger$"
                            % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                        )
                    else:
                        str_data.append(
                            "%.3E(%.3E)"
                            % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                        )
            else:
                if mean[col_names[i]][j] == np.max(mean[col_names[i]]):
                    str_format += "\cellcolor[rgb]{ .682,  .667,  .667}\\textbf{"
                    str_format += "%.3E(%.3E)" % (
                        float(mean[col_names[i]][j]),
                        std[col_names[i]][j],
                    )
                    str_format += "}"
                    str_data.append(str_format)
                else:
                    if significance[col_names[i]][row_names[j]] == "+":
                        str_data.append(
                            "%.3E(%.3E)$^\dagger$"
                            % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                        )
                    else:
                        str_data.append(
                            "%.3E(%.3E)"
                            % (float(mean[col_names[i]][j]), std[col_names[i]][j])
                        )
        test_name = col_names[i] + col_names[i]
        row_data = " & ".join(["\\texttt{" + f"{test_name}" + "}"] + str_data) + " \\\\"
        latex_code.append("        " + row_data)

    latex_code.append("        \\hline")
    latex_code.append("    \\end{tabular}")
    latex_code.append("    }")
    latex_code.append("    \\begin{tablenotes}")
    latex_code.append("        \\tiny")
    latex_code.append(
        "        \\item The labels in the first column are the combination of the first letter of test problem and the number of variables, e.g., A4 is Ackley problem with $n=4$."
    )
    latex_code.append(
        "        \\item $^\\dagger$ indicates that the best algorithm is significantly better than the other one according to the Wilcoxon signed-rank test at a 5\\% significance level."
    )
    latex_code.append("    \\end{tablenotes}")
    latex_code.append("\\end{table*}%")
    latex_code.append("\\end{document}")

    return "\n".join(latex_code)
