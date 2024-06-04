import os
import subprocess
import shutil


def compile_tex(tex_path, output_folder):
    # 保存当前工作目录
    original_cwd = os.getcwd()

    # 将路径转换为绝对路径
    tex_path = os.path.abspath(tex_path)
    output_folder = os.path.abspath(output_folder)

    # 获取文件名和文件夹路径
    folder, filename = os.path.split(tex_path)
    name, _ = os.path.splitext(filename)

    # 切换到tex文件所在的文件夹
    os.chdir(folder)

    try:
        # 编译tex文件
        subprocess.run(['pdflatex', filename], check=True)

        # 裁剪PDF文件
        pdf_path = os.path.join(folder, name + '.pdf')
        cropped_pdf_path = pdf_path.replace('.pdf', '-crop.pdf')
        subprocess.run(['pdfcrop', pdf_path, cropped_pdf_path], check=True)

        # 将裁剪后的PDF文件移动到输出文件夹，并去掉-crop
        output_pdf_path = os.path.join(output_folder, name + '.pdf')
        shutil.move(cropped_pdf_path, output_pdf_path)

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
    finally:
        # 切换回原始工作目录
        os.chdir(original_cwd)

    # 删除.aux和.log文件以及未裁剪的PDF文件
    aux_path = os.path.join(folder, name + '.aux')
    log_path = os.path.join(folder, name + '.log')
    if os.path.exists(aux_path):
        os.remove(aux_path)
    if os.path.exists(log_path):
        os.remove(log_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)