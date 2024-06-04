import os
from pdf2image import convert_from_path
from transopt.ResultAnalysis.ReportNote import Notes


def pdf_to_png(pictures_path):
    assert os.path.exists(pictures_path), "File 'Pictures' isn't exist!"
    pdf_files = [f for f in os.listdir(pictures_path) if f.endswith('.pdf')]
    pictures = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pictures_path, pdf_file)
        images = convert_from_path(pdf_path, dpi=1000, fmt='png')
        for image in images:
            image.save(os.path.join(pictures_path, f"{pdf_file.split('.')[0]}.png"), 'png')
        pictures.append(pdf_file.split('.')[0])
    return pictures


def create_details_report(details_folders, save_path):
    for details_folder in details_folders:
        html_begin = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="UTF-8">
                <title> {details_folder.title().replace('_', ' ')} </title>
        """
        html_begin += """
                <style>
                    body {
                        align-items: center;
                        text-align: center;
                    }

                    .title_container {
                        background-color: #024098;
                        height: 100px;
                        line-height: 100px;
                    }

                    .title {
                        color: white;
                        /* display: flex; */
                        align-items: center;
                        justify-content: center;
                        white-space: nowrap;
                    }

                    .container {
                        display: flex;
                        flex-wrap: wrap;
                        /* align-items: center; */
                        justify-content: center;
                    }

                    .figure {
                        margin-left: 10px;
                        margin-right: 10px;
                    }

                    .button {
                        display: inline-block;
                        border-radius: 7px;
                        background: #024098;
                        color: white;
                        text-align: center;
                        font-size: 20px;
                        width: 100px;
                        height: 30px;
                        cursor: pointer;
                        text-decoration: none;
                        margin-top: 15px;
                    }
                </style>
            </head>
        """
        html_begin += f"""
        <body>
            <div class="title_container">
                <h1 class="title">{details_folder.title().replace('_', ' ')}</h1>
            </div>

            <a class="button" href="../Report.html">Back</a>
        """
        html_end = """
        </body>
        </html>
        """

        pictures_path = save_path / details_folder
        pictures = pdf_to_png(pictures_path)
        function_name = set()
        html_content = """"""
        for picture in pictures:
            # 将同一种函数归为一类
            if picture.split('_')[0] not in function_name:
                if len(function_name) != 0:
                    html_content += """
                        </div>

                    """
                function_name.add(picture.split('_')[0])
                html_content += f"""
                    <h2>{picture.split('_')[0]}</h2>
                    <div class="container">
                """
            html_content += f"""
                    <a class="figure" href="{picture}.png"><IMG SRC="{picture}.png" width="350px"></a>
            """

        with open(pictures_path / f"{details_folder.title().replace('_', ' ')}.html", 'w', encoding='utf-8') as html:
            html.write(html_begin + html_content + html_end)


def create_table_report(save_path):
    html_begin = """
    <!DOCTYPE html>
    <html>

    <head>
        <meta charset="UTF-8">
        <title> Tables </title>

        <style>
            body {
                align-items: center;
                text-align: center;
            }

            .title_container {
                background-color: #024098;
                height: 100px;
                line-height: 100px;
            }

            .title {
                color: white;
                align-items: center;
                justify-content: center;
                white-space: nowrap;
            }

            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .report_container {
                padding: 50px;
            }

            .report {
                width: 1050px;
                border: 3px solid #024098;
                border-radius: 15px;
            }

            .report_title {
                color: white;
                background-color: #024098;
                border-radius: 10px 10px 0 0;
                height: 50px;
                line-height: 50px;
                margin: 0;
            }

            .content {
                padding-top: 15px;
                padding-left: 10px;
                padding-right: 10px;
            }

            .report_figure {
                align-items: center;
            }

            .report_note {
                text-align: justify;
                margin: 10px;
            }

            .button {
                display: inline-block;
                border-radius: 7px;
                background: #024098;
                color: white;
                text-align: center;
                font-size: 20px;
                width: 100px;
                height: 30px;
                cursor: pointer;
                text-decoration: none;
                margin-top: 15px;
            }
        </style>
    </head>

    <body>
        <div class="title_container">
            <h1 class="title">Tables</h1>
        </div>

        <a class="button" href="../../Report.html">Back</a>

        <div class="container">
    """
    html_end = """
        </div>
    </body>

    </html>
    """

    tables = pdf_to_png(save_path)
    html_content = """"""
    for table in tables:
        report_container = f"""
                <div class="report_container">
                    <div class="report">
                        <h2 class="report_title">{table.title().replace('_', ' ')}</h2>
                        <div class="content">
                            <div class="report_figure">
                                <a href="{table}.png"><IMG
                                        SRC="{table}.png" width="1000px"></a>
                            </div>
                        </div>
                    </div>
                </div>
        """
        html_content += report_container

    with open(save_path / 'Tables.html', 'w', encoding='utf-8') as html:
        html.write(html_begin + html_content + html_end)


def create_report(save_path):
    html_begin = """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title> Analysis Report </title>

            <style>
                body {
                    align-items: center;
                    text-align: center;
                }

                .title_container {
                    background-color: #024098;
                    height: 100px;
                    line-height: 100px;
                }

                .title {
                    color: white;
                    align-items: center;
                    justify-content: center;
                    white-space: nowrap;
                }

                .container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .report_container  {
                    padding: 50px;
                }

                .report {
                    width: 520px;
                    height: 600px;
                    border: 3px solid #024098;
                    border-radius: 15px;
                }

                .report_title {
                    color: white;
                    background-color: #024098;
                    border-radius: 10px 10px 0 0;
                    height: 50px;
                    line-height: 50px;
                    margin: 0;
                }

                .content {
                    padding-top: 15px;
                    padding-left: 10px;
                    padding-right: 10px;
                }

                .report_figure{
                    align-items: center;
                }

                .report_note {
                    text-align: justify;
                    margin: 10px;
                }

                .button {
                    display: inline-block;
                    border-radius: 7px;
                    background: #024098b0;
                    color: white;
                    text-align: center;
                    font-size: 20px;
                    width: 400px;
                    height: 30px;
                    cursor: pointer;
                    text-decoration: none;
                    margin-top: 15px;
                }
            </style>
        </head>
    <body>
        <div class="title_container">
            <h1 class="title">Analysis Rusults Report</h1>
        </div>

        <div class="container">
    """
    html_end = """
        </div>
    </body>
    </html>
    """

    # 读取生成的图片名，并写入html
    pictures_path = save_path / 'Overview' / 'Pictures'
    pictures = pdf_to_png(pictures_path)
    html_content = """"""
    for picture in pictures:
        report_container = f"""
        <div class="report_container">
            <div class="report">
                <h2 class="report_title">{picture.title().replace('_', ' ')}</h2>
                <div class="content">
                    <div class="report_figure">
                        <a href="Overview/Pictures/{picture}.png"><IMG SRC="Overview/Pictures/{picture}.png" width="450px"></a>
                    </div>
                    <div class="report_note">
                        <p><b>Note:</b> {Notes[picture]} </p>
                    </div>
                </div>
            </div>
        </div>
        """
        html_content += report_container

    # 更多 information 的链接
    table_path = save_path / 'Overview' / 'Table'
    create_table_report(table_path)
    report_container = """
        <div class="report_container">
            <div class="report">
                <h2 class="report_title">More Information</h2>
                <div class="content">
                    <a class="button" href="Overview/Table/Tables.html">Tables</a>
    """

    folders = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
    details_folders = [f for f in folders if f != 'Overview']
    if len(details_folders) != 0:
        create_details_report(details_folders, save_path)
        for details_folder in details_folders:
            report_container += f"""
                    <a class="button" href="{details_folder}/{details_folder.title().replace('_', ' ')}.html">{details_folder.title().replace('_', ' ')}</a>
            """

    report_container += """
                </div>
            </div>
        </div>
    """
    html_content += report_container

    with open(save_path / 'Report.html', 'w', encoding='utf-8') as html:
        html.write(html_begin + html_content + html_end)

