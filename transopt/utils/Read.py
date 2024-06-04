import os
import pandas as pd
import requests


def read_file(file_path)->pd.DataFrame:
    _, file_extension = os.path.splitext(file_path)

    if file_extension:
        # Determine and read based on file extension
        if file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension == '.txt':
            return pd.read_csv(file_path, sep='\t')  # Adjust delimiter as needed
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            unnamed_columns = [col for col in df.columns if "--unlimited" in col]
            df.drop(unnamed_columns, axis=1, inplace=True)
            return df
        elif file_extension in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    else:
        # No file extension, attempt different methods to read
        try:
            return pd.read_csv(file_path)
        except:
            pass  # Continue trying if CSV fails
        try:
            return pd.read_excel(file_path)
        except:
            pass  # Continue trying if Excel fails
        try:
            return pd.read_json(file_path)
        except:
            pass  # Continue trying if JSON fails

        try:
            return pd.read_csv(file_path, sep='\t')  # Assuming it might be a TXT file
        except:
            pass  # Continue trying if TXT fails

        raise ValueError("File could not be read with any method. Ensure the file format is correct.")


def read_url(url):
    # 定义UCI和OpenML的URL模式
    uci_pattern = "archive.ics.uci.edu"
    openml_pattern = "openml.org"

    # 初始化数据集来源
    data_source = None

    # 尝试从URL下载数据
    try:
        response = requests.get(url)
        data = response.text

        # 检测URL是否指向UCI或OpenML
        if uci_pattern in url:
            data_source = "UCI"
        elif openml_pattern in url:
            data_source = "OpenML"

        # 返回数据和数据来源信息
        return data, data_source

    except requests.RequestException as e:
        return None, data_source
