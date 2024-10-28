#!/bin/bash

# 检查是否提供了 antigen 参数
if [ -z "$1" ]; then
    echo "Usage: $0 <antigen>"
    exit 1
fi

ANTIGEN=$1
INSTALL_DIR=/usr/local/Absolut

# 确保工作目录存在
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# 获取文件名和下载 URL
info_output=$(AbsolutNoLib info_filenames $ANTIGEN)
filename=$(echo "$info_output" | grep -oP '(?<=Pre-calculated structures are in )[^\s]+')
url=$(echo "$info_output" | grep -oP '(?<=curl -O -J )[^\s]+')

# 检查文件是否已经存在
if [ -f "$INSTALL_DIR/${filename}" ]; then
    echo "File ${filename} already exists. Skipping download."
else
    if [ -n "$url" ]; then
        echo "Downloading from URL: $url"
        download_filename=$(basename $url)
        wget $url -O $INSTALL_DIR/$download_filename
        if [ $? -eq 0 ]; then
            unzip -o $INSTALL_DIR/$download_filename -d $INSTALL_DIR
            rm $INSTALL_DIR/$download_filename
        else
            echo "Download failed for $ANTIGEN"
            exit 1
        fi
    else
        echo "No URL found for antigen: $ANTIGEN"
        exit 1
    fi
fi