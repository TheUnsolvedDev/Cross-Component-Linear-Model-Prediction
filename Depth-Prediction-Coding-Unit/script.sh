#!/bin/bash

echo "+-------------------------+"
echo "|Script for making datset |"
echo "+-------------------------+"

echo "Step: 1. Downloading data"
echo "Visit @-> https://www.dropbox.com/sh/eo5dc3h27t41etl/AAADvFKoc5nYcZw6KO9XNycZa?dl=0"
echo "Download Info.rar and YUV_All.part01.rar"

echo ""

RAR_INFO=Info.zip
RAR_YUV=YUV_All.part01.zip

echo "Step: 2. Unziping data"
if [ -f $RAR_INFO ] && [ -f $RAR_YUV ]; then
    echo "Info.rar and YUV_All.part01.rar found"
else
    echo "Info.rar and YUV_All.part01.rar not found"
    echo "Download Info.rar and YUV_All.part01.rar"
    exit
fi

mkdir -p ./YUV_All
mkdir -p ./Info