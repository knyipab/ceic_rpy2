#!/usr/bin/env bash

sudo apt update
sudo apt upgrade -y


sudo apt install r-base -y

pip install -r requirements.txt

# This works well for x86-64 but not ARM64
sudo R -e 'install.packages(c("R6", "xml2", "zoo", "httr", "getPass"))'
sudo R -e 'install.packages("ceic", repos="https://downloads.ceicdata.com/R/", type="source")'
mkdir ~/R

