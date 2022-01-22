#!/bin/bash
cd /home/gauenk/Documents/packages/pdnn/
`nohup  /usr/bin/python3.8 -u ./script/train_pdnn.py > train_log.txt &`
