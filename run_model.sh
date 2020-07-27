#!/bin/bash

cd scr1_moyerej/ptnstrerrpredict/

#$ -o  scr1_moyerej/ptnstrerrpredict/test.out -e  scr1_moyerej/ptnstrerrpredict/test.err

source venv/bin/activate

python cnn.py
