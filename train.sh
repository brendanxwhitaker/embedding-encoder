#!/bin/bash
srun -J autoenco --mem 30000 -c 20 -w osmium python3 cuto.py ~/gigatext1.txt ../model_redux.ckpt
