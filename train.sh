#!/bin/bash
srun -J autoenco --mem 50000 -c 12 -w adamantium python3 cuto.py ~/gigatext1.txt ../model_10K.ckpt
