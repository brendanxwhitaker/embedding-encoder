#!/bin/bash
srun -J autoenco --mem 50000 -c 12 -w adamantium python3 cuto.py ~/binarygigatext.bin ../model_10K.ckpt
