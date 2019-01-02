#!/bin/bash
srun -J autoenco --mem 15000 -c 8 -w locomotion python3 ae.py ~/binarygigatext.bin ../model_20K.ckpt
