#!/bin/bash
srun -J autoenco --mem 30000 -c 20 -w osmium python3 get_dist_vecs.py ~/binarygigatext.bin ../model_20K.ckpt
