#!/bin/bash
#srun -J autoenco --mem 15000 -c 8 -w locomotion python3 ae.py ~/embeddings/binarygigatext.bin ../AE_models/model_20K.ckpt
srun -J autoenco --mem 10000 -c 4 -w vibranium python3 ae.py ~/geo-emb/pretrained-embeddings/wiki-news-300d-1M-subword.bin 