#!/bin/bash
#srun -J autoenco --mem 15000 -c 8 -w locomotion python3 ae.py ~/embeddings/binarygigatext.bin ../AE_models/model_20K.ckpt
srun -J autoenco --mem 25000 -c 4 -w adamantium python3 ae.py ~/geo-emb/pretrained-embeddings/glove.840B.300d.word2vec_clean.bin 
