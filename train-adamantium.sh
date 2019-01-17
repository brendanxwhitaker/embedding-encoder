#!/bin/bash
#srun -J autoenco --mem 15000 -c 8 -w locomotion python3 ae.py ~/embeddings/binarygigatext.bin ../AE_models/model_20K.ckpt
srun -J autoenco --mem 20000 -c 4 -w adamantium python3 ae.py ~/geo-emb/pretrained-embeddings/GoogleNews-vectors-negative300.bin  
#srun -J autoenco --mem 10000 -c 4 -w adamantium python3 ae.py ~/geo-emb/pretrained-embeddings/top_10000_emb.txt 
