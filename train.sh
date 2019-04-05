#!/bin/bash
#srun -J autoenco --mem 15000 -c 8 -w locomotion python3 ae.py ~/embeddings/binarygigatext.bin ../AE_models/model_20K.ckpt
#srun -J autoenco --mem 7000 -c 7 -w osmium python3 ae.py ~/geo-emb/pretrained-embeddings/wiki-news-300d-1M-subword.bin 
srun -J autoenco --mem 7000 -c 7 -w osmium python3 ae.py ~/geo-emb/pretrained-embeddings/parse-error-fix_glove.840B.300d.word2vec_clean.bin 
#srun -J autoenco --mem 7000 -c 7 -w osmium python3 ae.py ~/geo-emb/pretrained-embeddings/GoogleNews-vectors-negative300.bin 
