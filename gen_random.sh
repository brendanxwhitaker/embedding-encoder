#!/bin/bash

# Script to compute and save random embedding vectors for a given source vocab.  
#srun -J rando --mem 30000 -c 20 -w osmium python3 rand_vecs.py ../embeddings/top_10000_emb.txt
srun -J rando --mem 50000 -c 3 -w adamantium python3 rand_vecs.py ../embeddings/top_10000_emb.txt
