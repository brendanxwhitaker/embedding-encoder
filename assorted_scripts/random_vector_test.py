import numpy as np

def rand_embedding_generator(vectors):
    
    # shape [<num_inputs>,<dimensions>]
    rand_emb_array = []

    for i in range(len(embedding_tensor)):
        vec = np.random.rand(len(embedding_tensor[0]))
        vec = vec / np.linalg.norm(vec)
        rand_emb_array.append(vec)
