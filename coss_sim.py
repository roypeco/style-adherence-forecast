import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

res_list = list()
idx_list = list()

def cosine_similarity_2d(arr1, arr2):
    # numpy配列に変換
    arr1_np = np.array(arr1)
    arr2_np = np.array(arr2)

    # コサイン類似度を計算
    similarity_matrix = cosine_similarity(arr1_np, arr2_np)

    return similarity_matrix

with open('dataset/test_output.json') as f:
    d = json.load(f)

for i in d:
    idx_list.append(i)
    temp = list()
    for j in d[i]:
        if d[i][j][1] != 0:
            temp.append('{:.2f}'.format(d[i][j][0] / d[i][j][1]))
        else:
            temp.append(0)
    res_list.append(temp)

# np.set_printoptions(precision=3)
similarity_matrix = cosine_similarity_2d(res_list, res_list)

print("Cosine Similarity Matrix:")
print(pd.DataFrame(similarity_matrix, index=idx_list, columns=idx_list))
