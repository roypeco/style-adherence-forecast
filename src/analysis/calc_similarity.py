import json
import numpy as np

def jaccard(data1, data2):
    set_data1 = set(data1)
    set_data2 = set(data2)
    numerator = len(set_data1 & set_data2)
    denominator = len(set_data1 | set_data2)
    result = '{:.2f}'.format(
        numerator / denominator      
    )
    
    return result

def euclid(data1, data2):
    array1 = np.array(data1)
    array2 = np.array(data2)
    result = '{:.2f}'.format(
        np.linalg.norm(array1 - array2)
    )
    
    return result

def group_up(jaccard_border: float, distance_border: float):
    # 閾値
    jaccard_border = jaccard_border
    distance_border = distance_border
    print(f"jaccard_border: {jaccard_border} distance_border: {distance_border}")

    json_open = open('dataset/fix_rate.json')
    json_load = json.load(json_open)
    project_list = list(json_load)
    start_point = 1
    result_dict = {}
    
    for i in range(len(project_list)-1):
        result_list = []
        for j in range(start_point, len(project_list)):
            if i == j: continue
            # ジャッカード係数の計算
            result = jaccard(list(json_load[project_list[i]]), list(json_load[project_list[j]]))
            # じゃっカード係数が閾値を超えるものは距離も計算
            if float(result) >= jaccard_border:
                list1 = []
                list2 = []
                for wid in json_load[project_list[i]]:
                    if wid in json_load[project_list[j]]:
                        list1.append(float(json_load[project_list[i]][wid]))
                        list2.append(float(json_load[project_list[j]][wid]))
                dist = euclid(list1, list2)
                if float(dist) <= distance_border:
                    print(f"{project_list[i]} & {project_list[j]} ジャッカード係数: {result}, ユークリッド距離: {dist}")
                    result_list.append(project_list[j])
        # start_point += 1
        if result_list != []:
            result_dict[project_list[i]] = result_list
        
    return result_dict
