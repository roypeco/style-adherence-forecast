from src.modules import model_loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import collections
import csv

model_name = "Logistic"  # Logistic, RandomForest, SVM の３種類から選ぶ
dummys = model_loader.get_dummy()
project_name = "transitions"
id_dict = {}
list1 = []
list2 = []
list3 = []
list4 = []
model_all = model_loader.load_model("merge", model_name)
model_one = model_loader.load_model("conventional", model_name, project_name=project_name)
# model_dict = model_loader.load_model("cross", model_name)

df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
df_one = pd.concat(
        [
            pd.get_dummies(df_value["Warning ID"]),
            df_value.drop(columns="Warning ID"),
        ],
        axis=1,
    )
not_dummy_t, not_dummy = train_test_split(df_value, test_size=0.2, shuffle=False)
_, test_value = train_test_split(df_one, test_size=0.2, shuffle=False)
_, test_label = train_test_split(df_label, test_size=0.2, shuffle=False)
for i in list(dummys):
    id_dict[i] = []
for wid in not_dummy["Warning ID"]:
    if wid in id_dict:
        id_dict[wid].append(1)
        for i in id_dict:
            id_dict[i].append(0)
        id_dict[wid].pop(-1)
    else:
        for i in id_dict:
            id_dict[i].append(0)
id_df = pd.DataFrame(id_dict)
not_dummy = not_dummy.reset_index(drop=True)
test_df = pd.concat([id_df, not_dummy], axis=1)
metric_list = list(not_dummy.columns)
metric_one_list = list(df_one.columns)
label = dummys + metric_list
pre_list = model_one.predict(test_value.drop(["Project_name"], axis=1))
pre_list2 = model_all.predict(
        test_df.drop(["Warning ID", "Project_name"], axis=1)
    )
label.remove("Warning ID")
label.remove("Project_name")
metric_one_list.remove("Project_name")

if model_name == "Logistic":
    importances_all = model_all.coef_.squeeze().tolist()
else:
    importances_all = model_all.feature_importances_
dic = dict(zip(label, importances_all))

if model_name == "Logistic":
    importances = model_one.coef_.squeeze().tolist()
else:
    importances = model_one.feature_importances_

dic_all = dict(zip(label, importances_all))
dic = dict(zip(metric_one_list, importances))

sorted_dict_all = dict(sorted(dic_all.items(), key=lambda item: abs(item[1])))
sorted_dict = dict(sorted(dic.items(), key=lambda item: abs(item[1])))
length = 10
keys_all = list(sorted_dict_all.keys())
r_keys_all = list(reversed(keys_all))
values_all = list(sorted_dict_all.values())
keys = list(sorted_dict.keys())[-length:]
values = list(sorted_dict.values())[-length:]

# for i in reversed(keys):
#     print(f"従来:{i} -> {r_keys_all.index(i)+1}")

for i, j, k, l in zip(not_dummy["Warning ID"].to_list(), test_label[0].to_list(), pre_list, pre_list2):
    if j == k:
       list1.append(i)
    if j == l:
        list2.append(i)
    list3.append(i)

for i in not_dummy_t["Warning ID"].to_list():
    list4.append(i)

# print("従来")
# print(collections.Counter(list1))
# print("提案")
# print(collections.Counter(list2))
# print("テストデータ")
# print(collections.Counter(list3))
# print("学習データ")
# print(collections.Counter(list4))
# print(confusion_matrix(test_label[0].to_list(), pre_list))
# print(confusion_matrix(test_label[0].to_list(), pre_list2))

# 図示
plt.figure(figsize=(12, 8))  # グラフのサイズを変更
plt.barh(y=keys, width=values)  # y軸をカラム名(keys)に変更
plt.xlabel('係数', fontsize=20)  # x軸の軸名を「係数」に変更し、ラベルのサイズを拡大
plt.ylabel('カラム名', fontsize=20)  # y軸の軸名を「カラム名」に変更し、ラベルのサイズを拡大
plt.xticks(fontsize=14)  # x軸のラベルのサイズを拡大
plt.yticks(fontsize=14)  # y軸のラベルのサイズを拡大
plt.tight_layout()  # レイアウトを自動調整
plt.savefig(f'results/{project_name}_{model_name}.png', bbox_inches='tight')  # 画像を保存
plt.show()