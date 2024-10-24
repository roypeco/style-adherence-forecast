import copy

import numpy as np
import pandas as pd
from modules import model_loader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import Counter

model_name = "RandomForest"  # Logistic, RandomForest, SVM の３種類から選ぶ
dummys = model_loader.get_dummy()
model_all = model_loader.load_model("merge", model_name)
model_dict = model_loader.load_model("cross", model_name)
id_dict = {}
res_dict = {}
all_predict_list = []
all_answer_list = []
all_cluster_list = []
conv_list = []
pre1_list = []
pre2_list = []
test_num_list = []
ans_num_list = []

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

# project_list = [
#     "schema_salad",
#     # "serverless-application-model",
#     # "transitions",
#     # "django-fsm",
# ]  # 任意のプロジェクトだけを選択

for project_name in project_list:
    # プロジェクトごとのデータの読み込み
    df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
    df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
    df_cluster = pd.read_csv(
        f"./dataset/outputs/{project_name}_cluster.csv", header=None
    )
    df_merge = pd.concat(
        [pd.get_dummies(df_value["Warning ID"]), df_value.drop(columns="Warning ID")],
        axis=1,
    )
    _, X_test, _, Y_test, _, Z_test = train_test_split(
        df_merge, df_label, df_cluster, test_size=0.2, shuffle=False
    )
    _, not_dummy = train_test_split(df_value, test_size=0.2, shuffle=False)
    Y_test = Y_test.values.ravel()
    Z_test = Z_test.values.ravel()
    not_dummy["AnsTF"] = copy.deepcopy(Y_test)
    not_dummy["Cluster_num"] = copy.deepcopy(Z_test)
    predict_result = []
    ans_list = []
    cluster_list = []
    id_dict.clear()

    # 従来手法
    model = model_loader.load_model(
        "conventional", model_name, project_name=project_name
    )
    predicted = model.predict(X_test.drop(["Project_name"], axis=1))
    # print(confusion_matrix(Y_test, predicted))

    # 提案手法1
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
    predicted1 = model_all.predict(
        test_df.drop(["Warning ID", "Project_name", "Cluster_num", "AnsTF"], axis=1)
    )
    # print(confusion_matrix(Y_test, predicted1))

    # 提案手法2
    for i in range(10):
        try:
            if len(list(test_df[test_df["Cluster_num"] == i]["AnsTF"])) != 0:
                match model_name:
                    case "Logistic":
                        if (
                            model_dict[f"cluster_{i}"].__getattribute__("coef_")
                            is not None
                        ):
                            result_tmp = model_dict[f"cluster_{i}"].predict(
                                test_df[test_df["Cluster_num"] == i].drop(
                                    [
                                        "Warning ID",
                                        "Project_name",
                                        "Cluster_num",
                                        "AnsTF",
                                    ],
                                    axis=1,
                                )
                            )
                            predict_result.extend(result_tmp)
                            all_predict_list.extend(result_tmp)
                            ans_list.extend(
                                list(test_df[test_df["Cluster_num"] == i]["AnsTF"])
                            )
                            all_answer_list.extend(
                                list(test_df[test_df["Cluster_num"] == i]["AnsTF"])
                            )
                            for j in range(
                                len(list(test_df[test_df["Cluster_num"] == i]["AnsTF"]))
                            ):
                                cluster_list.append(i)
                                all_cluster_list.append(i)

                    case "RandomForest" | "SVM":
                        predict_result.extend(
                            model_dict[f"cluster_{i}"].predict(
                                test_df[test_df["Cluster_num"] == i].drop(
                                    [
                                        "Warning ID",
                                        "Project_name",
                                        "Cluster_num",
                                        "AnsTF",
                                    ],
                                    axis=1,
                                )
                            )
                        )
                        ans_list.extend(
                            list(test_df[test_df["Cluster_num"] == i]["AnsTF"])
                        )
                        all_answer_list.extend(
                            list(test_df[test_df["Cluster_num"] == i]["AnsTF"])
                        )
                        for j in range(
                            len(list(test_df[test_df["Cluster_num"] == i]["AnsTF"]))
                        ):
                            cluster_list.append(i)
                            all_cluster_list.append(i)

        except (AttributeError, KeyError) as e:
            print(f"{project_name}: skip cluster_{i} {e}")
    # print(np.array(predict_result))
    # print(confusion_matrix(np.array(ans_list), np.array(predict_result)))
    conv_list.append(sum(predicted))
    pre1_list.append(sum(predicted1))
    pre2_list.append(sum(predict_result))
    test_num_list.append(len(test_df['AnsTF'].to_list()))
    ans_num_list.append(sum(test_df['AnsTF'].to_list()))
    # print(f"{project_name}\nテストケース数: {len(test_df['AnsTF'].to_list())}\n従来サジェスト数: {sum(predicted)}\n提案1サジェスト数: {sum(predicted1)}\n提案2サジェスト数: {sum(predict_result)}\n必要修正数: {sum(test_df['AnsTF'].to_list())}")

# リストの中央値と平均値を計算してプリントする関数
def print_statistics(list_name, data_list):
    median_value = np.median(data_list)
    mean_value = np.mean(data_list)
    print(f"{list_name}の中央値: {median_value}")
    print(f"{list_name}の平均値: {mean_value}\n")

# 各リストの統計量を計算して表示
print_statistics("従来サジェスト数", conv_list)
print_statistics("提案1サジェスト数", pre1_list)
print_statistics("提案2サジェスト数", pre2_list)
print_statistics("テストケース数", test_num_list)
print_statistics("必要修正数", ans_num_list)
# df_analysis = pd.DataFrame(
#     {
#         "Warning ID": not_dummy["Warning ID"].to_list(),
#         "conventional": predicted,
#         "suggestion1": predicted1,
#         "answer": Y_test,
#     }
# )
# print(df_analysis)

# result_counter = {"w_correct": 0, "w_notcorrect": 0, "pre_correct": 0, "sug_correct": 0}
# waid = "R1705"
# for pre, pre1, ans in zip(df_analysis[df_analysis["Warning ID"]==waid]["conventional"].to_list(), df_analysis[df_analysis["Warning ID"]==waid]["suggestion1"].to_list(), df_analysis[df_analysis["Warning ID"]==waid]["answer"].to_list()):
#     if (pre == 1 and pre1 == 1 and ans == 1 ):
#         result_counter["w_correct"] += 1
#     elif (pre == 0 and pre1 == 0 and ans == 1):
#         result_counter["w_notcorrect"] += 1
#     elif (pre == 1 and pre1 == 0 and ans == 1 ):
#         result_counter["pre_correct"] += 1
#     elif (pre == 0 and pre1 == 1 and ans == 1):
#         result_counter["sug_correct"] += 1
# print(result_counter)

# result_counter = {"w_correct": 0, "w_notcorrect": 0, "pre_correct": 0, "sug_correct": 0}
# for pre, pre1, ans in zip(
#     df_analysis["conventional"].to_list(),
#     df_analysis["suggestion1"].to_list(),
#     df_analysis["answer"].to_list(),
# ):
#     if pre == 1 and pre1 == 1 and ans == 1:
#         result_counter["w_correct"] += 1
#     elif pre == 0 and pre1 == 0 and ans == 1:
#         result_counter["w_notcorrect"] += 1
#     elif pre == 1 and pre1 == 0 and ans == 1:
#         result_counter["pre_correct"] += 1
#     elif pre == 0 and pre1 == 1 and ans == 1:
#         result_counter["sug_correct"] += 1
# print(result_counter)

# tmp_list = df_analysis["Warning ID"].to_list()

# counter = Counter(tmp_list).most_common()

# # 結果を出力
# for element, count in counter:
#     print(f"{element}: {count}")

# for i in range(10):
# print(f"Cluster {i} {all_cluster_list.count(i)}")
# # print(df_analysis[df_analysis["cluster"]==i]["predict"].to_list())
# # print(df_analysis[df_analysis["cluster"]==i]["answer"].to_list())
# # print(f1_score(df_analysis[df_analysis["cluster"]==i]["answer"].to_list(), df_analysis[df_analysis["cluster"]==i]["predict"].to_list(), zero_division=np.nan))
# print("適合率", end=" ")
# print(
#     format(
#         precision_score(df_analysis[df_analysis["cluster"]==i]["answer"].to_list(), df_analysis[df_analysis["cluster"]==i]["predict"].to_list(), zero_division=np.nan), ".2f"
#     )
# )
# print("再現率", end=" ")
# print(
#     format(
#         recall_score(df_analysis[df_analysis["cluster"]==i]["answer"].to_list(), df_analysis[df_analysis["cluster"]==i]["predict"].to_list(), zero_division=np.nan), ".2f"
#     )
# )
# print("F1値", end=" ")
# print(
#     format(
#         f1_score(df_analysis[df_analysis["cluster"]==i]["answer"].to_list(), df_analysis[df_analysis["cluster"]==i]["predict"].to_list(), zero_division=np.nan), ".2f"
#     )
# )
