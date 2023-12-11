import pandas as pd

def label_catcher(col: int, start_num: int, edge: int, df: pd.DataFrame):
    if int(df['Line Number_' + str(edge)][col]) != -1:
        return False
    else:
        for i in range(start_num, edge + 1):
            if df['Status_' + str(i)][col] == 'Delete Warning Code':
                return False
            elif df['Status_' + str(i)][col] == 'Fix Warning':
                return True
