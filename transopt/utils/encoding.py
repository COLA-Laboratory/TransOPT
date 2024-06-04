import pandas as pds

def target_encoding(df:pds.DataFrame, column_name, target_name):
    """
    计算给定列的目标编码。

    参数:
    dataframe (pandas.DataFrame): 包含特征和目标列的DataFrame。
    column_name (str): 需要进行目标编码的列名。
    target_name (str): 目标变量的列名。

    返回:
    dict: 包含每个唯一值及其目标编码的字典。
    """
    # 计算每个唯一值的目标均值
    target_mean = df.groupby(column_name)[target_name].mean()
    target_rank = target_mean.rank(method='average')

    df[f'mean_encoding'] = df.groupby(column_name)[target_name].transform('mean')
    df[f'rank_encoding'] = df[column_name].map(target_rank)
    # target_rank = target_mean.rank
    print(df[[column_name, target_name, f'mean_encoding', f'rank_encoding']].head(10))

    encodings = {value: key for key, value in target_rank.to_dict().items()}

    # 返回结果字典
    return encodings

def multitarget_encoding(df:pds.DataFrame, column_name, target_names):
    encodings = {}
    for target in target_names:
        # 计算每个唯一值的目标均值
        target_mean = df.groupby(column_name)[target].mean()
        encodings[target] = target_mean.to_dict()
    return encodings