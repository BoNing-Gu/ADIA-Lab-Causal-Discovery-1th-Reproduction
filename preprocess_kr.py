import numpy as np
import pandas as pd
import multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt
import time

from tqdm.auto import tqdm

import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg

def preprocess_data(data):
    n_samples, p = data.shape
    idx_list = np.arange(p)
    edges = [(u, v) for u in range(p) for v in range(p) if u != v]
    preprocessed = np.zeros((len(edges), 3, n_samples))

    results = []

    for v in range(p):
        endog = data[:, v]
        exog = np.delete(data, v, axis=1)

        model = KernelReg(endog, exog, var_type='c' * exog.shape[1], reg_type='ll', bw=[0.3] * exog.shape[1], ckertype='gaussian')
        _, coeffs_matrix = model.fit()

        u_list = np.delete(idx_list, v)
        for u_idx, u in enumerate(u_list):
            results.append((u, v, coeffs_matrix[:, u_idx]))

    edge_indices = {(u, v): idx for idx, (u, v) in enumerate(edges)}
    for u, v, coeffs in results:
        idx = edge_indices[(u, v)]
        preprocessed[idx, 0, :] = data[:, u]
        preprocessed[idx, 1, :] = data[:, v]
        preprocessed[idx, 2, :] = coeffs
    
    return preprocessed

def process_dict_item(args):
    key, df = args
    data = df.values
    
    preprocessed = preprocess_data(data)
    return (key, preprocessed)

def preprocess_dict(X_dict, n_workers=8):
    processed_dict = {}
    
    # 准备参数列表并保持DataFrame格式
    items = list(X_dict.items())
    
    # 并行处理
    with mp.Pool(processes=n_workers) as pool:
        # 使用imap保持顺序并显示进度条
        results = list(tqdm(pool.imap(process_dict_item, items),
                      total=len(items),
                      desc="Processing dictionary"))
    
    # 收集结果
    for key, preprocessed in results:
        processed_dict[key] = preprocessed
    
    return processed_dict

def plot_preprocessed(data, preprocessed, u_col, v_col):
    # u,v 从变量名转换为索引值
    u = data.columns.get_loc(u_col)
    v = data.columns.get_loc(v_col)
    # 找到u, v对应的索引
    n_samples, p = data.shape
    edges = [(u, v) for u in range(p) for v in range(p) if u != v]
    edge_indices = {(u, v): idx for idx, (u, v) in enumerate(edges)}
    idx = edge_indices[(u, v)]

    if idx is None:
        print(f"没有找到特征对 ({u}, {v})")
        return
    
    # 提取数据
    u_data = preprocessed[idx, 0, :]
    v_data = preprocessed[idx, 1, :]
    coeffs = preprocessed[idx, 2, :]

    # 设置Seaborn样式
    sns.set(style="whitegrid")
    
    # 创建图表，包含两个子图
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # 子图1: u vs v
    sns.scatterplot(x=u_data, y=v_data, ax=axs[0], color='blue', alpha=0.6)
    axs[0].set_xlabel(f'Feature {u_col}')
    axs[0].set_ylabel(f'Feature {v_col}')
    axs[0].set_title(f'Scatter plot: {u_col} vs {v_col}')
    
    # 子图2: u vs coeffs
    sns.scatterplot(x=u_data, y=coeffs, ax=axs[1], color='green', alpha=0.6)
    axs[1].set_xlabel(f'Feature {u_col}')
    axs[1].set_ylabel('Coefficients')
    axs[1].set_title(f'Scatter plot: {u_col} vs Coef')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    # 假设data是形状为(1000, p)的numpy数组
    df = pd.read_csv('data/A.csv', index_col=0)
    data = df.values
    
    preprocessed = preprocess_data(data)
    print("预处理后的张量形状:", preprocessed.shape)

    plot_preprocessed(df, preprocessed, '0', 'Y')
    plot_preprocessed(df, preprocessed, '3', 'Y')
    plot_preprocessed(df, preprocessed, '7', 'Y')

    end_time = time.time()
    print(f'运行时间: {end_time - start_time}秒')