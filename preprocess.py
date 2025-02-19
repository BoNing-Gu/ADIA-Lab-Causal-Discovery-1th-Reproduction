import numpy as np
import pandas as pd
import multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt
import time

from tqdm.auto import tqdm

def gaussian_kernel(distances, tau):
    return np.exp(-distances / (2 * tau ** 2))

def compute_distances_and_weights(data, tau):
    # 计算样本间的欧氏距离平方
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    # 计算高斯核权重
    weights = gaussian_kernel(distances, tau)
    return weights

def process_edge(args):
    u, v, data, W = args
    n_samples, p = data.shape
    mask = [k for k in range(p) if k != v]
    if u not in mask:
        return (u, v, np.zeros(n_samples))
    X = data[:, mask]
    y = data[:, v]
    u_idx = mask.index(u)
    coeffs = np.zeros(n_samples)
    for i in range(n_samples):
        weights = W[i, :]
        # 加权最小二乘
        X_weighted = X * weights[:, np.newaxis]
        XtWX = X_weighted.T @ X
        XtWy = X_weighted.T @ y
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        coeffs[i] = beta[u_idx]
    return (u, v, coeffs)

def preprocess_data(data, tau):
    n_samples, p = data.shape
    edges = [(u, v) for u in range(p) for v in range(p) if u != v]
    preprocessed = np.zeros((len(edges), 3, n_samples))
    
    # 计算距离和权重矩阵
    W = compute_distances_and_weights(data, tau)
    
    # 顺序处理每个边
    results = [process_edge((u, v, data, W)) for (u, v) in edges]
    
    # 填充结果到preprocessed张量
    edge_indices = {(u, v): idx for idx, (u, v) in enumerate(edges)}
    for u, v, coeffs in results:
        idx = edge_indices[(u, v)]
        preprocessed[idx, 0, :] = data[:, u]
        preprocessed[idx, 1, :] = data[:, v]
        preprocessed[idx, 2, :] = coeffs
    
    return preprocessed

def process_dict_item(args):
    """用于并行处理的辅助函数（必须定义在顶层）"""
    key, df = args
    data = df.values
    
    # 计算tau
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    tau = np.sqrt(np.median(distances)) * 0.5
    
    preprocessed = preprocess_data(data, tau)
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
    
    # 计算tau，使用距离的中位数
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    tau = np.sqrt(np.median(distances)) * 0.5
    
    preprocessed = preprocess_data(data, tau)
    print("预处理后的张量形状:", preprocessed.shape)

    plot_preprocessed(df, preprocessed, '0', 'Y')
    plot_preprocessed(df, preprocessed, '3', 'Y')
    plot_preprocessed(df, preprocessed, '7', 'Y')

    end_time = time.time()
    print(f'运行时间: {end_time - start_time}秒')