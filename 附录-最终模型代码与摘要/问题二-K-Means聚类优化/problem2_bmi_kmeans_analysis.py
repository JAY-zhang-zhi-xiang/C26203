import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pygam import GAM, LinearGAM, s, te
import os

# --- 1. 数据加载与预处理 ---

def parse_ga(ga_str):
    """将孕周字符串 (如 '10W+3', '13w', '16W+1') 转换为天数。"""
    if not isinstance(ga_str, str):
        return np.nan
    
    ga_str = ga_str.lower().strip()
    
    if 'w+' in ga_str:
        try:
            weeks, days = ga_str.split('w+')
            return int(weeks) * 7 + int(days)
        except ValueError:
            return np.nan
    elif 'w' in ga_str:
        try:
            weeks = ga_str.replace('w', '')
            return int(weeks) * 7
        except ValueError:
            return np.nan
            
    return np.nan

def preprocess_data(filepath):
    """加载并预处理数据，并根据赛题要求筛选10-25周的样本。"""
    df = pd.read_excel(filepath)
    
    # 清理列名
    df.columns = df.columns.str.strip()
    
    # 重命名列以方便访问
    column_mapping = {
        '检测孕周': 'Gestational_week',
        'Y染色体浓度': 'Y_concentration_percentage',
        '高龄': 'AMA',
        '孕妇BMI': 'BMI',
        '单/双胎': 'Singleton_twin',
        '是否IVF': 'IVF'
    }
    df = df.rename(columns=column_mapping)
    
    # 转换Y浓度为数值，处理百分号
    if df['Y_concentration_percentage'].dtype == 'object':
        df['Y_concentration_percentage'] = df['Y_concentration_percentage'].str.replace('%', '', regex=False).astype(float) / 100
    
    # 过滤掉无效的Y浓度值
    df = df[df['Y_concentration_percentage'] > 0]
    
    # 计算Y的对数
    df['log_Y'] = np.log(df['Y_concentration_percentage'])
    
    # 解析孕周为天数
    df['GA_days'] = df['Gestational_week'].apply(parse_ga)
    
    # 移除在后续分析中绝对必要的列（孕周和Y浓度）的缺失值
    df.dropna(subset=['GA_days', 'log_Y', 'BMI'], inplace=True)
    
    # 根据赛题要求，筛选孕周在10周到25周之间的数据
    df = df[(df['GA_days'] >= 70) & (df['GA_days'] <= 175)]
    
    return df

# --- 2. K-Means 最佳聚类数 (Elbow Method) ---

def find_optimal_k(data, max_k=10):
    """使用肘部法则找到最佳的k值。"""
    sse = []
    # K-Means只处理二维或更高维度的数据，因此我们需要reshape
    # 这里的data已经是dropna过的了
    bmi_data = data[['BMI']].values
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(bmi_data)
        sse.append(kmeans.inertia_)
    
    return sse

def plot_elbow_method(sse, output_dir):
    """绘制肘部法则图。"""
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(sse) + 1), sse, 'bo-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Sum of Squared Errors (SSE)')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xticks(range(1, len(sse) + 1))
    
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, 'kmeans_elbow_plot.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Elbow method plot saved to {filepath}")

# --- 3. 执行K-Means并分析聚类结果 ---

def perform_kmeans(data, n_clusters):
    """执行K-Means聚类并返回带有聚类标签的数据。"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # K-Means期望一个二维数组
    bmi_data = data[['BMI']].values
    data['BMI_Cluster'] = kmeans.fit_predict(bmi_data)
    
    # 为了方便解读，我们根据BMI均值对簇进行排序和重命名
    cluster_means = data.groupby('BMI_Cluster')['BMI'].mean().sort_values()
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
    data['BMI_Cluster'] = data['BMI_Cluster'].map(cluster_mapping)
    
    return data

def plot_cluster_distribution(data, output_dir):
    """绘制BMI聚类分布图。"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=data, x='GA_days', y='BMI', hue='BMI_Cluster', palette='viridis', alpha=0.7, s=50)
    plt.title('BMI Distribution by Cluster across Gestational Age')
    plt.xlabel('Gestational Age (days)')
    plt.ylabel('BMI')
    plt.legend(title='BMI Cluster')
    
    filepath = os.path.join(output_dir, 'bmi_cluster_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Cluster distribution plot saved to {filepath}")

# --- 4. 分组GAM建模与预测 ---

def find_earliest_ga_for_threshold(gam_model, threshold=0.04, ga_range=None):
    """使用GAM模型和线性插值法找到达到浓度阈值的精确孕周。"""
    if ga_range is None:
        # 默认使用一个精细的网格
        ga_range = np.arange(70, 210, 0.1)

    log_threshold = np.log(threshold)
    
    # 预测浓度
    predicted_log_y = gam_model.predict(ga_range)
    
    # 找到第一个超过阈值的点的索引
    reaching_indices = np.where(predicted_log_y >= log_threshold)[0]
    
    if len(reaching_indices) > 0:
        index = reaching_indices[0]
        
        # 如果在范围的起点就已超过阈值，则返回起点
        if index == 0:
            return ga_range[0]
        
        # 线性插值，找到更精确的交点
        # y = mx + c => x = x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)
        x1, y1 = ga_range[index - 1], predicted_log_y[index - 1]
        x2, y2 = ga_range[index], predicted_log_y[index]
        
        precise_ga = x1 + (x2 - x1) * (log_threshold - y1) / (y2 - y1)
        return precise_ga
    else:
        return None # 在给定范围内未达到阈值

def calculate_gam_derivative(gam_model, ga_point):
    """计算GAM在某一点的导数（浓度增长率），单位是 %/天。"""
    h = 0.1 # 使用一个小的h来近似导数
    # 预测点周围的浓度（原始比例）
    y1 = np.exp(gam_model.predict(ga_point - h))
    y2 = np.exp(gam_model.predict(ga_point + h))
    # 计算斜率并转换为百分比
    derivative = (y2 - y1) / (2 * h) * 100
    return derivative

def plot_gam_fits(data, results_df, output_dir):
    """为每个聚类绘制GAM拟合曲线。"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    
    cluster_means = data.groupby('BMI_Cluster')['BMI'].mean()
    palette = sns.color_palette("viridis", n_colors=data['BMI_Cluster'].nunique())

    for cluster in sorted(data['BMI_Cluster'].unique()):
        cluster_data = data[data['BMI_Cluster'] == cluster]
        
        # 绘制散点
        plt.scatter(cluster_data['GA_days'], cluster_data['Y_concentration_percentage'], 
                    alpha=0.3, label=f'Cluster {cluster} (BMI ~{cluster_means[cluster]:.1f}) - Data', color=palette[cluster])
        
        # 绘制GAM曲线
        gam = LinearGAM(s(0, n_splines=20, lam=0.6)).fit(cluster_data['GA_days'], cluster_data['log_Y'])
        ga_grid = np.linspace(cluster_data['GA_days'].min(), cluster_data['GA_days'].max(), 200)
        y_pred = np.exp(gam.predict(ga_grid))
        
        plt.plot(ga_grid, y_pred, linewidth=2.5, 
                 label=f'Cluster {cluster} (BMI ~{cluster_means[cluster]:.1f}) - GAM Fit', color=palette[cluster])

    plt.axhline(y=0.04, color='r', linestyle='--', label='4% Threshold')
    plt.yscale('log')
    plt.title('GAM Fit of Y-Chromosome Concentration vs. Gestational Age by BMI Cluster')
    plt.xlabel('Gestational Age (days)')
    plt.ylabel('Y-Chromosome Concentration (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    filepath = os.path.join(output_dir, 'gam_fits_by_cluster.png')
    plt.savefig(filepath)
    plt.close()
    print(f"GAM fits plot saved to {filepath}")

def plot_gam_derivatives(data, output_dir):
    """计算并绘制每个聚类的GAM拟合曲线的一阶导数（增长率）。"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    
    cluster_means = data.groupby('BMI_Cluster')['BMI'].mean()
    palette = sns.color_palette("viridis", n_colors=data['BMI_Cluster'].nunique())
    
    ga_grid = np.linspace(data['GA_days'].min(), data['GA_days'].max(), 200)

    for cluster in sorted(data['BMI_Cluster'].unique()):
        cluster_data = data[data['BMI_Cluster'] == cluster]
        gam = LinearGAM(s(0, n_splines=20, lam=0.6)).fit(cluster_data['GA_days'], cluster_data['log_Y'])
        
        derivatives = [calculate_gam_derivative(gam, ga) for ga in ga_grid]
        plt.plot(ga_grid, derivatives, linewidth=2.5, 
                 label=f'Cluster {cluster} (BMI ~{cluster_means[cluster]:.1f})', color=palette[cluster])

    plt.axhline(y=0, color='r', linestyle='--', label='Zero Growth')
    plt.title('Concentration Growth Rate vs. Gestational Age by BMI Cluster')
    plt.xlabel('Gestational Age (days)')
    plt.ylabel('Growth Rate of Y-Concentration (%/day)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    filepath = os.path.join(output_dir, 'gam_derivatives_by_cluster.png')
    plt.savefig(filepath)
    plt.close()
    print(f"GAM derivatives plot saved to {filepath}")

def predict_concentration_at_milestones(gam_model, milestones_days):
    """在给定的孕周里程碑上预测浓度。"""
    log_preds = gam_model.predict(milestones_days)
    return np.exp(log_preds)


# --- 6. 三维GAM模型与可视化 ---

def build_and_plot_3d_gam(data, output_dir):
    """
    构建一个二维GAM模型 (Y ~ f(GA, BMI)) 并绘制三维响应曲面图。
    同时返回训练好的GAM模型以供后续使用。
    """
    print("\n--- Building and plotting 2D GAM for 3D visualization ---")
    
    # 准备数据，确保没有缺失值
    model_data = data.dropna(subset=['GA_days', 'BMI', 'log_Y'])
    X = model_data[['GA_days', 'BMI']].values
    y = model_data['log_Y'].values
    
    # 构建模型，使用张量积(te)来捕捉GA和BMI的交互作用
    gam = GAM(te(0, 1, n_splines=[25, 15], lam=[0.6, 0.6])).fit(X, y)
    
    # 创建用于绘图的网格
    ga_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    bmi_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    GA, BMI = np.meshgrid(ga_grid, bmi_grid)
    
    # 在网格上进行预测
    grid_X = np.vstack([GA.ravel(), BMI.ravel()]).T
    pred_log_y = gam.predict(grid_X)
    pred_y = np.exp(pred_log_y).reshape(GA.shape)
    
    # 绘制三维图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(GA, BMI, pred_y, cmap='viridis', alpha=0.7, rstride=5, cstride=5, linewidth=0.1, antialiased=True)
    ax.scatter(model_data['GA_days'], model_data['BMI'], model_data['Y_concentration_percentage'], 
               c=model_data['Y_concentration_percentage'], cmap='viridis', s=15, alpha=0.5, label='Original Data')

    ax.set_title('3D GAM Surface for Y-Concentration vs. GA and BMI', fontsize=16)
    ax.set_xlabel('Gestational Age (days)', fontsize=12, labelpad=10)
    ax.set_ylabel('BMI', fontsize=12, labelpad=10)
    ax.set_zlabel('Predicted Y-Concentration', fontsize=12, labelpad=10)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Y-Concentration')
    ax.view_init(elev=20, azim=-65)
    
    filepath = os.path.join(output_dir, 'gam_3d_surface_plot.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"3D GAM surface plot saved to {filepath}")
    
    return gam # 返回训练好的模型

# --- 7. 查找并绘制最佳检测孕周 ---

def find_and_plot_optimal_ga(gam_model, data, output_dir):
    """
    基于三维GAM模型，为每个BMI簇找到并绘制浓度达到峰值时的最佳孕周。

    Args:
        gam_model: 训练好的二维GAM模型 (Y ~ te(GA, BMI))。
        data: 包含'GA_days', 'BMI', 和 'BMI_Cluster'列的DataFrame。
        output_dir: 保存绘图的目录。
    """
    print("\n--- Finding and plotting optimal GA for each cluster ---")
    
    # 使用每个簇的平均BMI作为该簇的代表值
    cluster_bmis = data.groupby('BMI_Cluster')['BMI'].mean()
    
    # 定义孕周范围（10-25周，每天一个点）
    ga_range_days = np.arange(70, 176, 1)
    
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    palette = sns.color_palette("viridis", n_colors=len(cluster_bmis))
    
    optimal_points = []

    for i, (cluster_id, mean_bmi) in enumerate(cluster_bmis.items()):
        # 为当前簇的代表性BMI创建预测网格
        X_grid = np.array([[ga, mean_bmi] for ga in ga_range_days])
        
        # 预测Y浓度
        pred_y = np.exp(gam_model.predict(X_grid))
        
        # 找到浓度峰值点
        max_y_index = np.argmax(pred_y)
        optimal_ga_days = ga_range_days[max_y_index]
        max_y = pred_y[max_y_index]
        
        # 转换天数为周+天格式
        optimal_ga_weeks_float = optimal_ga_days / 7
        optimal_ga_week_part = int(optimal_ga_weeks_float)
        optimal_ga_day_part = int(round((optimal_ga_weeks_float - optimal_ga_week_part) * 7))

        optimal_points.append({
            'BMI Cluster': cluster_id,
            'Mean BMI': mean_bmi,
            'Optimal GA (Days)': optimal_ga_days,
            'Optimal GA (Week+Day)': f"{optimal_ga_week_part}W+{optimal_ga_day_part}d",
            'Peak Y Concentration': max_y
        })
        
        # 绘制该簇的浓度曲线
        ax.plot(ga_range_days / 7, pred_y, label=f'Cluster {cluster_id} (Mean BMI: {mean_bmi:.2f})', color=palette[i])
        # 标记峰值点
        ax.scatter(optimal_ga_days / 7, max_y, marker='*', s=200, zorder=5, color=palette[i], edgecolors='black')
        ax.annotate(f'{optimal_ga_week_part}W+{optimal_ga_day_part}d\n({max_y:.2f}%)', 
                    (optimal_ga_days / 7, max_y),
                    textcoords="offset points",
                    xytext=(0,15),
                    ha='center',
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    ax.set_xlabel('Gestational Age (Weeks)')
    ax.set_ylabel('Predicted Y Chromosome Concentration (%)')
    ax.set_title('Optimal NIPT Timing: Predicted Y Concentration Peak by BMI Cluster')
    ax.legend(title='BMI Cluster')
    ax.grid(True, which="both", ls="--")
    
    plot_path = os.path.join(output_dir, 'optimal_ga_by_cluster.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal GA plot saved to {plot_path}")

    # 打印总结表格
    summary_df = pd.DataFrame(optimal_points)
    print("\n--- Optimal NIPT Timing Summary ---")
    print(summary_df.to_string())


# --- 主执行流程 ---

def main():
    # 定义文件路径和参数
    data_filepath = 'newtest.xlsx'
    output_dir = 'problem2_analysis_plots_newtest'
    OPTIMAL_K = 6 # 基于肘部法则图的观察
    
    # --- Step 1: 加载和预处理数据 ---
    df = preprocess_data(data_filepath)
    
    if df.shape[0] < OPTIMAL_K:
        print(f"Error: Not enough data with BMI values to perform clustering. Found only {df.shape[0]} samples.")
        return
    print(f"Proceeding with k={OPTIMAL_K}")

    # --- Step 2: 执行K-Means并分析 ---
    df_clustered = perform_kmeans(df.copy(), n_clusters=OPTIMAL_K)
    plot_cluster_distribution(df_clustered, output_dir)
    
    cluster_summary = df_clustered.groupby('BMI_Cluster')['BMI'].describe()
    print("\n--- BMI Cluster Summary ---")
    print(cluster_summary)
    
    # --- Step 3: 分组建模与精细化预测 ---
    results = []
    milestones_weeks = [12, 14, 16, 18, 20]
    milestones_days = [w * 7 for w in milestones_weeks]
    
    ga_min = df_clustered['GA_days'].min()
    ga_max = df_clustered['GA_days'].max()
    fine_ga_grid = np.arange(ga_min, ga_max, 0.1)

    for cluster in sorted(df_clustered['BMI_Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['BMI_Cluster'] == cluster]
        gam = LinearGAM(s(0, n_splines=20, lam=0.6)).fit(cluster_data['GA_days'], cluster_data['log_Y'])
        earliest_ga = find_earliest_ga_for_threshold(gam, ga_range=fine_ga_grid)
        mean_ga_for_cluster = cluster_data['GA_days'].mean()
        growth_rate_at_mean_ga = calculate_gam_derivative(gam, mean_ga_for_cluster)
        milestone_preds = predict_concentration_at_milestones(gam, milestones_days)
        
        result_row = {
            'BMI Cluster': cluster, 'N Samples': len(cluster_data), 'Mean BMI': cluster_data['BMI'].mean(),
            'Earliest GA for 4% (weeks)': f"{int(earliest_ga // 7)}W+{earliest_ga % 7:.1f}d" if earliest_ga else "Not Reached",
            'Growth Rate at Mean GA (%/day)': growth_rate_at_mean_ga
        }
        for i, week in enumerate(milestones_weeks):
            result_row[f'Conc. at {week}W'] = milestone_preds[i]
        results.append(result_row)
        
    results_df = pd.DataFrame(results)
    cols = ['BMI Cluster', 'N Samples', 'Mean BMI', 'Earliest GA for 4% (weeks)', 'Growth Rate at Mean GA (%/day)']
    cols += [f'Conc. at {w}W' for w in milestones_weeks]
    results_df = results_df[cols]
    
    print("\n--- Detailed Analysis Results ---")
    print(results_df.to_string(formatters={col: '{:.4f}'.format for col in results_df.columns if 'Conc.' in col}))

    # --- Step 4: 结果可视化 (分组) ---
    plot_gam_fits(df_clustered, results_df, output_dir)
    plot_gam_derivatives(df_clustered, output_dir)

    # --- Step 5: 构建并绘制三维GAM图 ---
    # 注意这里我们使用df_clustered，因为它包含了聚类信息，尽管模型只用GA和BMI
    gam_3d = build_and_plot_3d_gam(df_clustered, output_dir)
    
    # --- Step 6: 查找并绘制每个簇的最佳孕周 ---
    find_and_plot_optimal_ga(gam_3d, df_clustered, output_dir)
    
    print("\nAnalysis complete. All plots and results are generated.")

if __name__ == '__main__':
    main()