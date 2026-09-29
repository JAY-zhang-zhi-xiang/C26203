import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
import os
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 字体与缓存处理 ---

def clear_matplotlib_cache():
    """清理Matplotlib的字体缓存，强制重新加载字体。"""
    try:
        cache_dir = fm.get_cachedir()
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.endswith(".json") or file.endswith(".ttf"):
                    os.remove(os.path.join(cache_dir, file))
            print("Matplotlib font cache cleared successfully.")
    except Exception as e:
        print(f"Error clearing Matplotlib cache: {e}")

def get_font_prop():
    """
    查找并设置一个可用的中文字体。
    按顺序尝试加载'SimHei', 'SimSun', 'Microsoft YaHei'。
    """
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc"
    ]
    
    installed_font_path = None
    for path in font_paths:
        if os.path.exists(path):
            installed_font_path = path
            break

    if installed_font_path:
        print(f"Found font: {installed_font_path}")
        prop = fm.FontProperties(fname=installed_font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        print(f"Set '{prop.get_name()}' as the primary sans-serif font.")
        return prop
    else:
        print("No suitable Chinese font found in C:/Windows/Fonts/. Using default font.")
        return None

# --- 数据加载与预处理 ---

def load_and_clean_data(filepath="./data.xlsx"):
    """
    加载并清洗男胎数据，仅保留问题2所需的核心列。
    """
    xls = pd.ExcelFile(filepath)
    
    # 仅加载男胎数据
    df = pd.read_excel(xls, sheet_name='男胎检测数据')

    # 清理列名中的空格
    df.columns = df.columns.str.replace(r'\s+', '', regex=True)
    
    print("男胎数据列名（清理后）:", df.columns.tolist())

    # 定义所需列和新列名
    required_cols = {
        '检测孕周': 'GA_str',
        '孕妇BMI': 'BMI',
        '21号染色体的Z值': 'Z_score', # 使用21号染色体的Z值
        '胎儿是否健康': 'abnormal_str'
    }
    
    # 检查所需列是否存在
    missing_cols = [col for col in required_cols.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少以下必需列: {missing_cols}")

    df = df[list(required_cols.keys())].rename(columns=required_cols)

    # --- 数据类型转换和解析 ---
    df['abnormal_flag'] = df['abnormal_str'].apply(lambda x: 1 if x == '是' else 0)

    # 解析孕周 (GA) -> total_days (e.g., '12W+3D' or '12W3D')
    ga_parts = df['GA_str'].str.extract(r'(\d+)[Ww](?:\+?(\d+))?')
    ga_parts[1] = ga_parts[1].fillna('0') # Fill NaN days with '0'
    df['GA_weeks'] = pd.to_numeric(ga_parts[0], errors='coerce')
    df['GA_days'] = pd.to_numeric(ga_parts[1], errors='coerce')
    df['total_days'] = df['GA_weeks'] * 7 + df['GA_days']

    df['Z_score'] = pd.to_numeric(df['Z_score'], errors='coerce')
    
    # 删除包含NaN的行
    df.dropna(subset=['total_days', 'BMI', 'Z_score', 'abnormal_flag'], inplace=True)

    # 仅保留最终需要的列
    final_cols = ['total_days', 'BMI', 'Z_score', 'abnormal_flag']
    df = df[final_cols]
    
    print(f"数据加载和清洗完成。总样本数: {len(df)}")
    
    return df

# --- P_false 计算 (LOWESS平滑) ---

def calculate_p_false_lowess(df):
    """
    使用LOWESS (局部加权回归) 平滑方法计算每个BMI簇在每个孕周的假阴性率 (P_false)。
    """
    print("\\n--- Calculating P_false using LOWESS smoothing ---")
    
    # 筛选出假阴性 (False Negative) 和真阴性 (True Negative) 的样本
    fn_cases = df[(df['abnormal_flag'] == 1) & (df['Z_score'] < 3)]
    tn_cases = df[(df['abnormal_flag'] == 0) & (df['Z_score'] < 3)]
    
    p_false_data = []

    for cluster in sorted(df['bmi_cluster'].unique()):
        cluster_df = df[df['bmi_cluster'] == cluster]
        
        # 计算每天的FN和TN数量
        fn_counts = fn_cases[fn_cases['bmi_cluster'] == cluster].groupby('total_days').size()
        tn_counts = tn_cases[tn_cases['bmi_cluster'] == cluster].groupby('total_days').size()
        
        # 合并并计算每天的经验P_false
        daily_stats = pd.DataFrame({'FN': fn_counts, 'TN': tn_counts}).fillna(0)
        daily_stats['empirical_p_false'] = daily_stats['FN'] / (daily_stats['FN'] + daily_stats['TN'])
        daily_stats = daily_stats.reindex(range(df['total_days'].min(), df['total_days'].max() + 1), fill_value=0).reset_index().rename(columns={'index': 'total_days'})

        # 准备LOWESS的数据
        x = daily_stats['total_days']
        y = daily_stats['empirical_p_false']
        
        # 应用LOWESS平滑
        # frac参数控制平滑窗口的大小，0.3表示使用30%的数据点
        smoothed = sm.nonparametric.lowess(y, x, frac=0.3, it=3, return_sorted=False)
        
        # 确保平滑后的概率值在[0, 1]范围内
        smoothed = np.clip(smoothed, 0, 1)

        # 存储结果
        for day, p_val in zip(x, smoothed):
            p_false_data.append({'bmi_cluster': cluster, 'total_days': day, 'p_false': p_val})

    p_false_table = pd.DataFrame(p_false_data)
    
    # 填充可能因LOWESS产生的NaN
    p_false_table['p_false'] = p_false_table.groupby('bmi_cluster')['p_false'].transform(
        lambda x: x.interpolate(method='linear').ffill().bfill()
    )
    p_false_table.fillna({'p_false': 1.0}, inplace=True) # 最后的保障

    print("P_false calculation complete.")
    print(p_false_table.head())
    return p_false_table

# --- BMI聚类 ---

def create_bmi_groups(df, n_clusters=4):
    """使用KMeans对BMI进行聚类。"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['bmi_cluster'] = kmeans.fit_predict(df[['BMI']])
    
    # 确保簇标签与BMI大小正相关
    cluster_means = df.groupby('bmi_cluster')['BMI'].mean().sort_values().index
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means)}
    df['bmi_cluster'] = df['bmi_cluster'].map(cluster_mapping)
    
    return df

# --- 风险函数与优化 ---

def calculate_risk(t, p_false_t, w1, w2, w3, t_min, t_max):
    """计算综合风险函数 R(t)。"""
    # 归一化时间
    t_normalized = (t - t_min) / (t_max - t_min) if t_max > t_min else 0
    
    r1 = p_false_t
    r2 = t_normalized
    r3 = 0  # R3(t) 暂不考虑
    
    return w1 * r1 + w2 * r2 + w3 * r3

def solve_optimization_problem(p_false_table, w1, w2, w3, theta=0.80):
    """求解优化问题，找到每个簇的最佳检测时间。"""
    results = []
    t_min = p_false_table['total_days'].min()
    t_max = p_false_table['total_days'].max()

    for cluster in sorted(p_false_table['bmi_cluster'].unique()):
        cluster_p_false = p_false_table[p_false_table['bmi_cluster'] == cluster]
        
        # 施加约束 P_false(t) <= 1 - theta
        feasible_times = cluster_p_false[cluster_p_false['p_false'] <= (1 - theta)]
        
        if feasible_times.empty:
            # 如果没有符合约束的时间点
            results.append({
                'BMI Cluster': cluster,
                'Optimal Time': 'N/A',
                'P_false at t_i': 'N/A',
                'Minimized Risk': 'N/A',
                'Note': f'No time satisfies P_false <= {1-theta:.2f}'
            })
            continue

        # 在可行的时间点中寻找最小化风险R(t)的时间
        min_risk = float('inf')
        optimal_day = -1

        for _, row in feasible_times.iterrows():
            t = row['total_days']
            p_false_t = row['p_false']
            risk = calculate_risk(t, p_false_t, w1, w2, w3, t_min, t_max)
            if risk < min_risk:
                min_risk = risk
                optimal_day = int(t)
        
        optimal_p_false = feasible_times.loc[feasible_times['total_days'] == optimal_day, 'p_false'].iloc[0]
        
        results.append({
            'BMI Cluster': cluster,
            'Optimal Time': f"{optimal_day // 7}W + {optimal_day % 7}D",
            'P_false at t_i': optimal_p_false,
            'Minimized Risk': min_risk,
            'Note': ''
        })
        
    return pd.DataFrame(results)

# --- 绘图函数 ---

def plot_risk_components(p_false_table, cluster_id, w1, w2, font_prop):
    """绘制单个簇的风险构成图。"""
    df = p_false_table[p_false_table['bmi_cluster'] == cluster_id]
    t_min, t_max = p_false_table['total_days'].min(), p_false_table['total_days'].max()
    
    days = df['total_days']
    weeks = days / 7
    r1 = w1 * df['p_false']
    r2 = w2 * (days - t_min) / (t_max - t_min)
    total_risk = r1 + r2
    
    plt.figure(figsize=(12, 7))
    plt.plot(weeks, r1, label=f'$R_1(t)$ (加权假阴性风险, $w_1={w1}$)', color='blue')
    plt.plot(weeks, r2, label=f'$R_2(t)$ (加权时间风险, $w_2={w2}$)', color='green')
    plt.plot(weeks, total_risk, label='总风险 $R(t)$', color='red', linewidth=2.5)
    
    # 标记最低点
    min_risk_idx = total_risk.idxmin()
    optimal_week = weeks[min_risk_idx]
    min_risk_value = total_risk[min_risk_idx]
    plt.scatter(optimal_week, min_risk_value, color='purple', s=100, zorder=5, label=f'最优点 ({optimal_week:.1f}周)')
    
    plt.title(f'BMI群组 {cluster_id} 的风险构成分析 ($w_1={w1}, w_2={w2}$)', fontproperties=font_prop, fontsize=16)
    plt.xlabel('孕周 (周)', fontproperties=font_prop, fontsize=12)
    plt.ylabel('风险值', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    output_dir = "problem2_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"risk_components_cluster_{cluster_id}_w1_{w1}_w2_{w2}.png")
    plt.savefig(filename)
    plt.close()
    print(f"图表已保存至: {filename}")

def plot_sensitivity_analysis(p_false_table, font_prop):
    """对w2进行敏感性分析并绘图。"""
    w2_values = np.linspace(0, 1, 21)
    optimal_days = {cluster: [] for cluster in sorted(p_false_table['bmi_cluster'].unique())}

    for w2 in w2_values:
        w1 = 1 - w2
        results_df = solve_optimization_problem(p_false_table, w1, w2, 0, theta=0.80)
        for _, row in results_df.iterrows():
            cluster = row['BMI Cluster']
            if row['Optimal Time'] != 'N/A':
                parts = row['Optimal Time'].replace('W', '').replace('D', '').split(' + ')
                day = int(parts[0]) * 7 + int(parts[1])
                optimal_days[cluster].append(day)
            else:
                optimal_days[cluster].append(np.nan) # 如果无解，则记录为NaN

    plt.figure(figsize=(12, 8))
    for cluster, days in optimal_days.items():
        plt.plot(w2_values, np.array(days) / 7, marker='o', linestyle='-', label=f'BMI群组 {cluster}')

    plt.title('最优检测孕周随权重$w_2$变化的敏感性分析', fontproperties=font_prop, fontsize=16)
    plt.xlabel('时间风险权重 $w_2$', fontproperties=font_prop, fontsize=12)
    plt.ylabel('最优检测孕周 (周)', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_dir = "problem2_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "sensitivity_analysis_optimal_day_vs_w2.png")
    plt.savefig(filename)
    plt.close()
    print(f"图表已保存至: {filename}")

# --- 主函数 ---

def main():
    """主执行流程。"""
    # 1. 清理缓存并设置字体
    clear_matplotlib_cache()
    font_prop = get_font_prop()

    # 2. 加载和处理数据
    df = load_and_clean_data()
    df = create_bmi_groups(df, n_clusters=4)
    
    # 打印各簇样本量
    print("\\n--- BMI Cluster Sample Sizes ---")
    print(df['bmi_cluster'].value_counts().sort_index().to_frame('n_samples'))

    # 3. 计算P_false
    p_false_table = calculate_p_false_lowess(df)
    
    # 4. 求解不同场景下的优化问题
    scenarios = {
        "仅考虑准确率": (1.0, 0.0, 0.0),
        "准确率与时效性均衡": (0.5, 0.5, 0.0)
    }
    
    theta = 0.80 # 约束条件 P_false <= 0.20

    for name, (w1, w2, w3) in scenarios.items():
        print(f"\\n--- 求解场景: '{name}' (权重 w1,w2,w3 = ({w1}, {w2}, {w3}), theta = {theta}) ---")
        results_df = solve_optimization_problem(p_false_table, w1, w2, w3, theta)
        
        # 补充样本量信息
        n_samples = df['bmi_cluster'].value_counts().sort_index().to_frame('n_samples')
        results_df = results_df.set_index('BMI Cluster').join(n_samples).reset_index()
        
        print("\\n--- 最优NIPT检测方案 ---")
        print(results_df[['BMI Cluster', 'n_samples', 'Optimal Time', 'P_false at t_i', 'Minimized Risk', 'Note']].to_string())
        
        # 为每个簇绘制风险构成图
        for cluster_id in sorted(df['bmi_cluster'].unique()):
            plot_risk_components(p_false_table, cluster_id, w1, w2, font_prop)

    # 5. 运行敏感性分析
    print("\\n--- 运行w2的敏感性分析 ---")
    plot_sensitivity_analysis(p_false_table, font_prop)

if __name__ == "__main__":
    main()