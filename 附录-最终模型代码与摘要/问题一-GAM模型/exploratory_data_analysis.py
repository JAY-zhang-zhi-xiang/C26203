import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# --- Font Settings ---
try:
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
except FileNotFoundError:
    print("SimSun font not found, using default font.")
    font = FontProperties(size=12)
plt.rcParams['font.sans-serif'] = [font.get_name()]
plt.rcParams['axes.unicode_minus'] = False

def convert_ga_to_weeks(ga_str):
    if isinstance(ga_str, str):
        ga_str = ga_str.strip().upper()
        match = re.match(r'(\d+)\s*W(?:\s*\+\s*(\d+))?', ga_str)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks + days / 7.0
    return np.nan

def get_cleaned_data():
    """Loads and cleans the data, returning a clean DataFrame."""
    file_path = 'data.xlsx'
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet1_name = excel_file.sheet_names[0]
        data = pd.read_excel(excel_file, sheet_name=sheet1_name)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_path}' 未找到。")
        return None

    male_fetus_data = data.copy()
    male_fetus_data['GA'] = male_fetus_data['检测孕周'].apply(convert_ga_to_weeks)
    cleaned_data = male_fetus_data.dropna(subset=['GA', '孕妇BMI'])
    cleaned_data = cleaned_data[(cleaned_data['GA'] >= 10) & (cleaned_data['GA'] <= 26)]
    cleaned_data['Y_concentration'] = pd.to_numeric(cleaned_data['Y染色体浓度'], errors='coerce')
    cleaned_data = cleaned_data.dropna(subset=['Y_concentration'])
    cleaned_data = cleaned_data[(cleaned_data['Y_concentration'] >= 0) & (cleaned_data['Y_concentration'] <= 0.15)]
    cleaned_data.rename(columns={'孕妇BMI': 'BMI'}, inplace=True)
    
    print(f"数据清洗完成，用于EDA的数据共: {len(cleaned_data)} 行")
    return cleaned_data

def run_exploratory_analysis(df):
    """Runs exploratory data analysis and generates plots."""
    if df is None:
        print("无法进行分析，因为数据为空。")
        return

    # --- 1. Plotting Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.distplot(df['Y_concentration'], ax=axes[0])
    axes[0].set_title('Y染色体浓度分布', fontproperties=font)
    axes[0].set_xlabel('Y染色体浓度', fontproperties=font)
    axes[0].set_ylabel('密度', fontproperties=font)
    sns.distplot(df['GA'], ax=axes[1])
    axes[1].set_title('检测孕周分布', fontproperties=font)
    axes[1].set_xlabel('检测孕周 (周)', fontproperties=font)
    sns.distplot(df['BMI'], ax=axes[2])
    axes[2].set_title('孕妇BMI分布', fontproperties=font)
    axes[2].set_xlabel('BMI', fontproperties=font)
    plt.tight_layout()
    dist_path = 'eda_distributions.png'
    plt.savefig(dist_path)
    print(f"已保存分布图: {dist_path}")
    plt.close(fig)

    # --- 2. BMI Grouping and Boxplot ---
    # Using Asian WHO standards
    bins = [0, 18.5, 23, 27.5, np.inf]
    labels = ['偏瘦', '正常', '超重', '肥胖']
    df['BMI_Group'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='BMI_Group', y='Y_concentration', data=df, order=labels)
    plt.title('不同BMI分组下的Y染色体浓度分布', fontproperties=font)
    plt.xlabel('BMI分组', fontproperties=font)
    plt.ylabel('Y染色体浓度', fontproperties=font)
    plt.xticks(fontproperties=font)
    boxplot_path = 'eda_bmi_boxplot.png'
    plt.savefig(boxplot_path)
    print(f"已保存BMI分组箱形图: {boxplot_path}")
    plt.close()

    # --- 3. Scatter plot of GA vs Y_concentration, colored by BMI Group ---
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='GA', y='Y_concentration', hue='BMI_Group', data=df, alpha=0.7, hue_order=labels)
    plt.title('孕周与Y浓度的关系 (按BMI分组)', fontproperties=font)
    plt.xlabel('检测孕周 (周)', fontproperties=font)
    plt.ylabel('Y染色体浓度', fontproperties=font)
    legend = plt.legend(title='BMI分组', prop=font)
    plt.setp(legend.get_title(), fontproperties=font)
    scatter_path = 'eda_ga_scatter_by_bmi.png'
    plt.savefig(scatter_path)
    print(f"已保存按BMI分组的散点图: {scatter_path}")
    plt.close()

if __name__ == '__main__':
    cleaned_df = get_cleaned_data()
    run_exploratory_analysis(cleaned_df)