"""多模型产前检测分析系统原创核心代码提交版。

本文件用于软件著作权代码材料展示，仅保留与说明书功能模块一致的
原创业务逻辑：数据导入校验、孕周转换、GAM 非线性分析、BMI 分层、
检测时机风险计算、纵向风险解释、异常分类评价和报告摘要生成。

通用 Web 框架页面、第三方库源码、样式代码、运行缓存、训练数据、
模型二进制文件均不放入本代码提交版。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("检测孕周", "孕妇BMI", "年龄")
Y_COLUMN = "Y染色体浓度"
ABNORMAL_COLUMN = "染色体异常"
Z_VALUE_COLUMNS = ("13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值")


@dataclass
class DataAudit:
    sample_count: int
    field_count: int
    missing_required_columns: list[str]
    converted_week_count: int
    warnings: list[str]


@dataclass
class BmiGroup:
    group_id: int
    sample_count: int
    mean_bmi: float
    mean_age: float
    recommended_week: float
    risk_score: float


def read_detection_table(path: str) -> pd.DataFrame:
    """读取 Excel 或 CSV 检测数据，并统一清理字段名。"""
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix in {"xlsx", "xls"}:
        frame = pd.read_excel(path)
    elif suffix == "csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("仅支持 xlsx、xls 或 csv 格式的检测数据文件")

    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def parse_gestational_week(value: Any) -> float | None:
    """将 12w+3、12周3天、12.5 等孕周写法转换为周数。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    week_day = re.match(r"^(\d+(?:\.\d+)?)\s*w(?:\s*\+\s*(\d+))?$", text)
    if week_day:
        weeks = float(week_day.group(1))
        days = float(week_day.group(2) or 0)
        return round(weeks + days / 7, 3)

    chinese_week = re.match(r"^(\d+(?:\.\d+)?)\s*周(?:\s*(\d+)\s*天)?$", text)
    if chinese_week:
        weeks = float(chinese_week.group(1))
        days = float(chinese_week.group(2) or 0)
        return round(weeks + days / 7, 3)

    try:
        numeric = float(text)
    except ValueError:
        return None

    return round(numeric / 7, 3) if numeric > 60 else round(numeric, 3)


def normalize_percent(series: pd.Series) -> pd.Series:
    """兼容百分号字符串和小数形式的浓度、比例字段。"""
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.dropna().gt(1).mean() > 0.5:
        numeric = numeric / 100
    return numeric


def prepare_detection_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataAudit]:
    """完成字段校验、孕周转换、浓度归一化和基础质量提示。"""
    prepared = frame.copy()
    prepared.columns = prepared.columns.astype(str).str.strip()

    warnings: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in prepared.columns]

    if "GA_week" not in prepared.columns and "检测孕周" in prepared.columns:
        prepared["GA_week"] = prepared["检测孕周"].map(parse_gestational_week)

    if Y_COLUMN in prepared.columns:
        prepared[Y_COLUMN] = normalize_percent(prepared[Y_COLUMN])

    for column in ("孕妇BMI", "年龄", *Z_VALUE_COLUMNS):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    converted_count = int(prepared["GA_week"].notna().sum()) if "GA_week" in prepared.columns else 0
    if "GA_week" in prepared.columns and prepared["GA_week"].isna().mean() > 0.2:
        warnings.append("超过20%的孕周无法转换，请核对检测孕周格式")
    if Y_COLUMN in prepared.columns and prepared[Y_COLUMN].isna().mean() > 0.2:
        warnings.append("超过20%的Y染色体浓度无法识别，请核对百分比或小数格式")

    audit = DataAudit(
        sample_count=len(prepared),
        field_count=prepared.shape[1],
        missing_required_columns=missing,
        converted_week_count=converted_count,
        warnings=warnings,
    )
    return prepared, audit


def fit_quadratic_gam_surrogate(frame: pd.DataFrame) -> dict[str, Any]:
    """用二次平滑项表达 GAM 分析核心，输出可解释的非线性贡献。

    实际运行环境可替换为 pygam.LinearGAM；本提交版保留原创特征构造、
    变量选择、指标计算和解释结果组织，不包含第三方库源码。
    """
    required = ("GA_week", "孕妇BMI", "年龄", Y_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"GAM分析缺少字段: {', '.join(missing)}")

    data = frame[list(required)].dropna()
    data = data[data[Y_COLUMN] > 0]
    if len(data) < 20:
        raise ValueError("GAM分析有效样本不足20条")

    x = data[["GA_week", "孕妇BMI", "年龄"]].to_numpy(dtype=float)
    y = np.log(data[Y_COLUMN].to_numpy(dtype=float))
    x_centered = x - x.mean(axis=0)
    design = np.column_stack([np.ones(len(x)), x_centered, x_centered**2])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_residual = float(np.sum(residual**2))
    explained_r2 = 1 - ss_residual / ss_total if ss_total else 0.0

    return {
        "model": "GAM非线性分析核心逻辑",
        "sample_count": int(len(data)),
        "features": ["孕周", "BMI", "年龄"],
        "coefficients": coefficients.round(6).tolist(),
        "explained_r2": round(float(explained_r2), 4),
        "residual_std": round(float(residual.std(ddof=1)), 4),
    }


def kmeans_bmi_groups(frame: pd.DataFrame, n_groups: int = 3, max_iter: int = 100) -> list[BmiGroup]:
    """基于 BMI 和年龄进行原创 K-Means 分层，并计算检测时机建议。"""
    if not {"孕妇BMI", "年龄"}.issubset(frame.columns):
        raise ValueError("BMI分层缺少孕妇BMI或年龄字段")

    data = frame[["孕妇BMI", "年龄"]].dropna().to_numpy(dtype=float)
    if len(data) < n_groups:
        raise ValueError("有效样本数小于分层数量")

    scaled = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    order = np.argsort(scaled[:, 0])
    centers = scaled[np.linspace(0, len(scaled) - 1, n_groups).astype(int)]
    labels = np.zeros(len(scaled), dtype=int)

    for _ in range(max_iter):
        distance = np.linalg.norm(scaled[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distance.argmin(axis=1)
        new_centers = centers.copy()
        for group_id in range(n_groups):
            members = scaled[new_labels == group_id]
            if len(members):
                new_centers[group_id] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels = new_labels
        centers = new_centers

    grouped_data = pd.DataFrame(data[order], columns=["孕妇BMI", "年龄"])
    grouped_data["group_id"] = labels[order]
    groups: list[BmiGroup] = []
    min_bmi = grouped_data["孕妇BMI"].min()
    for group_id, group in grouped_data.groupby("group_id"):
        mean_bmi = float(group["孕妇BMI"].mean())
        mean_age = float(group["年龄"].mean())
        risk_score = compute_detection_risk(mean_bmi=mean_bmi, mean_age=mean_age)
        recommended_week = optimize_detection_week(mean_bmi=mean_bmi, mean_age=mean_age)
        recommended_week = max(recommended_week, 12 + (mean_bmi - min_bmi) * 0.15)
        groups.append(
            BmiGroup(
                group_id=int(group_id),
                sample_count=int(len(group)),
                mean_bmi=round(mean_bmi, 2),
                mean_age=round(mean_age, 2),
                recommended_week=round(float(recommended_week), 1),
                risk_score=round(float(risk_score), 4),
            )
        )
    return sorted(groups, key=lambda item: item.mean_bmi)


def compute_detection_risk(mean_bmi: float, mean_age: float, week: float = 12.0) -> float:
    """计算过早检测失败和过晚发现异常的综合风险。"""
    early_failure = max(0.0, 13.0 - week) * (0.08 + max(mean_bmi - 28.0, 0) * 0.01)
    late_discovery = max(0.0, week - 13.0) * (0.05 + max(mean_age - 30.0, 0) * 0.006)
    baseline = 0.02 + max(mean_bmi - 25.0, 0) * 0.004
    return baseline + early_failure + late_discovery


def optimize_detection_week(mean_bmi: float, mean_age: float) -> float:
    """在候选孕周窗口内搜索综合风险最小的推荐检测时点。"""
    candidate_weeks = np.arange(11.0, 18.5, 0.5)
    risks = [compute_detection_risk(mean_bmi, mean_age, week) for week in candidate_weeks]
    return float(candidate_weeks[int(np.argmin(risks))])


def longitudinal_risk_explanation(frame: pd.DataFrame) -> dict[str, Any]:
    """生成纵向风险模块的核心解释结果。"""
    available = [column for column in ("GA_week", Y_COLUMN, "孕妇BMI", "年龄") if column in frame.columns]
    data = frame[available].dropna() if available else pd.DataFrame()
    if data.empty:
        return {"model": "纵向风险解释", "status": "缺少可用纵向字段"}

    trend = {}
    if {"GA_week", Y_COLUMN}.issubset(data.columns):
        sorted_data = data.sort_values("GA_week")
        trend["y_concentration_by_week"] = (
            sorted_data.groupby(pd.cut(sorted_data["GA_week"], bins=5, duplicates="drop"))[Y_COLUMN]
            .mean()
            .round(5)
            .astype(float)
            .to_dict()
        )

    return {
        "model": "纵向生存联合模型解释逻辑",
        "sample_count": int(len(data)),
        "available_fields": available,
        "trend_summary": trend,
        "explanation": "基于多次检测指标变化解释个体风险随孕周变化的趋势",
    }


def evaluate_abnormal_classification(frame: pd.DataFrame) -> dict[str, Any]:
    """执行染色体异常分类评价的核心指标计算。"""
    features = [column for column in ("年龄", "孕妇BMI", "GA_week", *Z_VALUE_COLUMNS) if column in frame.columns]
    if len(features) < 3:
        raise ValueError("异常分类可用特征少于3个")

    data = frame[features].copy()
    if ABNORMAL_COLUMN in frame.columns:
        label = pd.to_numeric(frame[ABNORMAL_COLUMN], errors="coerce").fillna(0).astype(int)
    elif "21号染色体的Z值" in frame.columns:
        label = (pd.to_numeric(frame["21号染色体的Z值"], errors="coerce").abs() >= 2.5).astype(int)
    else:
        raise ValueError("缺少异常分类标签或可推导标签的21号染色体Z值")

    data[ABNORMAL_COLUMN] = label
    data = data.dropna()
    if data[ABNORMAL_COLUMN].nunique() < 2:
        raise ValueError("异常分类标签只有一个类别，无法计算分类指标")

    x = data[features].to_numpy(dtype=float)
    y = data[ABNORMAL_COLUMN].to_numpy(dtype=int)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])

    for _ in range(300):
        probability = 1 / (1 + np.exp(-(design @ weights)))
        gradient = design.T @ (probability - y) / len(y)
        weights -= 0.08 * gradient

    scores = 1 / (1 + np.exp(-(design @ weights)))
    prediction = (scores >= 0.5).astype(int)
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / len(y)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "model": "染色体异常分类核心逻辑",
        "features": features,
        "sample_count": int(len(y)),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def generate_report_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """汇总各模块输出，形成说明书对应的结果摘要。"""
    prepared, audit = prepare_detection_data(frame)
    summary: dict[str, Any] = {
        "software": "多模型产前检测分析系统",
        "data_audit": audit,
        "modules": {},
    }

    if not audit.missing_required_columns:
        summary["modules"]["BMI分层"] = kmeans_bmi_groups(prepared)
    if {Y_COLUMN, "GA_week", "孕妇BMI", "年龄"}.issubset(prepared.columns):
        summary["modules"]["GAM分析"] = fit_quadratic_gam_surrogate(prepared)
    summary["modules"]["纵向风险"] = longitudinal_risk_explanation(prepared)
    if any(column in prepared.columns for column in (ABNORMAL_COLUMN, "21号染色体的Z值")):
        summary["modules"]["异常分类"] = evaluate_abnormal_classification(prepared)

    return summary


if __name__ == "__main__":
    print("本文件为软著源代码材料展示版，请在实际运行系统中按使用说明书加载数据。")
