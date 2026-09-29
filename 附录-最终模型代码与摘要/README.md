# 附录：最终模型代码与摘要

本附录仅保留多模型产前检测分析系统相关的原创业务代码、模型摘要和说明文件。提交包不包含通用 Web 框架源码、第三方开源库源码、运行缓存、训练数据、压缩包或模型二进制产物。

## 代码提交范围说明

按照初审意见，本次代码材料以原创业务逻辑为主，重点提交与设计说明书功能模块一致的代码：数据导入与字段校验、孕周转换、Y染色体浓度GAM非线性分析、BMI聚类分层、检测时机风险计算、纵向风险解释、染色体异常分类评价、模型指标汇总和结果摘要生成。

`提交材料/源代码与摘要/app.py` 已整理为“原创核心代码提交版”，不再包含 Streamlit 页面样式、登录表单、侧边栏导航等通用框架展示代码。各问题目录中的脚本用于佐证模型实现过程；其中调用 pandas、numpy、scikit-learn、pygam、imbalanced-learn 等第三方库的部分仅为接口调用，不包含第三方库源码。

不建议作为主要源代码提交的内容包括：前端页面装饰代码、通用框架模板、第三方依赖包源码、自动生成缓存、训练数据、模型二进制文件、旧项目示例、压缩包和与本系统功能无关的过程文件。

## 目录说明

- 问题一-GAM模型：Y染色体浓度非线性分析、交叉验证和外部验证脚本。
- 问题二-K-Means聚类优化：BMI分层和检测时机风险优化脚本。
- 问题三-纵向生存联合模型：纵向风险建模摘要。
- 问题四-逻辑回归模型：数据预处理、SMOTE均衡、异常分类训练与评价脚本。
- 代码介绍：逐个脚本的功能说明，便于审核材料和源代码对应。

## 文件清单

### 根目录
- `../app.py`（原创核心代码提交版）
- `README.md`
- `model_evaluation_comprehensive.md`
- `佐证材料文件列表.md`

### 代码介绍
- `代码介绍/problem1_exploratory_data_analysis介绍.txt`
- `代码介绍/problem1_gam_model_cv介绍.txt`
- `代码介绍/problem1_gam_model介绍.txt`
- `代码介绍/problem1_run_model_comparison_on_newtest介绍.txt`
- `代码介绍/problem2_bmi_kmeans_analysis介绍.txt`
- `代码介绍/problem2_risk_optimization介绍.txt`
- `代码介绍/problem3介绍.txt`
- `代码介绍/problem4_apply_smote_q4介绍.txt`
- `代码介绍/problem4_data_preprocessing_q4介绍.txt`
- `代码介绍/problem4_evaluate_model_lr_smote_q4介绍.txt`
- `代码介绍/problem4_evaluate_model_rf_q4介绍.txt`
- `代码介绍/problem4_evaluate_model_rf_smote_q4介绍.txt`
- `代码介绍/problem4_exploratory_data_analysis_q4介绍.txt`
- `代码介绍/problem4_train_model_lr_smote_q4介绍.txt`
- `代码介绍/problem4_train_model_rf_q4介绍.txt`
- `代码介绍/problem4_train_model_rf_smote_q4介绍.txt`

### 问题一-GAM模型
- `问题一-GAM模型/exploratory_data_analysis.py`
- `问题一-GAM模型/problem1_final_report_v4.md`
- `问题一-GAM模型/problem1_gam_model.py`
- `问题一-GAM模型/problem1_gam_model_cv.py`
- `问题一-GAM模型/problem1_gam_model_cv_summary.txt`
- `问题一-GAM模型/run_model_comparison_on_newtest.py`

### 问题三-纵向生存联合模型
- `问题三-纵向生存联合模型/问题三模型摘要.md`

### 问题二-K-Means聚类优化
- `问题二-K-Means聚类优化/problem2_bmi_kmeans_analysis.py`
- `问题二-K-Means聚类优化/problem2_final_report.md`
- `问题二-K-Means聚类优化/problem2_final_report_structured.md`
- `问题二-K-Means聚类优化/problem2_risk_optimization.py`

### 问题四-逻辑回归模型
- `问题四-逻辑回归模型/apply_smote_q4.py`
- `问题四-逻辑回归模型/data_preprocessing_q4.py`
- `问题四-逻辑回归模型/evaluate_model_lr_smote_q4.py`
- `问题四-逻辑回归模型/exploratory_data_analysis_q4.py`
- `问题四-逻辑回归模型/train_model_lr_smote_q4.py`

## 一致性说明

上述文件与设计说明书中的数据导入与清洗、GAM分析、BMI分层、纵向风险解释、异常分类、模型评价和结果展示模块一一对应。运行时若需要数据文件或模型输出文件，应由用户在本地按说明生成或上传，不作为本次软著源代码材料提交。代码材料仅展示本系统原创实现思路和业务处理流程，第三方依赖只保留必要的调用语句。
## 补正辅助材料
- `研发过程补充整理.md`：根据现有代码、说明书和当前版本记录整理的研发过程补充说明。
- `代码审查记录-当前版本.md`：当前版本的代码结构和材料范围检查记录，不是历史 PR 记录。
- `内部测试记录-当前版本.md`：当前版本的静态检查和结构检查记录，不是第三方测试报告。
- `CURRENT_VERSION_BASELINE.md`：当前版本基线及证据边界说明。
- `SOURCE_HASHES_SHA256.txt`：当前版本文件 SHA-256 清单。

