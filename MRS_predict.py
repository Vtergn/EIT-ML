"""
EIT 卒中预后分析脚本
针对3个月功能结局(mRS)的预测分析

功能：
1. 对卒中患者3个月改良Rankin评分(mRS)进行二分类预测
2. 采用解剖学先验知识进行特征初筛（背侧、左后侧、右后侧区域）
3. 结合Logistic Regression L1正则化进行二次特征选择
4. 使用留一交叉验证(LOOCV)评估模型性能
5. 生成完整的可视化分析报告

注意：此代码对应论文中"Exploratory observation of 3-month functional outcome"部分
样本量：n=13 (mRS 0-2 vs mRS ≥3)

作者：林敬珞
日期：2026年1月
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
from typing import List, Tuple, Optional, Dict

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, 
    roc_curve
)

# ==========================================
# 0. 基础配置
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 解剖学映射模块（与主分析一致）
# ==========================================

def get_pair_key(e1: int, e2: int) -> Tuple[int, int]:
    """标准化电极对，处理 (16, 1) 闭环情况"""
    if (e1 == 16 and e2 == 1) or (e1 == 1 and e2 == 16):
        return (16, 1)
    return tuple(sorted((e1, e2)))

# 定义 16 个相邻测量对的解剖归属
ZONE_MAP = {
    get_pair_key(15, 16): "腹侧", get_pair_key(16, 1): "腹侧", get_pair_key(1, 2): "腹侧",
    get_pair_key(2, 3): "左前侧", get_pair_key(3, 4): "左前侧",
    get_pair_key(4, 5): "左侧",
    get_pair_key(5, 6): "左后侧", get_pair_key(6, 7): "左后侧",
    get_pair_key(7, 8): "背侧", get_pair_key(8, 9): "背侧", get_pair_key(9, 10): "背侧",
    get_pair_key(10, 11): "右后侧", get_pair_key(11, 12): "右后侧",
    get_pair_key(12, 13): "右侧",
    get_pair_key(13, 14): "右前侧", get_pair_key(14, 15): "右前侧"
}

def decode_eit_channel(channel_index: int) -> dict:
    """将 0-191 通道索引解码为物理电极对和区域"""
    channel_num = channel_index + 1
    frame_idx = (channel_num - 1) // 12
    inj_1 = frame_idx + 1
    inj_2 = ((inj_1 + 8 - 1) % 16) + 1
    valid_pairs = []
    for i in range(1, 17):
        e_a = i
        e_b = (i % 16) + 1
        if not (e_a == inj_1 or e_b == inj_1 or e_a == inj_2 or e_b == inj_2):
            valid_pairs.append(get_pair_key(e_a, e_b))
    meas_idx_in_frame = (channel_num - 1) % 12
    meas_pair = valid_pairs[meas_idx_in_frame] if meas_idx_in_frame < len(valid_pairs) else (0, 0)
    return {"position": ZONE_MAP.get(meas_pair, "未知")}

def extract_channel_idx(name: str) -> Optional[int]:
    """从特征名提取 channel 数字"""
    match = re.search(r'channel[_\s]*(\d+)', name.lower())
    return int(match.group(1)) if match else None

# ==========================================
# 2. 数据加载与预处理
# ==========================================

def standardize_id(id_str):
    """标准化患者ID"""
    if pd.isna(id_str): 
        return None
    return os.path.basename(str(id_str).strip()).split('.')[0]

def load_prognosis_data():
    """
    加载预后分析数据
    返回：特征矩阵X，标签y，特征名列表
    注意：此函数需要实际的临床数据文件
    """
    # 脱敏路径 - 使用通用文件名
    clin_path = "stroke_prognosis_data.xlsx"
    feat_path = "Wavelet_Feature_Matrix.csv"

    
    if not os.path.exists(clin_path):
        raise FileNotFoundError(f"预后数据文件不存在: {clin_path}")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"特征矩阵文件不存在: {feat_path}")
    
    # 加载数据
    df_clin = pd.read_excel(clin_path)
    df_feat = pd.read_csv(feat_path, index_col=0)
    
    # 标准化ID
    df_clin['ID_Clean'] = df_clin['PatientID'].apply(standardize_id)
    df_feat.index = pd.Series(df_feat.index).apply(standardize_id)
    
    # 对齐样本
    common = sorted(list(set(df_clin['ID_Clean'].dropna()) & set(df_feat.index.dropna())))
    print(f"✅ 匹配成功样本数: {len(common)}")
    
    # 提取3个月mRS评分
    # 假设数据中有'3month_mRS'列
    df_clin_align = df_clin.set_index('ID_Clean').loc[common]
    df_feat_align = df_feat.loc[common]
    # 二分类：mRS 0-2 vs mRS ≥3
    y_raw = pd.to_numeric(df_clin_align['3个月'], errors='coerce').values
    valid_mask = ~np.isnan(y_raw)
    
    X = df_feat_align[valid_mask].values
    y = np.where(y_raw[valid_mask] >= 3, 1, 0)  # mRS ≥3 为阳性
    
    return X, y, df_feat_align.columns.tolist()

def filter_features_by_anatomy(feature_names: List[str], target_regions: List[str]) -> List[str]:
    """
    基于解剖学先验知识筛选特征
    根据论文发现，卒中预后相关信号集中在背侧区域
    """
    filtered = []
    for f_name in feature_names:
        idx = extract_channel_idx(f_name)
        if idx is None:
            continue
        try:
            pos = decode_eit_channel(idx)['position']
            if pos in target_regions:
                filtered.append(f_name)
        except:
            continue
    return filtered

# ==========================================
# 3. 双重筛选LOOCV分析
# ==========================================

def run_prognosis_analysis(X, y, feature_names):
    """
    执行预后分析流程：
    1. 解剖学筛选（第一阶段）
    2. L1正则化特征选择（第二阶段）
    3. LOOCV评估
    """
    # 第一阶段：解剖学筛选（基于论文发现的背侧敏感区域）
    target_regions = ['背侧', '左后侧', '右后侧']  # 根据论文发现
    selected_features = filter_features_by_anatomy(feature_names, target_regions)
    feature_indices = [i for i, n in enumerate(feature_names) if n in selected_features]
    X_selected = X[:, feature_indices]
    
    print(f"解剖学筛选完成: {len(feature_names)} → {len(selected_features)} 个特征")
    print(f"筛选区域: {', '.join(target_regions)}")
    
    # 第二阶段：LOOCV + L1特征选择
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []
    feature_selection_counts = {f: 0 for f in selected_features}
    
    # 构建Pipeline
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.5, 
                          class_weight='balanced', random_state=42),
        threshold="mean"
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('classifier', LogisticRegression(solver='liblinear', 
                                         class_weight='balanced', 
                                         random_state=42))
    ])
    
    # LOOCV
    n_samples = len(X_selected)
    print(f"开始LOOCV分析 (N={n_samples})...")
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(X_selected)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        
        # 统计特征选择
        support = pipeline.named_steps['selector'].get_support()
        for idx, selected in enumerate(support):
            if selected:
                feature_selection_counts[selected_features[idx]] += 1
        
        # 预测
        y_true.append(y_test[0])
        y_pred.append(pipeline.predict(X_test)[0])
        y_proba.append(pipeline.predict_proba(X_test)[0, 1])
        
        print(f"进度: {fold+1}/{n_samples}", end='\r')
    
    # 计算特征稳定性
    feature_stability = pd.DataFrame([
        {'Feature': f, 'Selection_Frequency': count/n_samples}
        for f, count in feature_selection_counts.items()
    ]).sort_values('Selection_Frequency', ascending=False)
    
    return (np.array(y_true), np.array(y_pred), np.array(y_proba), 
            feature_stability, selected_features)

# ==========================================
# 4. 可视化与报告
# ==========================================

def generate_prognosis_report(y_true, y_pred, y_proba, feature_stability, output_dir):
    """生成预后分析报告"""
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'F1_Score': f1_score(y_true, y_pred)
    }
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EIT-based Stroke Prognosis Analysis (3-month mRS)', fontsize=16, fontweight='bold')
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 0].plot(fpr, tpr, color='#2E86AB', lw=3, 
                    label=f'AUC = {metrics["AUC"]:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['mRS 0-2', 'mRS ≥3'],
                yticklabels=['mRS 0-2', 'mRS ≥3'],
                ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # 3. 特征稳定性
    top_features = feature_stability.head(10)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    axes[1, 0].barh(range(len(top_features)), top_features['Selection_Frequency'], color=colors)
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['Feature'].str[:30])  # 截断长特征名
    axes[1, 0].set_xlabel('Selection Frequency (LOOCV)')
    axes[1, 0].set_title('Top 10 Stable Features')
    axes[1, 0].invert_yaxis()
    
    # 4. 性能指标
    axes[1, 1].axis('off')
    metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存
    report_path = os.path.join(output_dir, 'prognosis_analysis_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果
    results_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    results_path = os.path.join(output_dir, 'prognosis_results.csv')
    results_df.to_csv(results_path, index=False)
    
    feature_stability_path = os.path.join(output_dir, 'feature_stability.csv')
    feature_stability.to_csv(feature_stability_path, index=False)
    
    return metrics

# ==========================================
# 5. 主函数
# ==========================================

def main():
    """主分析流程"""
    
    print("=" * 60)
    print("EIT Stroke Prognosis Analysis (3-month functional outcome)")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = 'EIT_Prognosis_Analysis_Results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        print("\n📊 加载数据...")
        X, y, feature_names = load_prognosis_data()
        
        print(f"样本量: {len(y)}")
        print(f"类别分布: {np.bincount(y)} (0: mRS 0-2, 1: mRS ≥3)")
        
        # 执行分析
        print("\n🔬 执行双重筛选预后分析...")
        y_true, y_pred, y_proba, feature_stability, selected_features = run_prognosis_analysis(
            X, y, feature_names
        )
        
        # 生成报告
        print("\n📈 生成分析报告...")
        metrics = generate_prognosis_report(
            y_true, y_pred, y_proba, feature_stability, output_dir
        )
        
        # 打印结果摘要
        print("\n" + "=" * 60)
        print("✨ 分析完成！")
        print("=" * 60)
        print(f"\n📊 性能指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\n📁 结果保存至: {output_dir}/")
        print("  - prognosis_analysis_report.png (可视化报告)")
        print("  - prognosis_results.csv (性能指标)")
        print("  - feature_stability.csv (特征稳定性)")
        
        print(f"\n💡 核心发现:")
        print(f"  • 基于背侧区域特征可预测3个月功能结局")
        print(f"  • AUC = {metrics['AUC']:.3f} (n={len(y)})")
        print(f"  • 最稳定的特征: {feature_stability.iloc[0]['Feature'][:40]}...")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()