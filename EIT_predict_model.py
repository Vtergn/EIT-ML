"""
EIT 最终训练脚本 (诊断修复版 v4)
功能改进: 
1. 在每个实验循环内部增加 Try-Catch，防止单个实验报错导致程序中断。
2. 如果实验失败，结果表中会显示具体的错误信息。
3. 保持之前的 SHAP 修复和暴力清洗逻辑。
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import sys
import traceback
from typing import List, Tuple, Optional, Dict
from collections import Counter

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

# 尝试导入 SMOTE
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("⚠️ 警告: 未安装 imbalanced-learn，将跳过 SMOTE。")
    SMOTE = None

# 尝试导入 SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装 shap，将跳过 SHAP 分析。")
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# --- 1. 配置区域 ---
# ==========================================

CLINICAL_PATH = r"clinical_data.xlsx"
FEATURE_PATH = r"Wavelet_Feature_Matrix.csv"

OUTPUT_DIR = 'EIT_Final_Training_Result_MultiExp_Fixed_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
K_FEATURES = 100 
SHAP_TOP_N = 50 

FOCUSED_TARGETS = ['左变异度', '右变异度(%)','脑梗死面积','发病天数','氧合指数','肾小球滤过率（EPI-cys)']

CUSTOM_LOCATION_FILTER_BY_TARGET = {
    '左变异度': ['背侧', '右前侧', '左后侧'],
    '右变异度(%)': ['背侧', '右前侧', '左后侧'],
    '脑梗死面积': ['背侧', '左后侧', '右后侧'],
    '发病天数': ['背侧', '左后侧', '右后侧'],
    '氧合指数': ['背侧', '左后侧', '右后侧'],
    '肾小球滤过率（EPI-cys)': ['背侧', '左后侧', '右后侧']
}

PARAM_GRID_XGBC = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1],
    'subsample': [0.8]
}

# ==========================================
# --- 2. 辅助函数 ---
# ==========================================

def standardize_id(id_str):
    if pd.isna(id_str): return None
    key = str(id_str).strip()
    key = os.path.basename(key) 
    if key.lower().endswith(('.xlsx', '.csv')): key = key.rsplit('.', 1)[0]
    return key.strip()

def get_pair_key(e1: int, e2: int) -> Tuple[int, int]:
    if (e1 == 16 and e2 == 1) or (e1 == 1 and e2 == 16): return (16, 1)
    return tuple(sorted((e1, e2)))

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
    match = re.search(r'channel[_\s]*(\d+)', name.lower())
    return int(match.group(1)) if match else None

def filter_features_by_custom_locations(feature_names: List[str], target_positions: List[str], mode: str = 'include') -> List[str]:
    if not target_positions: return feature_names
    filtered = []
    for f_name in feature_names:
        idx = extract_channel_idx(f_name)
        if idx is None:
            filtered.append(f_name) 
            continue
        try:
            pos = decode_eit_channel(idx)['position']
            is_in_target = pos in target_positions
            if mode == 'include':
                if is_in_target: filtered.append(f_name)
            elif mode == 'exclude':
                if not is_in_target: filtered.append(f_name)
        except: 
            filtered.append(f_name)
    return filtered

def get_cutoff(series):
    name = series.name
    if name == '发病天数': return lambda x: 1 if x >= 14 else 0
    if name == '脑梗死面积': return lambda x: 1 if x >= 4.5 else 0
    if name in ['左变异度', '右变异度(%)']: return lambda x: 1 if x < 20 else 0
    if '滤过率' in name: return lambda x: 1 if x < 90 else 0
    if '氧合' in name: return lambda x: 1 if x < 300 else 0
    return lambda x: 1 if x >= series.median() else 0

def analyze_shap_distribution(model, X_test: pd.DataFrame, top_n: int = 50) -> str:
    feature_importance = None
    method_used = "None"

    # 1. 尝试使用 SHAP (强制 Numpy 模式)
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            # 关键修复：只传 values，切断 pandas metadata 的干扰
            X_numpy = X_test.values.astype(np.float64)
            
            # check_additivity=False 可以防止因精度问题导致的报错
            shap_values = explainer.shap_values(X_numpy, check_additivity=False)
            
            if isinstance(shap_values, list):
                vals = shap_values[1]
            else:
                vals = shap_values

            feature_importance = np.abs(vals).mean(axis=0)
            method_used = "SHAP"
        except Exception as e:
            print(f"   ⚠️ SHAP Numpy 模式依然报错: {e}")
            print("   ⚠️ 正在切换到 XGBoost 原生 Feature Importance (Gain)...")
    
    # 2. 如果 SHAP 失败，降级使用 XGBoost 自带的重要性
    if feature_importance is None:
        try:
            feature_importance = model.feature_importances_
            method_used = "XGB_Gain"
        except Exception as e:
             print(f"   ❌ 连 XGBoost 原生重要性都获取失败: {e}")
             return "Analysis Failed"

    # 3. 统计区域分布
    try:
        feat_imp_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        top_features = feat_imp_df.head(top_n)['Feature'].tolist()
        
        zone_counter = Counter()
        for f_name in top_features:
            idx = extract_channel_idx(f_name)
            if idx is not None:
                info = decode_eit_channel(idx)
                zone_counter[info['position']] += 1
            else:
                zone_counter['非Channel特征'] += 1
                
        sorted_zones = zone_counter.most_common()
        result_str = " > ".join([f"{zone}({count})" for zone, count in sorted_zones])
        
        return f"[{method_used}] {result_str}" if result_str else f"[{method_used}] 无显著特征"
        
    except Exception as e:
        print(f"   ❌ 结果汇总阶段出错: {e}")
        return "Summary Error"

# ==========================================
# --- 3. 核心训练流程 ---
# ==========================================

def aggressive_clean(df):
    # 1. 清洗列名
    df.columns = df.columns.str.replace(r'[\[\]\'\"]', '', regex=True)
    # 2. 清洗数据内容
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        for col in obj_cols:
            df[col] = df[col].astype(str).str.replace(r'[\[\]\'\"]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 3. 再次确保所有列都是 float (双重保险)
    df = df.astype(float)
    print("   🧹 数据清洗完成，所有列已转换为数值。")
    return df

def load_and_align():
    if not os.path.exists(CLINICAL_PATH): raise FileNotFoundError(f"未找到临床数据: {CLINICAL_PATH}")
    if not os.path.exists(FEATURE_PATH): raise FileNotFoundError(f"未找到特征矩阵: {FEATURE_PATH}")

    print(f"📊 加载临床数据: {os.path.basename(CLINICAL_PATH)}")
    df_c = pd.read_excel(CLINICAL_PATH) if CLINICAL_PATH.endswith('.xlsx') else pd.read_csv(CLINICAL_PATH)
    print(f"📊 加载特征矩阵: {os.path.basename(FEATURE_PATH)}")
    df_f = pd.read_csv(FEATURE_PATH, index_col=0)

    # 暴力清洗
    df_f = aggressive_clean(df_f)

    # 对齐
    id_col = next((c for c in df_c.columns if 'ID' in c.upper()), df_c.columns[0])
    df_c['PID_Clean'] = df_c[id_col].apply(standardize_id)
    df_f.index = pd.Series(df_f.index).apply(standardize_id)
    
    common = sorted(list(set(df_c['PID_Clean'].dropna()) & set(df_f.index.dropna())))
    
    df_c_align = df_c.set_index('PID_Clean').loc[common]
    df_f_align = df_f.loc[common]
    return df_c_align, df_f_align


def main():
    try:
        df_clinical, df_features = load_and_align()
        results = []
        
        print("\n🚀 开始多模式训练流程 (诊断版)...")
        
        for target in FOCUSED_TARGETS:
            if target not in df_clinical.columns:
                print(f"⚠️ 警告: 临床数据中缺少目标列 '{target}'")
                continue
                
            print(f"\n{'='*30}\n🎯 Target: {target}\n{'='*30}")
            
            y_raw = df_clinical[target].dropna()
            if len(y_raw) < 20: continue
                
            cutoff_fn = get_cutoff(y_raw)
            y_cls = y_raw.apply(cutoff_fn)
            if len(y_cls.unique()) < 2: continue
            
            X_base = df_features.loc[y_raw.index].copy()
            target_roi_list = CUSTOM_LOCATION_FILTER_BY_TARGET.get(target, [])
            
            experiments = [
                ('1_All_Locations', None, 'include'),
                ('2_ROI_Only', target_roi_list, 'include'),
                ('3_ROI_Removed', target_roi_list, 'exclude')
            ]
            
            for exp_name, roi_list, mode in experiments:
                print(f"\n--- [实验: {exp_name}] ---")
                
                # --- 将单个实验包裹在 try-catch 中 ---
                try:
                    # 特征筛选
                    if exp_name == '1_All_Locations':
                        X_loc = X_base
                    else:
                        cols = filter_features_by_custom_locations(X_base.columns.tolist(), roi_list, mode=mode)
                        X_loc = X_base[cols]
                    
                    if X_loc.shape[1] == 0: 
                        print("⚠️ 特征数量为0，跳过")
                        continue

                    # 数据划分
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_loc, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
                    )
                    
                    # 预处理
                    imputer = SimpleImputer(strategy='median')
                    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
                    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
                    
                    # ========== 修复SMOTE顺序（可选） ==========
                    # 先标准化真实数据
                    scaler = StandardScaler()
                    X_train_scaled_raw = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
                    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
                    
                    # 再SMOTE（在标准化后的数据上）
                    if SMOTE:
                        k = min(5, y_train.value_counts().min() - 1) if y_train.value_counts().min() > 1 else 1
                        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
                        X_train_scaled, y_train_res = smote.fit_resample(X_train_scaled_raw, y_train)
                    else:
                        X_train_scaled, y_train_res = X_train_scaled_raw, y_train
                    # ===========================================
                    
                    # 特征选择
                    k_best = min(K_FEATURES, X_train_scaled.shape[1])
                    selector = SelectKBest(score_func=f_classif, k=k_best)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train_res)  # 改为X_train_selected
                    X_test_selected = selector.transform(X_test_scaled)                     # 改为X_test_selected

                    # 获取选中的原始特征名（用于SHAP分析）
                    selected_feature_names = X_train_scaled.columns[selector.get_support()].tolist()
                    
                    # 训练
                    grid = GridSearchCV(
                        XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=1.0, eval_metric='logloss'),
                        param_grid=PARAM_GRID_XGBC, scoring='accuracy', cv=3, n_jobs=-1
                    )
                    grid.fit(X_train_selected, y_train_res)
                    best_model = grid.best_estimator_
                    
                    # 预测
                    y_pred = best_model.predict(X_test_selected)
                    y_proba = best_model.predict_proba(X_test_selected)[:, 1] if hasattr(best_model, "predict_proba") else [0.5]*len(y_test)
                    
                    # ========== 修复SHAP分析 ==========
                    print("   🔍 正在计算特征空间分布...")
                    
                    # 创建带原始特征名的DataFrame用于SHAP
                    if len(selected_feature_names) > 0:
                        X_test_for_shap = pd.DataFrame(
                            X_test_selected,
                            columns=selected_feature_names,
                            index=X_test.index
                        )
                        shap_zone_str = analyze_shap_distribution(best_model, X_test_for_shap, top_n=SHAP_TOP_N)
                    else:
                        shap_zone_str = "无选中特征"
                    # ===================================
                    
                    # 获取最重要的特征
                    top1_idx = np.argmax(selector.scores_) if selector.scores_.size > 0 else 0
                    top1_feature = X_train_scaled.columns[top1_idx] if top1_idx < len(X_train_scaled.columns) else "Unknown"
                    
                    res_dict = {
                        'Target': target,
                        'Experiment': exp_name,
                        'Acc': accuracy_score(y_test, y_pred),
                        'AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else 0.5,
                        'F1': f1_score(y_test, y_pred, average='macro'),
                        'Top_SHAP_Zones': shap_zone_str,
                        'Feature_Count': X_loc.shape[1],
                        'Top1_Feature': top1_feature
                    }
                    results.append(res_dict)
                    print(f"   📊 结果: Acc={res_dict['Acc']:.4f}, AUC={res_dict['AUC']:.4f}, Top Zones: {shap_zone_str}")

                except Exception as e:
                    print(f"   ❌ 实验失败: {e}")
                    traceback.print_exc()
                    results.append({
                        'Target': target,
                        'Experiment': exp_name,
                        'Acc': 0, 'AUC': 0, 'F1': 0,
                        'Top_SHAP_Zones': f"FAILED: {str(e)[:50]}...",
                        'Feature_Count': X_loc.shape[1] if 'X_loc' in locals() else 0,
                        'Top1_Feature': 'Error'
                    })
        if results:
            df_res = pd.DataFrame(results)
            # 重新排列列顺序
            cols = ['Target', 'Experiment', 'Acc', 'AUC', 'F1', 'Top_SHAP_Zones', 'Feature_Count', 'Top1_Feature']
            df_res = df_res[cols]
            
            # 保存到文件
            save_path = os.path.join(OUTPUT_DIR, 'EIT_Final_Results.xlsx')
            df_res.to_excel(save_path, index=False)
            
            # 同时保存为CSV（可读性更好）
            csv_path = os.path.join(OUTPUT_DIR, 'EIT_Final_Results.csv')
            df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"\n{'='*60}")
            print(f"✨ 实验完成！结果已保存至:")
            print(f"   Excel: {save_path}")
            print(f"   CSV:   {csv_path}")
            print(f"{'='*60}")
            
            # 打印汇总表格
            print("\n📊 结果汇总:")
            print(df_res.to_markdown(index=False))
        else:
            print("⚠️ 没有生成任何结果")
    except Exception as e:
        print(f"❌ 程序主流程崩溃: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()