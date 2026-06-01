# -*- coding: utf-8 -*-
"""
Thyroid Cancer Lymph Node Metastasis Risk Calculator

运行：
    D:\myproject\venv\Scripts\python.exe -m streamlit run "D:\机器学习\app.py"
"""

# ============================================================
# 0. 基础导入
# ============================================================

import os
import textwrap
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import shap
except Exception:
    shap = None


# ============================================================
# 1. 主要配置区
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR

LAYER1_NAME = "Layer1_N0_vs_Nplus"
LAYER2_NAME = "Layer2_N1a_vs_N1b"

LAYER1_BEST_MODEL_NAME = "XGBoost_tuned"
LAYER2_BEST_MODEL_NAME = "CatBoost_native_tuned"

RISK_CUTOFFS_LAYER1 = (0.20, 0.50)
RISK_CUTOFFS_LAYER2 = (0.30, 0.60)

CLINICAL_LNM_THRESHOLD = 0.50
CLINICAL_N1B_THRESHOLD = 0.50

TOP_N_SHAP = 8


# ============================================================
# 2. 特征定义：必须和训练代码一致
# ============================================================

FEATURES_COMMON = [
    "age",
    "tumor_size_mm",
    "tumor_size_status",
    "sex",
    "race",
    "marital_status",
    "t_stage",
    "histology",
]

NUMERIC_FEATURES = [
    "age",
    "tumor_size_mm",
]

CATEGORICAL_FEATURES = [
    "tumor_size_status",
    "sex",
    "race",
    "marital_status",
    "t_stage",
    "histology",
]

CATBOOST_CAT_FEATURES = CATEGORICAL_FEATURES.copy()


# ============================================================
# 3. 页面基础设置 + CSS
# ============================================================

st.set_page_config(
    page_title="Thyroid LNM Risk Calculator",
    page_icon="🧬",
    layout="wide",
)


plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
})


def clean_panel(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#333333", width=0.8, length=3)
    ax.grid(True, linestyle="-", color="#EAEAEA", linewidth=0.6)
    ax.set_axisbelow(True)


# ============================================================
# 4. 数据准备函数
# ============================================================

def prepare_X(df):
    return df[FEATURES_COMMON].copy()


def prepare_catboost_X(df):
    X = df[FEATURES_COMMON].copy()

    for c in CATBOOST_CAT_FEATURES:
        X[c] = X[c].fillna("Unknown").astype(str)

    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def logit_transform(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


def apply_platt_calibrator(calibrator, y_prob):
    y_prob = np.asarray(y_prob, dtype=float)

    if calibrator is None:
        return y_prob

    return calibrator.predict_proba(logit_transform(y_prob))[:, 1]


# ============================================================
# 5. 模型加载
# ============================================================

def infer_model_type_from_name(model_name):
    name = str(model_name).lower()

    if "catboost_native" in name:
        return "catboost_native"

    return "pipeline"


def load_model_info_from_files(layer_name, model_name):
    layer_dir = Path(OUTPUT_DIR) / layer_name

    model_path = layer_dir / f"{model_name}.joblib"
    calibrator_path = layer_dir / f"{model_name}_Platt_calibrator.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}\n"
            f"请检查 OUTPUT_DIR、layer_name 或 model_name 是否正确。"
        )

    model = joblib.load(model_path)

    calibrator = None
    calibration_method = "none"

    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        calibration_method = "Platt"

    return {
        "name": model_name,
        "type": infer_model_type_from_name(model_name),
        "model": model,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "threshold_f1": 0.5,
    }


def load_best_model_info(layer_name, fallback_model_name):
    layer_dir = Path(OUTPUT_DIR) / layer_name
    best_info_path = layer_dir / f"{layer_name}_BEST_MODEL_INFO.joblib"

    if best_info_path.exists():
        model_info = joblib.load(best_info_path)

        if "name" not in model_info:
            model_info["name"] = fallback_model_name

        if "type" not in model_info:
            model_info["type"] = infer_model_type_from_name(model_info["name"])

        if "calibrator" not in model_info:
            model_info["calibrator"] = None

        if "calibration_method" not in model_info:
            model_info["calibration_method"] = (
                "Platt" if model_info.get("calibrator") is not None else "none"
            )

        return model_info

    return load_model_info_from_files(layer_name, fallback_model_name)


@st.cache_resource
def load_models():
    layer1_info = load_best_model_info(
        layer_name=LAYER1_NAME,
        fallback_model_name=LAYER1_BEST_MODEL_NAME,
    )

    layer2_info = load_best_model_info(
        layer_name=LAYER2_NAME,
        fallback_model_name=LAYER2_BEST_MODEL_NAME,
    )

    return layer1_info, layer2_info


# ============================================================
# 6. 预测函数
# ============================================================

def predict_model_proba(model_info, df):
    model_type = model_info["type"]
    model = model_info["model"]

    if model_type == "pipeline":
        X = prepare_X(df)
        prob = model.predict_proba(X)[:, 1]

    elif model_type == "catboost_native":
        from catboost import Pool

        X = prepare_catboost_X(df)
        pool = Pool(X, cat_features=CATBOOST_CAT_FEATURES)
        prob = model.predict_proba(pool)[:, 1]

    else:
        raise ValueError(f"未知模型类型：{model_type}")

    calibrator = model_info.get("calibrator", None)

    if calibrator is not None:
        prob = apply_platt_calibrator(calibrator, prob)

    return prob


def hierarchical_predict(input_df, layer1_info, layer2_info):
    p_nplus = float(predict_model_proba(layer1_info, input_df)[0])
    p_n1b_given_nplus = float(predict_model_proba(layer2_info, input_df)[0])

    p_n0 = 1 - p_nplus
    p_n1a = p_nplus * (1 - p_n1b_given_nplus)
    p_n1b = p_nplus * p_n1b_given_nplus

    probs = {
        "N0": p_n0,
        "N1a": p_n1a,
        "N1b": p_n1b,
    }

    highest_single_class = max(probs, key=probs.get)

    return {
        "P(N0)": p_n0,
        "P(N1a)": p_n1a,
        "P(N1b)": p_n1b,
        "P(N+)": p_nplus,
        "P(N1b | N+)": p_n1b_given_nplus,
        "Highest single class": highest_single_class,
    }


def classify_three_level_risk(prob, cutoffs):
    low_cut, high_cut = cutoffs

    if prob < low_cut:
        return "Low"
    elif prob < high_cut:
        return "Intermediate"
    else:
        return "High"


def risk_color_label(risk):
    if risk == "Low":
        return "🟢 Low"
    if risk == "Intermediate":
        return "🟡 Intermediate"
    return "🔴 High"


def render_main_clinical_conclusion(
    pred,
    layer1_threshold=CLINICAL_LNM_THRESHOLD,
    layer2_threshold=CLINICAL_N1B_THRESHOLD,
):
    """
    使用 Streamlit 原生组件渲染醒目的主结论栏目。
    不使用 HTML，避免 <div> 被当成代码显示。
    """

    p_nplus = pred["P(N+)"]
    p_n1b_given_nplus = pred["P(N1b | N+)"]

    # --------------------------------------------------------
    # 1. 是否发生淋巴结转移
    # --------------------------------------------------------
    if p_nplus >= layer1_threshold:
        lnm_title = "🔴 可能发生淋巴结转移"
        lnm_desc = "模型判断：该患者整体上更倾向于存在淋巴结转移风险。"
        lnm_box_type = "error"
    else:
        lnm_title = "🟢 淋巴结转移可能性较低"
        lnm_desc = "模型判断：该患者整体上更倾向于无淋巴结转移。"
        lnm_box_type = "success"

    # --------------------------------------------------------
    # 2. 中央区 vs 侧区
    # --------------------------------------------------------
    if p_n1b_given_nplus >= layer2_threshold:
        region_title = "🟠 若发生转移，更偏向侧区转移 N1b"
        region_desc = "在已经发生淋巴结转移的前提下，模型认为侧区转移风险更高。"
        region_box_type = "warning"
    else:
        region_title = "🔵 若发生转移，更偏向中央区转移 N1a"
        region_desc = "在已经发生淋巴结转移的前提下，模型认为中央区转移可能性更高。"
        region_box_type = "info"

    # --------------------------------------------------------
    # 3. 页面渲染
    # --------------------------------------------------------
    st.markdown("## 主要临床判断 / Main Clinical Conclusion")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Step 1：是否发生淋巴结转移？")

            if lnm_box_type == "error":
                st.error(
                    f"""
                    ### {lnm_title}

                    **P(N+) = {p_nplus:.1%}**

                    {lnm_desc}
                    """
                )
            else:
                st.success(
                    f"""
                    ### {lnm_title}

                    **P(N+) = {p_nplus:.1%}**

                    {lnm_desc}
                    """
                )

        with col2:
            st.markdown("### Step 2：若发生转移，更偏向哪里？")

            if region_box_type == "warning":
                st.warning(
                    f"""
                    ### {region_title}

                    **P(N1b | N+) = {p_n1b_given_nplus:.1%}**

                    {region_desc}
                    """
                )
            else:
                st.info(
                    f"""
                    ### {region_title}

                    **P(N1b | N+) = {p_n1b_given_nplus:.1%}**

                    {region_desc}
                    """
                )

        if p_nplus < layer1_threshold:
            st.caption(
                "说明：模型首先判断该患者淋巴结转移可能性较低，因此中央区/侧区判断应作为附加参考，而不是主要结论。"
            )
        else:
            st.caption(
                "说明：模型首先判断该患者存在较高淋巴结转移风险，随后基于 Layer 2 模型进一步判断转移区域更偏向中央区或侧区。"
            )


# ============================================================
# 7. SHAP 相关函数
# ============================================================

def get_value_based_colors(values, cmap_name="Blues", min_light=0.28, max_dark=0.88):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return []

    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        scaled = np.full(values.shape, 0.55, dtype=float)
    else:
        vmin = np.nanmin(values[finite_mask])
        vmax = np.nanmax(values[finite_mask])

        if np.isclose(vmin, vmax):
            scaled = np.full(values.shape, 0.60, dtype=float)
        else:
            scaled = (values - vmin) / (vmax - vmin)
            scaled = np.where(np.isfinite(scaled), scaled, 0.0)

    scaled = min_light + scaled * (max_dark - min_light)
    cmap = plt.get_cmap(cmap_name)

    return [cmap(float(x)) for x in scaled]


def canonicalize_shap_feature_name(feature):
    feat = str(feature).strip()

    while "__" in feat:
        feat = feat.split("__", 1)[-1]

    categorical_bases = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)

    for base in categorical_bases:
        if feat == base or feat.startswith(base + "_"):
            return base

    for base in NUMERIC_FEATURES:
        if feat == base:
            return base

    return feat


def extract_binary_shap_values(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])[0]
        return np.asarray(shap_values[0])[0]

    arr = np.asarray(shap_values)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        return arr[0]

    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[0, :, 1]

        if arr.shape[1] == 2:
            return arr[0, 1, :]

        return arr[0, :, 0]

    raise ValueError(f"无法识别 SHAP values 形状：{arr.shape}")


def aggregate_local_shap(shap_df, input_df):
    df = shap_df.copy()

    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
    df = df.dropna(subset=["shap_value"])

    df["original_feature"] = df["feature"].apply(canonicalize_shap_feature_name)

    agg = (
        df.groupby("original_feature", as_index=False)
        .agg(
            signed_shap=("shap_value", "sum"),
            importance=("shap_value", lambda x: float(np.sum(np.abs(x)))),
        )
        .sort_values("importance", ascending=False)
    )

    input_values = input_df.iloc[0].to_dict()

    agg["input_value"] = agg["original_feature"].map(
        lambda x: str(input_values.get(x, ""))
    )

    return agg


def calculate_local_shap(model_info, input_df):
    if shap is None:
        raise ImportError("未安装 shap，请先运行：pip install shap")

    model_type = model_info["type"]
    model = model_info["model"]

    if model_type == "catboost_native":
        from catboost import Pool

        X = prepare_catboost_X(input_df)
        pool = Pool(X, cat_features=CATBOOST_CAT_FEATURES)

        shap_values = model.get_feature_importance(
            pool,
            type="ShapValues",
        )

        values = np.asarray(shap_values)[0, :-1]
        feature_names = list(X.columns)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": values,
        })

        return aggregate_local_shap(shap_df, input_df)

    if model_type == "pipeline":
        X = prepare_X(input_df)

        if not hasattr(model, "named_steps"):
            raise ValueError("pipeline 类型模型应当是 sklearn Pipeline，但当前模型没有 named_steps。")

        if "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
            raise ValueError("Pipeline 中需要包含 named_steps['preprocessor'] 和 named_steps['clf']。")

        preprocessor = model.named_steps["preprocessor"]
        clf = model.named_steps["clf"]

        X_trans = preprocessor.transform(X)

        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        else:
            X_trans = np.asarray(X_trans)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)

        values = extract_binary_shap_values(shap_values)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": values,
        })

        return aggregate_local_shap(shap_df, input_df)

    raise ValueError(f"未知模型类型：{model_type}")


def add_direction_label(shap_df, positive_label, negative_label):
    df = shap_df.copy()

    df["direction"] = np.where(
        df["signed_shap"] > 0,
        positive_label,
        negative_label,
    )

    return df


def plot_local_shap_bar(shap_df, title, cmap_name="Blues", top_n=8):
    plot_df = shap_df.copy().head(top_n)

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.text(0.5, 0.5, "No SHAP values available", ha="center", va="center")
        ax.axis("off")
        return fig

    plot_df = plot_df.sort_values("importance", ascending=True)

    colors = get_value_based_colors(
        plot_df["importance"].values,
        cmap_name=cmap_name,
    )

    fig, ax = plt.subplots(figsize=(7.0, max(4.8, 0.48 * len(plot_df) + 1.8)))

    ax.barh(
        plot_df["original_feature"],
        plot_df["signed_shap"],
        color=colors,
    )

    ax.axvline(0, linestyle="--", linewidth=1.0, color="#777777")
    ax.set_xlabel("Local SHAP value")
    ax.set_title(title)

    clean_panel(ax)

    fig.tight_layout()

    return fig


# ============================================================
# 8. 概率图
# ============================================================

def plot_probability_bar(pred):
    prob_df = pd.DataFrame({
        "Class": ["N0", "N1a", "N1b"],
        "Probability": [pred["P(N0)"], pred["P(N1a)"], pred["P(N1b)"]],
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    values = prob_df["Probability"].values
    colors = get_value_based_colors(values, cmap_name="Greens")

    ax.bar(
        prob_df["Class"],
        values,
        color=colors,
    )

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, min(1.0, max(values) + 0.18))
    ax.set_ylabel("Predicted probability")
    ax.set_title("Predicted probabilities of N stage")

    clean_panel(ax)

    fig.tight_layout()

    return fig


# ============================================================
# 9. 输入界面
# ============================================================

def build_input_dataframe():
    st.sidebar.header("Input clinical features")

    age = st.sidebar.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=45,
        step=1,
    )

    sex = st.sidebar.selectbox(
        "Sex",
        ["Female", "Male", "Unknown"],
        index=0,
    )

    race = st.sidebar.selectbox(
        "Race",
        [
            "White",
            "Black",
            "Asian or Pacific Islander",
            "American Indian/Alaska Native",
            "Unknown",
        ],
        index=0,
    )

    marital_status = st.sidebar.selectbox(
        "Marital status",
        [
            "Married",
            "Single",
            "Divorced",
            "Widowed",
            "Separated",
            "Unmarried or Domestic Partner",
            "Not_available",
            "Unknown",
        ],
        index=0,
    )

    tumor_size_mm = st.sidebar.number_input(
        "Tumor size, mm",
        min_value=0.0,
        max_value=300.0,
        value=10.0,
        step=1.0,
    )

    tumor_size_status = "Observed"

    t_stage = st.sidebar.selectbox(
        "T stage",
        [
            "T1",
            "T1a",
            "T1b",
            "T2",
            "T3",
            "T3a",
            "T3b",
            "T4",
            "T4a",
            "T4b",
            "TX",
        ],
        index=1,
    )

    histology = st.sidebar.selectbox(
        "Histology",
        [
            "PTC",
            "FTC",
            "MTC",
            "ATC",
            "Other",
            "Unknown",
        ],
        index=0,
    )

    input_df = pd.DataFrame([{
        "age": age,
        "tumor_size_mm": tumor_size_mm,
        "tumor_size_status": tumor_size_status,
        "sex": sex,
        "race": race,
        "marital_status": marital_status,
        "t_stage": t_stage,
        "histology": histology,
    }])

    return input_df


# ============================================================
# 10. 主页面
# ============================================================

def main():
    st.title("🧬 Thyroid Cancer Lymph Node Metastasis Risk Calculator")

    st.markdown(
        """
        This web tool estimates the probabilities of **N0**, **N1a**, and **N1b**
        using a hierarchical machine learning model.
        """
    )

    st.caption(
        "Layer 1 estimates P(N+). Layer 2 estimates P(N1b | N+). "
        "The final N-stage probabilities are calculated hierarchically."
    )

    try:
        layer1_info, layer2_info = load_models()
    except Exception as e:
        st.error("模型加载失败。请检查 OUTPUT_DIR 和模型文件路径。")
        st.exception(e)
        return

    with st.expander("Loaded model information", expanded=False):
        st.write({
            "Layer 1 model": layer1_info.get("name", ""),
            "Layer 1 type": layer1_info.get("type", ""),
            "Layer 1 calibration": layer1_info.get("calibration_method", "none"),
            "Layer 2 model": layer2_info.get("name", ""),
            "Layer 2 type": layer2_info.get("type", ""),
            "Layer 2 calibration": layer2_info.get("calibration_method", "none"),
        })

    input_df = build_input_dataframe()

    st.subheader("Input summary")
    st.dataframe(input_df, width="stretch")

    calculate = st.sidebar.button("Calculate risk", type="primary")

    if not calculate:
        st.info("请在左侧输入患者特征，然后点击 **Calculate risk**。")
        return

    try:
        pred = hierarchical_predict(input_df, layer1_info, layer2_info)
    except Exception as e:
        st.error("预测失败。请检查输入特征格式和模型文件是否匹配。")
        st.exception(e)
        return

    p_nplus = pred["P(N+)"]
    p_n1b_given_nplus = pred["P(N1b | N+)"]

    layer1_risk = classify_three_level_risk(
        p_nplus,
        RISK_CUTOFFS_LAYER1,
    )

    layer2_risk = classify_three_level_risk(
        p_n1b_given_nplus,
        RISK_CUTOFFS_LAYER2,
    )

    render_main_clinical_conclusion(
        pred,
        layer1_threshold=CLINICAL_LNM_THRESHOLD,
        layer2_threshold=CLINICAL_N1B_THRESHOLD,
    )

    st.subheader("Predicted risk details")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("P(N0)", f"{pred['P(N0)']:.1%}")
    col2.metric("P(N1a)", f"{pred['P(N1a)']:.1%}")
    col3.metric("P(N1b)", f"{pred['P(N1b)']:.1%}")
    col4.metric("最高单类概率", pred["Highest single class"])

    col5, col6 = st.columns(2)

    col5.metric(
        "Layer 1: P(N+)",
        f"{p_nplus:.1%}",
        risk_color_label(layer1_risk),
    )

    col6.metric(
        "Layer 2: P(N1b | N+)",
        f"{p_n1b_given_nplus:.1%}",
        risk_color_label(layer2_risk),
    )

    fig_prob = plot_probability_bar(pred)
    st.pyplot(fig_prob)
    plt.close(fig_prob)

    with st.expander("Probability details", expanded=False):
        prob_table = pd.DataFrame({
            "Item": [
                "P(N0)",
                "P(N1a)",
                "P(N1b)",
                "P(N+)",
                "P(N1b | N+)",
                "Layer 1 risk group",
                "Layer 2 risk group",
                "Highest single class",
            ],
            "Value": [
                f"{pred['P(N0)']:.4f}",
                f"{pred['P(N1a)']:.4f}",
                f"{pred['P(N1b)']:.4f}",
                f"{pred['P(N+)']:.4f}",
                f"{pred['P(N1b | N+)']:.4f}",
                layer1_risk,
                layer2_risk,
                pred["Highest single class"],
            ],
        })

        st.dataframe(prob_table, width="stretch")

    st.subheader("Main contributing factors")

    if shap is None:
        st.warning("当前环境未安装 shap，因此无法显示 local SHAP。请运行：pip install shap")
        return

    tab1, tab2 = st.tabs([
        "Layer 1 explanation: N0 vs N+",
        "Layer 2 explanation: N1a vs N1b",
    ])

    with tab1:
        st.markdown(
            """
            **Interpretation:** positive SHAP values push the prediction toward **N+**,
            while negative SHAP values push the prediction toward **N0**.
            """
        )

        try:
            shap_l1 = calculate_local_shap(layer1_info, input_df)

            shap_l1_display = add_direction_label(
                shap_l1,
                positive_label="Increase N+ risk",
                negative_label="Decrease N+ risk",
            )

            shap_l1_display = shap_l1_display[[
                "original_feature",
                "input_value",
                "signed_shap",
                "importance",
                "direction",
            ]].copy()

            shap_l1_display["signed_shap"] = shap_l1_display["signed_shap"].map(lambda x: f"{x:.4f}")
            shap_l1_display["importance"] = shap_l1_display["importance"].map(lambda x: f"{x:.4f}")

            st.dataframe(
                shap_l1_display,
                width="stretch",
            )

            fig_l1 = plot_local_shap_bar(
                shap_l1,
                title="Local SHAP explanation: N0 vs N+",
                cmap_name="Blues",
                top_n=TOP_N_SHAP,
            )

            st.pyplot(fig_l1)
            plt.close(fig_l1)

        except Exception as e:
            st.error("Layer 1 SHAP 计算失败。")
            st.exception(e)

    with tab2:
        st.markdown(
            """
            **Interpretation:** positive SHAP values push the prediction toward **N1b**,
            while negative SHAP values push the prediction toward **N1a**.
            """
        )

        try:
            shap_l2 = calculate_local_shap(layer2_info, input_df)

            shap_l2_display = add_direction_label(
                shap_l2,
                positive_label="Increase N1b risk",
                negative_label="Decrease N1b risk",
            )

            shap_l2_display = shap_l2_display[[
                "original_feature",
                "input_value",
                "signed_shap",
                "importance",
                "direction",
            ]].copy()

            shap_l2_display["signed_shap"] = shap_l2_display["signed_shap"].map(lambda x: f"{x:.4f}")
            shap_l2_display["importance"] = shap_l2_display["importance"].map(lambda x: f"{x:.4f}")

            st.dataframe(
                shap_l2_display,
                width="stretch",
            )

            fig_l2 = plot_local_shap_bar(
                shap_l2,
                title="Local SHAP explanation: N1a vs N1b",
                cmap_name="Oranges",
                top_n=TOP_N_SHAP,
            )

            st.pyplot(fig_l2)
            plt.close(fig_l2)

        except Exception as e:
            st.error("Layer 2 SHAP 计算失败。")
            st.exception(e)

    st.divider()

    st.warning(
        """
        This calculator is intended for research and model demonstration only.
        It should not replace ultrasound assessment, pathology, clinical guidelines,
        or physician judgment.
        """
    )


if __name__ == "__main__":
    main()
