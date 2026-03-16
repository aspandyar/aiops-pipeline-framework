"""
Streamlit dashboard for the CI/CD Build Failure Prediction Framework.
Dissertation: "AI-Powered Intelligent Framework for CI/CD Pipeline Optimization and Visualization"
Multi-tab: Dataset Overview, Model Performance, SHAP Explainability, Live Prediction.
All heavy computation is done in notebooks and saved; dashboard only loads and displays.
Run with: streamlit run dashboard/app.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


st.set_page_config(
    page_title="CI/CD Build Failure Prediction — XAI Dashboard",
    page_icon="🔧",
    layout="wide",
)


@st.cache_data
def load_splits():
    """Load test set and metadata from processed data."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = np.load(data_dir / "y_test.npy")
    return X_test, y_test


@st.cache_data
def load_encoders():
    """Load feature list and class names."""
    encoders = joblib.load(PROJECT_ROOT / "models" / "encoders.joblib")
    return encoders.get("feature_list", []), encoders.get("class_names", [])


@st.cache_resource
def load_model():
    """Load trained XGBoost model."""
    return joblib.load(PROJECT_ROOT / "models" / "build_failure_model.pkl")


@st.cache_resource
def load_metrics():
    """Load saved test metrics."""
    return joblib.load(PROJECT_ROOT / "models" / "metrics.joblib")


@st.cache_resource
def load_shap_data():
    """Load precomputed SHAP values and sample (no recomputation)."""
    return joblib.load(PROJECT_ROOT / "models" / "shap_data.joblib")


@st.cache_resource
def get_shap_explainer():
    """Recreate TreeExplainer from model + sample (explainer is not saved)."""
    model = load_model()
    shap_data = load_shap_data()
    return shap.TreeExplainer(model, shap_data["X_sample"])


def main():
    st.title("🔧 CI/CD Build Failure Prediction — XAI Dashboard")
    st.markdown(
        "Real-time, XAI-driven, human-in-the-loop visualization. "
        "TravisTorrent dataset · XGBoost · SHAP explanations."
    )

    # Sidebar: model metrics, dataset, classes, reference
    st.sidebar.header("Model & dataset")
    try:
        metrics = load_metrics()
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_weighted", 0)
        class_names = metrics.get("class_names", [])
        st.sidebar.metric("Accuracy", f"{acc:.2%}")
        st.sidebar.metric("F1 (weighted)", f"{f1:.4f}")
        st.sidebar.success("F1 ≥ 0.85 (dissertation target)" if f1 >= 0.85 else "F1 < 0.85")
    except Exception:
        acc = f1 = 0
        class_names = []
        st.sidebar.warning("Run notebooks 2–3 to generate model and metrics.")
    st.sidebar.markdown("**Dataset:** TravisTorrent (Travis CI builds)")
    st.sidebar.markdown("**Classes:** " + ", ".join(class_names) if class_names else "—")
    st.sidebar.markdown("---")
    st.sidebar.caption("Reference: Al-Barhami et al. (2026) — 95.9% accuracy benchmark")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Overview",
        "Model Performance",
        "SHAP Explainability",
        "Live Prediction",
    ])

    # ——— Tab 1: Dataset Overview ———
    with tab1:
        st.header("Dataset Overview")
        try:
            X_test, y_test = load_splits()
            feature_list, _ = load_encoders()
            st.write(f"Test set: **{len(X_test):,}** rows × **{len(X_test.columns)}** features")
            # Class distribution
            unique, counts = np.unique(y_test, return_counts=True)
            dist = pd.DataFrame({"Class index": unique, "Count": counts})
            if class_names:
                dist["Class"] = [class_names[i] for i in unique]
            st.bar_chart(dist.set_index("Class" if class_names else "Class index")["Count"])
            st.subheader("Feature summary")
            st.dataframe(X_test.describe(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load data: {e}")

    # ——— Tab 2: Model Performance ———
    with tab2:
        st.header("Model Performance")
        try:
            metrics = load_metrics()
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_weighted", 0)
            st.metric("Accuracy", f"{acc:.2%}")
            st.metric("F1 (weighted)", f"{f1:.4f}")
            st.markdown("**Evaluation target:** F1 > 0.85 (dissertation)")
            if f1 >= 0.85:
                st.success("Target met.")
            else:
                st.info("Below target; consider more data or hyperparameter tuning.")
            X_test, y_test = load_splits()
            model = load_model()
            feature_list, class_names = load_encoders()
            y_pred = model.predict(X_test)
            from sklearn.metrics import classification_report, confusion_matrix
            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred, target_names=class_names))
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Could not load model or metrics: {e}")

    # ——— Tab 3: SHAP Explainability ———
    with tab3:
        st.header("SHAP Explainability")
        try:
            shap_data = load_shap_data()
            X_sample = shap_data["X_sample"]
            shap_values = shap_data["shap_values"]
            feature_list = shap_data["feature_list"]
            class_names = list(shap_data["class_names"])
            # Class selector: which class to explain
            chosen_class = st.selectbox("Select class to explain", class_names, index=min(1, len(class_names) - 1))
            class_idx = class_names.index(chosen_class)
            sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
            # Global summary bar for chosen class
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, feature_names=feature_list, class_names=class_names, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()
            # Slider: pick build index for waterfall
            build_idx = st.slider("Build index (for waterfall)", 0, len(X_sample) - 1, 0)
            explainer = get_shap_explainer()
            ev = shap_data.get("expected_value", explainer.expected_value)
            base_val = float(ev[class_idx]) if isinstance(ev, (list, np.ndarray)) else float(ev)
            vals = sv[build_idx]
            ex = shap.Explanation(values=vals, base_values=base_val, data=X_sample.iloc[build_idx], feature_names=feature_list)
            fig2, ax2 = plt.subplots()
            shap.waterfall_plot(ex, show=False)
            st.pyplot(fig2)
            plt.close()
        except Exception as e:
            st.error(f"Could not load SHAP data. Run Notebook 4 first: {e}")

    # ——— Tab 4: Live Prediction ———
    with tab4:
        st.header("Live Prediction")
        st.markdown("Enter build parameters and get a real-time prediction with a SHAP waterfall explanation.")
        try:
            feature_list, class_names = load_encoders()
            model = load_model()
            shap_data = load_shap_data()
            explainer = get_shap_explainer()
            X_sample = shap_data["X_sample"]
            shap_values = shap_data["shap_values"]
            # Build a single row from user inputs (one input per feature)
            inputs = {}
            for i, f in enumerate(feature_list):
                if f == "gh_lang_enc":
                    # Use sample mode or dropdown of known labels if we had them
                    inputs[f] = st.number_input(f, value=int(X_sample[f].mode().iloc[0]) if len(X_sample) else 0, min_value=0, key=f"inp_{i}")
                else:
                    def_val = float(X_sample[f].median()) if f in X_sample.columns else 0
                    inputs[f] = st.number_input(f, value=def_val, key=f"inp_{i}")
            row = pd.DataFrame([inputs])[feature_list]
            if st.button("Predict"):
                pred_proba = model.predict_proba(row)[0]
                pred_class = int(np.argmax(pred_proba))
                st.subheader("Prediction")
                for k, name in enumerate(class_names):
                    st.write(f"**{name}:** {pred_proba[k]:.2%}")
                st.write(f"**Predicted class:** {class_names[pred_class]}")
                # SHAP waterfall for this single prediction
                single_shap = explainer.shap_values(row)
                if isinstance(single_shap, list):
                    vals = single_shap[pred_class][0]
                else:
                    vals = single_shap[0]
                ev = shap_data.get("expected_value", explainer.expected_value)
                base_val = float(ev[pred_class]) if isinstance(ev, (list, np.ndarray)) else float(ev)
                ex = shap.Explanation(values=vals, base_values=base_val, data=row.iloc[0], feature_names=feature_list)
                st.subheader("Why this prediction? (SHAP waterfall)")
                fig, ax = plt.subplots()
                shap.waterfall_plot(ex, show=False)
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.error(f"Live prediction error: {e}")


if __name__ == "__main__":
    main()
