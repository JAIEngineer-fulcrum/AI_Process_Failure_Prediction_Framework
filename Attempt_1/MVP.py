import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =============================================================================
# Utility: Synthetic data generator to simulate workflows (e.g., insurance claims)
# =============================================================================

def generate_synthetic_workflows(n_rows: int = 500) -> pd.DataFrame:
    """
    Generate synthetic workflow data to simulate the kinds of signals
    described in the FRD (documentation, data quality, compliance, resources, patterns).

    Columns:
        - channel: source system/channel
        - amount: claim/transaction amount
        - docs_missing: count of missing required documents
        - invalid_fields: number of invalid fields
        - compliance_flags: how many compliance/policy checks failed
        - resource_load: resource utilization level (0-1)
        - past_failures: historical failures for this customer/process
        - processing_time_minutes: time taken so far
        - priority: business priority (1–3)
        - will_fail: label (0/1) – whether the workflow ultimately failed
    """

    rng = np.random.default_rng(seed=42)
    n = n_rows

    channel = rng.integers(0, 4, size=n)  # 0:web, 1:agent, 2:api, 3:batch
    amount = rng.normal(loc=5000, scale=3000, size=n).clip(100, 50000)
    docs_missing = rng.poisson(lam=0.5, size=n)
    invalid_fields = rng.poisson(lam=0.8, size=n)
    compliance_flags = rng.poisson(lam=0.4, size=n)
    resource_load = rng.uniform(0.2, 0.95, size=n)
    past_failures = rng.poisson(lam=0.3, size=n)
    processing_time_minutes = rng.normal(loc=45, scale=20, size=n).clip(5, 240)
    priority = rng.integers(1, 4, size=n)

    # "True" failure risk (not visible to model directly)
    # Higher when docs_missing, invalid_fields, compliance_flags,
    # resource_load, past_failures are higher.
    risk_raw = (
        0.25 * docs_missing
        + 0.25 * invalid_fields
        + 0.25 * compliance_flags
        + 1.5 * np.maximum(resource_load - 0.8, 0)
        + 0.4 * past_failures
        + 0.00002 * amount
    )

    # Convert continuous risk into probability, then sample failures
    risk_prob = 1 / (1 + np.exp(-(risk_raw - 1.5)))  # sigmoid
    will_fail = rng.binomial(1, risk_prob)

    df = pd.DataFrame(
        {
            "channel": channel,
            "amount": amount.round(2),
            "docs_missing": docs_missing,
            "invalid_fields": invalid_fields,
            "compliance_flags": compliance_flags,
            "resource_load": resource_load.round(3),
            "past_failures": past_failures,
            "processing_time_minutes": processing_time_minutes.round(1),
            "priority": priority,
            "will_fail": will_fail,
        }
    )
    return df

# =============================================================================
# ML training + risk decomposition
# =============================================================================

def train_model(df: pd.DataFrame):
    """Train a RandomForest classifier on the workflow data with safe handling
    when only one class exists in the dataset."""
    
    feature_cols = [
        "channel",
        "amount",
        "docs_missing",
        "invalid_fields",
        "compliance_flags",
        "resource_load",
        "past_failures",
        "processing_time_minutes",
        "priority",
    ]

    X = df[feature_cols]
    y = df["will_fail"]

    # Handle case where dataset has only one class
    unique_classes = y.unique()
    if len(unique_classes) == 1:
        # Create a dummy model wrapper
        class DummyModel:
            def predict(self, X):
                return np.full(len(X), unique_classes[0])

            def predict_proba(self, X):
                if unique_classes[0] == 1:
                    return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
                else:
                    return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

        model = DummyModel()

        metrics = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "avg_predicted_risk": float(unique_classes[0]),
            "note": "Single-class dataset detected. Dummy model created."
        }

        return model, feature_cols, metrics

    # Normal training if dataset has 2 classes
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "avg_predicted_risk": float(np.mean(y_prob)),
        "note": "Normal model trained.",
    }

    return model, feature_cols, metrics



def compute_risk_breakdown(row: pd.Series) -> dict:
    """
    Compute per-category risk scores in [0, 1] to mimic the parallel
    failure analysis pipeline in the FRD.

    Categories:
        - documentation
        - data_quality
        - compliance
        - resources
        - patterns
    """
    # Normalize heuristically into 0–1 risk scores
    doc_risk = min(1.0, 0.4 * row["docs_missing"])
    data_quality_risk = min(1.0, 0.35 * row["invalid_fields"])
    compliance_risk = min(1.0, 0.5 * row["compliance_flags"])
    resource_risk = float(
        np.clip((row["resource_load"] - 0.7) / 0.3, 0.0, 1.0)
    )
    pattern_risk = float(
        np.clip(0.3 * row["past_failures"] + 0.00002 * row["amount"], 0.0, 1.0)
    )

    category_scores = {
        "Documentation": doc_risk,
        "Data Quality": data_quality_risk,
        "Compliance": compliance_risk,
        "Resources": resource_risk,
        "Patterns / Behaviour": pattern_risk,
    }

    # Aggregate overall risk (simple weighted average)
    weights = {
        "Documentation": 0.2,
        "Data Quality": 0.2,
        "Compliance": 0.2,
        "Resources": 0.2,
        "Patterns / Behaviour": 0.2,
    }
    overall = sum(category_scores[k] * weights[k] for k in category_scores)

    return {
        "categories": category_scores,
        "overall_risk_score": float(overall),
    }


def risk_to_priority(risk_score: float) -> str:
    if risk_score >= 0.7:
        return "CRITICAL"
    elif risk_score >= 0.4:
        return "HIGH"
    elif risk_score >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"


def recommended_actions(row: pd.Series, risk_breakdown: dict) -> list:
    """
    Generate human-readable remediation suggestions based on
    prominent risk signals in the row.
    """
    actions = []

    if row["docs_missing"] > 0:
        actions.append(
            f"Upload {int(row['docs_missing'])} missing document(s) and re-run validation."
        )
    if row["invalid_fields"] > 0:
        actions.append(
            "Correct invalid or incomplete fields (e.g., dates, IDs, numeric ranges)."
        )
    if row["compliance_flags"] > 0:
        actions.append(
            "Review policy/compliance rules violated and obtain necessary approvals."
        )
    if row["resource_load"] > 0.8:
        actions.append(
            "Route workflow to an alternate queue or schedule during lower load hours."
        )
    if row["past_failures"] > 0:
        actions.append(
            "Review previous failure history and ensure known issues are addressed."
        )

    # If nothing stands out
    if not actions:
        actions.append(
            "Proceed with standard processing; monitor for anomalies in downstream steps."
        )

    # Add generic action if overall risk is high
    if risk_breakdown["overall_risk_score"] >= 0.7:
        actions.append(
            "Escalate for manual review and notify supervisor / quality team."
        )

    return actions


# =============================================================================
# Streamlit UI helpers
# =============================================================================

def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = []  # list of dicts: {risk, pred, actual}


def layout_header():
    st.title("🧠 Enterprise Process Failure Predictor – MVP")
    st.caption(
        "POC implementation of the Process Failure Predictor Framework "
        "(real-time risk prediction, parallel failure analysis, and recommendations)."
    )

# =============================
# AUTO COLUMN MAPPER + DEFAULTS
# =============================

EXPECTED_COLS = {
    "channel": "channel",
    "amount": "amount",
    "docs_missing": "docs_missing",
    "invalid_fields": "invalid_fields",
    "compliance_flags": "compliance_flags",
    "resource_load": "resource_load",
    "past_failures": "past_failures",
    "processing_time_minutes": "processing_time_minutes",
    "priority": "priority",
    "will_fail": "will_fail"
}

# Optional mapping synonyms → expected
COLUMN_SYNONYMS = {
    "doc_missing": "docs_missing",
    "missing_docs": "docs_missing",
    "missing_documents": "docs_missing",

    "invalids": "invalid_fields",
    "bad_fields": "invalid_fields",

    "compliance_issues": "compliance_flags",
    "policy_flags": "compliance_flags",

    "res_load": "resource_load",
    "utilization": "resource_load",

    "history_failures": "past_failures",

    "process_time": "processing_time_minutes",
    "duration": "processing_time_minutes",
    "time_taken": "processing_time_minutes",

    "prio": "priority",
    "importance": "priority",

    "failed": "will_fail",
    "is_failed": "will_fail",
}

def normalize_columns(df: pd.DataFrame):
    df = df.copy()
    lower_cols = {c.lower(): c for c in df.columns}

    # Step 1: Rename using synonyms
    for col in list(lower_cols.keys()):
        if col in COLUMN_SYNONYMS:
            original = lower_cols[col]
            df.rename(columns={original: COLUMN_SYNONYMS[col]}, inplace=True)

    # Step 2: Fill missing required columns
    for required in EXPECTED_COLS:
        if required not in df.columns:
            # Default filler values
            if required == "channel": df[required] = 0
            elif required == "amount": df[required] = df.get("amount", pd.Series([100]*len(df)))
            elif required == "processing_time_minutes": df[required] = 30
            elif required == "priority": df[required] = 2
            elif required == "will_fail": df[required] = 0  # Assume success if unknown
            else: df[required] = 0

    return df

def page_data_setup():
    st.subheader("Step 1 – Connect & Prepare Data")

    st.markdown(
        """
This page simulates the **Data Ingestion Layer**:
- Upload workflow data as CSV **or** generate synthetic demo data.
- The app validates structure and trains an ML model for prediction.
"""
    )

    uploaded = st.file_uploader("Upload workflow CSV", type=["csv"])
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Demo Data (Synthetic)"):
            df = generate_synthetic_workflows(800)
            st.session_state.df = df
            st.success("Generated synthetic demo data.")
    with col2:
        st.write("")

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            df_upload = normalize_columns(df_upload)
            st.session_state.df = df_upload
            st.success("Uploaded CSV loaded & normalized successfully.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")


    df = st.session_state.df
    if df is None:
        st.info("Upload a CSV or click 'Generate Demo Data' to continue.")
        return

    st.markdown("### Sample of Ingested Workflows")
    st.dataframe(df.head(20), use_container_width=True)

    required_cols = {
        "channel",
        "amount",
        "docs_missing",
        "invalid_fields",
        "compliance_flags",
        "resource_load",
        "past_failures",
        "processing_time_minutes",
        "priority",
        "will_fail",
    }

    if not required_cols.issubset(set(df.columns)):
        st.warning(
            "Your dataset is missing some expected columns.\n\n"
            f"Expected at least: {sorted(list(required_cols))}\n\n"
            "You can still adapt the training code, but the demo model "
            "assumes these columns exist (demo data already matches this)."
        )
        return

    if st.button("Train / Refresh Prediction Model"):
        with st.spinner("Training ML model (RandomForestClassifier)..."):
            model, feature_cols, metrics = train_model(df)
            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.metrics = metrics

        st.success("Model trained successfully.")
        st.markdown("### Model Performance (Hold-out Test Set)")
        m = metrics
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        cols[1].metric("Precision", f"{m['precision']:.3f}")
        cols[2].metric("Recall", f"{m['recall']:.3f}")
        cols[3].metric("F1 Score", f"{m['f1']:.3f}")

    if st.session_state.metrics is not None:
        st.markdown("### Current Model Snapshot")
        m = st.session_state.metrics
        st.write(
            f"- Trained on **{len(df)}** workflows\n"
            f"- Average predicted failure risk: **{m['avg_predicted_risk']:.2f}**"
        )


def page_realtime_prediction():
    st.subheader("Step 2 – Real-Time Process Failure Prediction")

    if st.session_state.df is None or st.session_state.model is None:
        st.warning("Please set up data and train the model on the previous tab first.")
        return

    df = st.session_state.df
    model = st.session_state.model
    feature_cols = st.session_state.feature_cols

    st.markdown(
        """
This page simulates the **real-time sequence**:
1. Select or configure a workflow.
2. The app runs **parallel failure analysis** (documentation, data quality, etc.).
3. It aggregates risk into an overall score and recommends actions.
"""
    )

    # Choose a specific workflow index
    row_index = st.slider(
        "Select a workflow index to analyze",
        min_value=0,
        max_value=int(len(df) - 1),
        value=0,
        step=1,
    )
    base_row = df.iloc[row_index].copy()

    st.markdown("#### Input Signal Configuration")
    cols = st.columns(3)
    with cols[0]:
        channel = st.selectbox(
            "Channel (0:web, 1:agent, 2:api, 3:batch)",
            options=[0, 1, 2, 3],
            index=int(base_row["channel"]),
        )
        amount = st.number_input(
            "Amount",
            value=float(base_row["amount"]),
            min_value=0.0,
            step=100.0,
            format="%.2f",
        )
        docs_missing = st.number_input(
            "Missing Documents",
            value=int(base_row["docs_missing"]),
            min_value=0,
            step=1,
        )
    with cols[1]:
        invalid_fields = st.number_input(
            "Invalid Fields",
            value=int(base_row["invalid_fields"]),
            min_value=0,
            step=1,
        )
        compliance_flags = st.number_input(
            "Compliance Flags",
            value=int(base_row["compliance_flags"]),
            min_value=0,
            step=1,
        )
        resource_load = st.slider(
            "Resource Load (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=float(base_row["resource_load"]),
            step=0.01,
        )
    with cols[2]:
        past_failures = st.number_input(
            "Past Failures",
            value=int(base_row["past_failures"]),
            min_value=0,
            step=1,
        )
        processing_time_minutes = st.number_input(
            "Processing Time (minutes)",
            value=float(base_row["processing_time_minutes"]),
            min_value=0.0,
            step=1.0,
            format="%.1f",
        )
        priority = st.selectbox(
            "Business Priority (1=Low, 3=High)",
            options=[1, 2, 3],
            index=int(base_row["priority"]) - 1,
        )

    if st.button("Run Prediction"):
        input_row = pd.DataFrame(
            [
                {
                    "channel": channel,
                    "amount": amount,
                    "docs_missing": docs_missing,
                    "invalid_fields": invalid_fields,
                    "compliance_flags": compliance_flags,
                    "resource_load": resource_load,
                    "past_failures": past_failures,
                    "processing_time_minutes": processing_time_minutes,
                    "priority": priority,
                }
            ]
        )

        with st.spinner("Running AI/ML prediction and parallel failure analysis..."):
            prob_fail = float(model.predict_proba(input_row[feature_cols])[:, 1])
            pred_label = int(prob_fail >= 0.5)

            breakdown = compute_risk_breakdown(input_row.iloc[0])
            overall_risk = breakdown["overall_risk_score"]
            priority_label = risk_to_priority(overall_risk)
            actions = recommended_actions(input_row.iloc[0], breakdown)

        st.markdown("### Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Failure Probability", f"{prob_fail:.2%}")
        c2.metric("Aggregated Risk Score", f"{overall_risk:.2f}")
        c3.metric("Risk Priority", priority_label)

        st.markdown("#### Parallel Failure Analysis (Per Category Risk)")
        cat_cols = st.columns(len(breakdown["categories"]))
        for (name, score), col in zip(breakdown["categories"].items(), cat_cols):
            col.metric(name, f"{score:.2f}")

        st.markdown("#### Recommended Actions")
        for i, act in enumerate(actions, start=1):
            st.write(f"{i}. {act}")

        st.session_state.last_prediction = {
            "input_row": input_row,
            "prob_fail": prob_fail,
            "pred_label": pred_label,
            "overall_risk": overall_risk,
        }

    if "last_prediction" in st.session_state:
        with st.expander("Show Input Payload (for API / integration discussions)"):
            st.json(
                {
                    "workflow": st.session_state.last_prediction["input_row"]
                    .iloc[0]
                    .to_dict(),
                    "prediction": {
                        "failure_probability": st.session_state.last_prediction[
                            "prob_fail"
                        ],
                        "overall_risk_score": st.session_state.last_prediction[
                            "overall_risk"
                        ],
                        "priority": risk_to_priority(
                            st.session_state.last_prediction["overall_risk"]
                        ),
                    },
                }
            )


def page_dashboard():
    st.subheader("Step 3 – Batch Analysis & KPI Dashboard")

    if st.session_state.df is None or st.session_state.model is None:
        st.warning("Please set up data and train the model on the first tab.")
        return

    df = st.session_state.df
    model = st.session_state.model
    feature_cols = st.session_state.feature_cols

    st.markdown(
        """
This page is **Operations Dashboard**
- See predicted vs actual failures.
- Understand the distribution of risk across workflows.
- Get a high-level view of operational health.
"""
    )

    # Predict for all rows
    probs = model.predict_proba(df[feature_cols])[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_view = df.copy()
    df_view["predicted_failure_prob"] = probs
    df_view["predicted_will_fail"] = preds

    st.markdown("### Portfolio View (Top 200 Workflows)")
    st.dataframe(df_view.head(200), use_container_width=True)

    st.markdown("### Portfolio KPIs")
    total = len(df_view)
    actual_fail_rate = df_view["will_fail"].mean()
    predicted_fail_rate = df_view["predicted_will_fail"].mean()
    high_risk_rate = (df_view["predicted_failure_prob"] >= 0.7).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Workflows", f"{total}")
    col2.metric("Actual Failure Rate", f"{actual_fail_rate:.1%}")
    col3.metric("Predicted Failure Rate", f"{predicted_fail_rate:.1%}")
    col4.metric("High-Risk Workflows", f"{high_risk_rate:.1%}")

    st.markdown("### Risk Buckets")
    bucket_labels = ["Low (0-0.2)", "Medium (0.2-0.4)", "High (0.4-0.7)", "Critical (0.7-1.0)"]
    bucket_edges = [0.0, 0.2, 0.4, 0.7, 1.0]
    bucket_counts = np.histogram(probs, bins=bucket_edges)[0]
    bucket_df = pd.DataFrame(
        {
            "Risk Bucket": bucket_labels,
            "Count": bucket_counts,
        }
    )
    st.bar_chart(bucket_df.set_index("Risk Bucket"))


def page_feedback():
    st.subheader("Step 4 – Feedback & Continuous Improvement")

    if "last_prediction" not in st.session_state:
        st.info(
            "Run at least one prediction on the 'Real-Time Prediction' tab "
            "to start logging feedback."
        )

    st.markdown(
        """
This page simulates the **feedback loop & reinforcement learning**:
- After each prediction, a user can mark whether the prediction was correct.
- We track simple online KPIs: accuracy, precision, recall, and failure rates over time.
"""
    )

    last_pred = st.session_state.get("last_prediction", None)
    if last_pred is not None:
        st.markdown("#### Last Prediction Feedback")
        col1, col2 = st.columns(2)
        with col1:
            st.write(
                f"Predicted failure probability: **{last_pred['prob_fail']:.2%}** "
                f"(label={last_pred['pred_label']})"
            )
        with col2:
            actual = st.radio(
                "What actually happened?",
                options=["Unknown", "Process Succeeded", "Process Failed"],
                horizontal=True,
            )
        if st.button("Log Feedback"):
            if actual == "Unknown":
                st.warning("Please select an actual outcome before logging.")
            else:
                actual_label = 1 if actual == "Process Failed" else 0
                st.session_state.feedback.append(
                    {
                        "pred_label": last_pred["pred_label"],
                        "actual_label": actual_label,
                        "overall_risk": last_pred["overall_risk"],
                    }
                )
                st.success("Feedback logged. Future versions could retrain the model on this.")

    if len(st.session_state.feedback) > 0:
        st.markdown("### Feedback KPIs (Based on Logged Cases)")
        fb = pd.DataFrame(st.session_state.feedback)
        acc = accuracy_score(fb["actual_label"], fb["pred_label"])
        prec = precision_score(fb["actual_label"], fb["pred_label"], zero_division=0)
        rec = recall_score(fb["actual_label"], fb["pred_label"], zero_division=0)
        f1 = f1_score(fb["actual_label"], fb["pred_label"], zero_division=0)
        avg_risk = fb["overall_risk"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Feedback Cases", len(fb))
        c2.metric("Accuracy", f"{acc:.2f}")
        c3.metric("Precision", f"{prec:.2f}")
        c4.metric("Recall", f"{rec:.2f}")
        st.metric("Avg Overall Risk (Logged Cases)", f"{avg_risk:.2f}")

        st.markdown("#### Raw Feedback Log")
        st.dataframe(fb, use_container_width=True)
    else:
        st.info("No feedback logged yet. Use the real-time prediction page to add some.")


# =============================================================================
# Main app
# =============================================================================

def main():
    st.set_page_config(
        page_title="Process Failure Predictor MVP",
        page_icon="🧠",
        layout="wide",
    )
    init_session_state()
    layout_header()

    tab_names = [
        "1️⃣ Data Setup",
        "2️⃣ Real-Time Prediction",
        "3️⃣ Dashboard",
        "4️⃣ Feedback & Learning",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        page_data_setup()
    with tabs[1]:
        page_realtime_prediction()
    with tabs[2]:
        page_dashboard()
    with tabs[3]:
        page_feedback()


if __name__ == "__main__":
    main()
