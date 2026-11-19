
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="XAI-PAL | Research Demonstrator", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent
TEACHER_FILE = BASE / "Teachers form (Responses) (1).xlsx"
PARENT_FILE = BASE / "Parent_ Guardian form (Responses) (1).xlsx"

st.title("Counterfactual-Guided Explainable AI for Personalized Adaptive Learning")
st.caption("Research demonstrator • NEP 2020 context • Teacher-in-the-loop")

st.warning(
    "Research integrity note: The uploaded workbooks contain stakeholder questionnaire responses, "
    "not student-level learning histories or verified learning outcomes. The survey dashboard below "
    "uses the uploaded responses. The interactive prediction sandbox uses explicitly synthetic "
    "demonstration data and is NOT a trained or validated model from this study."
)

@st.cache_data
def load_workbook(path):
    df = pd.read_excel(path)
    df = df.dropna(how="all").copy()
    # Do not expose direct identifiers in the app.
    drop_cols = [c for c in df.columns if str(c).strip().lower() in {"timestamp", "email address", "email"}]
    return df.drop(columns=drop_cols, errors="ignore")

@st.cache_resource
def build_demo_model():
    rng = np.random.default_rng(42)
    n = 6000
    X = pd.DataFrame({
        "Previous assessment score": rng.uniform(10, 100, n),
        "Attendance": rng.uniform(40, 100, n),
        "Practice completion": rng.uniform(0, 100, n),
        "Topic accuracy": rng.uniform(10, 100, n),
    })
    # Synthetic teaching-demo relationship; not learned from the survey workbooks.
    z = (
        0.030 * (X["Previous assessment score"] - 55)
        + 0.022 * (X["Attendance"] - 70)
        + 0.026 * (X["Practice completion"] - 55)
        + 0.035 * (X["Topic accuracy"] - 55)
    )
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=160, max_depth=7, min_samples_leaf=12, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }
    return model, X_test, y_test, metrics

def pick_col(df, fragment):
    matches = [c for c in df.columns if fragment.lower() in str(c).lower()]
    return matches[0] if matches else None

def show_distribution(df, col, title):
    s = df[col].dropna().astype(str).str.strip()
    if s.empty:
        st.info("No non-empty responses for this item.")
        return
    counts = s.value_counts().reset_index()
    counts.columns = ["Response", "Count"]
    fig = px.bar(counts, x="Count", y="Response", orientation="h", text="Count",
                 title=title, color="Count", color_continuous_scale="Blues")
    fig.update_layout(height=max(300, 42 * len(counts)), yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

teachers = load_workbook(TEACHER_FILE) if TEACHER_FILE.exists() else None
parents = load_workbook(PARENT_FILE) if PARENT_FILE.exists() else None

tab1, tab2, tab3 = st.tabs(["Research survey dashboard", "Interactive AI demonstrator", "Method & limitations"])

with tab1:
    st.subheader("Uploaded questionnaire responses")
    if teachers is None or parents is None:
        st.error("The two Excel workbooks must be placed beside app.py. See README.")
    else:
        t_n = len(teachers)
        p_n = len(parents)
        c1, c2, c3 = st.columns(3)
        c1.metric("Teacher responses in uploaded workbook", f"{t_n}")
        c2.metric("Parent/guardian responses in uploaded workbook", f"{p_n}")
        c3.metric("Combined workbook records", f"{t_n + p_n}")
        st.info(
            "Count check: these uploaded workbook versions contain "
            f"{t_n} teacher records and {p_n} parent/guardian records. "
            "If your presentation reports 97 teachers and 203 parents/guardians (300 total), "
            "reconcile the workbook versions and slide figures before presenting."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Teacher questionnaire**")
            teacher_options = [c for c in teachers.columns if teachers[c].notna().any()]
            default_t = pick_col(teachers, "How useful would an AI system be")
            t_choice = st.selectbox("Explore a teacher question", teacher_options,
                                    index=teacher_options.index(default_t) if default_t in teacher_options else 0,
                                    key="t_question")
            show_distribution(teachers, t_choice, "Teacher responses")
        with right:
            st.markdown("**Parent/guardian questionnaire**")
            parent_options = [c for c in parents.columns if parents[c].notna().any()]
            default_p = pick_col(parents, "Would you trust an AI-based system")
            p_choice = st.selectbox("Explore a parent/guardian question", parent_options,
                                    index=parent_options.index(default_p) if default_p in parent_options else 0,
                                    key="p_question")
            show_distribution(parents, p_choice, "Parent/guardian responses")

        st.markdown("#### Selected descriptive indicators")
        # Identify columns from actual question text; display counts, not inferred scores.
        t_use = pick_col(teachers, "How useful would an AI system be")
        t_override = pick_col(teachers, "allowed to override AI suggestions")
        p_privacy = pick_col(parents, "concerned about the privacy")
        p_language = pick_col(parents, "How important is it that AI learning recommendations")
        cards = st.columns(4)
        def pct_contains(series, terms):
            s = series.dropna().astype(str).str.lower()
            if len(s) == 0:
                return None
            mask = s.apply(lambda x: any(term in x for term in terms))
            return 100 * mask.mean()
        items = [
            ("Teachers: AI usefulness", pct_contains(teachers[t_use], ["very", "extremely"]) if t_use else None),
            ("Teachers: override support", pct_contains(teachers[t_override], ["agree", "strongly agree"]) if t_override else None),
            ("Parents: privacy concern", pct_contains(parents[p_privacy], ["concerned"]) if p_privacy else None),
            ("Parents: language importance", None),
        ]
        # Numeric 1–5 parent scale: show distribution in question chart; do not assume labels.
        if p_language:
            vals = pd.to_numeric(parents[p_language], errors="coerce").dropna()
            items[3] = ("Parents: language rating (mean)", float(vals.mean()) if len(vals) else None)
        for card, (label, value) in zip(cards, items):
            if value is None:
                card.metric(label, "Not available")
            elif "mean" in label:
                card.metric(label, f"{value:.2f}/5")
            else:
                card.metric(label, f"{value:.1f}%")
        st.caption("These are descriptive summaries of the uploaded workbooks. Response wording is matched from the questionnaire; verify any recoding against the original form before publication.")

with tab2:
    st.subheader("Student competency prediction — illustrative prototype")
    st.write("Adjust a hypothetical learner profile. The model returns a probability, feature-level context and a counterfactual scenario.")
    model, X_test, y_test, metrics = build_demo_model()

    a, b = st.columns([1, 1])
    with a:
        score = st.slider("Previous assessment score (%)", 0, 100, 42)
        attendance = st.slider("Attendance (%)", 0, 100, 65)
        practice = st.slider("Practice completion (%)", 0, 100, 30)
        accuracy = st.slider("Topic accuracy (%)", 0, 100, 40)
        language = st.selectbox("Preferred learning language", ["Assamese", "English", "Hindi", "Other"])
        st.caption("Language is shown as context for accessible recommendations; it is not used as a risk penalty in this demo model.")
    row = pd.DataFrame([{
        "Previous assessment score": score,
        "Attendance": attendance,
        "Practice completion": practice,
        "Topic accuracy": accuracy
    }])
    p_success = float(model.predict_proba(row)[0, 1])
    p_risk = 1 - p_success
    with b:
        st.markdown("#### Model output")
        st.metric("Predicted probability of target competency", f"{p_success*100:.1f}%")
        st.metric("Illustrative probability of needing additional support", f"{p_risk*100:.1f}%")
        if p_success >= 0.70:
            st.success("Model classification: higher predicted likelihood of achieving the target competency")
        elif p_success >= 0.45:
            st.warning("Model classification: uncertain / monitor progress")
        else:
            st.error("Model classification: additional support may be warranted")
        st.caption("Thresholds are demonstration settings, not validated educational decision thresholds.")

    st.markdown("#### Feature context")
    importances = pd.Series(model.feature_importances_, index=row.columns).sort_values()
    imp_df = importances.rename("Relative importance").reset_index()
    imp_df.columns = ["Feature", "Relative importance"]
    fig = px.bar(imp_df, x="Relative importance", y="Feature", orientation="h",
                 title="Global model feature importance (not SHAP values)",
                 color="Relative importance", color_continuous_scale="Teal")
    fig.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Counterfactual what-if exploration")
    st.write("The app searches for the smallest tested change in actionable learning-support inputs that flips the model classification. This is a model-based scenario, not a causal claim.")
    threshold = 0.50
    current_class = int(p_success >= threshold)
    candidates = []
    # Search discrete alternatives for actionable variables; keep prior score fixed.
    for att in range(attendance, 101, 5):
        for prac in range(practice, 101, 5):
            for acc in range(accuracy, 101, 5):
                candidate = pd.DataFrame([{
                    "Previous assessment score": score,
                    "Attendance": att,
                    "Practice completion": prac,
                    "Topic accuracy": acc
                }])
                cp = float(model.predict_proba(candidate)[0, 1])
                if int(cp >= threshold) != current_class:
                    distance = (att-attendance) + (prac-practice) + (acc-accuracy)
                    candidates.append((distance, att, prac, acc, cp))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, ca, cpra, cacc, cprob = candidates[0]
        cf1, cf2, cf3 = st.columns(3)
        cf1.metric("Attendance scenario", f"{attendance}% → {ca}%")
        cf2.metric("Practice scenario", f"{practice}% → {cpra}%")
        cf3.metric("Topic accuracy scenario", f"{accuracy}% → {cacc}%")
        st.info(f"Under this synthetic model, the predicted competency probability changes from {p_success*100:.1f}% to {cprob*100:.1f}%.")
    else:
        st.info("No classification-flipping scenario was found in the tested upward changes. Try a different learner profile.")
    st.markdown("#### Suggested teacher-reviewed support")
    suggestions = []
    if accuracy < 60:
        suggestions.append("Provide topic-specific revision and short formative checks.")
    if practice < 60:
        suggestions.append("Offer additional guided practice with feedback.")
    if attendance < 75:
        suggestions.append("Discuss attendance barriers with the learner and family before interpreting missed learning.")
    if score < 50:
        suggestions.append("Review prerequisite concepts and identify the exact skill gap.")
    if language in ["Assamese", "Hindi", "Other"]:
        suggestions.append(f"Offer accessible explanations or learning materials in {language}, where available.")
    if not suggestions:
        suggestions.append("Continue monitoring and provide extension activities where appropriate.")
    for item in suggestions:
        st.write("• " + item)
    st.caption("Recommendations are rule-based demonstration prompts. A real system would need curriculum mapping, teacher review, consent, privacy safeguards and evaluation.")

    with st.expander("Demonstrator model evaluation (synthetic test split)"):
        st.write("These metrics describe only the generated synthetic demonstration dataset. They are not research findings.")
        mcols = st.columns(len(metrics))
        for col, (name, val) in zip(mcols, metrics.items()):
            col.metric(name, f"{val:.3f}")

with tab3:
    st.subheader("What this prototype does — and does not — establish")
    st.markdown("""
**Uses actual study material**
- Teacher and parent/guardian questionnaire workbooks are used for descriptive summaries.
- Direct email and timestamp columns are excluded from the displayed analysis.

**Uses synthetic data**
- The prediction model is trained on generated, artificial learner records.
- The feature importance chart is a global Random Forest importance measure, not a SHAP explanation.
- Counterfactual search explores selected upward changes in attendance, practice completion and topic accuracy.

**Not established by this prototype**
- It does not demonstrate that AI improves student achievement.
- It does not validate a real student-risk prediction model.
- It does not establish causal effects of changing attendance or practice.
- It does not replace teachers or make decisions about real students.

**Data needed for a research-grade model**
- Consented, de-identified student-level longitudinal records.
- Clearly defined competency outcomes and assessment rubrics.
- Curriculum/topic mapping and intervention records.
- Adequate representation across language, location and access groups.
- Train/validation/test separation at student or school level to prevent leakage.
- Calibration, subgroup fairness, external validation and classroom evaluation.
""")
    st.markdown("**Suggested panel explanation:**")
    st.info(
        "This is a proof-of-concept demonstrator. The survey dashboard reflects stakeholder perceptions from our questionnaire. "
        "The interactive prediction component uses synthetic data to demonstrate the proposed pipeline—prediction, explanation, "
        "counterfactual exploration and teacher-reviewed support. A research-grade model would require consented student-level "
        "learning data and independent validation."
    )

st.divider()
st.caption("Academic prototype • Do not use this demonstrator to make decisions about real students.")
