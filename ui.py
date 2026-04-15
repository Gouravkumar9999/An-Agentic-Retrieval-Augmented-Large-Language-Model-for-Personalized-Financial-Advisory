import streamlit as st
from src.agents.query_rewriter import rewrite_query
import plotly.express as px
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Finance Advisor", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>Personalized Financial Advisor</h1>
    <p style='text-align: center; color: grey;'>AI-powered budgeting, investment & risk insights</p>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("⚙️ User Inputs")

query = st.sidebar.text_area("Enter your finance query", height=80)
income = st.sidebar.number_input("Monthly Income", min_value=0, value=50000)
risk_pref = st.sidebar.selectbox("Risk Preference", ["low", "medium", "high"])

# -----------------------------
# IMPORTS
# -----------------------------
from src.data.load_data import (
    load_transactions,
    load_stock_data,
    load_economic_data,
    load_knowledge_base
)

from src.preprocessing.preprocess import (
    preprocess_transactions,
    preprocess_stock_data,
    preprocess_economic_data
)

from src.rag.retriever import RAGSystem
from src.agents.orchestrator import run_agents
from src.utils.analysis import analyze_transactions, generate_market_summary
from src.evaluation import evaluate_response

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(income, expenses):
    savings = income - expenses
    return {
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "savings": round(savings, 2),
        "expense_ratio": round(expenses / income if income else 0, 3),
        "savings_ratio": round(savings / income if income else 0, 3),
        "emergency_months": round(savings / expenses if expenses else 0, 2)
    }

def compute_risk_level(metrics, user_pref):
    exp = metrics["expense_ratio"]
    emergency = metrics["emergency_months"]
    if exp > 0.7 or emergency < 3:
        base = "HIGH"
    elif exp > 0.5 or emergency < 6:
        base = "MEDIUM"
    else:
        base = "LOW"
    # USER OVERRIDE
    if user_pref == "low":
        return "LOW"
    elif user_pref == "medium":
        return "MEDIUM"
    else:
        return base
def compute_risks(metrics, user_data):
    risks = []
    structured = {}

    expense_ratio = metrics["expense_ratio"]
    emergency = metrics["emergency_months"]
    discretionary = user_data.get("discretionary_ratio", 0)
    volatility = user_data.get("volatility", 0)

    # Emergency risk
    if emergency < 3:
        level = "HIGH"
    elif emergency < 6:
        level = "MEDIUM"
    else:
        level = "LOW"

    risks.append(f"{level} emergency risk (months={emergency})")
    structured["emergency_risk"] = level

    # Expense risk
    if expense_ratio > 0.7:
        level = "HIGH"
    else:
        level = "LOW"

    risks.append(f"{level} expense risk (ratio={expense_ratio})")
    structured["expense_risk"] = level

    # Behavioral risk
    if discretionary > 0.6:
        risks.append(f"HIGH behavioral risk (discretionary={discretionary})")
        structured["behavioral_risk"] = "HIGH"
    else:
        structured["behavioral_risk"] = "LOW"

    # Volatility risk
    if volatility > 300:
        risks.append(f"HIGH volatility risk ({volatility})")
        structured["volatility_risk"] = "HIGH"
    else:
        structured["volatility_risk"] = "LOW"

    return {
        "list": risks[:3],         
        "dict": structured         
    }
# -----------------------------
# CACHE
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_only():
    transactions = preprocess_transactions(load_transactions())
    txn_analysis = analyze_transactions(transactions)

    stocks = preprocess_stock_data(load_stock_data())
    inflation, interest = preprocess_economic_data(*load_economic_data())

    knowledge = load_knowledge_base()
    market_summary = generate_market_summary(inflation, interest)

    return transactions, txn_analysis, knowledge, market_summary

@st.cache_resource(show_spinner=False)
def load_rag_system(knowledge):
    return RAGSystem(knowledge)

transactions, txn_analysis, knowledge, market_summary = load_data_only()
rag_system = load_rag_system(knowledge)

# -----------------------------
# SESSION STATE
# -----------------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = []

if "current_session" not in st.session_state:
    st.session_state.current_session = None

# -----------------------------
# SIDEBAR HISTORY
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 History")

for i, session in enumerate(reversed(st.session_state.sessions[-10:])):
    idx = len(st.session_state.sessions) - 1 - i
    truncated_query = session["query"] if len(session["query"]) < 50 else session["query"][:47] + "..."
    if st.sidebar.button(truncated_query, key=f"session_{idx}", help=session["query"]):
        st.session_state.current_session = idx

# -----------------------------
# RUN
# -----------------------------
if st.button("🚀 Get Financial Advice"):
    if not query.strip():
        st.error("Enter a query")
    else:
        with st.spinner("Analyzing..."):

            expenses = txn_analysis["monthly_spending"]

            # GET EXTRA FEATURES
            extra_features = transactions.attrs.get("features", {})

            # COMPUTE METRICS
            metrics_data = compute_metrics(income, expenses)
            
            # ADD FINAL RISK
            final_risk = compute_risk_level(metrics_data, risk_pref)

            user_data = {
                "income": income,
                "monthly_spending": expenses,
                "category_breakdown": txn_analysis["category_breakdown"],
                "risk_preference": risk_pref,
                "final_risk": final_risk,

                # metrics
                "estimated_savings": metrics_data["savings"],
                "expense_ratio": metrics_data["expense_ratio"],
                "savings_ratio": metrics_data["savings_ratio"],
                "emergency_months": metrics_data["emergency_months"],

                # NEW FEATURES
                "top3_ratio": extra_features.get("top3_ratio", 0),
                "discretionary_ratio": extra_features.get("discretionary_ratio", 0),
                "essential_ratio": extra_features.get("essential_ratio", 0),
                "volatility": extra_features.get("volatility", 0),
                "txn_frequency": extra_features.get("txn_frequency", 0)
            }

            optimized_query = rewrite_query(query)
            results = rag_system.retrieve(optimized_query, k=3)
            context = "\n".join(results)
            computed_risks = compute_risks(metrics_data, user_data)
            final_report = run_agents(user_data, market_summary, context, metrics_data, computed_risks)
            eval_scores = evaluate_response(final_report, context, query)

            st.session_state.sessions.append({
                "query": query,
                "report": final_report,
                "user_data": user_data,
                "metrics": metrics_data,
                "eval": eval_scores,
                "context": results
            })

            st.session_state.current_session = len(st.session_state.sessions) - 1

# -----------------------------
# DISPLAY
# -----------------------------
if st.session_state.current_session is not None:

    session = st.session_state.sessions[st.session_state.current_session]
    col1, col2 = st.columns([2, 1])

    # --------- FINANCIAL REPORT ----------
    with col1:
        st.subheader("📄 Financial Report")
        report_text = session["report"]

        # Format bullets and newlines properly
        report_text = re.sub(r"•\s*", "<li>", report_text)
        report_text = re.sub(r"\n", "<br>", report_text)
        report_text = f"<div style='line-height:1.6'>{report_text}</div>"

        st.markdown(report_text, unsafe_allow_html=True)
        # Investment Allocation Pie Chart
        alloc = session["user_data"]
        alloc_dict = {
            "Equities": alloc.get("equities", 30),
            "Stocks": alloc.get("stocks", 30),
            "Bonds": alloc.get("bonds", 20),
            "Index Funds": alloc.get("top3_ratio", 40),
            "Cash": alloc.get("essential_ratio", 10)
        }
        values = list(alloc_dict.values())

        # Avoid division by zero
        total = sum(values) if sum(values) != 0 else 1
        normalized_values = [v / total for v in values]

        fig = px.pie(
            names=list(alloc_dict.keys()),
            values=normalized_values,
            title="Investment Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)



    # --------- KEY METRICS ----------
    with col2:
        st.subheader("📊 Key Metrics")
        m = session["metrics"]
        st.metric("Savings", f"${m['savings']:,}")
        st.metric("Expense Ratio", f"{m['expense_ratio']*100:.1f}%")
        st.metric("Savings Ratio", f"{m['savings_ratio']*100:.1f}%")
        st.metric("Emergency Months", m['emergency_months'])

        # Color-coded risk
        risk_color = {"LOW":"green", "MEDIUM":"orange", "HIGH":"red"}
        st.markdown(f"**Risk Level:** <span style='color:{risk_color.get(session['user_data']['final_risk'], 'black')};'>{session['user_data']['final_risk']}</span>", unsafe_allow_html=True)

    # --------- EVALUATION ----------
    st.subheader("📊 Evaluation")
    eval_data = session["eval"]

    st.markdown("**Combined Scores (Embedding + LLM)**")
    for k, v in eval_data["final_scores"].items():
        st.progress(min(max(v, 0), 1))
        st.write(f"{k}: {v}")

    with st.expander("🔹 Embedding-based Scores"):
        for k, v in eval_data["embedding_scores"].items():
            st.write(f"{k}: {v}")

    with st.expander("🔹 LLM-based Scores & Explanation"):
        llm = eval_data["llm_scores"]
        for k in ["faithfulness", "relevance", "consistency"]:
            st.write(f"{k}: {llm.get(k, 0)}")
        st.markdown(f"**Explanation:** {llm.get('explanation', 'N/A')}")

    with st.expander("🧠 Retrieved Knowledge"):
        for r in session["context"]:
            st.write("•", r)

# -----------------------------
# CLEAR HISTORY
# -----------------------------
if st.sidebar.button("🗑 Clear History"):
    st.session_state.sessions = []
    st.session_state.current_session = None