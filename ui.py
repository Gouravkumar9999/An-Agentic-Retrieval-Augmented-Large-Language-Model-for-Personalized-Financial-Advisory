import streamlit as st
from src.agents.query_rewriter import rewrite_query
import os
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

import pandas as pd
import plotly.express as px

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

    #  USER OVERRIDE
    if user_pref == "low":
        return "LOW"
    elif user_pref == "medium":
        return "MEDIUM"
    else:
        return base

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
    display_text = session["query"][:40] + ("..." if len(session["query"]) > 40 else "")
    if st.sidebar.button(display_text, key=f"session_{idx}", help=session["query"]):
        st.session_state.current_session = idx

# -----------------------------
# RUN QUERY
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
                "estimated_savings": metrics_data["savings"],
                "expense_ratio": metrics_data["expense_ratio"],
                "savings_ratio": metrics_data["savings_ratio"],
                "emergency_months": metrics_data["emergency_months"],
                "top3_ratio": extra_features.get("top3_ratio", 0),
                "discretionary_ratio": extra_features.get("discretionary_ratio", 0),
                "essential_ratio": extra_features.get("essential_ratio", 0),
                "volatility": extra_features.get("volatility", 0),
                "txn_frequency": extra_features.get("txn_frequency", 0)
            }

            optimized_query = rewrite_query(query)
            results = rag_system.retrieve(optimized_query, k=3)
            context = "\n".join(results)

            final_report = run_agents(user_data, market_summary, context, metrics_data)
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
# DISPLAY SESSION
# -----------------------------
if st.session_state.current_session is not None:

    session = st.session_state.sessions[st.session_state.current_session]

    st.subheader("📄 Financial Report")

    # Use columns for budget vs metrics
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(session["report"].replace("\n", "<br>"), unsafe_allow_html=True)

        # Investment Allocation Pie Chart
        alloc = session["user_data"]
        alloc_dict = {
            "Equities": alloc.get("equities", 30),
            "Stocks": alloc.get("stocks", 30),
            "Bonds": alloc.get("bonds", 20),
            "Index Funds": alloc.get("top3_ratio", 40),
            "Cash": alloc.get("essential_ratio", 10)
        }
        fig = px.pie(names=list(alloc_dict.keys()), values=list(alloc_dict.values()),
                     title="Investment Allocation")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Key Metrics")
        m = session["metrics"]
        risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
        st.metric("Savings", f"${m['savings']:,}")
        st.metric("Expense Ratio", f"{m['expense_ratio']*100:.1f}%", delta_color="inverse")
        st.metric("Savings Ratio", f"{m['savings_ratio']*100:.1f}%")
        st.metric("Emergency Months", m['emergency_months'])
        st.markdown(f"**Risk Level:** <span style='color:{risk_color.get(session['user_data']['final_risk'], 'black')};'>{session['user_data']['final_risk']}</span>", unsafe_allow_html=True)

    # Evaluation
    st.subheader("📊 Evaluation")
    eval_data = session["eval"]

    st.markdown("**Combined Scores (Embedding + LLM)**")
    for k, v in eval_data["final_scores"].items():
        st.progress(min(max(v, 0), 1))
        st.write(f"{k}: {v:.3f}")

    with st.expander("🔹 Embedding-based Scores"):
        for k, v in eval_data["embedding_scores"].items():
            st.write(f"{k}: {v:.3f}")

    with st.expander("🔹 LLM-based Scores & Explanation"):
        llm = eval_data["llm_scores"]
        for k in ["faithfulness", "relevance", "consistency"]:
            st.write(f"{k}: {llm.get(k, 0):.3f}")
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