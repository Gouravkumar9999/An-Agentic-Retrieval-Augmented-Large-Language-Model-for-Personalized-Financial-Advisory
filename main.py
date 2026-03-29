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
from src.utils.analysis import analyze_transactions
from src.evaluation import evaluate_response
from src.utils.analysis import generate_market_summary

def main():
    print("Loading data...")

    transactions = load_transactions()
    stocks = load_stock_data()
    inflation, interest = load_economic_data()
    knowledge = load_knowledge_base()

    print("Preprocessing data...")

    transactions = preprocess_transactions(transactions)
    txn_analysis = analyze_transactions(transactions)
    stocks = preprocess_stock_data(stocks)
    inflation, interest = preprocess_economic_data(inflation, interest)

    print("Data Loaded Successfully")

    # -----------------------------
    # RAG SYSTEM
    # -----------------------------
    print("Initializing RAG system...")

    rag = RAGSystem(knowledge)

    query = "How should I manage my savings during high inflation?"

    results = rag.retrieve(query, k=5)

    print("\nRetrieved Context:")
    for r in results:
        print("-", r)

    user_data = {
        "income": 50000,
        "monthly_spending": txn_analysis["monthly_spending"],
        "category_breakdown": txn_analysis["category_breakdown"],
        "risk_preference": "medium"
}

    market_data = generate_market_summary(inflation, interest)

    # Convert retrieved chunks into context
    context = "\n".join(results)

    # Run agents
    final_report = run_agents(user_data, market_data, context)

    print("\n\n===== FINAL OUTPUT =====\n")
    print(final_report)
    metrics = evaluate_response(final_report, context, query)
    print("\nEvaluation Metrics:", metrics)


if __name__ == "__main__":
    main()