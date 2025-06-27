#!/usr/bin/env python3

# Test script to verify query classification
from rag_agent import RAGAgent

def test_classification():
    agent = RAGAgent()
    
    test_queries = [
        "What is an LLM?",
        "What does this code do?",
        "What does this code do?\n\nimport numpy as np\nA = np.random.rand(10, 10)",
        "Analyze this code:\n```python\nimport numpy as np\nA = np.random.rand(10, 10)\n```",
        "How can I optimize this numpy code for GPU?",
        "Hello, how are you?",
        "Tell me about CuPy optimization"
    ]
    
    print("🧪 Testing Query Classification:")
    print("=" * 50)
    
    for query in test_queries:
        classification = agent._classify_query(query)
        print(f"Query: {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"Classification: {classification}")
        print("-" * 30)

if __name__ == "__main__":
    test_classification()
