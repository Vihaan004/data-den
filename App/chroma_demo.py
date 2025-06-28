# ChromaDB Demo Script
# A simple script to demonstrate the ChromaDB vector store functionality
# Run this script to see how the persistent vector store works

import os
import sys
from document_loader import DocumentLoader, VectorStore
from config import USE_PERSISTENT_VECTORSTORE, VECTORSTORE_PERSIST_DIRECTORY

def main():
    print("🚀 ChromaDB Demo Script")
    print(f"Using persistent storage: {USE_PERSISTENT_VECTORSTORE}")
    print(f"Storage directory: {VECTORSTORE_PERSIST_DIRECTORY}")
    
    # Load documents
    print("\n📚 Loading documents...")
    doc_loader = DocumentLoader()
    docs = doc_loader.load_documents()
    doc_splits = doc_loader.split_documents(docs)
    print(f"Total document splits: {len(doc_splits)}")
    
    # Create vector store
    print("\n🔍 Creating/Loading vector store...")
    vector_store = VectorStore()
    retriever = vector_store.create_vectorstore(doc_splits)
    
    # Check if we loaded an existing store or created a new one
    print(f"Using persistent store: {vector_store.using_persistent}")
    print(f"Document count: {vector_store.get_document_count()}")
    
    # Test retrieval
    print("\n🔎 Testing retrieval...")
    test_queries = [
        "How do I convert NumPy code to CuPy?",
        "What are the performance benefits of using cuDF?",
        "When should I not use GPU acceleration?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.invoke({"query": query})
        print(f"Found {len(results)} results")
        if results:
            # Print the first result
            if hasattr(results[0], 'page_content'):
                print(f"First result (sample): {results[0].page_content[:100]}...")
            else:
                print(f"First result (sample): {str(results[0])[:100]}...")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()
