from typing import Dict, List, Any
import re
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
import socket
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class RAGAgent:
    """Core RAG agent for GPU acceleration knowledge."""
    
    def __init__(self):
        self.llm_model = None
        self.retriever_tool = None
        self.rag_graph = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Initialize the local LLM model."""
        try:
            self.llm_model = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                base_url=OLLAMA_BASE_URL
            )
            print("✅ LLM model initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize LLM: {e}")
    
    def set_retriever_tool(self, retriever_tool):
        """Set the retriever tool for the RAG system."""
        self.retriever_tool = retriever_tool
        self._build_graph()
    
    def _build_graph(self):
        """Build the RAG graph with retrieval and generation."""
        from langgraph.graph import StateGraph, START, END
        
        # Build the graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._decide_to_retrieve,
            {
                "retrieve": "grade_documents",
                "respond": "generate_response"
            }
        )
        workflow.add_edge("grade_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        
        self.rag_graph = workflow.compile()
        print("✅ RAG graph compiled successfully")
    
    def _generate_query_or_respond(self, state: MessagesState):
        """Generate a response or decide to retrieve documents."""
        if not self.llm_model:
            return {"messages": [AIMessage(content="LLM model not available")]}
        
        try:
            response = self.llm_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
            # Clean up response content
            content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
            response.content = content
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error generating response: {str(e)}")]}
    
    def _grade_documents(self, state: MessagesState):
        """Grade retrieved documents for relevance."""
        # For simplicity, assume all retrieved documents are relevant
        # In production, implement proper document grading
        return {"messages": state["messages"]}
    
    def _generate_response(self, state: MessagesState):
        """Generate final response based on retrieved documents."""
        if not self.llm_model:
            return {"messages": [AIMessage(content="LLM model not available")]}
        
        try:
            # Get the last message and generate response
            last_message = state["messages"][-1]
            
            # If there are tool calls, handle them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Retrieve documents
                tool_call = last_message.tool_calls[0]
                query = tool_call['args']['query']
                
                # Get retrieved documents
                retrieved_docs = self.retriever_tool.invoke({"query": query})
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generate response with context
                response_prompt = f"""
                Based on the following context about GPU acceleration, answer the user's question:
                
                Context:
                {context}
                
                Question: {query}
                
                Provide a comprehensive answer focusing on practical GPU acceleration techniques.
                """
                
                response = self.llm_model.invoke([HumanMessage(content=response_prompt)])
                return {"messages": [response]}
            else:
                # Direct response without retrieval
                response = self.llm_model.invoke(state["messages"])
                return {"messages": [response]}
                
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error generating response: {str(e)}")]}
    
    def _decide_to_retrieve(self, state: MessagesState) -> str:
        """Decide whether to retrieve documents or respond directly."""
        last_message = state["messages"][-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "retrieve"
        else:
            return "respond"
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.rag_graph:
            return "RAG system not initialized"
        
        try:
            result = self.rag_graph.invoke({
                "messages": [HumanMessage(content=question)]
            })
            return result["messages"][-1].content
        except Exception as e:
            return f"Error processing query: {str(e)}"
