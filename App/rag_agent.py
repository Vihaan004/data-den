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
            # Try different connection methods for supercomputer environment
            import socket
            host_node = socket.gethostname()
            
            # Try the supercomputer-style connection first
            try:
                self.llm_model = ChatOllama(
                    model=OLLAMA_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=f"http://vpatel69@{host_node}:11437/"  # Updated port
                )
                print("✅ LLM model initialized (supercomputer style)")
                return
            except:
                pass
            
            # Fallback to standard connection
            self.llm_model = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                base_url=OLLAMA_BASE_URL
            )
            print("✅ LLM model initialized (standard)")
        except Exception as e:
            print(f"⚠️ Could not initialize LLM: {e}")
            self.llm_model = None
    
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
            return {"messages": [AIMessage(content="LLM model not available. Please check Ollama connection.")]}
        
        try:
            response = self.llm_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
            # Clean up response content
            content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
            response.content = content
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return {"messages": [AIMessage(content=f"Error generating response. Please check if Ollama is running with the model {OLLAMA_MODEL}")]}
    
    def _grade_documents(self, state: MessagesState):
        """Grade retrieved documents for relevance."""
        # For simplicity, assume all retrieved documents are relevant
        # In production, implement proper document grading
        return {"messages": state["messages"]}
    
    def _generate_response(self, state: MessagesState):
        """Generate final response based on retrieved documents."""
        if not self.llm_model:
            return {"messages": [AIMessage(content="LLM model not available. Please check Ollama connection.")]}
        
        try:
            # Get the last message and generate response
            last_message = state["messages"][-1]
            
            # If there are tool calls, handle them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Retrieve documents
                tool_call = last_message.tool_calls[0]
                query = tool_call['args']['query']
                
                # Get retrieved documents
                try:
                    retrieved_docs = self.retriever_tool.invoke({"query": query})
                    print(f"DEBUG: Retrieved docs type: {type(retrieved_docs)}")
                    if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                        print(f"DEBUG: First doc type: {type(retrieved_docs[0])}")
                except Exception as e:
                    print(f"Error retrieving documents: {e}")
                    retrieved_docs = []
                
                # Handle different types of retrieved content
                if isinstance(retrieved_docs, list):
                    # If it's a list of documents or strings
                    context_parts = []
                    for doc in retrieved_docs:
                        if hasattr(doc, 'page_content'):
                            context_parts.append(doc.page_content)
                        elif isinstance(doc, str):
                            context_parts.append(doc)
                        else:
                            context_parts.append(str(doc))
                    context = "\n\n".join(context_parts)
                elif hasattr(retrieved_docs, 'page_content'):
                    # Single document
                    context = retrieved_docs.page_content
                else:
                    # Fallback - convert to string
                    context = str(retrieved_docs)
                
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
            print(f"Error in response generation: {e}")
            return {"messages": [AIMessage(content=f"Error generating response. Please check if Ollama is running with the model {OLLAMA_MODEL}")]}
    
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
            # Determine query type and handle appropriately
            query_type = self._classify_query(question)
            
            if query_type == "general_chat":
                # Handle general conversation without retrieval
                return self._handle_general_chat(question)
            elif query_type == "gpu_question":
                # Handle GPU-specific questions with retrieval
                result = self.rag_graph.invoke({
                    "messages": [HumanMessage(content=question)]
                })
                return result["messages"][-1].content
            else:
                # Default to RAG for technical questions
                result = self.rag_graph.invoke({
                    "messages": [HumanMessage(content=question)]
                })
                return result["messages"][-1].content
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query to determine appropriate handling."""
        question_lower = question.lower().strip()
        
        # General chat patterns
        general_patterns = [
            r'\b(hello|hi|hey|greetings)\b',
            r'\bwho are you\b',
            r'\bwhat are you\b',
            r'\bhow are you\b',
            r'\bwhat can you do\b',
            r'\bhelp\b',
            r'\bthank you\b',
            r'\bthanks\b',
            r'\bbye\b',
            r'\bgoodbye\b',
            r'\bwhat does this code do\b',
            r'\bexplain this code\b',
            r'\bwhat is this\b'
        ]
        
        # Check for general chat patterns
        for pattern in general_patterns:
            if re.search(pattern, question_lower):
                return "general_chat"
        
        # GPU/technical patterns
        gpu_patterns = [
            r'\bgpu\b',
            r'\bcuda\b',
            r'\bcupy\b',
            r'\bcudf\b',
            r'\bcuml\b',
            r'\brapids\b',
            r'\bacceler\w+\b',
            r'\boptimiz\w+\b',
            r'\bperformance\b',
            r'\bspeedup\b',
            r'\bparallel\b'
        ]
        
        # Check for GPU/technical patterns
        for pattern in gpu_patterns:
            if re.search(pattern, question_lower):
                return "gpu_question"
        
        # If question contains code blocks, treat as technical
        if "```" in question or "import " in question:
            return "gpu_question"
        
        # Default to general chat for short, simple questions
        if len(question.split()) < 10 and not any(word in question_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return "general_chat"
        
        return "gpu_question"
    
    def _handle_general_chat(self, question: str) -> str:
        """Handle general conversation without document retrieval."""
        question_lower = question.lower().strip()
        
        # Predefined responses for common queries
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey']):
            return """Hello! I'm GPU Mentor, your AI assistant for GPU acceleration with NVIDIA Rapids libraries. 

I can help you:
- Analyze Python code for GPU optimization opportunities
- Convert NumPy, Pandas, and scikit-learn code to use CuPy, cuDF, and cuML
- Explain GPU acceleration concepts and best practices
- Provide performance optimization recommendations

Feel free to ask me questions about GPU acceleration or paste some code for analysis!"""
        
        elif any(phrase in question_lower for phrase in ['who are you', 'what are you']):
            return """I'm GPU Mentor, an AI-powered assistant specialized in helping developers accelerate their Python code using NVIDIA Rapids libraries.

My expertise includes:
- **CuPy**: GPU acceleration for NumPy operations
- **cuDF**: GPU acceleration for Pandas DataFrames  
- **cuML**: GPU acceleration for machine learning with scikit-learn
- **Performance optimization**: Memory management, data transfer optimization, and best practices

I can analyze your code, suggest optimizations, and help you learn GPU acceleration techniques!"""
        
        elif 'what can you do' in question_lower:
            return """I can help you accelerate your Python code with GPU computing! Here's what I can do:

🔍 **Code Analysis**: Analyze your Python code to identify GPU acceleration opportunities

⚡ **Optimization Suggestions**: Convert NumPy → CuPy, Pandas → cuDF, scikit-learn → cuML

📊 **Performance Estimates**: Predict potential speedups from GPU acceleration

🎓 **Learning Support**: Generate tutorials and answer questions about GPU programming

💡 **Best Practices**: Share memory management tips and optimization techniques

Just paste your Python code or ask me questions about GPU acceleration!"""
        
        elif any(phrase in question_lower for phrase in ['what does this code do', 'explain this code', 'what is this']):
            return """I'd be happy to explain code for you! However, I don't see any code in your message. 

Please paste the Python code you'd like me to explain, and I'll:
- Describe what the code does step by step
- Identify the libraries and functions being used
- Explain the logic and data flow
- Suggest any potential GPU acceleration opportunities

You can paste code directly in the chat or use the "Code to Analyze" box below."""
        
        elif any(phrase in question_lower for phrase in ['thank you', 'thanks']):
            return "You're welcome! I'm here to help with your GPU acceleration journey. Feel free to ask more questions or share code for optimization!"
        
        elif any(phrase in question_lower for phrase in ['bye', 'goodbye']):
            return "Goodbye! Feel free to come back anytime you need help with GPU acceleration. Happy coding! 🚀"
        
        elif 'how are you' in question_lower:
            return "I'm doing great and ready to help you accelerate your Python code with GPUs! What would you like to work on today?"
        
        else:
            # For other general questions, try to be helpful
            if len(question.split()) < 15:  # Short questions
                return f"""I'm here to help with GPU acceleration using NVIDIA Rapids libraries! 

Your question: "{question}"

I specialize in:
- Converting Python code to use GPU libraries (CuPy, cuDF, cuML)
- Performance optimization and best practices
- Explaining GPU acceleration concepts

Could you provide more details about what you'd like to know, or share some code you'd like to optimize?"""
            else:
                # Longer questions - try to answer with LLM but without retrieval
                try:
                    if self.llm_model:
                        response = self.llm_model.invoke([HumanMessage(content=f"""You are GPU Mentor, an AI assistant specialized in GPU acceleration with NVIDIA Rapids libraries. Answer this question in a helpful, friendly way:

{question}

If the question is about GPU acceleration, programming, or code optimization, provide a technical but accessible answer. If it's a general question, respond conversationally while mentioning your specialty in GPU acceleration.""")])
                        return response.content
                    else:
                        return "I'd be happy to help! Could you be more specific about what you'd like to know regarding GPU acceleration or code optimization?"
                except:
                    return "I'd be happy to help! Could you be more specific about what you'd like to know regarding GPU acceleration or code optimization?"
