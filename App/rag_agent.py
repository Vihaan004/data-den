from typing import Dict, List, Any
import re
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
import socket
from config import OLLAMA_BASE_URL, CHAT_MODEL, CODE_MODEL, LLM_TEMPERATURE, OLLAMA_PORT

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class RAGAgent:
    """Core RAG agent for GPU acceleration knowledge."""
    
    def __init__(self):
        self.chat_llm_model = None  # For general chat and RAG
        self.code_llm_model = None  # For code analysis and optimization
        self.retriever_tool = None
        self.rag_graph = None
        self.conversation_memory = []  # Store conversation history for context
        self.max_history_length = 10  # Keep last 10 exchanges to avoid token limits
        self._setup_llm()
    
    def clear_conversation_memory(self):
        """Clear the conversation memory."""
        self.conversation_memory = []
        print("🧹 Conversation memory cleared")
    
    def add_to_conversation_memory(self, user_message: str, assistant_response: str):
        """Add an exchange to conversation memory."""
        self.conversation_memory.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the last max_history_length exchanges
        if len(self.conversation_memory) > self.max_history_length:
            self.conversation_memory = self.conversation_memory[-self.max_history_length:]
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation history for context."""
        if not self.conversation_memory:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, exchange in enumerate(self.conversation_memory, 1):
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
            if i < len(self.conversation_memory):
                context_parts.append("---")
        
        return "\n".join(context_parts)

    def _setup_llm(self):
        """Initialize both LLM models - chat model and code model."""
        try:
            # Try different connection methods for supercomputer environment
            import socket
            host_node = socket.gethostname()
            
            # Setup Chat Model (for general conversation and RAG)
            try:
                self.chat_llm_model = ChatOllama(
                    model=CHAT_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=f"http://vpatel69@{host_node}:{OLLAMA_PORT}/"  # Use configurable port
                )
                print(f"✅ Chat LLM model ({CHAT_MODEL}) initialized (supercomputer style)")
            except:
                # Fallback to standard connection
                self.chat_llm_model = ChatOllama(
                    model=CHAT_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=OLLAMA_BASE_URL
                )
                print(f"✅ Chat LLM model ({CHAT_MODEL}) initialized (standard)")
            
            # Setup Code Model (for code analysis and optimization)
            try:
                self.code_llm_model = ChatOllama(
                    model=CODE_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=f"http://vpatel69@{host_node}:{OLLAMA_PORT}/"  # Use configurable port
                )
                print(f"✅ Code LLM model ({CODE_MODEL}) initialized (supercomputer style)")
            except:
                # Fallback to standard connection
                self.code_llm_model = ChatOllama(
                    model=CODE_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=OLLAMA_BASE_URL
                )
                print(f"✅ Code LLM model ({CODE_MODEL}) initialized (standard)")
                
        except Exception as e:
            print(f"⚠️ Could not initialize LLM models: {e}")
            self.chat_llm_model = None
            self.code_llm_model = None
    
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
        if not self.chat_llm_model:
            return {"messages": [AIMessage(content="Chat LLM model not available. Please check Ollama connection.")]}
        
        try:
            response = self.chat_llm_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
            # Clean up response content
            content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
            response.content = content
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return {"messages": [AIMessage(content=f"Error generating response. Please check if Ollama is running with the chat model {CHAT_MODEL}")]}
    
    def _grade_documents(self, state: MessagesState):
        """Grade retrieved documents for relevance."""
        # For simplicity, assume all retrieved documents are relevant
        # In production, implement proper document grading
        return {"messages": state["messages"]}
    
    def _generate_response(self, state: MessagesState):
        """Generate final response based on retrieved documents."""
        if not self.chat_llm_model:
            return {"messages": [AIMessage(content="Chat LLM model not available. Please check Ollama connection.")]}
        
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
                
                response = self.chat_llm_model.invoke([HumanMessage(content=response_prompt)])
                return {"messages": [response]}
            else:
                # Direct response without retrieval
                response = self.chat_llm_model.invoke(state["messages"])
                return {"messages": [response]}
                
        except Exception as e:
            print(f"Error in response generation: {e}")
            return {"messages": [AIMessage(content=f"Error generating response. Please check if Ollama is running with the chat model {CHAT_MODEL}")]}
    
    def _decide_to_retrieve(self, state: MessagesState) -> str:
        """Decide whether to retrieve documents or respond directly."""
        last_message = state["messages"][-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "retrieve"
        else:
            return "respond"
    
    def query(self, question: str, use_conversation_context: bool = True) -> str:
        """Query the RAG system with a question, optionally including conversation context."""
        try:
            print(f"DEBUG: RAGAgent.query called with question length: {len(question)}")
            
            # Determine query type and handle appropriately
            query_type = self._classify_query(question)
            print(f"DEBUG: Query classified as: {query_type}")
            
            # Store the original question for memory
            original_question = question
            conversation_context_used = False
            
            # Add conversation context if enabled and available
            if use_conversation_context and self.conversation_memory:
                conversation_context = self.get_conversation_context()
                question = f"{conversation_context}\n\nCurrent question: {question}"
                conversation_context_used = True
                print(f"DEBUG: Added conversation context, total length: {len(question)}")
            
            response = None
            
            if query_type == "general_chat":
                # Handle general conversation without retrieval - use direct LLM
                print("DEBUG: Handling as general chat")
                response = self._handle_general_chat(question)
                print(f"DEBUG: General chat result length: {len(response) if response else 0}")
            elif query_type == "code_analysis":
                # Handle code analysis with the dedicated code model
                print("DEBUG: Handling as code analysis")
                response = self.query_code_analysis(question)
                print(f"DEBUG: Code analysis result length: {len(response) if response else 0}")
            elif query_type == "gpu_question":
                # Handle GPU-specific questions with RAG retrieval
                print("DEBUG: Handling as GPU question with RAG")
                if not self.rag_graph:
                    print("ERROR: RAG graph not initialized")
                    response = "RAG system not initialized for GPU queries"
                else:
                    print("DEBUG: Invoking RAG graph")
                    result = self.rag_graph.invoke({
                        "messages": [HumanMessage(content=question)]
                    })
                    print(f"DEBUG: RAG graph completed, result messages: {len(result.get('messages', []))}")
                    response = result["messages"][-1].content
                    print(f"DEBUG: Final response length: {len(response) if response else 0}")
            else:
                # This shouldn't happen with our current classification, but fallback to general chat
                print("DEBUG: Fallback to general chat")
                response = self._handle_general_chat(question)
            
            # Add this exchange to conversation memory
            if use_conversation_context and response:
                self.add_to_conversation_memory(original_question, response)
            
            return response
                
        except Exception as e:
            print(f"ERROR in query processing: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {str(e)}"
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query to determine appropriate handling."""
        question_lower = question.lower().strip()
        
        # Check if question contains code blocks - these should be analyzed with code model
        if "```" in question:
            return "code_analysis"
        
        # Check for explicit code analysis patterns with actual code content
        code_analysis_patterns = [
            r'what does this code do',
            r'explain this code',
            r'analyze.*code',
            r'what is this code',
            r'how does this code work',
            r'what does.*code.*do',
            r'optimize.*code',
            r'accelerate.*code',
            r'make.*code.*faster',
            r'convert.*code'
        ]
        
        # Only classify as code analysis if the pattern matches AND there's actual code context
        for pattern in code_analysis_patterns:
            if re.search(pattern, question_lower):
                # Check if there's code-like content in the question
                if any(indicator in question for indicator in ['import ', 'def ', 'class ', 'np.', 'pd.', 'for ', 'if ', '=', '()', 'numpy', 'pandas', 'sklearn']):
                    return "code_analysis"
                # If no code content but asking about code, treat as general chat
                else:
                    return "general_chat"
        
        # GPU/technical patterns - Check for GPU-specific questions (without code)
        gpu_patterns = [
            r'\bgpu acceleration\b',
            r'\bcuda programming\b',
            r'\bcupy\b',
            r'\bcudf\b',
            r'\bcuml\b',
            r'\brapids\b',
            r'\bnvidia\b',
            r'\boptimize.*gpu\b',
            r'\baccelerating.*gpu\b',
            r'\bgpu.*performance\b',
            r'\bconvert.*cupy\b',
            r'\bconvert.*cudf\b',
            r'\bparallel.*gpu\b',
            r'\bgpu.*computing\b',
            r'\bconvert.*to.*gpu\b',
            r'\bmake.*faster.*gpu\b'
        ]
        
        # Check for GPU-specific questions
        for pattern in gpu_patterns:
            if re.search(pattern, question_lower):
                return "gpu_question"
        
        # If question mentions specific libraries in a general context (not code analysis)
        if any(lib in question_lower for lib in ['numpy', 'pandas', 'sklearn']) and not any(code_word in question_lower for code_word in ['analyze', 'optimize', 'convert', 'explain']):
            return "gpu_question"
        
        # Everything else should be general chat
        return "general_chat"
    
    def _handle_general_chat(self, question: str) -> str:
        """Handle general conversation without document retrieval."""
        print(f"DEBUG: _handle_general_chat called with question length: {len(question)}")
        
        # Extract the current question if conversation context was added
        current_question = question
        if "Current question:" in question:
            current_question = question.split("Current question:")[-1].strip()
        
        question_lower = current_question.lower().strip()
        
        # Check if this is a follow-up question that references previous conversation
        is_followup = any(phrase in question_lower for phrase in [
            'what about', 'how about', 'and what', 'can you also', 'what if',
            'tell me more', 'explain more', 'go into more detail', 'elaborate',
            'that', 'this', 'it', 'them', 'those', 'these'
        ])
        
        # If it's a follow-up and we have conversation context, use LLM for context-aware response
        if is_followup and "Previous conversation:" in question:
            print("DEBUG: Detected follow-up question with context, using LLM")
            try:
                if self.chat_llm_model:
                    context_prompt = f"""You are GPU Mentor, a friendly AI assistant specialized in GPU acceleration with NVIDIA Rapids libraries. 

The user is asking a follow-up question based on our previous conversation. Please provide a helpful, context-aware response that builds upon what we've already discussed.

{question}

Provide a clear, helpful answer that acknowledges the conversation context. If the question relates to GPU acceleration, provide specific guidance. Keep your response conversational and informative."""
                    
                    response = self.chat_llm_model.invoke([HumanMessage(content=context_prompt)])
                    return response.content
                else:
                    return f"""I'd be happy to help with your follow-up question: "{current_question}"

However, I need my language model to be available to provide context-aware responses. Please check if Ollama is running with the chat model.

In the meantime, could you rephrase your question to be more specific? I specialize in GPU acceleration with NVIDIA Rapids libraries."""
            except Exception as e:
                print(f"Error in follow-up LLM: {e}")
                return f"""I understand you're asking a follow-up question: "{current_question}"

However, I'm having trouble accessing my language model for context-aware responses. Could you please rephrase your question more specifically? 

I specialize in GPU acceleration with NVIDIA Rapids libraries and can help with:
- Converting Python code to use GPU libraries (CuPy, cuDF, cuML)
- Performance optimization techniques
- GPU acceleration best practices"""
        
        # Handle predefined responses for common queries (without context dependency)
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
            # Check if there's actual code in the question
            if "```" in question or any(lib in question for lib in ['import ', 'def ', 'class ', 'numpy', 'pandas', 'sklearn', 'np.', 'pd.']):
                # This should have been classified as code_analysis, but handle it here as fallback
                return self.query_code_analysis(question)
            else:
                return """I'd be happy to explain code for you! However, I don't see any code in your message. 

Please paste the Python code you'd like me to explain, and I'll:
- Describe what the code does step by step
- Identify the libraries and functions being used
- Explain the logic and data flow
- Suggest any potential GPU acceleration opportunities

You can paste code directly in the chat or use the "Code Analysis & Optimization" tab for detailed analysis."""
        
        elif any(phrase in question_lower for phrase in ['thank you', 'thanks']):
            return "You're welcome! I'm here to help with your GPU acceleration journey. Feel free to ask more questions or share code for optimization!"
        
        elif any(phrase in question_lower for phrase in ['bye', 'goodbye']):
            return "Goodbye! Feel free to come back anytime you need help with GPU acceleration. Happy coding! 🚀"
        
        elif 'how are you' in question_lower:
            return "I'm doing great and ready to help you accelerate your Python code with GPUs! What would you like to work on today?"
        
        # Handle specific knowledge questions
        elif re.search(r'what is (an? )?llm', question_lower):
            return """An **LLM** stands for **Large Language Model**. It's a type of artificial intelligence model that has been trained on vast amounts of text data to understand and generate human-like text.

Key characteristics of LLMs:
- **Large Scale**: Trained on billions or trillions of text tokens
- **Transformer Architecture**: Most modern LLMs use transformer neural networks
- **Generative**: Can create new text based on prompts
- **Versatile**: Can perform various tasks like writing, coding, analysis, and conversation

Popular examples include GPT (like ChatGPT), Claude, Llama, and others.

As GPU Mentor, I use LLM capabilities to help analyze your code and provide GPU acceleration recommendations! Would you like to know how I can help optimize your Python code for GPU acceleration?"""
        
        elif re.search(r'what is (artificial intelligence|ai)', question_lower):
            return """**Artificial Intelligence (AI)** is the simulation of human intelligence in machines that are programmed to think and learn like humans.

Key aspects of AI:
- **Machine Learning**: Algorithms that improve automatically through experience
- **Deep Learning**: Neural networks with multiple layers
- **Natural Language Processing**: Understanding and generating human language
- **Computer Vision**: Interpreting and analyzing visual information

AI is heavily used in GPU acceleration because:
- Training neural networks requires massive parallel computation
- GPUs excel at the matrix operations used in AI/ML
- NVIDIA Rapids libraries (like cuML) accelerate many AI algorithms

Would you like to learn how to accelerate your AI/ML code using GPU libraries like cuML?"""
        
        elif re.search(r'what is (machine learning|ml)', question_lower):
            return """**Machine Learning (ML)** is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed for every task.

Types of Machine Learning:
- **Supervised Learning**: Learning from labeled data (classification, regression)
- **Unsupervised Learning**: Finding patterns in unlabeled data (clustering, dimensionality reduction)
- **Reinforcement Learning**: Learning through interaction and rewards

ML algorithms that benefit greatly from GPU acceleration:
- **Neural Networks**: Deep learning models
- **Clustering**: K-means, DBSCAN
- **Regression**: Linear, logistic regression
- **Tree-based models**: Random Forest, XGBoost

As GPU Mentor, I can help you accelerate your ML code using cuML (GPU-accelerated scikit-learn)! Would you like to see how to convert your scikit-learn code to use GPU acceleration?"""
        
        elif re.search(r'what is python', question_lower):
            return """**Python** is a high-level, interpreted programming language known for its simplicity and versatility.

Why Python is popular:
- **Easy to Learn**: Simple, readable syntax
- **Versatile**: Used for web development, data science, AI, automation, and more
- **Rich Ecosystem**: Extensive libraries for almost everything
- **Community**: Large, supportive developer community

Python libraries that can be GPU-accelerated:
- **NumPy** → **CuPy**: Array operations on GPU
- **Pandas** → **cuDF**: DataFrame operations on GPU
- **Scikit-learn** → **cuML**: Machine learning on GPU

As GPU Mentor, I specialize in helping you accelerate your Python code using NVIDIA Rapids! Want to see how to make your Python code run faster on GPUs?"""
        
        else:
            # For other questions, use LLM without RAG retrieval
            try:
                if self.chat_llm_model:
                    response = self.chat_llm_model.invoke([HumanMessage(content=f"""You are GPU Mentor, a friendly AI assistant specialized in GPU acceleration with NVIDIA Rapids libraries. Answer this question in a helpful, conversational way:

{question}

Provide a clear, accurate answer. If it's not directly related to GPU acceleration, give a good general answer but also mention how GPU acceleration might be relevant. Keep your response concise and helpful. Don't use information from GPU documentation unless the question specifically asks about GPU topics.""")])
                    return response.content
                else:
                    return f"""I'd be happy to help answer your question: "{question}"

However, my specialty is GPU acceleration with NVIDIA Rapids libraries. While I can provide general assistance, I'm most knowledgeable about:
- Converting Python code to use GPU libraries (CuPy, cuDF, cuML)
- Performance optimization techniques
- GPU acceleration best practices

Could you let me know if you have any Python code you'd like to optimize for GPU acceleration?"""
            except Exception as e:
                print(f"Error in general chat LLM: {e}")
                return f"""I'd be happy to help with your question: "{question}"

As GPU Mentor, I specialize in GPU acceleration with NVIDIA Rapids libraries. While I can assist with general programming questions, my expertise is in:
- Analyzing Python code for GPU optimization opportunities
- Converting NumPy, Pandas, and scikit-learn code to GPU equivalents
- Performance optimization and best practices

Is there any Python code you'd like me to help optimize for GPU acceleration?"""
    
    def query_code_analysis(self, prompt: str) -> str:
        """Query the code-specific LLM for code analysis and optimization."""
        try:
            print(f"DEBUG: query_code_analysis called with prompt length: {len(prompt)}")
            
            if not self.code_llm_model:
                print("DEBUG: Code model not available, falling back to chat model")
                if self.chat_llm_model:
                    response = self.chat_llm_model.invoke([HumanMessage(content=f"""You are GPU Mentor, an expert in Python and GPU acceleration. {prompt}
                    
Please provide a clear explanation focusing on:
1. What the code does
2. How it works
3. Any GPU acceleration opportunities
4. Suggestions for optimization

Be helpful and educational in your response.""")])
                    return response.content
                else:
                    return "Neither code analysis model nor chat model is available. Please check Ollama connection."
            
            print(f"DEBUG: Using code model {CODE_MODEL} for analysis")
            
            # Enhance the prompt for better code analysis
            enhanced_prompt = f"""You are GPU Mentor, an expert in Python programming and GPU acceleration with NVIDIA Rapids libraries.

{prompt}

Please provide a comprehensive analysis that includes:
1. **Code Explanation**: What the code does and how it works
2. **Current Approach**: The libraries and methods being used
3. **GPU Opportunities**: How this code could benefit from GPU acceleration
4. **Optimization Suggestions**: Specific improvements using CuPy, cuDF, or cuML if applicable
5. **Performance Considerations**: Expected speedups and when GPU acceleration makes sense

Focus on being educational and practical in your response."""
            
            response = self.code_llm_model.invoke([HumanMessage(content=enhanced_prompt)])
            
            print(f"DEBUG: Code analysis response length: {len(response.content) if response.content else 0}")
            return response.content
            
        except Exception as e:
            print(f"ERROR in code analysis: {e}")
            import traceback
            traceback.print_exc()
            return f"Error in code analysis: {str(e)}"
