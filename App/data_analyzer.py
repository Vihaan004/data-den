import os
import glob
import pandas as pd
import gradio as gr
import base64
import io
from PIL import Image
from job_executor import run_data_analysis

class DataAnalyzer:
    """Component for analyzing datasets with GPU-accelerated operations."""
    
    def __init__(self, rag_agent):
        self.rag_agent = rag_agent
        self.current_dataset = None
        self.current_dataset_path = None
        self.dataset_info = None
        
    def get_available_datasets(self):
        """Get list of available CSV datasets from the datasets folder."""
        datasets_dir = "./datasets"
        if not os.path.exists(datasets_dir):
            return []
        
        csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
        return [os.path.basename(f) for f in csv_files]
    
    def load_dataset(self, dataset_file, uploaded_file):
        """Load a dataset from file selection or upload."""
        try:
            if uploaded_file is not None:
                # Use uploaded file
                df = pd.read_csv(uploaded_file.name)
                self.current_dataset_path = uploaded_file.name
                dataset_name = os.path.basename(uploaded_file.name)
            elif dataset_file and dataset_file != "None":
                # Use selected sample dataset
                dataset_path = os.path.join("./datasets", dataset_file)
                df = pd.read_csv(dataset_path)
                self.current_dataset_path = dataset_path
                dataset_name = dataset_file
            else:
                return "❌ Please select a dataset or upload a file."
            
            self.current_dataset = df
            
            # Store dataset info for LLM
            self.dataset_info = {
                "name": dataset_name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head(5).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
            
            return f"✅ Dataset '{dataset_name}' loaded successfully! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns. You can now generate analysis suggestions."
            
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}"
    
    def generate_suggestions(self):
        """Generate analysis suggestions using the RAG agent."""
        if self.current_dataset is None or self.dataset_info is None:
            return "❌ Please load a dataset first."
        
        try:
            # Prepare simple prompt for LLM
            prompt = f"""Based on this dataset, suggest exactly 5 specific data analysis tasks that would benefit from GPU acceleration:

Dataset: {self.dataset_info['name']}
Columns: {', '.join(self.dataset_info['columns'])}
Numeric columns: {', '.join(self.dataset_info['numeric_columns'])}
Categorical columns: {', '.join(self.dataset_info['categorical_columns'])}
Sample data: {pd.DataFrame(self.dataset_info['sample_data']).to_string()}

Provide ONLY a numbered list of 5 simple analysis tasks that could be performed on this dataset to gain insights, with clear, specific titles. No explanations, no descriptions, just the analysis names. For example:
1. K-Means Customer Segmentation
2. Sales Correlation Analysis
3. Employee Performance Prediction
4. Salary Distribution Analysis
5. Department Efficiency Clustering"""

            # Try direct LLM call first, bypass RAG if it's causing issues
            print(f"DEBUG: Sending simplified prompt to RAG agent, length: {len(prompt)}")
            try:
                # Try to use the chat model directly for more reliable responses
                if hasattr(self.rag_agent, 'chat_llm_model') and self.rag_agent.chat_llm_model:
                    from langchain_core.messages import HumanMessage
                    response = self.rag_agent.chat_llm_model.invoke([HumanMessage(content=prompt)])
                    response = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = self.rag_agent.query(prompt)
            except Exception as e:
                print(f"DEBUG: LLM call failed: {e}")
                response = ""
            
            print(f"DEBUG: RAG agent response length: {len(response) if response else 0}")
            print(f"DEBUG: RAG agent response preview: {response[:200] if response else 'None'}...")
            
            # If still empty, provide a simple fallback
            if not response or len(response.strip()) == 0:
                print("DEBUG: Using simple fallback suggestions")
                response = self._get_simple_fallback_suggestions()
            
            return f"## 🔍 Suggested GPU-Accelerated Analyses\n\n{response}"
            
        except Exception as e:
            return f"❌ Error generating suggestions: {str(e)}"
    
    def generate_code(self, selected_suggestion):
        """Generate Python code for the selected analysis suggestion."""
        if self.current_dataset is None or self.dataset_info is None:
            return "❌ Please load a dataset first.", ""
        
        if not selected_suggestion.strip():
            return "❌ Please provide a specific analysis topic or method.", ""
        
        try:
            # Prepare detailed prompt for code generation
            dataset_summary = f"""
Dataset Name: {self.dataset_info['name']}
Shape: {self.dataset_info['shape'][0]} rows × {self.dataset_info['shape'][1]} columns
Columns: {', '.join(self.dataset_info['columns'])}
Numeric Columns: {', '.join(self.dataset_info['numeric_columns'])}
Categorical Columns: {', '.join(self.dataset_info['categorical_columns'])}

Sample Data:
{pd.DataFrame(self.dataset_info['sample_data']).to_string()}
"""
            
            prompt = f"""Generate a complete Python script for data analysis. The analysis task is: "{selected_suggestion}"

Dataset Information:
{dataset_summary}

SOL ENVIRONMENT SPECIFICATIONS:
- Python 3.12.9 with conda-forge
- RAPIDS 25.02 (cuDF 25.02.02, cuML 25.02.01, CuPy 13.4.1)
- CUDA 12.8 with NVIDIA A100 GPUs (80GB VRAM each)
- pandas 2.3.0, numpy 1.26.4, matplotlib 3.10.1, sklearn 1.7.0
- CRITICAL: cuDF cannot handle mixed data types in single operations
- CRITICAL: cuDF Series cannot be used directly with matplotlib
- GPU memory pool available but must be managed properly

IMPORTANT INSTRUCTIONS:
- The script will run in the directory: /home/vpatel69/R1/App/output/
- Load the dataset using: df = pd.read_csv('../datasets/{self.dataset_info['name']}')
- Import ALL necessary libraries at the top (pandas, cudf, cuml, cupy, matplotlib, numpy, sklearn, etc.)
- Write a complete analysis script with proper data loading, analysis, and visualization
- Use GPU libraries (cuDF, cuML, CuPy) when appropriate for acceleration
- Include detailed print statements showing progress and results
- Create and save visualizations using matplotlib (save to current directory with plt.savefig(), NOT plt.show())
- DO NOT use try-except blocks (error handling is done by the execution environment)
- Focus on GPU-accelerated operations that showcase performance benefits

CRITICAL SOL ENVIRONMENT CONSTRAINTS:
- cuDF CANNOT create DataFrames with mixed data types (strings + numbers in same operation)
- ALWAYS separate string/categorical columns from numeric operations
- cuDF DataFrame creation will fail with "Cannot create column with mixed types" error
- Solution: Create cuDF only with numeric columns, handle strings separately

GPU LIBRARY USAGE GUIDELINES:
- Use cudf.DataFrame for GPU DataFrames, NOT cuml.DataFrame (cuML has no DataFrame class)
- For correlations: use gdf.corr() ONLY on numeric columns, filter first: gdf.select_dtypes(include=['number']).corr()
- For machine learning: import specific algorithms like 'from cuml.linear_model import LinearRegression'
- For clustering: 'from cuml.cluster import KMeans'
- For dimensionality reduction: 'from cuml.decomposition import PCA'
- Always convert pandas to cuDF: gdf = cudf.from_pandas(df)
- IMPORTANT: cuDF operations work best with numeric data, filter out string/object columns before GPU operations

DATA HANDLING BEST PRACTICES:
- Always select numeric columns before correlations: gdf.select_dtypes(include=['number'])
- String/object columns (like names, departments) should be handled separately or converted to categorical
- Check data types before GPU operations: print(gdf.dtypes)
- Use gdf.describe(include='all') to see all column statistics

MATPLOTLIB PLOTTING WITH cuDF:
- cuDF Series/DataFrames cannot be used directly with matplotlib
- Convert to numpy for plotting: gdf['column'].to_numpy() or gdf['column'].values
- For machine learning with cuML, convert to cupy arrays: gdf['column'].to_cupy()
- Example: plt.scatter(gdf['x'].to_numpy(), gdf['y'].to_numpy())

Generate a complete, standalone Python script:

EXAMPLE STRUCTURE:
```python
import pandas as pd
import cudf
import matplotlib.pyplot as plt
import numpy as np
from cuml.linear_model import LinearRegression  # Import specific algorithms

# Load dataset
df = pd.read_csv('../datasets/{self.dataset_info['name']}')
print(f"Dataset loaded: {{df.shape}}")

# CRITICAL: Separate numeric and string columns for Sol environment
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
string_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Numeric columns: {{numeric_columns}}")
print(f"String columns: {{string_columns}}")

# Convert ONLY numeric data to cuDF (mixed types will fail on Sol)
numeric_df = df[numeric_columns]
gdf = cudf.from_pandas(numeric_df)
print("Numeric data converted to cuDF format")

# Your analysis code here
# Use gdf.corr() for correlations (only numeric data in gdf)
# Use proper cuML imports for machine learning algorithms  
# For plotting: use gdf['column'].to_numpy() to convert cuDF to numpy
# Handle string columns separately with pandas if needed
# Create visualizations with plt.savefig('filename.png') NOT plt.show()

print("Analysis completed successfully")
```"""
            
            # Get code from RAG agent
            code_response = self.rag_agent.query(prompt)
            
            # Clean up the response to extract just the code
            code = self._extract_code_from_response(code_response)
            
            # Generate explanation
            explanation_prompt = f"""Provide a clear, educational explanation for this GPU-accelerated data analysis code:

Analysis Topic: {selected_suggestion}
Dataset: {self.dataset_info['name']}

Explain:
1. What this analysis does and what insights it provides
2. Which GPU acceleration techniques are used and why
3. Expected performance benefits compared to CPU
4. Key libraries and functions being utilized

Keep the explanation concise but informative, suitable for someone learning about GPU acceleration in data science."""
            
            explanation = self.rag_agent.query(explanation_prompt)
            
            return code, explanation
            
        except Exception as e:
            return f"❌ Error generating code: {str(e)}", ""
    
    def _extract_code_from_response(self, response):
        """Extract clean Python code from LLM response and validate syntax."""
        if not response:
            return "# No code generated\nprint('Error: No code was generated')"
        
        # Remove markdown code blocks if present
        code = response
        if "```python" in response:
            # Extract content between ```python and ```
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                code = response[start:end].strip()
            else:
                code = response[start:].strip()
        elif "```" in response:
            # Extract content between ``` blocks
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                code = response[start:end].strip()
            else:
                code = response[start:].strip()
        else:
            code = response.strip()
        
        # Basic syntax validation and cleanup
        try:
            # Try to compile the code to check for syntax errors
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError as e:
            print(f"DEBUG: Syntax error in generated code: {e}")
            # Return a safe fallback code that is a complete script
            return f"""# Fallback analysis script
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('../datasets/{self.dataset_info['name']}')
print(f"Dataset loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

# Basic dataset analysis
print("\\nDataset Info:")
print(df.info())

print("\\nDataset Columns:")
print(df.columns.tolist())

print("\\nSummary Statistics:")
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())

# Create visualizations
if len(numeric_cols) > 0:
    # Distribution plots
    plt.figure(figsize=(12, 8))
    df[numeric_cols].hist(bins=20, alpha=0.7)
    plt.tight_layout()
    plt.savefig('data_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data distribution plot saved as 'data_distributions.png'")
    
    # Correlation heatmap if we have multiple numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Correlation matrix plot saved as 'correlation_matrix.png'")

print("Fallback analysis completed successfully")"""
    
    def execute_analysis(self, code):
        """Execute the analysis code using the job executor."""
        if self.current_dataset is None or self.current_dataset_path is None:
            return "❌ Please load a dataset first.", None
        
        if not code.strip():
            return "❌ Please generate code first.", None
        
        try:
            # Run the analysis job
            result = run_data_analysis(code, self.current_dataset_path, "./output")
            
            # Always show results with static header (success/failure determined by warnings in error file)
            execution_time_text = f"{result['execution_time']:.3f} seconds" if result.get('execution_time') else "Unknown"
            
            output_text = f"""## 📊 Analysis Results

### Output:
```
{result['output']}
```
"""
            
            # Handle plots
            plot_images = []
            if result.get("plots"):
                for i, plot_data in enumerate(result["plots"]):
                    try:
                        # Decode base64 plot data
                        image_data = base64.b64decode(plot_data)
                        image = Image.open(io.BytesIO(image_data))
                        plot_images.append(image)
                    except Exception as e:
                        print(f"Error processing plot {i}: {e}")
            
            return output_text, plot_images[0] if plot_images else None
                
        except Exception as e:
            return f"❌ Error executing analysis: {str(e)}", None
    
    def create_interface(self):
        """Create the Gradio interface for data analysis."""
        gr.Markdown("# 📊 Data Analysis")
        gr.Markdown("Upload datasets and perform GPU-accelerated analysis with AI-generated suggestions.")
        
        # Top Row: Dataset Selection and AI Suggestions
        with gr.Row():
            # Left Column: Dataset Selection
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Dataset Selection")
                dataset_dropdown = gr.Dropdown(
                    choices=["None"] + self.get_available_datasets(),
                    label="Sample Datasets",
                    value="None"
                )
                dataset_upload = gr.File(
                    label="Or Upload CSV File",
                    file_types=[".csv"]
                )
                load_btn = gr.Button("📊 Load Dataset", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
            
            # Right Column: AI Suggestions
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 AI-Generated Analysis Suggestions")
                suggestions_btn = gr.Button("🔍 Generate Suggestions", variant="secondary", size="lg")
                suggestions_output = gr.Markdown("Load a dataset and click 'Generate Suggestions' to get AI recommendations.", elem_classes=["suggestions-box"])
        
        gr.Markdown("---")
        
        # Code Generation Section
        with gr.Row():
            with gr.Column():
                selected_topic = gr.Textbox(
                    label="Selected Analysis Topic",
                    placeholder="Enter or copy one of the suggested analysis topics above...",
                    lines=2
                )
                generate_code_btn = gr.Button("⚡ Generate Analysis", variant="secondary")
        
        # Generated Code and Explanation
        with gr.Row():
            with gr.Column(scale=2):
                generated_code = gr.Code(
                    label="📝 Generated Python Code",
                    language="python",
                    lines=20
                )
            with gr.Column(scale=1):
                code_explanation = gr.Markdown("### Analysis Overview:")
        
        # Execution Section
        with gr.Row():
            execute_btn = gr.Button("🚀 Execute Analysis", variant="primary", size="lg")
        
        # Results Section
        with gr.Row():
            with gr.Column(scale=2):
                execution_output = gr.Markdown("### Execute analysis to see results")
            with gr.Column(scale=1):
                plot_output = gr.Image(label="📈 Generated Plots")
    
        
        # Event handlers
        load_btn.click(
            fn=self.load_dataset,
            inputs=[dataset_dropdown, dataset_upload],
            outputs=[load_status]
        )
        
        suggestions_btn.click(
            fn=self.generate_suggestions,
            inputs=[],
            outputs=[suggestions_output]
        )
        
        generate_code_btn.click(
            fn=self.generate_code,
            inputs=[selected_topic],
            outputs=[generated_code, code_explanation]
        )
        
        execute_btn.click(
            fn=self.execute_analysis,
            inputs=[generated_code],
            outputs=[execution_output, plot_output]
        )
    
    def _get_simple_fallback_suggestions(self):
        """Provide simple fallback suggestions when LLM fails."""
        if self.dataset_info is None:
            return "❌ No dataset information available."
        
        # Analyze dataset characteristics to provide relevant suggestions
        numeric_cols = len(self.dataset_info['numeric_columns'])
        categorical_cols = len(self.dataset_info['categorical_columns'])
        
        suggestions = []
        
        # Generate simple topic names based on dataset structure
        if numeric_cols >= 2:
            suggestions.append("1. K-Means Clustering Analysis")
        
        if numeric_cols >= 3:
            suggestions.append("2. Correlation Matrix Analysis")
        
        if numeric_cols >= 2:
            suggestions.append("3. Linear Regression Analysis")
        
        if numeric_cols >= 4:
            suggestions.append("4. Principal Component Analysis (PCA)")
        
        if categorical_cols >= 1 and numeric_cols >= 1:
            suggestions.append("5. Statistical Aggregations by Groups")
        
        # Ensure we always have 5 suggestions
        if len(suggestions) < 5:
            suggestions.append("5. Data Distribution Analysis")
        
        return "\n".join(suggestions[:5])  # Return exactly 5 suggestions
