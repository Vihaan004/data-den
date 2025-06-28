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
                return "❌ Please select a dataset or upload a file.", "", ""
            
            self.current_dataset = df
            
            # Generate dataset info
            info = f"""## 📊 Dataset: {dataset_name}
            
**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns
**Size:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

**Columns:**
{self._format_column_info(df)}

**Data Types:**
{df.dtypes.to_string()}

**Missing Values:**
{df.isnull().sum().to_string()}
"""
            
            # Generate sample data preview
            sample_data = df.head(10).to_string(max_cols=10, max_colwidth=30)
            
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
            
            return info, sample_data, "✅ Dataset loaded successfully! You can now generate analysis suggestions."
            
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}", "", ""
    
    def _format_column_info(self, df):
        """Format column information nicely."""
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            column_info.append(f"• **{col}** ({dtype}) - {unique_count:,} unique, {null_count:,} nulls")
        return "\\n".join(column_info)
    
    def generate_suggestions(self):
        """Generate analysis suggestions using the RAG agent."""
        if self.current_dataset is None or self.dataset_info is None:
            return "❌ Please load a dataset first."
        
        try:
            # Prepare prompt for LLM
            dataset_summary = f"""
Dataset Name: {self.dataset_info['name']}
Shape: {self.dataset_info['shape'][0]} rows × {self.dataset_info['shape'][1]} columns
Columns: {', '.join(self.dataset_info['columns'])}
Numeric Columns: {', '.join(self.dataset_info['numeric_columns'])}
Categorical Columns: {', '.join(self.dataset_info['categorical_columns'])}
Data Types: {self.dataset_info['dtypes']}

Sample Data (first 5 rows):
{pd.DataFrame(self.dataset_info['sample_data']).to_string()}
"""
            
            prompt = f"""I need help with GPU acceleration for data analysis. Please analyze this dataset and suggest 5-7 specific data analysis operations that would be particularly effective with GPU acceleration using libraries like CuPy, cuDF, cuML, and RAPIDS. Focus on operations that would showcase GPU performance benefits over CPU implementations.

{dataset_summary}

For each suggestion, provide:
1. A clear, specific title (e.g. "K-Means Clustering Analysis", "Time Series Correlation Analysis")
2. A brief description of what insights it would provide
3. Why GPU acceleration with RAPIDS/CuPy/cuDF would be beneficial
4. What specific GPU computing advantages it would demonstrate

Format your response as a numbered list with clear titles and descriptions. Focus on practical, executable analyses that would work well with this specific dataset's structure and content and would benefit from GPU computing power."""

            # Get suggestions from RAG agent
            print(f"DEBUG: Sending prompt to RAG agent, length: {len(prompt)}")
            response = self.rag_agent.query(prompt)
            print(f"DEBUG: RAG agent response length: {len(response) if response else 0}")
            print(f"DEBUG: RAG agent response preview: {response[:200] if response else 'None'}...")
            
            return f"## 🔍 Suggested GPU-Accelerated Analyses\\n\\n{response}"
            
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
            
            prompt = f"""Generate a complete Python program for GPU-accelerated data analysis based on this request: "{selected_suggestion}"

Dataset Information:
{dataset_summary}

Requirements:
1. Write a complete, executable Python program optimized for GPU using RAPIDS libraries (cuDF, cuML, CuPy)
2. The dataset is already loaded as 'df' (pandas DataFrame) - convert to cuDF as needed
3. Include data visualization using matplotlib (save plots, don't show them)
4. Add clear comments explaining each step
5. Include performance-oriented code that showcases GPU acceleration benefits
6. Handle potential data issues gracefully
7. Include print statements to show intermediate results and insights

Focus on practical, working code that demonstrates GPU acceleration advantages. Make sure the code is complete and can run independently.

Provide ONLY the Python code without any markdown formatting or explanation text - just clean, executable Python code."""
            
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
        """Extract clean Python code from LLM response."""
        # Remove markdown code blocks if present
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
        
        return code
    
    def execute_analysis(self, code):
        """Execute the analysis code using the job executor."""
        if self.current_dataset is None or self.current_dataset_path is None:
            return "❌ Please load a dataset first.", None, ""
        
        if not code.strip():
            return "❌ Please generate code first.", None, ""
        
        try:
            # Run the analysis job
            result = run_data_analysis(code, self.current_dataset_path, "./output")
            
            if result["success"]:
                output_text = f"""## ✅ Analysis Completed Successfully

**Execution Time:** {result['execution_time']:.3f} seconds
**Job ID:** {result['job_id']}

### Output:
```
{result['output']}
```
"""
                
                # Handle plots
                plot_images = []
                if result["plots"]:
                    for i, plot_data in enumerate(result["plots"]):
                        try:
                            # Decode base64 plot data
                            image_data = base64.b64decode(plot_data)
                            image = Image.open(io.BytesIO(image_data))
                            plot_images.append(image)
                        except Exception as e:
                            print(f"Error processing plot {i}: {e}")
                
                return output_text, plot_images[0] if plot_images else None, ""
            else:
                error_text = f"""## ❌ Analysis Failed

**Job ID:** {result.get('job_id', 'N/A')}

### Error:
```
{result['error']}
```

### Output:
```
{result['output']}
```
"""
                return error_text, None, ""
                
        except Exception as e:
            return f"❌ Error executing analysis: {str(e)}", None, ""
    
    def create_interface(self):
        """Create the Gradio interface for data analysis."""
        gr.Markdown("# 📊 Data Analysis")
        gr.Markdown("Upload datasets and perform GPU-accelerated analysis with AI-generated suggestions.")
        
        # Dataset Selection Section
        with gr.Row():
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
            
            with gr.Column(scale=2):
                dataset_info = gr.Markdown("### Dataset Info\\nNo dataset loaded.")
                load_status = gr.Textbox(label="Status", interactive=False)
        
        # Dataset Preview
        with gr.Row():
            dataset_preview = gr.Textbox(
                label="📋 Dataset Preview (First 10 rows)",
                lines=10,
                interactive=False
            )
        
        gr.Markdown("---")
        
        # Analysis Suggestions Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 AI-Generated Analysis Suggestions")
                suggestions_btn = gr.Button("🔍 Generate Suggestions", variant="secondary")
                suggestions_output = gr.Markdown("Click 'Generate Suggestions' to get AI recommendations.")
        
        # Code Generation Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💻 Code Generation")
                selected_topic = gr.Textbox(
                    label="Selected Analysis Topic",
                    placeholder="Enter or copy one of the suggested analysis topics above...",
                    lines=2
                )
                generate_code_btn = gr.Button("⚡ Generate GPU Code", variant="secondary")
        
        # Generated Code and Explanation
        with gr.Row():
            with gr.Column(scale=2):
                generated_code = gr.Code(
                    label="📝 Generated Python Code",
                    language="python",
                    lines=20
                )
            with gr.Column(scale=1):
                code_explanation = gr.Markdown("### Code Explanation\\nGenerate code to see explanation.")
        
        # Execution Section
        with gr.Row():
            execute_btn = gr.Button("🚀 Execute Analysis", variant="primary", size="lg")
        
        # Results Section
        with gr.Row():
            with gr.Column(scale=2):
                execution_output = gr.Markdown("### Execution Results\\nExecute code to see results.")
            with gr.Column(scale=1):
                plot_output = gr.Image(label="📈 Generated Plots")
        
        execution_status = gr.Textbox(label="Execution Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=self.load_dataset,
            inputs=[dataset_dropdown, dataset_upload],
            outputs=[dataset_info, dataset_preview, load_status]
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
            outputs=[execution_output, plot_output, execution_status]
        )
