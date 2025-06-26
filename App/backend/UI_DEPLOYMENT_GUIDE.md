# 🚀 GPU Mentor Frontend & UI Options

## 🎉 **Congratulations! Your Backend is Ready**

Your GPU Mentor backend has successfully passed all tests on Sol and is ready for frontend integration. Here are your options:

## 🎯 **UI Options Available**

### **Option 1: Gradio Web UI (Recommended for Quick Testing)**

**Benefits:**
- ✅ Quick to deploy and test
- ✅ Built-in components for ML applications
- ✅ Works directly on Sol compute nodes
- ✅ No complex setup required

**Deploy on Sol:**
```bash
# 1. Submit the UI job
sbatch start_ui.slurm

# 2. Check the job output for the URL
tail -f gpu_mentor_ui_*.out

# 3. Access via SSH tunnel (see instructions below)
```

### **Option 2: FastAPI Web Interface**

**Benefits:**
- ✅ Full REST API for external integration
- ✅ Custom HTML/CSS/JavaScript interface
- ✅ Production-ready architecture
- ✅ Easy to integrate with external frontend

**Deploy on Sol:**
```bash
# Start the FastAPI server
python fastapi_ui.py
# Access via: http://compute_node_ip:8000
```

### **Option 3: Connect to External Frontend**

**Benefits:**
- ✅ Full control over UI/UX
- ✅ Modern frontend frameworks (React, Vue, etc.)
- ✅ Mobile-responsive design
- ✅ Advanced features and integrations

## 📡 **Network Access Methods**

### **Method 1: SSH Tunneling (Most Common)**

```bash
# From your local machine:
ssh -L 7860:compute_node:7860 username@sol.asu.edu

# Then access: http://localhost:7860
```

### **Method 2: VPN Access (If Available)**

```bash
# If ASU provides VPN access to Sol network:
# Access directly: http://compute_node_ip:7860
```

### **Method 3: Port Forwarding Setup**

```bash
# Set up port forwarding through Sol login node
ssh -L 7860:localhost:7860 username@sol.asu.edu
# Then from Sol: ssh -L 7860:compute_node:7860 compute_node
```

## 🛠️ **Step-by-Step Deployment**

### **Quick Start with Gradio UI:**

1. **Submit the UI job:**
   ```bash
   cd ~/R1/App/backend/
   sbatch start_ui.slurm
   ```

2. **Get the compute node info:**
   ```bash
   # Check job status
   squeue -u $USER
   
   # Get compute node name from job output
   tail -f gpu_mentor_ui_*.out
   ```

3. **Set up SSH tunnel:**
   ```bash
   # From your local machine (replace NODE_NAME):
   ssh -L 7860:NODE_NAME:7860 username@sol.asu.edu
   ```

4. **Access the UI:**
   - Open browser to: `http://localhost:7860`
   - You should see the GPU Mentor interface!

### **Production Deployment with FastAPI:**

1. **Create a production SLURM job:**
   ```bash
   # Create: production_ui.slurm
   #!/bin/bash
   #SBATCH --job-name=gpu_mentor_prod
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=08:00:00
   
   module load genai25.06
   cd ~/R1/App/backend/
   export PYTHONPATH=$PWD:$PYTHONPATH
   
   python fastapi_ui.py
   ```

2. **Submit and access:**
   ```bash
   sbatch production_ui.slurm
   # Access via: http://compute_node:8000
   ```

## 🔌 **API Integration Examples**

### **For External Frontend Integration:**

```javascript
// Example API calls to your backend
const API_BASE = 'http://sol-compute-node:8000';

// Optimize code
async function optimizeCode(code, question = '') {
    const response = await fetch(`${API_BASE}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, question })
    });
    return await response.json();
}

// Get sample code
async function getSampleCode(category, operation) {
    const response = await fetch(`${API_BASE}/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category, operation })
    });
    return await response.json();
}

// Chat with mentor
async function askMentor(question) {
    const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });
    return await response.json();
}
```

### **Python Client Example:**

```python
import requests

API_BASE = "http://sol-compute-node:8000"

# Optimize code
def optimize_code(code, question=""):
    response = requests.post(f"{API_BASE}/optimize", 
                           json={"code": code, "question": question})
    return response.json()

# Example usage
code = """
import numpy as np
result = []
for i in range(1000):
    result.append(i * 2)
"""

optimized = optimize_code(code, "How to vectorize this loop?")
print(optimized['optimized_code'])
```

## 🎨 **Frontend Framework Integration**

### **React.js Integration:**

```jsx
import React, { useState } from 'react';

function CodeOptimizer() {
    const [code, setCode] = useState('');
    const [optimized, setOptimized] = useState('');
    
    const handleOptimize = async () => {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });
        const result = await response.json();
        setOptimized(result.optimized_code);
    };
    
    return (
        <div className="code-optimizer">
            <h2>GPU Mentor - Code Optimizer</h2>
            <textarea value={code} onChange={(e) => setCode(e.target.value)} />
            <button onClick={handleOptimize}>Optimize</button>
            <textarea value={optimized} readOnly />
        </div>
    );
}
```

### **Vue.js Integration:**

```vue
<template>
  <div class="gpu-mentor">
    <h2>GPU Mentor</h2>
    <textarea v-model="code" placeholder="Enter your code..."></textarea>
    <button @click="optimize">Optimize Code</button>
    <textarea v-model="optimizedCode" readonly></textarea>
  </div>
</template>

<script>
export default {
  data() {
    return {
      code: '',
      optimizedCode: ''
    }
  },
  methods: {
    async optimize() {
      const response = await this.$http.post('/api/optimize', {
        code: this.code
      });
      this.optimizedCode = response.data.optimized_code;
    }
  }
}
</script>
```

## 🚀 **Next Steps**

### **Immediate (Next 30 minutes):**
1. ✅ Test Gradio UI on Sol
2. ✅ Verify all backend functions work
3. ✅ Test SSH tunneling setup

### **Short Term (Next few hours):**
1. 🔄 Set up FastAPI for production use
2. 🔄 Create external frontend (React/Vue)
3. 🔄 Add user authentication
4. 🔄 Implement job queueing

### **Long Term (Next few days):**
1. 📊 Add advanced analytics dashboard
2. 🔐 Implement security measures
3. 📱 Create mobile-responsive design
4. ⚡ Add real-time collaboration features

## 🆘 **Troubleshooting**

### **UI Won't Start:**
```bash
# Check if Gradio is installed
pip install gradio --user

# Check job logs
tail -f gpu_mentor_ui_*.err
```

### **Can't Access UI:**
```bash
# Verify compute node name
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"

# Test SSH tunnel
ssh -v -L 7860:compute_node:7860 username@sol.asu.edu
```

### **Backend Errors:**
```bash
# Check backend status
python -c "
from core.enhanced_gpu_mentor import EnhancedGPUMentor
mentor = EnhancedGPUMentor()
print('Backend OK')
"
```

## 📋 **Available Endpoints (FastAPI)**

- `GET /` - Main web interface
- `POST /optimize` - Code optimization
- `POST /sample` - Get sample code
- `POST /benchmark` - Run benchmarks
- `POST /chat` - Chat with mentor
- `GET /health` - Health check

Your GPU Mentor is now ready for production use on Sol! 🎉
