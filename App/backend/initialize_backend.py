#!/usr/bin/env python3
"""
Initialize GPU Mentor Backend
Standalone script to initialize the RAG pipeline and backend components.
"""
import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_gpu_mentor import EnhancedGPUMentor

async def main():
    """Initialize the GPU Mentor backend."""
    print("🚀 Initializing GPU Mentor Backend...")
    print("=" * 50)
    
    try:
        # Create and initialize the GPU Mentor
        gpu_mentor = EnhancedGPUMentor()
        await gpu_mentor.initialize()
        
        print("✅ GPU Mentor backend initialized successfully!")
        print("🎯 Ready to use with gradio_ui.py")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        print("Please check your dependencies and configuration.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
