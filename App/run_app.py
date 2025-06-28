#!/usr/bin/env python3
"""
GPU Mentor Application Runner

This script starts the GPU Mentor application with all necessary components.
Make sure you have installed all requirements and have Ollama running.

Requirements:
1. Install Python dependencies: pip install -r requirements.txt
2. Start Ollama server: ollama serve
3. Make sure you have the qwen2.5:14b model: ollama pull qwen2.5:14b

Usage:
    python run_app.py [--port PORT] [--share] [--host HOST]
"""

import argparse
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'gradio',
        'langchain_core',
        'langchain_community',
        'langchain_ollama',
        'langgraph',
        'sentence_transformers',
        'transformers',
        'torch',
        'pydantic',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        from config import OLLAMA_PORT
        response = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("⚠️ Ollama server responded with unexpected status")
            return False
    except Exception as e:
        print(f"❌ Ollama server is not accessible: {e}")
        print("Please start Ollama server:")
        print("  ollama serve")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run GPU Mentor Application")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--skip-checks", action="store_true", help="Skip requirement checks")
    
    args = parser.parse_args()
    
    print("🚀 Starting GPU Mentor Application...")
    
    if not args.skip_checks:
        print("🔍 Checking requirements...")
        
        if not check_requirements():
            sys.exit(1)
        
        if not check_ollama():
            print("⚠️ Continuing without Ollama (some features may not work)")
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from app import GPUMentorApp
        
        print(f"🌐 Starting application...")
        if args.share:
            print("🔗 Creating public shareable link...")
        
        app = GPUMentorApp()
        app.launch(share=args.share)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
