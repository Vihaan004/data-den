#!/usr/bin/env python3
"""
Simple GPU Mentor App Launcher

This script provides a simple way to launch the GPU Mentor application.
"""

import sys
import os
import subprocess
import signal
import atexit

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main launcher function."""
    print("🚀 GPU Mentor Application Launcher")
    print("=" * 40)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(lambda: print("👋 GPU Mentor application has been shut down."))
    
    # Check if requirements are installed
    try:
        import gradio
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        if not install_requirements():
            sys.exit(1)
    
    # Check if Ollama is available
    try:
        import requests
        import socket
        from config import OLLAMA_PORT
        
        # Try supercomputer-style connection first
        host_node = socket.gethostname()
        try:
            response = requests.get(f"http://vpatel69@{host_node}:{OLLAMA_PORT}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running (supercomputer style)")
            else:
                print("⚠️ Ollama server responded with unexpected status")
        except:
            # Try standard connection
            try:
                response = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama server is running (standard)")
                else:
                    print("⚠️ Ollama server responded with unexpected status")
            except:
                print(f"⚠️ Ollama server not detected on port {OLLAMA_PORT} (some features may be limited)")
    except:
        print("⚠️ Could not check Ollama status")
    
    # Launch the app
    print("\n🌐 Starting GPU Mentor Application...")
    print("💡 Tip: Use Ctrl+C to stop the application cleanly")
    
    app = None
    try:
        # Import and run the app
        from app import GPUMentorApp
        app = GPUMentorApp()
        print("🎉 Application ready! Opening in your browser...")
        
        # Launch with proper error handling - let Gradio find an available port
        app.launch(
            share=True,
            server_name="0.0.0.0",
            quiet=False,
            show_error=True,
            inbrowser=False  # Don't auto-open browser to avoid issues
        )
        
    except KeyboardInterrupt:
        print("\n� Keyboard interrupt received (Ctrl+C)")
        print("�👋 Shutting down GPU Mentor application...")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all requirements are installed: pip install -r requirements.txt")
        print("2. Start Ollama server: ollama serve")
        print("3. Install Ollama model: ollama pull qwen2.5-coder:14b")
        
    finally:
        # Ensure clean shutdown
        if app and hasattr(app, 'close'):
            try:
                app.close()
            except:
                pass
        print("🔄 Application cleanup complete")
        sys.exit(0)

if __name__ == "__main__":
    main()
