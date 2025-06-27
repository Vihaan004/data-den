#!/usr/bin/env python3
"""
GPU Mentor Kill Script

This script helps kill any lingering GPU Mentor processes.
Use this when Ctrl+C doesn't properly shut down the application.
"""

import subprocess
import sys
import os
import signal

def kill_gpu_mentor_processes():
    """Kill any lingering GPU Mentor processes."""
    print("🔍 Looking for GPU Mentor processes...")
    
    try:
        # For Unix-like systems (Linux, macOS)
        if os.name != 'nt':
            # Find processes related to our application
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            gpu_mentor_processes = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['launch.py', 'app.py', 'gpu_mentor', 'gradio']):
                    if 'python' in line.lower():
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            gpu_mentor_processes.append(pid)
                            print(f"Found process: PID {pid}")
            
            if gpu_mentor_processes:
                print(f"🛑 Killing {len(gpu_mentor_processes)} GPU Mentor process(es)...")
                for pid in gpu_mentor_processes:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"✅ Killed process {pid}")
                    except:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"✅ Force killed process {pid}")
                        except:
                            print(f"❌ Could not kill process {pid}")
            else:
                print("✅ No GPU Mentor processes found")
        
        else:
            # For Windows
            print("💡 On Windows, use Task Manager or run:")
            print("   tasklist | findstr python")
            print("   taskkill /F /IM python.exe")
            
    except Exception as e:
        print(f"❌ Error finding processes: {e}")
        print("\n💡 Manual cleanup options:")
        print("1. Check running jobs: jobs")
        print("2. Kill background job: kill %1 (replace 1 with job number)")
        print("3. Kill all Python processes: pkill -f python")
        print("4. Find specific processes: ps aux | grep gpu_mentor")

def main():
    """Main function."""
    print("🚫 GPU Mentor Process Killer")
    print("=" * 30)
    
    kill_gpu_mentor_processes()
    
    print("\n💡 To prevent this in the future:")
    print("1. Always use Ctrl+C to stop the application")
    print("2. Wait a few seconds for graceful shutdown")
    print("3. If stuck, use this script or manual process killing")

if __name__ == "__main__":
    main()
