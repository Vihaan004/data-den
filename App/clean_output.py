#!/usr/bin/env python3
"""
Clean Output Directory Script

This script removes all files from the output directory while preserving subdirectories.
Useful for cleaning up generated plots, analysis results, and temporary files.

Usage:
    python clean_output.py
    
Options:
    --dry-run    Show what would be deleted without actually deleting
    --verbose    Show detailed information about each file
    --help       Show this help message
"""

import os
import sys
import glob
import argparse
from pathlib import Path

def clean_output_directory(output_dir="./output", dry_run=False, verbose=False):
    """
    Clean all files from the output directory.
    
    Args:
        output_dir (str): Path to the output directory
        dry_run (bool): If True, only show what would be deleted
        verbose (bool): If True, show detailed information
    
    Returns:
        tuple: (files_deleted, total_size_freed)
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory '{output_dir}' does not exist.")
        return 0, 0
    
    if not output_path.is_dir():
        print(f"❌ '{output_dir}' is not a directory.")
        return 0, 0
    
    # Find all files (not directories) in the output directory
    all_files = []
    for item in output_path.iterdir():
        if item.is_file():
            all_files.append(item)
    
    if not all_files:
        print(f"✅ Output directory '{output_dir}' is already clean (no files found).")
        return 0, 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in all_files)
    
    print(f"🗂️  Found {len(all_files)} files in '{output_dir}'")
    if verbose or dry_run:
        print(f"📊 Total size: {format_size(total_size)}")
        print()
    
    # Group files by extension for better organization
    files_by_ext = {}
    for file_path in all_files:
        ext = file_path.suffix.lower() or 'no extension'
        if ext not in files_by_ext:
            files_by_ext[ext] = []
        files_by_ext[ext].append(file_path)
    
    # Show summary by file type
    if verbose or dry_run:
        print("📋 Files by type:")
        for ext, files in sorted(files_by_ext.items()):
            size = sum(f.stat().st_size for f in files)
            print(f"   {ext}: {len(files)} files ({format_size(size)})")
        print()
    
    if dry_run:
        print("🔍 DRY RUN - Files that would be deleted:")
        for file_path in sorted(all_files):
            size = file_path.stat().st_size
            print(f"   📄 {file_path.name} ({format_size(size)})")
        print(f"\n📊 Total: {len(all_files)} files, {format_size(total_size)}")
        return 0, 0
    
    # Actually delete the files
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    for file_path in all_files:
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()  # Delete the file
            deleted_count += 1
            deleted_size += file_size
            
            if verbose:
                print(f"   ✅ Deleted: {file_path.name} ({format_size(file_size)})")
                
        except Exception as e:
            error_msg = f"Failed to delete {file_path.name}: {str(e)}"
            errors.append(error_msg)
            if verbose:
                print(f"   ❌ {error_msg}")
    
    # Show results
    print(f"🧹 Cleanup completed!")
    print(f"   ✅ Deleted: {deleted_count} files")
    print(f"   💾 Freed: {format_size(deleted_size)}")
    
    if errors:
        print(f"   ⚠️  Errors: {len(errors)}")
        if verbose:
            for error in errors:
                print(f"      {error}")
    
    return deleted_count, deleted_size

def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_bytes = float(size_bytes)
    i = 0
    
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clean all files from the output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_output.py                    # Clean the output directory
    python clean_output.py --dry-run          # Show what would be deleted
    python clean_output.py --verbose          # Show detailed information
    python clean_output.py --dry-run --verbose # Detailed dry run
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./output",
        help="Path to the output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Show detailed information about each file"
    )
    
    args = parser.parse_args()
    
    print("🚀 GPU Mentor Output Directory Cleaner")
    print("=" * 40)
    
    try:
        deleted_count, deleted_size = clean_output_directory(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if not args.dry_run and deleted_count > 0:
            print(f"\n🎉 Successfully cleaned output directory!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
