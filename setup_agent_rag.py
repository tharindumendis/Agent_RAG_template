#!/usr/bin/env python3
"""
setup_agent_rag.py — Interactive installer for Agent_rag with dependency selection.

This script helps you choose the right dependency set for your use case:
  - Ollama embeddings (lightweight, remote)
  - Local ONNX embeddings (standalone, CPU-friendly)
  - GPU embeddings with PyTorch (fast, requires NVIDIA)

Usage:
    python setup_agent_rag.py [--non-interactive] [--choice=ollama|local|gpu|all]
"""

import subprocess
import sys
import os
from pathlib import Path

# Choices and their descriptions
CHOICES = {
    "ollama": {
        "name": "🚀 Ollama (Lightweight, Remote Embeddings)",
        "size": "~50 MB",
        "description": "Use Ollama for embeddings. Minimal install, requires Ollama running.",
        "command": "uv install .[ollama]",
        "requirements": "Ollama server running on localhost:11434",
    },
    "local": {
        "name": "💻 Local ONNX (Offline, CPU-Friendly)",
        "size": "~300 MB",
        "description": "Fully self-contained. Embeddings run locally, no external service.",
        "command": "uv install .[local]",
        "requirements": "Modern CPU (fast embeddings in ~1-2s)",
    },
    "gpu": {
        "name": "⚡ GPU (PyTorch + CUDA)",
        "size": "~1.2 GB",
        "description": "GPU-accelerated embeddings. Fastest but largest footprint.",
        "command": "uv install .[gpu]",
        "requirements": "NVIDIA GPU + CUDA 12.1+ drivers",
    },
    "all": {
        "name": "📦 All Features (Development)",
        "size": "~1.5 GB",
        "description": "All providers available. Use for dev/testing multiple setups.",
        "command": "uv install .[all]",
        "requirements": "Disk space & optional NVIDIA GPU",
    },
}

def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("Agent_rag — Dependency Installer")
    print("=" * 70)
    print("\nChoose your embedding provider:")
    print()

def print_options():
    """Print all available options."""
    for i, (key, info) in enumerate(CHOICES.items(), 1):
        print(f"\n  [{i}] {info['name']}")
        print(f"      📊 Size: {info['size']}")
        print(f"      📝 {info['description']}")
        print(f"      ✓  {info['requirements']}")

def print_separator():
    """Print separator."""
    print("\n" + "-" * 70)

def interactive_mode():
    """Run interactive installer."""
    print_header()
    print_options()
    print_separator()
    
    while True:
        choice_input = input("\nEnter your choice (1-4, or 'ollama'/'local'/'gpu'/'all'): ").strip().lower()
        
        # Map number to choice
        if choice_input in ["1", "ollama"]:
            choice = "ollama"
        elif choice_input in ["2", "local"]:
            choice = "local"
        elif choice_input in ["3", "gpu"]:
            choice = "gpu"
        elif choice_input in ["4", "all"]:
            choice = "all"
        else:
            print("❌ Invalid choice. Please try again.")
            continue
        
        break
    
    return choice

def confirm_choice(choice):
    """Confirm user's choice before proceeding."""
    info = CHOICES[choice]
    print_separator()
    print(f"\n✓ You chose: {info['name']}")
    print(f"  📊 Size: {info['size']}")
    print(f"  ✓  Requirements: {info['requirements']}")
    print_separator()
    
    confirm = input("\nProceed with installation? (y/n): ").strip().lower()
    return confirm in ["y", "yes"]

def install(choice):
    """Run the installation command."""
    info = CHOICES[choice]
    command = info["command"]
    
    print(f"\n🔄 Installing: {command}")
    print(f"   (This may take a few minutes...)\n")
    
    # Run the command
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        print_separator()
        print(f"\n✅ Installation complete!")
        print(f"\n📋 Next steps:")
        print(f"   1. Start the RAG server: python Agent_rag/server.py")
        
        if choice == "ollama":
            print(f"   2. Ensure Ollama is running: ollama serve")
            print(f"   3. (Optional) Pull model: ollama pull nomic-embed-text")
        
        print(f"\n📖 For detailed configuration, see: INSTALLATION.md")
        print_separator()
    else:
        print_separator()
        print(f"\n❌ Installation failed (exit code: {result.returncode})")
        print(f"   Please check the error messages above.")
        sys.exit(1)

def main():
    """Main entry point."""
    # Check if uv is available
    if subprocess.run(["uv", "--version"], capture_output=True).returncode != 0:
        print("❌ Error: 'uv' command not found.")
        print("   Please install uv first: https://docs.astral.sh/uv/getting-started/")
        sys.exit(1)
    
    # Check if in correct directory
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print(f"❌ Error: pyproject.toml not found in {os.getcwd()}")
        print("   Please run this script from the Agent_rag directory.")
        sys.exit(1)
    
    # Get choice from command line or interactive
    if len(sys.argv) > 1:
        if "--non-interactive" in sys.argv:
            # Use --choice parameter
            for arg in sys.argv[1:]:
                if arg.startswith("--choice="):
                    choice = arg.split("=")[1].lower()
                    if choice not in CHOICES:
                        print(f"❌ Invalid choice: {choice}")
                        sys.exit(1)
                    break
            else:
                print("❌ --non-interactive requires --choice parameter")
                sys.exit(1)
        else:
            # Try first argument as choice
            choice = sys.argv[1].lower()
            if choice not in CHOICES:
                print(f"❌ Invalid choice: {choice}")
                print("Usage: python setup_agent_rag.py [ollama|local|gpu|all]")
                sys.exit(1)
    else:
        choice = interactive_mode()
    
    # Confirm and install
    if confirm_choice(choice):
        install(choice)
    else:
        print("\n⏭️  Installation cancelled.")
        sys.exit(0)

if __name__ == "__main__":
    main()
