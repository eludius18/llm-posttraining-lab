#!/usr/bin/env python3
"""
Setup script for SFT Pipeline
Automatically creates virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        print(f"Command: {command}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function"""
    print("SFT Pipeline Setup")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        sys.exit(1)

    print(f"Python {sys.version.split()[0]} detected")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Create virtual environment
    venv_path = Path("sft_env")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() == "y":
            print(" Removing existing environment...")
            import shutil

            shutil.rmtree(venv_path)
        else:
            print("Using existing environment")
            return

    if not run_command(
        f"{sys.executable} -m venv sft_env", "Creating virtual environment"
    ):
        sys.exit(1)

    # Determine activation script
    if platform.system() == "Windows":
        activate_script = "sft_env\\Scripts\\activate"
        pip_path = "sft_env\\Scripts\\pip"
    else:
        activate_script = "sft_env/bin/activate"
        pip_path = "sft_env/bin/pip"

    # Install dependencies
    requirements_path = Path("../../requirements.txt")
    if not requirements_path.exists():
        print("requirements.txt not found")
        return 1

    if not run_command(
        f"{pip_path} install -r {requirements_path}", "Installing dependencies"
    ):
        return 1

    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate environment: source {activate_script}")
    print("2. Create .env file: python create_env.py")
    print("3. Run pipeline: python __main__.py")
    print("\ Or run everything at once:")
    print(f"   source {activate_script} && python __main__.py")


if __name__ == "__main__":
    main()
