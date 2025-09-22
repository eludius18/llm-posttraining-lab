"""
Script to create .env file from env_example.txt
"""

import os
import shutil


def create_env_file():
    """
    Create .env file by copying env_example.txt
    """
    example_file = "helpers/env_example.txt"
    env_file = ".env"

    if os.path.exists(env_file):
        print(f" .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != "y":
            print("Cancelled. .env file not created.")
            return False

    if os.path.exists(example_file):
        shutil.copy2(example_file, env_file)
        print(f"Created .env file from {example_file}")
        print(f"You can now edit .env to customize your configuration")
        return True
    else:
        print(f"Error: {example_file} not found!")
        return False


if __name__ == "__main__":
    print("Creating .env file...")
    create_env_file()
