"""
Main entry point for the SFT module.
"""

import sys
import os

# Add helpers directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))

from sft_pipeline import main

if __name__ == "__main__":
    main()
