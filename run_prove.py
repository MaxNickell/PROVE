#!/usr/bin/env python3
"""
PROVE: Probabilistic Reasoning Over Visual Evidence
Simple test script.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.prove import PROVE

# =============================================================================
# Configuration
# =============================================================================

IMAGE_A = "test_images/dev-473-3-img0.png"
IMAGE_B = "test_images/dev-473-3-img1.png"
QUESTION = "Is there a white bird on top of another animal in both images and are there an equal number of birds in image A and image B?"

SAVE_LOGS = True
VERBOSE = True

# =============================================================================
# Run PROVE
# =============================================================================

if __name__ == "__main__":
    model = PROVE(verbose=VERBOSE)

    result = model.predict_with_details(
        image_a_path=IMAGE_A,
        image_b_path=IMAGE_B,
        question=QUESTION,
        save_logs=SAVE_LOGS
    )

    print(f"\n{'='*80}")
    print("ANSWER")
    print('='*80)
    print(result['answer'])
    print('='*80)
