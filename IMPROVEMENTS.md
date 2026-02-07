# RESEARCH GOALS
- Isolate effect of perception level confidence with deterministic and probabilistic versions
- How do results differ when propagating perception confidence?
    - METHOD 1 (Loss)
        - Deterministic = 1.0, Probabilistic = 1 - p
    - METHOD 2 (Thresholding)
        - Final Answer Threshold for Probabilistic
        - Rounding Threshold for Deterministic

## IMPROVEMENTS NEEDED
### Single Image Support
- Support both single and dual images


### ACTUAL PROBLEMS
- Qwen issue generating bullshit


# Calibration
1. Investigate why failed
    - poor problog generation
2. Investigate why so many at 0
    - Most 0 probabilities are from the lack of object detections
