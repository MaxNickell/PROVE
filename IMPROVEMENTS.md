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

## OBJECT DETECTION CALIBRATION
- Current calibration is a fixed monotone transform from logits to probability (not well suited)
- Findings:
    - Florence 2 is by far the best open vocab object detector but no confidences
    - OWL VIT, GroundingDino do not work well for detecting or calibrating
    - BLIP ITM, CLIP, and SIGLIP Calibration after Florence detection not working either


### UNIFIED AGENT
- Sometimes reverifying facts -> could possibly have a list of verified actions
- Sometimes only focusing on one image
- Still sometimes only using perception
- Qwen issue generating bullshit

### ATTR/REL VERIFICATION CALIBRATION

# PROBLOG GENERATION