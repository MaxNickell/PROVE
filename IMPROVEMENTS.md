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
- Perception is currently tied to a single object so it will always crop to that object
- Make sure counts are only on object classes
- Spatial Relationship Calibration?

### ATTR/REL VERIFICATION CALIBRATION
- Calibration is working very well with BLIP ITM
- Question format is not good and forming illogical grammar for BLIP ITM

# PROBLOG GENERATION
- Need to get a final probability
    - Cannot force subquestions to be true (Ultimate question: is there at least 1 dog?)
- LLM Problog generation is fragile
    - Pass only the facts for that subquestion to the problog generator?
    - Maybe structured outputs?
