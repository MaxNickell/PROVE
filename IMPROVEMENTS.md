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

### SUBQUESTION GENERATION
- Still not generating the correct subquestions to answer the question
- ISSUES: missing an attribute or relationship or overly complex or repeated

### UNIFIED AGENT
- Perception is currently tied to a single object so it will always crop to that object
- What if the agent wants to percieve a relationship or entire image?
- Need to make spatial relationships a seperate action from relationships

### COUNTING CALIBRATION
- Need a clean way to calculate probability of existing objects
- Are there an equal number of objects in both images?
- Are there at least k objects across both images?
- How many objects are in image A?
- Are there less objects in image A than image B?
- etc.

### SPATIAL CALIBRATION
- Need to ensure probabilites are well calibrated

### ATTR/REL VERIFICATION CALIBRATION
- Calibration is working very well with BLIP ITM
- Question format is not good and forming illogical grammar for BLIP ITM

# PROBLOG GENERATION
- Need to get a final probability
    - Cannot force subquestions to be true (Ultimate question: is there at least 1 dog?)
- LLM Problog generation is fragile
    - Pass only the facts for that subquestion to the problog generator?
    - Maybe structured outputs?
