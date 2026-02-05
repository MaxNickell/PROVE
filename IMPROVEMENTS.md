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
- Make sure counts are only on object classes and not non existent object classes or objects and attributes (X)
- I believe relationships are restricted to spatial only right now (X)
- Make sure it grabs every attribute, every relationship necessary to be sure (X)
- Are string relationships or string attributes fine? ()
- Percieve needs to be used when deciding which facts it should verify and to collect image context (X)
- Sometimes reverifying facts -> could possibly have a list of verified actions
- Sometimes only focusing on one image
- Remove all temperature values


### ATTR/REL VERIFICATION CALIBRATION
- Calibration is working very well with BLIP ITM
- Question format is not good and forming illogical grammar for BLIP ITM

# PROBLOG GENERATION
- LLM Problog generation is fragile
    - Pass only the facts for that subquestion to the problog generator?
    - Maybe structured outputs?

    - Do we really need the sugar rules?
    - Only use the listed facts and do not try to use facts that dont exist. If a fact is missing than just omit it
    - Most form the correct logical sequence

