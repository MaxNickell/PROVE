# Goal
- Isolate perception level confidence
- How do results differ when propagating perception confidence 


## Single Image Support
- Need to support single and dual images


# How do we combine problog results so we can preserve probability
- If we enforce the subquestions needing to be true?
- Some how creating one problog that answers the ultimate question?


## CALIBRATION
- COUNTING: we keep the probability of k out of n existing is the min of the top k
- OBJECT DETECTION: Using weird math right now that is not well suited
- VERIFICATION PROBABLITIES: Not well calibrated with log probs (SigLIP, XVLM, CLIP, etc.)

# PROBLOG GENERATION
- Pass only the facts for that subquestion to the problog generator
- LLM generating problog is fragile
- LLM comparing counts is a mess right now
    - more, less, equal, 2 more, etc.

## UNIFIED AGENT
- Getting stuck in a loop again fuhhhk
- The perception is asking questions with cropped images which is causing the VLM to return answers that dont make sense
    - Are there multiple dogs together in this part of the image
- Need a better strategy for percieving and verification because the current strategy is ass
    - is dog_a_0 chasing dog_a_1: it clearly is but we are getting probability like 0.1

