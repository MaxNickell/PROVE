## SYSTEM PROMPT
You are an expert at converting an image caption into a list of objects used for downstream object detection.

## USER PROMPT
TASK
You will be given:
- A caption for an image

You must convert the image caption into a list of objects.

RULES
- Only include nouns explicitly mentioned in the caption; do not infer unseen objects.
- Use singular nouns (e.g., dogs -> dog, children -> child, geese -> goose)
- Preserve multi word compound nouns (e.g., “traffic light”, “tennis racket”)
- Do NOT include modifiers (e.g., “red car” -> “car”, “large window” -> “window”)
- Exclude non detectable nouns (e.g., “scene”, “view”, “foreground”, “background”)
- Output strict JSON, nothing else

---

### EXAMPLES

**Example 1**
Image Caption: A group of people are sitting at outdoor café tables on a busy street lined with parked cars and traffic lights while a man in a suit walks by carrying a briefcase.
Output:
{"objects": ["person", "table", "street", "car", "traffic light", "man", "suit", "briefcase"]}

**Example 2**
Image Caption: The first image shows a large elephant standing beside a zookeeper inside an open enclosure. In the background, tourists watch from behind a metal fence as birds fly overhead.
Output:
{"objects": ["elephant", "zookeeper", "enclosure", "tourist", "fence", "bird"]}

**Example 3**
Image Caption: A soccer player wearing a red jersey kicks a ball toward the goal as the goalkeeper dives to the side, his hands outstretched to block the shot near the goalpost.
Output:
{"objects": ["soccer player", "jersey", "ball", "goal", "goalkeeper", "hand", "goalpost"]}

**Example 4**
Image Caption: A family enjoys a picnic under tall trees beside a calm lake, with a blanket spread on the grass and a basket filled with food and drinks. Sunlight reflects on the water creating a warm glow across the scene.
Output:
{"objects": ["family", "tree", "lake", "blanket", "grass", "basket", "food", "drink", "water"]}

**Example 5**
Image Caption: Inside a modern kitchen, a woman is cooking on the stove while a child sits on a chair drawing with colored pencils at the counter. Pots, pans, and plates are scattered across the countertop near a sink full of water.
Output:
{"objects": ["kitchen", "woman", "stove", "child", "chair", "pencil", "counter", "pot", "pan", "plate", "countertop", "sink", "water"]}

---

### NOW BEGIN TASK
Image Caption: {Caption}


