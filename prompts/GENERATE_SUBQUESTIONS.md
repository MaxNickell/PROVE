## SYSTEM PROMPT:
You are an expert at breaking down an ambiguous comparative question about an image pair into piecewise binary (Yes or No) subquestions using the provided visual context.

## USER PROMPT:
TASK
You will be given:
- An ultimate question about two images
- A caption for each image
- An object list for each image
Given the visual context, you must reason through the ultimate question. Break down the ultimate question into a set of binary subquestions that, when answered, can collectively resolve the ultimate question.

SUBQUESTION CATEGORIES
- **attribute**: Specific visual attributes of objects
- **relationship**: Spatial relationships or interactions
- **scene_attribute**: Visually observable property of the entire scene
- **count**: Which object categories’ counts must be determined

RULES
- Each subquestion must be answerable with Yes or No.
- Attribute questions should specify which attribute class or value must be verified.
- Relationship questions should ask one explicit visual relation.
- Count questions must explicitly ask about the number of objects of a certain class.
- Scene attribute questions must ask an observable, image-level visual property.
- Only reference objects from the object list using their exact IDs in “referenced_objects”.
- The combined subquestions must collectively contain all the information needed to answer the ultimate question.
- Output strict JSON, nothing else.

---

### EXAMPLES

**Example 1**
Ultimate Question: Which scene depicts more power?

IMAGE A
Caption: A king sits on a golden throne in a grand hall surrounded by four guards holding spears. Red carpets line the floor and tall stained glass windows cast colorful light over the crown resting beside him. Three subjects bow before the throne while two musicians stand by holding trumpets.
Objects:
{
  "king_a_0": "king",
  "throne_a_1": "throne",
  "guard_a_2": "guard",
  "guard_a_3": "guard",
  "guard_a_4": "guard",
  "guard_a_5": "guard",
  "spear_a_6": "spear",
  "spear_a_7": "spear",
  "spear_a_8": "spear",
  "spear_a_9": "spear",
  "crown_a_10": "crown",
  "subject_a_11": "subject",
  "subject_a_12": "subject",
  "subject_a_13": "subject"
}

IMAGE B
Caption: A man sits cross-legged on the sidewalk with torn clothes and an empty cup beside him. Two people walk past without looking as a gust of wind scatters some coins near his feet. Behind him, a cracked wall with faded posters leans into shadow.
Objects:
{
  "man_b_0": "man",
  "sidewalk_b_1": "sidewalk",
  "clothing_b_2": "clothing",
  "cup_b_3": "cup",
  "coin_b_4": "coin",
  "coin_b_5": "coin",
  "wall_b_6": "wall",
  "poster_b_7": "poster"
}

Output:
{
  "subquestions": [
    {
      "question": "Is the king sitting on the throne?",
      "subquery_type": "relationship",
      "referenced_objects": ["king_a_0", "throne_a_1"]
    },
    {
      "question": "Is the king wearing the crown?",
      "subquery_type": "relationship",
      "referenced_objects": ["king_a_0", "crown_a_10"]
    },
    {
      "question": "Do the guards appear to be facing or serving the king?",
      "subquery_type": "relationship",
      "referenced_objects": ["guard_a_2", "guard_a_3", "guard_a_4", "guard_a_5", "king_a_0"]
    },
    {
      "question": "Are the subjects bowing toward the king?",
      "subquery_type": "relationship",
      "referenced_objects": ["subject_a_11", "subject_a_12", "subject_a_13", "king_a_0"]
    },
    {
      "question": "How many subjects are there?",
      "subquery_type": "count",
      "referenced_objects": ["subject_a_11", "subject_a_12", "subject_a_13"]
    },
    {
      "question": "How many guards are there?",
      "subquery_type": "count",
      "referenced_objects": ["guard_a_2", "guard_a_3", "guard_a_4", "guard_a_5"]
    },
    {
      "question": "Is the man sitting on the sidewalk?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "sidewalk_b_1"]
    },
    {
      "question": "Is the man wearing clothing?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "clothing_b_2"]
    },
    {
      "question": "Does the clothing appear torn or worn out?",
      "subquery_type": "attribute",
      "referenced_objects": ["clothing_b_2"]
    },
    {
      "question": "Does the man appear to be poor?",
      "subquery_type": "attribute",
      "referenced_objects": ["man_b_0"]
    },
    {
      "question": "Is the man holding or sitting beside a cup for donations (begging)?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "cup_b_3"]
    },
    {
      "question": "Is the environment of image A bright and ornate?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    },
    {
      "question": "Is the environment of image B dimly lit and worn down?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }
  ]
}

---

**Example 2**
Ultimate Question: What is the difference between the two images?

IMAGE A
Caption: Several dogs of different breeds run freely through a sunny dog park. Two chase tennis balls, one leaps through a sprinkler, and three others roll in the grass while two owners watch from benches. Water bowls and toys are scattered across the open field.
Objects:
{
  "dog_a_0": "dog",
  "dog_a_1": "dog",
  "dog_a_2": "dog",
  "dog_a_3": "dog",
  "ball_a_4": "ball",
  "ball_a_5": "ball",
  "sprinkler_a_6": "sprinkler",
  "owner_a_7": "owner",
  "owner_a_8": "owner"
}

IMAGE B
Caption: A crowd of dogs races down a marked track during a dog competition. Four trainers stand at the sidelines holding leashes and stopwatches. A banner with the competition logo waves in the background as spectators cheer from bleachers.
Objects:
{
  "dog_b_0": "dog",
  "dog_b_1": "dog",
  "dog_b_2": "dog",
  "trainer_b_3": "trainer",
  "trainer_b_4": "trainer",
  "track_b_5": "track",
  "leash_b_6": "leash"
}

Output:
{
  "subquestions": [
    {
      "question": "Do any of the dogs appear relaxed?",
      "subquery_type": "attribute",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3"]
    },
    {
      "question": "Do the dogs appear to be playing with each other?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3"]
    },
    {
      "question": "Do the dogs appear to be playing with balls?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3", "ball_a_4", "ball_a_5"]
    },
    {
      "question": "How many dogs are there?",
      "subquery_type": "count",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3", "dog_b_0", "dog_b_1", "dog_b_2"]
    },
    {
      "question": "Are the dogs in image B running on the track?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_b_0", "dog_b_1", "dog_b_2", "track_b_5"]
    },
    {
      "question": "Are the trainers holding leashes?",
      "subquery_type": "relationship",
      "referenced_objects": ["trainer_b_3", "trainer_b_4", "leash_b_6"]
    },
    {
      "question": "Are the dogs in image B competing or racing with each other?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_b_0", "dog_b_1", "dog_b_2"]
    },
    {
      "question": "Is the environment in image A open and natural?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    },
    {
      "question": "Is the environment in image B structured and man-made?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }
  ]
}

---

**Example 3**
Ultimate Question: Which image appears more fair?

IMAGE A
Caption: Five children sit in a circle dividing colorful candies evenly among themselves. Each child smiles and places pieces into small cups. The table is neatly arranged, and everyone receives the same amount. A teacher stands nearby supervising.
Objects:
{
  "child_a_0": "child",
  "child_a_1": "child",
  "child_a_2": "child",
  "child_a_3": "child",
  "child_a_4": "child",
  "candy_a_5": "candy",
  "candy_a_6": "candy",
  "cup_a_7": "cup",
  "cup_a_8": "cup"
}

IMAGE B
Caption: Several animals gather around two water troughs under the sun. Three horses drink from a full container while two goats stand beside an empty trough. A farmer watches from the distance without intervening.
Objects:
{
  "horse_b_0": "horse",
  "horse_b_1": "horse",
  "horse_b_2": "horse",
  "goat_b_3": "goat",
  "goat_b_4": "goat",
  "trough_b_5": "trough",
  "trough_b_6": "trough"
}

Output:
{
  "subquestions": [
    {
      "question": "Do the children each have candies in front of them?",
      "subquery_type": "relationship",
      "referenced_objects": ["child_a_0", "child_a_1", "child_a_2", "child_a_3", "child_a_4", "candy_a_5", "candy_a_6"]
    },
    {
      "question": "How many candies are there?",
      "subquery_type": "count",
      "referenced_objects": ["candy_a_5", "candy_a_6"]
    },
    {
      "question": "How many children are there?",
      "subquery_type": "count",
      "referenced_objects": ["child_a_0", "child_a_1", "child_a_2", "child_a_3", "child_a_4"]
    },
    {
      "question": "Are the horses drinking from the full trough?",
      "subquery_type": "relationship",
      "referenced_objects": ["horse_b_0", "horse_b_1", "horse_b_2", "trough_b_5"]
    },
    {
      "question": "Are the goats standing beside the empty trough?",
      "subquery_type": "relationship",
      "referenced_objects": ["goat_b_3", "goat_b_4", "trough_b_6"]
    },
    {
      "question": "Do the goats appear thirsty or waiting for water?",
      "subquery_type": "attribute",
      "referenced_objects": ["goat_b_3", "goat_b_4"]
    },
    {
      "question": "Is the environment in image A organized?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    },
    {
      "question": "Is the environment in image B dry?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }
  ]
}

---

### NOW BEGIN TASK

IMAGE A
Image Caption: {Caption_A}
Object List: {ID: Object_ID - Object Class: Object_Class}

IMAGE B
Image Caption: {Caption_B}
Object List: {ID: Object_ID - Object Class: Object_Class}

Ultimate Question: {Ultimate_Question}"""
