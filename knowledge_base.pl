% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).


% entity facts
0.906::entity(image_a, field_a_0, field, 2, 4, 443, 637).
0.857::entity(image_a, camera_a_1, camera, 92, 182, 402, 597).
0.874::entity(image_a, buffalo_a_2, buffalo, 93, 182, 402, 597).
0.846::entity(image_a, shoulder_a_3, shoulder, 93, 182, 402, 597).
0.938::entity(image_a, bird_a_4, bird, 196, 96, 270, 202).
0.867::entity(image_a, sky_a_5, sky, 1, 1, 443, 117).
0.858::entity(image_a, head_a_6, head, 97, 181, 354, 370).
0.883::entity(image_a, grass_a_7, grass, 2, 4, 443, 637).
0.881::entity(image_b, camera_b_0, camera, 297, 34, 336, 103).
0.864::entity(image_b, cow_b_1, cow, 27, 79, 559, 358).
0.895::entity(image_b, horns_b_2, horns, 386, 151, 519, 189).
0.895::entity(image_b, horns_b_3, horns, 387, 156, 425, 189).
0.895::entity(image_b, horns_b_4, horns, 484, 151, 518, 186).
0.788::entity(image_b, shoulder_b_5, shoulder, 92, 80, 402, 150).
0.907::entity(image_b, shrubs_b_6, shrubs, 2, 3, 586, 397).
0.895::entity(image_b, egrets_b_7, egrets, 210, 225, 286, 331).
0.895::entity(image_b, egrets_b_8, egrets, 296, 34, 337, 105).
0.871::entity(image_b, bird_b_9, bird, 210, 226, 286, 327).
0.871::entity(image_b, bird_b_10, bird, 293, 35, 340, 98).
0.834::entity(image_b, legs_b_11, legs, 97, 246, 167, 360).
0.908::entity(image_b, grass_b_12, grass, 2, 3, 587, 398).

% relation facts
0.3822470665278754::relation(image_a, buffalo_a_2, grass_a_7, touching).
0.8255236302712332::relation(image_a, buffalo_a_2, grass_a_7, on).
0.6272021909085183::relation(image_a, buffalo_a_2, field_a_0, inside).
0.2475982557557056::relation(image_a, shoulder_a_3, buffalo_a_2, part_of).
0.3204144262091577::relation(image_a, buffalo_a_2, shoulder_a_3, has_part).
0.3413482980297394::relation(image_a, bird_a_4, sky_a_5, above).
0.41506445777018913::relation(image_a, bird_a_4, sky_a_5, inside).
0.9504597623308491::relation(image_a, head_a_6, buffalo_a_2, part_of).
0.92097087904897::relation(image_a, buffalo_a_2, head_a_6, has_part).

% attribute facts
0.8788102136631378::attribute(image_a, field_a_0, green).
0.86136826874415::attribute(image_a, field_a_0, not green).
0.2286794479953697::attribute(image_a, camera_a_1, person).
0.26569370400519043::attribute(image_a, camera_a_1, no_one).
0.2512924723437087::attribute(image_a, camera_a_1, animal).
0.32398685818358564::attribute(image_a, camera_a_1, yes).
0.5173814168198083::attribute(image_a, camera_a_1, no).
0.3517146162201499::attribute(image_a, camera_a_1, camera).
0.44381962653022655::attribute(image_a, camera_a_1, photography equipment).
0.3524859534655921::attribute(image_a, camera_a_1, optical instrument).
0.3846342422414725::attribute(image_a, camera_a_1, imaging device).
0.9227478399463634::attribute(image_a, grass_a_7, green).
0.7928735828401379::attribute(image_a, grass_a_7, brown).
0.958440702336362::attribute(image_a, grass_a_7, yellowish-green).
0.9380389636838466::attribute(image_b, grass_b_12, green).
0.5766632817527407::attribute(image_b, grass_b_12, brown).
0.5379932333905569::attribute(image_b, camera_b_0, DSLR).
0.5379932333905569::attribute(image_b, camera_b_0, DSLR).
0.5379932333905569::attribute(image_b, camera_b_0, DSLR).
0.6222609579948615::attribute(image_b, camera_b_0, Action Camera).
0.5817044427555914::attribute(image_b, camera_b_0, Surveillance Camera).
0.533147011306608::attribute(image_b, camera_b_0, Point-and-Shoot Camera).
0.5503081551222349::attribute(image_b, camera_b_0, DSLR Camera).

% scene_attr facts
0.9519086852316345::scene_attr(image_a, blue).
0.9387277804147227::scene_attr(image_b, blue).
