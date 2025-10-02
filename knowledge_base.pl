% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).


% entity facts
0.938::entity(image_a, bird_a_0, bird, 196, 96, 270, 202).
0.883::entity(image_a, grass_a_1, grass, 2, 4, 443, 637).
0.858::entity(image_a, head_a_2, head, 97, 181, 354, 370).
0.867::entity(image_a, sky_a_3, sky, 1, 1, 443, 117).
0.857::entity(image_a, camera_a_4, camera, 92, 182, 402, 597).
0.906::entity(image_a, field_a_5, field, 2, 4, 443, 637).
0.874::entity(image_a, buffalo_a_6, buffalo, 93, 182, 402, 597).
0.846::entity(image_a, shoulder_a_7, shoulder, 93, 182, 402, 597).
0.871::entity(image_b, bird_b_0, bird, 210, 226, 286, 327).
0.871::entity(image_b, bird_b_1, bird, 293, 35, 340, 98).
0.908::entity(image_b, grass_b_2, grass, 2, 3, 587, 398).
0.887::entity(image_b, head_b_3, head, 351, 151, 559, 288).
0.907::entity(image_b, shrubs_b_4, shrubs, 2, 3, 586, 397).
0.881::entity(image_b, camera_b_5, camera, 297, 34, 336, 103).
0.864::entity(image_b, cow_b_6, cow, 27, 79, 559, 358).
0.788::entity(image_b, shoulder_b_7, shoulder, 92, 80, 402, 150).
0.912::entity(image_b, field_b_8, field, 2, 3, 587, 398).
0.895::entity(image_b, horns_b_9, horns, 386, 151, 519, 189).
0.895::entity(image_b, horns_b_10, horns, 387, 156, 425, 189).
0.895::entity(image_b, horns_b_11, horns, 484, 151, 518, 186).
0.867::entity(image_b, image_b_12, image, 3, 5, 584, 396).
0.895::entity(image_b, egrets_b_13, egrets, 210, 225, 286, 331).
0.895::entity(image_b, egrets_b_14, egrets, 296, 34, 337, 105).
0.834::entity(image_b, legs_b_15, legs, 97, 246, 167, 360).

% relation facts
0.4109205049685359::relation(image_a, camera_a_4, bird_a_0, near).

% attribute facts
0.9296076237500793::attribute(image_a, bird_a_0, facing_right).
0.8046394468123945::attribute(image_a, bird_a_0, facing_slightly_left).
0.9334031739256955::attribute(image_a, bird_a_0, profile).
0.883092448827513::attribute(image_a, head_a_2, facing_camera).
0.7840098697217348::attribute(image_a, head_a_2, facing_slightly_left).
0.6596617205635571::attribute(image_a, head_a_2, profile).
0.9227478399463634::attribute(image_a, grass_a_1, green).
0.9295663421426746::attribute(image_a, grass_a_1, similar green).
0.7353917931467832::attribute(image_a, grass_a_1, same green).
0.9420508659695352::attribute(image_a, grass_a_1, coarse).
0.8230528673590459::attribute(image_a, grass_a_1, smooth).
0.8332960057632293::attribute(image_a, grass_a_1, dense).
0.8719515501568575::attribute(image_b, head_b_3, facing_camera).
0.6705099483513486::attribute(image_b, head_b_3, profile).
0.8581424461539788::attribute(image_b, head_b_3, slightly_turned_away).
0.9380389636838466::attribute(image_b, grass_b_2, green).
0.5766632817527407::attribute(image_b, grass_b_2, brown).
0.5833275347137609::attribute(image_b, grass_b_2, yellow).
0.690956555861926::attribute(image_b, grass_b_2, smooth).
0.8945922131575793::attribute(image_b, grass_b_2, coarse).
0.6679780915096143::attribute(image_b, grass_b_2, dense).

% scene_attr facts
0.9519086852316345::scene_attr(image_a, blue).
0.96521526212605::scene_attr(image_a, field).
0.9387277804147227::scene_attr(image_b, blue).
0.9501788748833543::scene_attr(image_b, field).

% count facts
0.062000000000000055::count(image_a, bird, 0).
0.938::count(image_a, bird, 1).
0.016641::count(image_b, bird, 0).
0.224718::count(image_b, bird, 1).
0.758641::count(image_b, bird, 2).
