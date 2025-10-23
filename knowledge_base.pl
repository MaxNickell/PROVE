% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).


% entity facts
0.857::entity(image_a, camera_a_0, camera, 92, 182, 402, 597).
0.867::entity(image_a, sky_a_1, sky, 1, 1, 443, 117).
0.906::entity(image_a, field_a_2, field, 2, 4, 443, 637).
0.846::entity(image_a, shoulder_a_3, shoulder, 93, 182, 402, 597).
0.874::entity(image_a, buffalo_a_4, buffalo, 93, 182, 402, 597).
0.883::entity(image_a, grass_a_5, grass, 2, 4, 443, 637).
0.858::entity(image_a, head_a_6, head, 97, 181, 354, 370).
0.938::entity(image_a, bird_a_7, bird, 196, 96, 270, 202).
0.864::entity(image_b, cow_b_0, cow, 27, 79, 559, 358).
0.881::entity(image_b, horn_b_1, horn, 387, 156, 425, 189).
0.912::entity(image_b, field_b_2, field, 2, 3, 587, 398).
0.905::entity(image_b, shrub_b_3, shrub, 2, 3, 586, 397).
0.788::entity(image_b, shoulder_b_4, shoulder, 92, 80, 402, 150).
0.908::entity(image_b, grass_b_5, grass, 2, 3, 587, 398).
0.887::entity(image_b, head_b_6, head, 351, 151, 559, 288).
0.886::entity(image_b, egret_b_7, egret, 210, 224, 285, 329).
0.803::entity(image_b, leg_b_8, leg, 97, 247, 167, 360).
0.871::entity(image_b, bird_b_9, bird, 210, 226, 286, 327).
0.871::entity(image_b, bird_b_10, bird, 293, 35, 340, 98).

% relation facts
0.8484208673716215::relation(image_a, bird_a_7, shoulder_a_3, perched_on).
0.48432259135324135::relation(image_b, bird_b_9, shoulder_b_4, perched_on).

% attribute facts
0.6084716415249373::attribute(image_a, buffalo_a_4, brown).
0.6110517647638952::attribute(image_a, buffalo_a_4, large).
0.8574602862986039::attribute(image_b, cow_b_0, large).
0.3338039702585453::attribute(image_b, cow_b_0, brown).
