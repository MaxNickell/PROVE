% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).


% entity facts
0.906::entity(image_a, sidewalk_a_0, sidewalk, 2, 99, 597, 454).
0.929::entity(image_a, dog_a_1, dog, 55, 96, 545, 391).
0.877::entity(image_a, leg_a_2, leg, 261, 275, 315, 392).
0.9::entity(image_a, bandage_a_3, bandage, 274, 299, 313, 373).
0.873::entity(image_a, leash_a_4, leash, 239, 119, 597, 200).
0.879::entity(image_a, hook_a_5, hook, 256, 119, 269, 136).
0.861::entity(image_a, harness_a_6, harness, 195, 129, 336, 290).
0.858::entity(image_a, sign_a_7, sign, 477, 42, 598, 184).
0.902::entity(image_a, trash can_a_8, trash can, 16, 1, 378, 316).
0.884::entity(image_b, dog_b_0, dog, 60, 0, 157, 176).
0.906::entity(image_b, shrub_b_1, shrub, 4, 0, 231, 176).
0.903::entity(image_b, knee pad_b_2, knee pad, 86, 90, 103, 122).
0.822::entity(image_b, camera_b_3, camera, 101, 39, 141, 63).
0.856::entity(image_b, leg_b_4, leg, 113, 113, 139, 176).
0.872::entity(image_b, collar_b_5, collar, 101, 39, 140, 62).
0.921::entity(image_b, lawn_b_6, lawn, 5, 0, 231, 176).
0.879::entity(image_b, leash_b_7, leash, 101, 39, 231, 74).
0.863::entity(image_b, buckle_b_8, buckle, 120, 113, 139, 144).
0.873::entity(image_b, harness_b_9, harness, 101, 39, 141, 63).
0.919::entity(image_b, tree_b_10, tree, 0, 0, 6, 176).

% relation facts
0.8657495654653493::relation(image_a, dog_a_1, sidewalk_a_0, standing_on).
0.4629105151003525::relation(image_a, dog_a_1, bandage_a_3, wearing).
0.8762266468636289::relation(image_b, dog_b_0, lawn_b_6, sitting_on).
0.8897339016205222::relation(image_b, dog_b_0, knee pad_b_2, wearing).

% attribute facts
0.9082656495999696::attribute(image_a, dog_a_1, Jack Russell Terrier mix).
0.9712152514509116::attribute(image_b, dog_b_0, slender).
0.9668637748858844::attribute(image_b, dog_b_0, long).
0.978306336185776::attribute(image_b, dog_b_0, narrow).

% scene_attr facts
0.9240867879977541::scene_attr(image_a, outdoor).
0.9102819360855492::scene_attr(image_b, outdoor).
