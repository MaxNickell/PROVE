% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).


% entity facts
0.565::entity(image_a, bird_a_0, bird, 196, 96, 270, 202).
0.348::entity(image_a, cattle_a_1, cattle, 94, 185, 400, 595).
0.15::entity(image_a, cattle_a_2, cattle, 0, 134, 214, 400).
0.169::entity(image_a, cattle_a_3, cattle, 139, 29, 408, 202).
0.251::entity(image_a, cattle_a_4, cattle, 0, 83, 242, 210).
0.269::entity(image_b, animal_b_0, animal, 28, 80, 560, 362).
0.394::entity(image_b, bird_b_1, bird, 210, 226, 287, 328).
0.232::entity(image_b, bird_b_2, bird, 293, 35, 340, 99).

% relation facts
0.9349268095742377::relation(image_a, cattle_a_1, bird_a_0, near).
0.9443661954890298::relation(image_a, cattle_a_2, bird_a_0, near).
0.4447272068542485::relation(image_b, bird_b_1, bird_b_2, same_image).

% attribute facts
0.95003003572103::attribute(image_a, bird_a_0, pointed).
0.9727950613327562::attribute(image_a, bird_a_0, rounded).
0.5103949530722968::attribute(image_a, bird_a_0, irregular).
0.9296076237500793::attribute(image_a, bird_a_0, facing_right).
0.8046394468123945::attribute(image_a, bird_a_0, facing_slightly_left).
0.9334031739256955::attribute(image_a, bird_a_0, profile).
0.9327896498419032::attribute(image_a, cattle_a_1, curved).
0.7146104849382701::attribute(image_a, cattle_a_1, pointed).
0.5526677193026593::attribute(image_a, cattle_a_1, oval).
0.9265141193931442::attribute(image_a, cattle_a_1, large).
0.684132195534571::attribute(image_a, cattle_a_1, medium).
0.7035021910914062::attribute(image_a, cattle_a_1, small).
0.93816325107104::attribute(image_a, cattle_a_1, black).
0.6900627741426635::attribute(image_a, cattle_a_1, brown).
0.9134039237746202::attribute(image_a, cattle_a_1, dark).
0.914200835708351::attribute(image_a, cattle_a_1, coarse).
0.7637609907728693::attribute(image_a, cattle_a_1, smooth).
0.9339707016593403::attribute(image_a, cattle_a_1, rough).
0.7055198039093818::attribute(image_a, cattle_a_1, solid).
0.3460535157487628::attribute(image_a, cattle_a_1, mottled).
0.40523805265637974::attribute(image_a, cattle_a_1, spotted).
0.9598660580367058::attribute(image_b, bird_b_1, long).
0.49871069356880343::attribute(image_b, bird_b_1, slender).
0.9470640537748818::attribute(image_b, bird_b_1, curved).
0.7524104872624603::attribute(image_b, bird_b_2, pointed).
0.9769312264464902::attribute(image_b, bird_b_2, rounded).
0.9593948301747462::attribute(image_b, bird_b_2, hooked).

% scene_attr facts
0.9691981049950249::scene_attr(image_b, animals).
