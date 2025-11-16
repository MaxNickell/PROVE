% PROVE Pipeline - Unified ProbLog Program
% Generated for 6 subquestions
% Ultimate question: Are there more birds in image A than in image B and are all birds orange?

% Facts from visual evidence
% entity facts
0.874::entity(image_a, buffalo_a_0, buffalo, 93, 182, 402, 597).
0.883::entity(image_a, grass_a_1, grass, 2, 4, 443, 637).
0.867::entity(image_a, sky_a_2, sky, 1, 1, 443, 117).
0.906::entity(image_a, field_a_3, field, 2, 4, 443, 637).
0.857::entity(image_a, camera_a_4, camera, 92, 182, 402, 597).
0.846::entity(image_a, shoulder_a_5, shoulder, 93, 182, 402, 597).
0.938::entity(image_a, bird_a_6, bird, 196, 96, 270, 202).
0.858::entity(image_a, head_a_7, head, 97, 181, 354, 370).
0.908::entity(image_b, grass_b_0, grass, 2, 3, 587, 398).
0.905::entity(image_b, shrub_b_1, shrub, 2, 3, 586, 397).
0.881::entity(image_b, horn_b_2, horn, 387, 156, 425, 189).
0.803::entity(image_b, leg_b_3, leg, 97, 247, 167, 360).
0.912::entity(image_b, field_b_4, field, 2, 3, 587, 398).
0.788::entity(image_b, shoulder_b_5, shoulder, 92, 80, 402, 150).
0.871::entity(image_b, bird_b_6, bird, 210, 226, 286, 327).
0.871::entity(image_b, bird_b_7, bird, 293, 35, 340, 98).
0.864::entity(image_b, cow_b_8, cow, 27, 79, 559, 358).
0.887::entity(image_b, head_b_9, head, 351, 151, 559, 288).
0.886::entity(image_b, egret_b_10, egret, 210, 224, 285, 329).

% attribute facts
0.44788083454157057::attribute(image_a, bird_a_6, orange).
0.3130379734334475::attribute(image_b, bird_b_6, orange).
0.23788564712722604::attribute(image_b, egret_b_10, orange).
0.23788564712722604::attribute(image_b, egret_b_10, orange).

% count facts
0.062000000000000055::count(image_a, bird, 0).
0.938::count(image_a, bird, 1).
0.016641::count(image_b, bird, 0).
0.224718::count(image_b, bird, 1).
0.758641::count(image_b, bird, 2).


% Sugar rules
% Helper predicates for easier rule writing
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

% Generated rules for all subquestions
% Rule for: How many birds are there in image A?

bird_count(I, N) :- count(I, bird, N).

% Rule for: How many birds are there in image B?

bird_count_in_image(I, N) :- count(I, bird, N).

% Rule for: Is the bird in image A orange?

bird_is_orange(I) :-
is_category(I,B,bird),
has_attribute(I,B,orange).

% Rule for: Is the bird on the cow's head in image B orange?

bird_on_cow_head_is_orange(I) :-
is_category(I,B,bird),
is_category(I,C,cow),
is_category(I,H,head),
has_relationship(I,B,H,on),
has_relationship(I,H,C,part_of),
has_attribute(I,B,orange).

% Rule for: Is the egret in front of the cow in image B orange?

egret_in_front_of_cow_is_orange(I) :-
is_category(I,E,egret),
is_category(I,C,cow),
has_relationship(I,E,C,in_front_of),
has_attribute(I,E,orange).

% Rule for: Is the egret on the cow's front legs in image B orange?

egret_on_cow_front_legs_orange(I) :-
is_category(I, E, egret),
is_category(I, C, cow),
is_category(I, L, leg),
has_relationship(I, E, L, on),
has_relationship(I, L, C, part_of),
has_attribute(I, E, orange).


% Ultimate composition rule

ultimate_answer :-
birds_in_image_a(CountA),
birds_in_image_b(CountB),
CountA > CountB,
bird_in_image_a_orange,
bird_on_cows_head_in_image_b_orange,
egret_in_front_of_cow_in_image_b_orange,
egret_on_cows_front_legs_in_image_b_orange.

% Queries for all subquestions
query(bird_count(image_a, 1)).
query(bird_count_in_image(image_b, N)).
query(bird_is_orange(image_a)).
query(bird_on_cow_head_is_orange(image_b)).
query(egret_in_front_of_cow_is_orange(image_b)).
query(egret_on_cow_front_legs_orange(image_b)).
query(ultimate_answer).