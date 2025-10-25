# demo_problog.py
# Minimal single-file ProbLog demo using your schema.

from problog.program import PrologString
from problog import get_evaluatable

PROBLOG_MODEL = r"""
% ==== Facts (trimmed & normalized) ====
% entity(image_id, entity_id, category, x1,y1,x2,y2).
0.938::entity(image_a, bird_a_5, bird, 196, 96, 270, 202).
0.874::entity(image_a, buffalo_a_2, buffalo, 93, 182, 402, 597).
0.865129078852486::relation(image_a, bird_a_5, buffalo_a_2, perched_on).
0.71993141937532815::attribute(image_a, buffalo_a_2, neon_blue).

0.871::entity(image_b, bird_b_9, bird, 293, 35, 340, 98).
0.864::entity(image_b, cow_b_2, cow, 27, 79, 559, 358).
0.8291954988947496::relation(image_b, bird_b_9, cow_b_2, perched_on).
0.93729935806822924::attribute(image_b, cow_b_2, neon_green).

% ==== Tiny ontology / sugar ====
animal(buffalo).
animal(cow).

has_attr(I,E,A) :- attribute(I,E,A).
is_cat(I,E,C)   :- entity(I,E,C,_,_,_,_).
rel(I,A,B,R)    :- relation(I,A,B,R).

% ==== Subquestion: "bird perched on a neon green animal?" ====
perched_on_bird_on_neon_animal(I) :-
    is_cat(I, B, bird),
    is_cat(I, A, C), animal(C),
    rel(I, B, A, perched_on),
    has_attr(I, A, neon_green).

in_both_images :-
    perched_on_bird_on_neon_animal(image_a),
    perched_on_bird_on_neon_animal(image_b).

% ==== Queries ====
query(perched_on_bird_on_neon_animal(image_a)).
query(perched_on_bird_on_neon_animal(image_b)).
query(in_both_images).
"""

def main():
    result = get_evaluatable().create_from(PrologString(PROBLOG_MODEL)).evaluate()
    # Pretty print in a stable order
    for q in sorted(result.keys(), key=str):
        print(f"{q}: {result[q]:.6f}")

if __name__ == "__main__":
    main()
