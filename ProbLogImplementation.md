## Input
- Problog Facts
- Problog Sugar
- Each Subquery

## Implementation
- For each subquestion
- Use LLM to Read all the problog facts
- Use LLM Generate problog rules to be used for the query
- Combine all problog Facts, problog sugar, problog rules, and problog queries into problog string and run

## Output
- We will then have a probablity for each subquestion
- Store the probability that comes from each associated query with each subquestion

### Problog Sugar
- We should always include these because they will be helpful ins constructing other queries and rules:
```
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).
```

### In Context Examples
- We should use these incontext examples to help the LLM with this task
- We should format these incontext examples in the same way we will feed the content to the LLM so that it is well aligned with the task

**Example 1**
Subquestions
1. Is the dog in image A wearing a green harness?
2. Is the dog in image B wearing a black collar?

Problog Facts
```
0.861::entity(image_a, harness_a_0, harness, 195,129,336,290).
0.929::entity(image_a, dog_a_4, dog, 55,96,545,391).
0.873::entity(image_b, dog_b_3, dog, 60,0,157,176).
0.872::entity(image_b, collar_b_4, collar, 101,39,140,62).

0.854::relation(image_a, harness_a_0, dog_a_4, wearing).
0.875::relation(image_b, collar_b_4, dog_b_3, wearing).

0.954::attribute(image_a, harness_a_0, green).
0.885::attribute(image_b, collar_b_4, black).
```

Expected Problog Program
```
0.861::entity(image_a, harness_a_0, harness, 195,129,336,290).
0.929::entity(image_a, dog_a_4, dog, 55,96,545,391).
0.873::entity(image_b, dog_b_3, dog, 60,0,157,176).
0.872::entity(image_b, collar_b_4, collar, 101,39,140,62).

0.854::relation(image_a, harness_a_0, dog_a_4, wearing).
0.875::relation(image_b, collar_b_4, dog_b_3, wearing).

0.954::attribute(image_a, harness_a_0, green).
0.885::attribute(image_b, collar_b_4, black).

has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

dog_wearing_green_harness(I) :-
    is_category(I,D,dog),
    is_category(I,H,harness),
    has_relationship(I,H,D,wearing),
    has_attribute(I,H,green).

dog_wearing_black_collar(I) :-
    is_category(I,D,dog),
    is_category(I,C,collar),
    has_relationship(I,C,D,wearing),
    has_attribute(I,C,black).

query(dog_wearing_green_harness(image_a)).
query(dog_wearing_black_collar(image_b)).
```

**Example 2**
Subquestions
1. In image A, is there a man to the left of a woman?
2. In image A, is a woman holding an umbrella?
3. In image A, is the umbrella red?
4. In image B, is there a man to the left of a woman?
5. In image B, is a woman holding an umbrella?
6. In image B, is the umbrella black?

Problog Facts
```
0.881::entity(image_a, man_a_0, man, 150,150,300,400).
0.887::entity(image_a, woman_a_1, woman, 300,150,450,400).
0.905::entity(image_a, umbrella_a_2, umbrella, 320,80,420,200).
0.884::entity(image_b, man_b_0, man, 400,150,520,400).
0.879::entity(image_b, woman_b_1, woman, 200,150,320,400).
0.881::entity(image_b, umbrella_b_2, umbrella, 220,90,300,200).

0.892::relation(image_a, woman_a_1, umbrella_a_2, holding).
0.846::relation(image_a, man_a_0, woman_a_1, left_of).
0.897::relation(image_b, woman_b_1, umbrella_b_2, holding).
0.823::relation(image_b, man_b_0, woman_b_1, right_of).

0.943::attribute(image_a, umbrella_a_2, red).
0.916::attribute(image_b, umbrella_b_2, black).
```

Expected Problog Program
```
0.881::entity(image_a, man_a_0, man, 150,150,300,400).
0.887::entity(image_a, woman_a_1, woman, 300,150,450,400).
0.905::entity(image_a, umbrella_a_2, umbrella, 320,80,420,200).
0.884::entity(image_b, man_b_0, man, 400,150,520,400).
0.879::entity(image_b, woman_b_1, woman, 200,150,320,400).
0.881::entity(image_b, umbrella_b_2, umbrella, 220,90,300,200).

0.892::relation(image_a, woman_a_1, umbrella_a_2, holding).
0.846::relation(image_a, man_a_0, woman_a_1, left_of).
0.897::relation(image_b, woman_b_1, umbrella_b_2, holding).
0.823::relation(image_b, man_b_0, woman_b_1, right_of).

0.943::attribute(image_a, umbrella_a_2, red).
0.916::attribute(image_b, umbrella_b_2, black).

has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

man_left_of_woman(I) :-
    is_category(I,M,man),
    is_category(I,W,woman),
    has_relationship(I,M,W,left_of).

woman_holding_umbrella(I) :-
    is_category(I,W,woman),
    is_category(I,U,umbrella),
    has_relationship(I,W,U,holding).

umbrella_is_red(I) :-
    is_category(I,U,umbrella),
    has_attr(I,U,red).

umbrella_is_black(I) :-
    is_category(I,U,umbrella),
    has_attr(I,U,black).

query(man_left_of_woman(image_a)).
query(woman_holding_umbrella(image_a)).
query(umbrella_is_red(image_a)).
query(man_left_of_woman(image_b)).
query(woman_holding_umbrella(image_b)).
query(umbrella_is_black(image_b)).
```

**Example 3**
Subquestions
1. Is image A indoor?
2. Is image A lit by artificial light?
3. Does image A contain four students?
4. Is image B outdoor?
5. Is image B sunny?
6. Does image B contain three people?

Problog Facts
```
0.931::entity(image_a, student_a_0, student, 50,120,200,300).
0.905::entity(image_a, student_a_1, student, 210,120,350,300).
0.915::entity(image_a, student_a_2, student, 360,120,480,300).
0.927::entity(image_a, student_a_3, student, 490,120,620,300).
0.912::entity(image_a, student_a_4, student, 640,120,780,300).

0.886::entity(image_b, person_b_0, person, 200,200,400,400).
0.894::entity(image_b, person_b_1, person, 420,200,600,400).
0.881::entity(image_b, person_b_2, person, 640,200,820,400).

0.954::scene_attr(image_a, indoor).
0.945::scene_attr(image_a, artificial_light).
0.922::scene_attr(image_b, outdoor).
0.933::scene_attr(image_b, sunny).

0.010::count(image_a, student, 0).
0.015::count(image_a, student, 1).
0.033::count(image_a, student, 2).
0.048::count(image_a, student, 3).
0.894::count(image_a, student, 4).

0.012::count(image_b, person, 0).
0.028::count(image_b, person, 1).
0.060::count(image_b, person, 2).
0.900::count(image_b, person, 3).
```

Expected Problog Program
```
0.931::entity(image_a, student_a_0, student, 50,120,200,300).
0.905::entity(image_a, student_a_1, student, 210,120,350,300).
0.915::entity(image_a, student_a_2, student, 360,120,480,300).
0.927::entity(image_a, student_a_3, student, 490,120,620,300).
0.912::entity(image_a, student_a_4, student, 640,120,780,300).

0.886::entity(image_b, person_b_0, person, 200,200,400,400).
0.894::entity(image_b, person_b_1, person, 420,200,600,400).
0.881::entity(image_b, person_b_2, person, 640,200,820,400).

0.954::scene_attr(image_a, indoor).
0.945::scene_attr(image_a, artificial_light).
0.922::scene_attr(image_b, outdoor).
0.933::scene_attr(image_b, sunny).

0.010::count(image_a, student, 0).
0.015::count(image_a, student, 1).
0.033::count(image_a, student, 2).
0.048::count(image_a, student, 3).
0.894::count(image_a, student, 4).

0.012::count(image_b, person, 0).
0.028::count(image_b, person, 1).
0.060::count(image_b, person, 2).
0.900::count(image_b, person, 3).

has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

scene_is_indoor(I) :- scene_attr(I, indoor).
scene_has_artificial_light(I) :- scene_attr(I, artificial_light).
scene_is_outdoor(I) :- scene_attr(I, outdoor).
scene_is_sunny(I) :- scene_attr(I, sunny).

student_count_four(I) :- count(I, student, 4).
person_count_three(I) :- count(I, person, 3).

query(scene_is_indoor(image_a)).
query(scene_has_artificial_light(image_a)).
query(student_count_four(image_a)).
query(scene_is_outdoor(image_b)).
query(scene_is_sunny(image_b)).
query(person_count_three(image_b)).
```

