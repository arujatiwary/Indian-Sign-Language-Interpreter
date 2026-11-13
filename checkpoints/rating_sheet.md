# Story rating sheet

Rater: ______________________   Date: ____________

Fill one sheet per rater, save as rating_sheet_<name>.md, then:
    python src/score_ratings.py checkpoints/rating_sheet_*.md

Rate each 1 to 5. Criteria:

- **meaning**: Does the sentence say what the glosses said? 1 = wrong meaning, 5 = exactly right
- **grammar**: Is it grammatical English? 1 = not a sentence, 5 = fluent
- **context**: Does it fit the story so far? 1 = contradicts it, 5 = follows naturally
- **no_extra**: Did it avoid adding information? 1 = invents events, 5 = adds nothing

---

## 1
Signs: `GOOD_MORNING TEACHER`
Output: **Good morning, teacher.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 2
Signs: `I HAPPY`
Output: **I am happy.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 3
Signs: `BROTHER DOG BLACK`
Output: **My brother's dog is black.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 4
Signs: `MONDAY I HOUSE`
Output: **On Monday I was at the house.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 5
Signs: `FATHER PRIEST`
Output: **The father is a priest.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 6
Signs: `GIRL BOY HAPPY`
Output: **The girl and the boy are happy.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 7
Signs: `DOG BIG_LARGE`
Output: **The dog is large.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 8
Signs: `BIRD WHITE`
Output: **The bird is white.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 9
Signs: `I STORE_OR_SHOP`
Output: **I am at the store.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 10
Signs: `TEACHER GOOD`
Output: **The teacher is good.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 11
Signs: `COW NEW`
Output: **The cow is new.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 12
Signs: `HELLO FATHER`
Output: **Hello, father.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 13
Signs: `YOU_(PLURAL) GOOD`
Output: **You are good.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 14
Signs: `THANK_YOU TEACHER`
Output: **Thank you, teacher.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 15
Signs: `BROTHER HOUSE RED`
Output: **My brother's house is red.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 16
Signs: `GOOD_MORNING TEACHER`
Output: **Good morning, teacher.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 17
Signs: `I HAPPY`
Story so far: Good morning, teacher.
Output: **I am happy.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 18
Signs: `THANK_YOU TEACHER`
Story so far: Good morning, teacher. I am happy.
Output: **Thank you, teacher.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 19
Signs: `MONDAY I HOUSE`
Output: **On Monday I was at the house.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 20
Signs: `BROTHER DOG BLACK`
Story so far: On Monday I was at the house.
Output: **My brother's dog is black.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 21
Signs: `DOG BIG_LARGE`
Story so far: On Monday I was at the house. My brother's dog is black.
Output: **The dog is big.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 22
Signs: `HELLO FATHER`
Output: **Hello, father.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 23
Signs: `FATHER PRIEST`
Story so far: Hello, father.
Output: **Father is a priest.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |

## 24
Signs: `I HAPPY`
Story so far: Hello, father. Father is a priest.
Output: **I am happy.**

| meaning | grammar | context | no_extra |
|---|---|---|---|
|   |   |   |   |
