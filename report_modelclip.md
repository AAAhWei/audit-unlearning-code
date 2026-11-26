# Model clipping: forget-set design and calibration budget

## 1. Question

The audit deletes 3 of 6 forget batches and then predicts which batches the model
deleted. The better the prediction, the higher the audited lower bound ε_LB.

Can the auditor select the forget points, and put them into batches, so that ε_LB is
higher for the same unlearning algorithm?

Two design choices:

- **Pool:** which 4,500 training points are the forget points.
- **Assignment:** which batch each forget point goes into.

This report gives the same experiments as the R2D report, for model clipping.

## 2. Main results

1. **Class-pure batches help, but much less than for R2D.** At unlearning step 20 they
increase ε_LB by 0.2 to 1.0. For R2D they moved ε_LB from approximately 1 to the
maximum of the test.
2. **The pool has an effect only with a random assignment.** There the easy pool is
clearly lower (1.44 against 1.96 at step 20). With class-pure batches the three pools
give the same ε_LB.
3. **The global hard pool is not bad here.** Its points are in all 100 classes, and it
gives the same ε_LB as the random pool.
4. **The decoder `ours_a` needs approximately 2 times fewer calibration draws** than
the decoder of the paper. **The linear decoder `ours_b` fails**: it gives less than
half of the ε_LB of the other two decoders.
5. **Each design is one run.** Differences below approximately 0.2 are not reliable.

## 3. Settings

### Data and audit


| Item       | Value                                                                     |
| ---------- | ------------------------------------------------------------------------- |
| Data       | CIFAR-100, 45,000 training points, 100 classes, 450 points for each class |
| Forget set | 6 forget batches of 750 points (4,500 points, 10 % of the data)           |
| Retain set | the other 40,500 points                                                   |
| Deleted    | 3 of the 6 batches. There are 20 such selections (sign vectors).          |
| Models     | one trained model for each of the 20 sign vectors                         |
| Trials     | 150 unlearning trials for each trained model, each with new noise         |


### Unlearning: model clipping


| Item              | Value                                                                              |
| ----------------- | ---------------------------------------------------------------------------------- |
| Training          | mini-batch SGD, 400 epochs, batch size 128, one-cycle learning rate, maximum 0.06 |
| Training data     | the retain set and the 3 forget batches that stay (42,750 points)                  |
| Unlearning step 0 | clip the weights to the norm 30. No noise.                                         |
| Steps 1 to 30     | one SGD step on 128 retain points (learning rate 0.001), clip the weights to the norm 5, add Gaussian noise (σ = 1e-4) |
| Variant           | not certified. There is no claimed ε.                                              |
| Audited steps     | 10, 20 and 30                                                                      |
| Accuracy          | the model is at chance level at the audited steps (test accuracy 1.4 % at step 30) |


After more steps, the model keeps less information about the deleted batches, thus ε_LB decreases with  
the step.

### Auditor


| Item                    | Value                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| Score of a forget point | φ = log(p / (1 − p)), p = probability of the correct label                                                     |
| Decoder                 | joint Gaussian decoder of the paper (`b2`): one mean and one variance of φ for each point and each sign vector |
| Calibration             | 50 trials for each sign vector                                                                                 |
| Evaluation              | 100 trials for each sign vector (2,000 trials)                                                                 |
| Lower bound             | average-v test, confidence 95 %                                                                                |
| Maximum of ε\_LB        | 4.72 (the test cannot give more with 2,000 trials)                                                             |


### Difficulty score

The score of a training point is its per-instance privacy loss (Sepahvand et al.), the
same score as for R2D. The reference model is one model clipping training run on all
45,000 points (training accuracy 97 %, test accuracy 53 %). For each point, the score
uses the gradient norm at 35 of the 400 epochs.

Differences from the R2D score:

- The training uses mini-batches, thus the score uses the mini-batch form of the
definition (sampling rate 128 / 45,000).
- The reference model learns the data. The R2D reference model stays at chance level.
- The hard pools of the two methods are different: 680 of the 4,500 points are in the
two hard pools. Random selection gives 450.

### Pools and assignments


| Pool        | Forget points                          |
| ----------- | -------------------------------------- |
| random      | random, 45 points for each class       |
| easy        | the 45 lowest scores in each class     |
| hard        | the 45 highest scores in each class    |
| global hard | the 4,500 highest scores of all points |


The random pool is the random pool 0 of the R2D report.


| Assignment | Batch of each point                                           |
| ---------- | ------------------------------------------------------------- |
| random     | random                                                        |
| label      | samples in one superclass goes into the same batch            |
| grad       | points with a similar gradient direction go into one batch    |
| spread     | the opposite of grad: each batch gets all gradient directions |


A design is one pool with one assignment, for example "hard pool, label assignment".

## 4. Pool selection (random assignment)


| Pool        | Assignment | Runs | Step 10 | Step 20 | Step 30 |
| ----------- | ---------- | ---- | ------- | ------- | ------- |
| random      | random     | 1    | 4.72    | 1.96    | 0.91    |
| easy        | random     | 1    | 4.72    | 1.44    | 0.56    |
| hard        | random     | 1    | 4.72    | 2.18    | 0.92    |
| global hard | random     | 1    | 4.72    | 2.03    | 1.05    |
| hard        | spread     | 1    | 4.72    | 2.08    | 0.93    |


The value 4.72 is the maximum of the test. At step 10 each design is at the maximum,
thus step 10 does not compare the designs.

The split of the paper gives 1.98 at step 20 (run of a team member). The random pool
agrees with it.

### What the numbers show

- **The easy pool is clearly lower:** −0.52 at step 20 and −0.35 at step 30, against
the random pool.
- **Hard is not clearly above random:** +0.22 at step 20 and +0.01 at step 30.
- **Spread gives no gain:** 2.08 against 2.18 for the hard pool. For R2D spread
increased ε_LB.
- There is one run for each pool, thus the spread between runs is not known. Designs
of the same type differ by up to 0.2 in section 5. Thus only the easy pool is a safe
result.

### Sanity check: does the difficulty score predict the audit signal?

**Audit signal of a forget point x in batch b.** It shows how well this one point tells
the auditor if batch b was deleted.

```
d_x = | μ⁺ − μ⁻ | / sqrt( (v⁺ + v⁻) / 2 )
```


| Symbol    | Meaning                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| "+" group | the 10 sign vectors that delete batch b                                                                            |
| "−" group | the 10 sign vectors that keep batch b                                                                              |
| μ⁺, μ⁻    | the mean of φ(x) in each group                                                                                     |
| v⁺, v⁻    | the variance of φ(x) in each group: the noise variance, plus the variance between the 10 sign vectors of the group |


**Check.** The Spearman rank correlation between d_x and the difficulty score of x.
We calculate it for the points of each class and then take the mean of the 100 classes.

**Result** (random pool with 3 assignments, mean ± standard error):


| Assignment | Step 10      | Step 20      | Step 30      |
| ---------- | ------------ | ------------ | ------------ |
| random     | +0.01 ± 0.01 | +0.02 ± 0.02 | 0.00 ± 0.02  |
| label      | +0.03 ± 0.02 | +0.02 ± 0.02 | +0.04 ± 0.02 |
| grad       | −0.02 ± 0.02 | −0.04 ± 0.02 | +0.01 ± 0.02 |


The correlation is approximately 0, as for R2D. A point with a high score does not have
a larger audit signal.

The median of d_x is 0.29 (step 10), 0.20 (step 20) and 0.14 (step 30).

### Why the global hard pool is not bad here

For R2D the global hard pool gave less than half of the ε_LB of the random pool. For
model clipping it gives the same ε_LB (2.03 against 1.96 at step 20).


| Property of the global hard pool | R2D       | Model clipping |
| -------------------------------- | --------- | -------------- |
| Classes in the pool              | 32 of 100 | 100 of 100     |
| Share of the 9 largest classes   | 67 %      | 22 %           |
| Points of the largest class      | 446       | 145            |
| ε\_LB against the random pool    | 0.43 against 0.95 (ε = 1000) | 2.03 against 1.96 (step 20) |


- The model clipping score comes from a model that learned the data. Each class has
hard points, thus the pool has all classes.
- The R2D score comes from a model at chance level. Whole classes have a large
gradient, thus the pool has few classes.

This agrees with the explanation of the R2D report: the R2D pool is bad because it
has few classes, not because its points are hard.

## 5. Batch assignment (class-pure batches)


| Pool   | Assignment | Step 10 | Step 20  | Step 30  |
| ------ | ---------- | ------- | -------- | -------- |
| hard   | random     | 4.72    | 2.18     | 0.92     |
| hard   | label      | 4.72    | **2.48** | **1.11** |
| hard   | grad       | 4.72    | 2.37     | 1.05     |
| random | random     | 4.72    | 1.96     | 0.91     |
| random | label      | 4.72    | 2.21     | 0.96     |
| random | grad       | 4.72    | **2.41** | **1.00** |
| easy   | random     | 4.72    | 1.44     | 0.56     |
| easy   | label      | 4.72    | **2.41** | **1.12** |
| easy   | grad       | 4.72    | 2.22     | 1.09     |


Each row is one run.

### What the numbers show

- **Class-pure batches are higher for each pool.** The 6 class-pure designs are in
[2.21, 2.48] at step 20. The 3 designs with a random assignment are in [1.44, 2.18].
- **The gain depends on the pool.** At step 20: +0.97 (easy), +0.45 (random) and +0.30
(hard), for the better of label and grad.
- **Class-pure batches remove the pool effect.** The order easy < random < hard holds
only with the random assignment. With label, the random pool is the lowest. With grad,
the random pool is the highest. These differences are 0.2 or less.
- **Label and grad are equal.** For the same pool they differ by 0.11 to 0.20, with no
constant direction.

### Why class-pure batches gain less than for R2D

Median over the forget points, at step 20:


| Quantity                                         | Random or spread assignment | Label or grad assignment |
| ------------------------------------------------ | --------------------------- | ------------------------ |
| Shift of φ when the own batch is deleted         | \[0.009, 0.012\]            | \[0.009, 0.014\]         |
| Shift of φ when a different batch is deleted     | \[0.011, 0.013\]            | \[0.011, 0.014\]         |
| Shift of φ between two random halves of the 20 models | \[0.008, 0.009\]       | \[0.009, 0.010\]         |
| Noise of φ (standard deviation between trials)   | \[0.038, 0.039\]            | \[0.038, 0.040\]         |


- **The own batch does not give a larger shift than a different batch**, for each
assignment. For R2D with class-pure batches the own batch gave 3 times the shift.
- **The shifts are at the level of two random halves.** The 20 trained models differ
from each other by this amount, with no relation to the deleted batch.

A possible explanation (not tested): each sign vector has one training run of 400
epochs with mini-batches. A different training set changes the full training run, not
only the points of the deleted batch. Thus each trained model has its own values of φ.
The decoder can identify the model, but the difference does not follow the batches.

The gradient directions are the same as for R2D, thus they do not explain the smaller
gain:


| Assignment      | Alignment in a batch | Cosine between two batches |
| --------------- | -------------------- | -------------------------- |
| random / spread | \[0.05, 0.07\]       | \[0.45, 0.67\]             |
| label / grad    | \[0.17, 0.23\]       | \[0.10, 0.17\]             |


- Alignment in a batch: take the mean of the 750 normalized gradient directions in the batch. The alignment is the length of this mean.
- Cosine between two batches: cosine similarity between the mean gradient directions of the two batches.

### How similar are the label and grad assignments?

They are the same type of design, found in two different ways.


| Property                                   | label                                  | grad                                           |
| ------------------------------------------ | -------------------------------------- | ---------------------------------------------- |
| Input                                      | the class and superclass of each point | the gradient direction of each point, no label |
| Classes in one batch                       | 16 or 17                               | 15 to 17                                       |
| Share of a class that is in one batch      | 98.7 %                                 | \[98.2 %, 98.7 %\]                             |
| Share of a superclass that is in one batch | 93 %                                   | \[40 %, 42 %\]                                 |
| Alignment in a batch                       | \[0.17, 0.19\]                         | \[0.21, 0.23\]                                 |
| Cosine between two batches                 | \[0.10, 0.15\]                         | \[0.12, 0.17\]                                 |
| Points in the same batch for the two       | \[26 %, 31 %\]                         |                                                |
| ε\_LB at step 20 (hard, random, easy pool) | 2.48, 2.21, 2.41                       | 2.37, 2.41, 2.22                               |


- **Same:** the two assignments keep each class in one batch. The gradient clustering
finds the classes without the labels.
- **Different:** which classes go together. Label puts the 5 classes of a superclass into one batch. Grad does not follow the rule.
- **Result:** no difference in ε_LB. For R2D label was higher by 0.25 to 0.46.

## 6. Calibration budget and decoder

The budget is the number of calibration trials for each sign vector. A lower budget is
a cheaper audit.


| Decoder      | Mean of φ                 | Variance of φ            | Parameters for each point                          |
| ------------ | ------------------------- | ------------------------ | -------------------------------------------------- |
| `b2` (paper) | one for each sign vector  | one for each sign vector | 20 means, 20 variances                             |
| `ours_a`     | one for each sign vector  | one for all sign vectors | 20 means, 1 variance                               |
| `ours_b`     | linear in the sign vector | one for all sign vectors | 1 offset, 1 coefficient for each batch, 1 variance |


- Budgets 2 and 5: mean ± standard deviation of 10 fits on disjoint calibration trials.
- Budget 10: 5 fits. Budget 20: 2 fits. Budget 50: one fit.
- `b2` with the budget 50 is the audit of sections 4 and 5.
- The sparse design of `ours_b` (5 pairs) was not run for model clipping.

### Random assignment


| Pool   | Step | Decoder  | Budget 2        | Budget 5        | Budget 10       | Budget 20       | Budget 50 |
| ------ | ---- | -------- | --------------- | --------------- | --------------- | --------------- | --------- |
| random | 10   | `b2`     | 0.15 ± 0.04     | 2.40 ± 0.27     | 4.02 ± 0.65     | 4.34 ± 0.54     | 4.72      |
|        |      | `ours_a` | **3.18 ± 0.32** | **4.21 ± 0.56** | **4.72 ± 0.00** | **4.72 ± 0.00** | 4.72      |
|        |      | `ours_b` | 0.91 ± 0.07     | 1.03 ± 0.04     | 1.08 ± 0.03     | 1.09 ± 0.00     | 1.12      |
| random | 20   | `b2`     | 0.01 ± 0.01     | 0.65 ± 0.05     | 1.38 ± 0.02     | 1.72 ± 0.02     | 1.96      |
|        |      | `ours_a` | **0.87 ± 0.07** | **1.42 ± 0.07** | **1.67 ± 0.07** | **1.81 ± 0.04** | 1.95      |
|        |      | `ours_b` | 0.40 ± 0.04     | 0.57 ± 0.03     | 0.64 ± 0.04     | 0.68 ± 0.02     | 0.73      |
| random | 30   | `b2`     | 0.00 ± 0.00     | 0.15 ± 0.03     | 0.45 ± 0.03     | 0.69 ± 0.01     | 0.91      |
|        |      | `ours_a` | **0.17 ± 0.03** | **0.43 ± 0.03** | **0.63 ± 0.04** | **0.79 ± 0.01** | **0.97**  |
|        |      | `ours_b` | 0.14 ± 0.01     | 0.26 ± 0.03     | 0.32 ± 0.01     | 0.37 ± 0.00     | 0.42      |
| hard   | 10   | `b2`     | 0.20 ± 0.10     | 2.96 ± 0.52     | 4.57 ± 0.34     | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_a` | **3.75 ± 0.60** | **4.65 ± 0.24** | **4.72 ± 0.00** | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_b` | 1.29 ± 0.07     | 1.42 ± 0.03     | 1.45 ± 0.03     | 1.48 ± 0.02     | 1.49      |
| hard   | 20   | `b2`     | 0.01 ± 0.01     | 0.81 ± 0.10     | 1.63 ± 0.05     | 2.01 ± 0.04     | **2.18**  |
|        |      | `ours_a` | **1.05 ± 0.11** | **1.60 ± 0.05** | **1.86 ± 0.07** | **2.08 ± 0.05** | 2.15      |
|        |      | `ours_b` | 0.58 ± 0.05     | 0.78 ± 0.03     | 0.85 ± 0.03     | 0.88 ± 0.00     | 0.91      |
| hard   | 30   | `b2`     | 0.00 ± 0.00     | 0.15 ± 0.02     | 0.43 ± 0.01     | 0.72 ± 0.00     | 0.92      |
|        |      | `ours_a` | **0.17 ± 0.03** | **0.41 ± 0.03** | **0.60 ± 0.02** | **0.80 ± 0.04** | 0.92      |
|        |      | `ours_b` | 0.15 ± 0.02     | 0.25 ± 0.01     | 0.31 ± 0.02     | 0.36 ± 0.01     | 0.42      |


### Label assignment


| Pool   | Step | Decoder  | Budget 2        | Budget 5        | Budget 10       | Budget 20       | Budget 50 |
| ------ | ---- | -------- | --------------- | --------------- | --------------- | --------------- | --------- |
| random | 10   | `b2`     | 0.21 ± 0.08     | 2.91 ± 0.38     | 4.72 ± 0.00     | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_a` | **3.92 ± 0.48** | **4.72 ± 0.00** | 4.72 ± 0.00     | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_b` | 1.27 ± 0.10     | 1.38 ± 0.08     | 1.43 ± 0.06     | 1.42 ± 0.04     | 1.45      |
| random | 20   | `b2`     | 0.01 ± 0.01     | 0.78 ± 0.13     | 1.63 ± 0.03     | 2.06 ± 0.02     | 2.21      |
|        |      | `ours_a` | **1.11 ± 0.10** | **1.69 ± 0.04** | **1.95 ± 0.06** | **2.12 ± 0.00** | **2.31**  |
|        |      | `ours_b` | 0.47 ± 0.05     | 0.65 ± 0.03     | 0.72 ± 0.02     | 0.78 ± 0.00     | 0.80      |
| random | 30   | `b2`     | 0.00 ± 0.00     | 0.16 ± 0.03     | 0.52 ± 0.01     | 0.77 ± 0.04     | 0.96      |
|        |      | `ours_a` | **0.20 ± 0.04** | **0.49 ± 0.02** | **0.71 ± 0.02** | **0.88 ± 0.03** | **0.98**  |
|        |      | `ours_b` | 0.14 ± 0.02     | 0.24 ± 0.02     | 0.31 ± 0.02     | 0.35 ± 0.01     | 0.40      |
| hard   | 10   | `b2`     | 0.24 ± 0.11     | 3.19 ± 0.46     | 4.72 ± 0.00     | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_a` | **4.34 ± 0.52** | **4.72 ± 0.00** | 4.72 ± 0.00     | 4.72 ± 0.00     | 4.72      |
|        |      | `ours_b` | 1.18 ± 0.07     | 1.30 ± 0.03     | 1.33 ± 0.02     | 1.35 ± 0.02     | 1.36      |
| hard   | 20   | `b2`     | 0.01 ± 0.01     | 1.06 ± 0.09     | 1.89 ± 0.11     | 2.30 ± 0.01     | 2.48      |
|        |      | `ours_a` | **1.26 ± 0.09** | **1.85 ± 0.07** | **2.17 ± 0.11** | **2.41 ± 0.11** | **2.52**  |
|        |      | `ours_b` | 0.61 ± 0.05     | 0.80 ± 0.04     | 0.88 ± 0.03     | 0.93 ± 0.01     | 0.96      |
| hard   | 30   | `b2`     | 0.00 ± 0.00     | 0.20 ± 0.02     | 0.59 ± 0.03     | 0.87 ± 0.03     | 1.11      |
|        |      | `ours_a` | **0.25 ± 0.04** | **0.58 ± 0.03** | **0.82 ± 0.04** | **0.99 ± 0.05** | **1.15**  |
|        |      | `ours_b` | 0.19 ± 0.02     | 0.32 ± 0.02     | 0.39 ± 0.03     | 0.43 ± 0.04     | 0.46      |


### All designs at step 20

Budgets 2, 10 and 50 for each decoder.


| Pool        | Assignment | `b2` 2      | `b2` 10     | `b2` 50 | `ours_a` 2  | `ours_a` 10 | `ours_a` 50 | `ours_b` 2  | `ours_b` 10 | `ours_b` 50 |
| ----------- | ---------- | ----------- | ----------- | ------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| random      | random     | 0.01 ± 0.01 | 1.38 ± 0.02 | 1.96    | 0.87 ± 0.07 | 1.67 ± 0.07 | 1.95        | 0.40 ± 0.04 | 0.64 ± 0.04 | 0.73        |
| easy        | random     | 0.00 ± 0.00 | 0.98 ± 0.07 | 1.44    | 0.54 ± 0.06 | 1.18 ± 0.02 | 1.41        | 0.30 ± 0.03 | 0.47 ± 0.00 | 0.53        |
| hard        | random     | 0.01 ± 0.01 | 1.63 ± 0.05 | 2.18    | 1.05 ± 0.11 | 1.86 ± 0.07 | 2.15        | 0.58 ± 0.05 | 0.85 ± 0.03 | 0.91        |
| global hard | random     | 0.01 ± 0.01 | 1.51 ± 0.09 | 2.03    | 0.97 ± 0.09 | 1.76 ± 0.07 | 1.98        | 0.51 ± 0.05 | 0.72 ± 0.04 | 0.79        |
| hard        | spread     | 0.01 ± 0.01 | 1.49 ± 0.06 | 2.08    | 0.98 ± 0.05 | 1.81 ± 0.04 | 2.11        | 0.51 ± 0.02 | 0.76 ± 0.01 | 0.84        |
| random      | label      | 0.01 ± 0.01 | 1.63 ± 0.03 | 2.21    | 1.11 ± 0.10 | 1.95 ± 0.06 | 2.31        | 0.47 ± 0.05 | 0.72 ± 0.02 | 0.80        |
| easy        | label      | 0.01 ± 0.01 | 1.80 ± 0.11 | 2.41    | 1.22 ± 0.09 | 2.02 ± 0.04 | 2.43        | 0.58 ± 0.05 | 0.81 ± 0.02 | 0.89        |
| hard        | label      | 0.01 ± 0.01 | 1.89 ± 0.11 | 2.48    | 1.26 ± 0.09 | 2.17 ± 0.11 | 2.52        | 0.61 ± 0.05 | 0.88 ± 0.03 | 0.96        |
| random      | grad       | 0.01 ± 0.01 | 1.68 ± 0.07 | 2.41    | 1.08 ± 0.08 | 2.06 ± 0.05 | 2.37        | 0.48 ± 0.03 | 0.73 ± 0.03 | 0.80        |
| easy        | grad       | 0.00 ± 0.01 | 1.75 ± 0.08 | 2.22    | 1.13 ± 0.06 | 1.96 ± 0.09 | 2.19        | 0.48 ± 0.04 | 0.76 ± 0.04 | 0.82        |
| hard        | grad       | 0.02 ± 0.01 | 1.70 ± 0.10 | 2.37    | 1.10 ± 0.06 | 1.97 ± 0.05 | 2.29        | 0.58 ± 0.04 | 0.87 ± 0.01 | 0.95        |


### All designs at step 30


| Pool        | Assignment | `b2` 2      | `b2` 10     | `b2` 50 | `ours_a` 2  | `ours_a` 10 | `ours_a` 50 | `ours_b` 2  | `ours_b` 10 | `ours_b` 50 |
| ----------- | ---------- | ----------- | ----------- | ------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| random      | random     | 0.00 ± 0.00 | 0.45 ± 0.03 | 0.91    | 0.17 ± 0.03 | 0.63 ± 0.04 | 0.97        | 0.14 ± 0.01 | 0.32 ± 0.01 | 0.42        |
| easy        | random     | 0.00 ± 0.00 | 0.22 ± 0.01 | 0.56    | 0.08 ± 0.01 | 0.33 ± 0.03 | 0.59        | 0.09 ± 0.01 | 0.22 ± 0.01 | 0.29        |
| hard        | random     | 0.00 ± 0.00 | 0.43 ± 0.01 | 0.92    | 0.17 ± 0.03 | 0.60 ± 0.02 | 0.92        | 0.15 ± 0.02 | 0.31 ± 0.02 | 0.42        |
| global hard | random     | 0.00 ± 0.00 | 0.56 ± 0.03 | 1.05    | 0.24 ± 0.02 | 0.76 ± 0.02 | 1.03        | 0.21 ± 0.03 | 0.38 ± 0.01 | 0.45        |
| hard        | spread     | 0.00 ± 0.00 | 0.45 ± 0.03 | 0.93    | 0.18 ± 0.02 | 0.63 ± 0.05 | 0.97        | 0.15 ± 0.03 | 0.33 ± 0.02 | 0.42        |
| random      | label      | 0.00 ± 0.00 | 0.52 ± 0.01 | 0.96    | 0.20 ± 0.04 | 0.71 ± 0.02 | 0.98        | 0.14 ± 0.02 | 0.31 ± 0.02 | 0.40        |
| easy        | label      | 0.00 ± 0.00 | 0.65 ± 0.04 | 1.12    | 0.29 ± 0.04 | 0.88 ± 0.04 | 1.19        | 0.21 ± 0.03 | 0.42 ± 0.01 | 0.50        |
| hard        | label      | 0.00 ± 0.00 | 0.59 ± 0.03 | 1.11    | 0.25 ± 0.04 | 0.82 ± 0.04 | 1.15        | 0.19 ± 0.02 | 0.39 ± 0.03 | 0.46        |
| random      | grad       | 0.00 ± 0.00 | 0.54 ± 0.04 | 1.00    | 0.21 ± 0.02 | 0.74 ± 0.02 | 1.05        | 0.15 ± 0.01 | 0.34 ± 0.02 | 0.42        |
| easy        | grad       | 0.00 ± 0.00 | 0.56 ± 0.01 | 1.09    | 0.22 ± 0.04 | 0.76 ± 0.03 | 1.10        | 0.14 ± 0.03 | 0.33 ± 0.01 | 0.43        |
| hard        | grad       | 0.00 ± 0.00 | 0.50 ± 0.04 | 1.05    | 0.22 ± 0.01 | 0.72 ± 0.01 | 1.07        | 0.16 ± 0.02 | 0.35 ± 0.01 | 0.46        |


### What the numbers show

- **`ours_a` is equal to `b2` at the full budget and higher at each smaller budget**,
for each design at steps 20 and 30. At step 10 the two decoders are at the maximum for
most designs. For two designs (hard pool with spread and with grad) `ours_a` gives
3.96 and 3.70 with the full budget, where `b2` is at 4.72. This is one fit near the
maximum, where one wrong prediction changes the bound much.
- **`ours_a` needs approximately 2 times fewer trials.**
  
  | Pool, assignment, step | `ours_a`, budget 5 | `b2`, budget 10 | `ours_a`, budget 10 | `b2`, budget 20 |
  | ---------------------- | ------------------ | --------------- | ------------------- | --------------- |
  | random, random, 20     | 1.42               | 1.38            | 1.67                | 1.72            |
  | hard, random, 20       | 1.60               | 1.63            | 1.86                | 2.01            |
  | random, label, 20      | 1.69               | 1.63            | 1.95                | 2.06            |
  | hard, label, 20        | 1.85               | 1.89            | 2.17                | 2.30            |
  
- **`b2` fails with the budget 2** (ε_LB approximately 0 at steps 20 and 30). `ours_a`
with the budget 2 gives approximately half of the full-budget value at step 20.
- **`ours_b` fails for model clipping.** With the full budget it gives 0.53 to 0.96 at
step 20, where `b2` gives 1.44 to 2.48. More trials do not help: the value
changes little between the budgets 10 and 50. For R2D `ours_b` was the best decoder.
- **Why `ours_b` fails (possible explanation, not tested):** `ours_b` assumes that the
mean of φ is linear in the sign vector. Section 5 shows that the 20 trained models do
not differ in a way that follows the batches. A model with one mean for each sign
vector (`b2`, `ours_a`) does not need this assumption.
- **The design gain and the decoder gain add.** The hard pool with the label assignment
gives the highest ε_LB for `b2` and `ours_a` at step 20, for the budgets 10 and 50.

## 7. Limits

- Each design is one run: one forget set and one training run for each sign vector.
There is no measurement of the spread between runs.
- Step 10 is at the maximum of the test for each design. A step between 10 and 20 can
compare the designs where the audit is strong.
- The difficulty scores come from one reference model and 35 of its 400 epochs.
- The explanations in sections 5 and 6 agree with the measured shifts of φ. No
experiment tests them directly.
- The variant is not certified, thus there is no claimed ε to compare with ε_LB.
