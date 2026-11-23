# R2D: forget-set design and calibration budget

## 1. Question

The audit deletes 3 of 6 forget batches and then predicts which batches the model
deleted. The better the prediction, the higher the audited lower bound ε_LB.

Can the auditor select the forget points, and put them into batches, so that ε_LB is
higher for the same unlearning algorithm and the same claimed ε?

Two design choices:

- **Pool:** which 4,500 training points are the forget points.
- **Assignment:** which batch each forget point goes into.

## 2. Main results

1. **The assignment is the important choice.** Batches that keep each class in one
 batch increase ε\_LB at claimed ε = 100 from approximately 0.03 to approximately 2.
2. **The pool is a small effect.** Hard points are above easy points, but random pools
 differ from each other by as much.
3. **The difficulty score does not predict the audit signal of a point.** The rank
 correlation is approximately 0.
4. **The linear decoder `ours_b` needs approximately 10 times fewer calibration draws**
 than the decoder of the paper for the same ε\_LB.

## 3. Settings

### Data and audit


| Item       | Value                                                                     |
| ---------- | ------------------------------------------------------------------------- |
| Data       | CIFAR-100, 45,000 training points, 100 classes, 450 points for each class |
| Forget set | 6 forget batches of 750 points (4,500 points, 10 % of the data)           |
| Retain set | the other 40,500 points                                                   |
| Deleted    | 3 of the 6 batches. There are 20 such selections (sign vectors).          |
| Models     | one trained and unlearned model for each of the 20 sign vectors           |


### Unlearning: rewind-to-delete (R2D)


| Item       | Value                                                                       |
| ---------- | --------------------------------------------------------------------------- |
| Training   | full-batch gradient descent, 35 steps, learning rate 0.01                   |
| Unlearning | rewind, then 5 full-batch steps on the data without the deleted batches     |
| Noise      | Gaussian noise on the weights, calibrated for the claimed ε with δ = 1e-3   |
| Claimed ε  | 100, 1000, 2000 and ∞ (no noise). 10, 20 and 50 for the class-pure batches. |


### Auditor


| Item                    | Value                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| Score of a forget point | φ = log(p / (1 − p)), p = probability of the correct label                                                     |
| Decoder                 | joint Gaussian decoder of the paper (`b2`): one mean and one variance of φ for each point and each sign vector |
| Calibration             | 100 noise draws for each sign vector                                                                           |
| Evaluation              | 500 trials with new noise                                                                                      |
| Lower bound             | average-v test, confidence 95 %                                                                                |
| Maximum of ε\_LB        | 4.03 (the test cannot give more with 500 trials)                                                               |


### Difficulty score

The score of a training point is its per-instance privacy loss (Sepahvand et al.). In
practice it is a weighted sum of the gradient norm of the point at each training step
of a reference model. A point with a large gradient is a "hard" point.

### Pools and assignments


| Pool        | Forget points                          |
| ----------- | -------------------------------------- |
| paper split | the split of the paper                 |
| random      | random, 45 points for each class       |
| easy        | the 45 lowest scores in each class     |
| hard        | the 45 highest scores in each class    |
| global hard | the 4,500 highest scores of all points |



| Assignment | Batch of each point                                           |
| ---------- | ------------------------------------------------------------- |
| random     | random                                                        |
| label      | one superclass goes into one batch                            |
| grad       | points with a similar gradient direction go into one batch    |
| spread     | the opposite of grad: each batch gets all gradient directions |


A design is one pool with one assignment, for example "hard pool, label assignment".

## 4. Pool selection (random assignment)


| Pool        | Assignment | Runs          | ε = 100 | ε = 1000            | ε = 2000            |
| ----------- | ---------- | ------------- | ------- | ------------------- | ------------------- |
| paper split | random     | 1             | 0.05    | 1.04                | 1.93                |
| random      | random     | 6 pools       | 0.03    | 0.95 (0.82 to 1.20) | 1.79 (1.52 to 2.10) |
| easy        | random     | 3 assignments | 0.02    | 0.79 (0.77 to 0.82) | 1.62 (1.58 to 1.65) |
| hard        | random     | 3 assignments | 0.05    | 1.16 (1.14 to 1.18) | 2.12 (2.03 to 2.18) |
| global hard | random     | 1             | 0.03    | 0.43                | 0.78                |
| hard        | spread     | 1             | 0.04    | 1.41                | 2.49                |


The cells give the mean, and the minimum and the maximum in brackets.
With ε = ∞ each configuration is at the maximum 4.03.

The 6 random pools:


| Pool     | 0    | 1    | 2    | 3    | 4    | 5    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- |
| ε = 1000 | 0.82 | 0.82 | 1.16 | 1.20 | 0.84 | 0.86 |
| ε = 2000 | 1.73 | 1.78 | 2.03 | 2.10 | 1.60 | 1.52 |


### What the numbers show

- **Hard is above easy:** +0.37 (ε = 1000) and +0.50 (ε = 2000).
- **Hard is not clearly above random.** Two of the 6 random pools (2 and 3) are as high
as the hard pool.
- **The pool changes more than the assignment.** A different random pool changes ε_LB
by up to 0.38. A different random assignment of the same pool changes it by 0.04.

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


- The numerator is the shift of φ(x) when the auditor deletes the batch of x.
- The denominator is the spread of φ(x) that the shift must exceed.
- d_x = 0: the point gives no information. d_x = 2: the two groups are 2 standard
deviations apart.

**Check.** The Spearman rank correlation between d_x and the difficulty score of x.
We calculate it for the points of each class and then take the mean of the 100 classes.
The pools select points in each class, thus the correlation in a class is the
applicable one.

**Result** (3 random pools with a random assignment, mean ± standard error):


| Random pool | ε = 100      | ε = 1000     | ε = 2000     | ε = ∞        |
| ----------- | ------------ | ------------ | ------------ | ------------ |
| 0           | −0.01 ± 0.02 | −0.04 ± 0.02 | −0.04 ± 0.02 | −0.01 ± 0.02 |
| 1           | −0.02 ± 0.02 | −0.03 ± 0.02 | −0.03 ± 0.02 | 0.00 ± 0.02  |
| 2           | −0.03 ± 0.02 | −0.01 ± 0.02 | −0.01 ± 0.02 | 0.02 ± 0.02  |


The correlation is approximately 0. A point with a high score does not have a larger
audit signal. The median of d_x is small for all points: 0.06 (ε = 100), 0.19
(ε = 1000) and 0.26 (ε = 2000) for pool 0.

**Why.** The score and the signal measure two different things.

- The score: how large is the gradient of the point?
- The signal: does φ of the point change *only* when its own batch is deleted?

With a random assignment, the points of each class are in all 6 batches. The deletion
of a different batch then changes φ of a point as much as the deletion of its own batch
(median shift 0.0035 against 0.0032 for a random pool).

### Why the global hard pool is bad

The global hard pool takes the 4,500 highest scores with no limit for each class.

1. **The pool has few classes.** 32 of the 100 classes. 9 classes are 67 % of the pool.
 One class gives 446 of its 450 training points.
2. **All 6 batches look the same.** The assignment is random, thus each batch is the
 same mixture of the same few classes.
3. **The auditor cannot tell the batches apart.** The deletion of a different batch
 changes φ of a point more (0.0080) than the deletion of its own batch (0.0066).
4. **The points repeat each other.** Noise moves all points of one class together.
 750 points of a batch give as much information as approximately 18 independent
 points. For the random, easy and hard pools, the points are almost independent.

In one sentence: many points of few classes in each batch give the auditor the same
information many times, and that information does not identify the batch.

## 5. Batch assignment (class-pure batches)


| Pool            | Assignment | ε = 100  | ε = 1000 | ε = 2000 |
| --------------- | ---------- | -------- | -------- | -------- |
| hard            | random     | 0.05     | 1.16     | 2.12     |
| hard            | label      | **2.55** | 4.03     | 4.03     |
| hard            | grad       | 2.15     | 4.03     | 4.03     |
| random (pool 0) | random     | 0.02     | 0.82     | 1.73     |
| random (pool 0) | label      | **2.39** | 4.03     | 4.03     |
| random (pool 0) | grad       | 1.93     | 4.03     | 4.03     |
| easy            | random     | 0.02     | 0.79     | 1.62     |
| easy            | label      | **2.03** | 4.03     | 4.03     |
| easy            | grad       | 1.78     | 4.03     | 4.03     |


The value 4.03 is the maximum of the test. Each label and grad row is one run.

Lower claimed ε for the label assignment:


| Pool            | ε = 10 | ε = 20 | ε = 50 | ε = 100 |
| --------------- | ------ | ------ | ------ | ------- |
| hard            | 0.07   | 0.27   | 1.26   | 2.55    |
| random (pool 0) | 0.05   | 0.23   | 1.05   | 2.39    |


Class-pure batches at ε = 50 give approximately the same ε_LB as mixed batches at
ε = 1000 (1.05 to 1.26 against 0.82 to 1.17).

### Why class-pure batches work

Median over the forget points:


| Quantity                                     | Random assignment | Label or grad assignment |
| -------------------------------------------- | ----------------- | ------------------------ |
| Shift of φ when the own batch is deleted     | 0.003 to 0.004    | 0.044 to 0.056           |
| Shift of φ when a different batch is deleted | 0.003 to 0.005    | 0.013 to 0.018           |
| Noise of φ at ε = 1000 (standard deviation)  | 0.015 to 0.017    | 0.016 to 0.018           |
| Shift of the own batch divided by the noise  | 0.20 to 0.25      | 2.7 to 3.1               |


1. **A class-pure batch contains almost all forget points of 17 or 18 classes.**
 When the auditor deletes the batch, the model loses these classes together.
2. **The signal is 13 times larger.** The noise is the same.
3. **The signal identifies the batch.** The own batch gives 3 times the shift of a
 different batch. With a random assignment the two shifts are equal.

The gradient directions show the same picture before any audit:


| Assignment       | Alignment of the directions in a batch | Cosine between two batches |
| ---------------- | -------------------------------------- | -------------------------- |
| random or spread | 0.06                                   | 0.61 to 0.65               |
| label or grad    | 0.23                                   | 0.14 to 0.15               |


- Alignment: how much the points of a batch pull the model in the same direction.
Random directions give 0.037.
- Cosine between two batches: how similar the pull of two batches is. 0 is fully
different.
- Class-pure batches pull 4 times more strongly, and each batch pulls in its own
direction. The pool (hard, random, easy) does not change these values.

### How similar are the label and grad assignments?

They are the same type of design, found in two different ways.


| Property                                   | label                                  | grad                                           |
| ------------------------------------------ | -------------------------------------- | ---------------------------------------------- |
| Input                                      | the class and superclass of each point | the gradient direction of each point, no label |
| Classes in one batch                       | 17 or 18                               | 17 or 18                                       |
| Share of a class that is in one batch      | 98.7 %                                 | 98.7 %                                         |
| Share of a superclass that is in one batch | 93 %                                   | 43 %                                           |
| Alignment in a batch                       | 0.23                                   | 0.23                                           |
| Cosine between two batches                 | 0.14                                   | 0.15                                           |
| ε\_LB at ε = 100 (hard, random, easy pool) | 2.55, 2.39, 2.03                       | 2.15, 1.93, 1.78                               |


- **Same:** the two assignments keep each class in one batch. The gradient clustering
finds the classes without the labels.
- **Different:** which classes go together. Label puts the 5 classes of a superclass
into one batch. Grad does not follow the superclasses. Only 26 % of the points are
in the same batch in the two assignments.
- **Result:** label is higher by 0.25 to 0.46.

Conclusion: "each class in one batch" gives most of the gain. "Similar classes in one
batch" adds a smaller gain.

### The pool with class-pure batches

The order easy < random < hard holds at ε = 100: 2.03 < 2.39 < 2.55 (label) and
1.78 < 1.93 < 2.15 (grad). Each value is one run. The pool effect (approximately 0.4)
is small compared with the assignment effect (approximately 2).

## 6. Calibration budget and decoder

The budget is the number of calibration noise draws for each sign vector. A lower
budget is a cheaper audit.


| Decoder      | Mean of φ                 | Variance of φ            | Parameters for each point                          |
| ------------ | ------------------------- | ------------------------ | -------------------------------------------------- |
| `b2` (paper) | one for each sign vector  | one for each sign vector | 20 means, 20 variances                             |
| `ours_a`     | one for each sign vector  | one for all sign vectors | 20 means, 1 variance                               |
| `ours_b`     | linear in the sign vector | one for all sign vectors | 1 offset, 1 coefficient for each batch, 1 variance |


- Budgets 2 and 10: mean ± standard deviation of 10 fits on disjoint calibration draws.
- Budget 100: one fit. `b2` with the budget 100 is the audit of sections 4 and 5.
- `ours_b` (5 pairs): a sparse design that uses 10 of the 20 sign vectors for the
calibration.

### Random assignment (mixed batches)


| Pool            | ε    | Decoder            | Budget 2        | Budget 10       | Budget 100 |
| --------------- | ---- | ------------------ | --------------- | --------------- | ---------- |
| random (pool 0) | 1000 | `b2`               | 0.01 ± 0.01     | 0.30 ± 0.04     | 0.82       |
|                 |      | `ours_a`           | 0.13 ± 0.03     | 0.41 ± 0.04     | 0.92       |
|                 |      | `ours_b`           | **0.34 ± 0.04** | **0.73 ± 0.04** | **1.12**   |
|                 |      | `ours_b` (5 pairs) | 0.15 ± 0.04     | 0.44 ± 0.05     | 0.91       |
| random (pool 0) | 2000 | `b2`               | 0.03 ± 0.02     | 0.71 ± 0.05     | 1.73       |
|                 |      | `ours_a`           | 0.38 ± 0.05     | 0.99 ± 0.06     | 1.88       |
|                 |      | `ours_b`           | **0.93 ± 0.08** | **1.58 ± 0.08** | **2.07**   |
|                 |      | `ours_b` (5 pairs) | 0.36 ± 0.07     | 1.00 ± 0.11     | 1.81       |
| hard            | 1000 | `b2`               | 0.01 ± 0.01     | 0.49 ± 0.03     | 1.17       |
|                 |      | `ours_a`           | 0.25 ± 0.03     | 0.67 ± 0.05     | 1.19       |
|                 |      | `ours_b`           | **0.59 ± 0.06** | **1.10 ± 0.08** | **1.41**   |
|                 |      | `ours_b` (5 pairs) | 0.23 ± 0.04     | 0.66 ± 0.07     | 1.29       |
| hard            | 2000 | `b2`               | 0.07 ± 0.03     | 1.13 ± 0.07     | 2.15       |
|                 |      | `ours_a`           | 0.66 ± 0.08     | 1.46 ± 0.09     | 2.15       |
|                 |      | `ours_b`           | **1.40 ± 0.07** | **2.02 ± 0.10** | 2.15       |
|                 |      | `ours_b` (5 pairs) | 0.55 ± 0.08     | 1.44 ± 0.13     | 2.15       |


With ε = 100 each cell is 0.07 or less.

### Label assignment (class-pure batches)


| Pool            | ε   | Decoder            | Budget 2        | Budget 10       | Budget 100 |
| --------------- | --- | ------------------ | --------------- | --------------- | ---------- |
| random (pool 0) | 50  | `b2`               | 0.00 ± 0.01     | 0.38 ± 0.03     | 1.05       |
|                 |     | `ours_a`           | 0.18 ± 0.03     | 0.52 ± 0.03     | 1.11       |
|                 |     | `ours_b`           | **0.46 ± 0.05** | **0.99 ± 0.09** | **1.31**   |
|                 |     | `ours_b` (5 pairs) | 0.20 ± 0.04     | 0.55 ± 0.09     | 1.21       |
| random (pool 0) | 100 | `b2`               | 0.02 ± 0.01     | 1.07 ± 0.02     | 2.39       |
|                 |     | `ours_a`           | 0.60 ± 0.06     | 1.44 ± 0.07     | 2.55       |
|                 |     | `ours_b`           | **1.42 ± 0.14** | **2.18 ± 0.13** | **2.78**   |
|                 |     | `ours_b` (5 pairs) | 0.52 ± 0.07     | 1.40 ± 0.17     | 2.55       |
| hard            | 50  | `b2`               | 0.00 ± 0.00     | 0.47 ± 0.03     | 1.26       |
|                 |     | `ours_a`           | 0.23 ± 0.04     | 0.63 ± 0.03     | 1.37       |
|                 |     | `ours_b`           | **0.58 ± 0.08** | **1.15 ± 0.11** | **1.53**   |
|                 |     | `ours_b` (5 pairs) | 0.24 ± 0.04     | 0.66 ± 0.10     | 1.34       |
| hard            | 100 | `b2`               | 0.01 ± 0.01     | 1.29 ± 0.04     | 2.55       |
|                 |     | `ours_a`           | 0.74 ± 0.06     | 1.69 ± 0.09     | 2.61       |
|                 |     | `ours_b`           | **1.70 ± 0.17** | **2.60 ± 0.27** | **3.31**   |
|                 |     | `ours_b` (5 pairs) | 0.59 ± 0.10     | 1.66 ± 0.17     | 2.78       |


- ε = 10: each cell is 0.11 or less. ε = 20: each cell is 0.41 or less.
- ε = 1000 and 2000: each cell is at the maximum 4.03, with one exception. `b2` with
the budget 2 gives 0.8 to 0.9 (ε = 1000) and 1.6 to 1.9 (ε = 2000).

### What the numbers show

- **`ours_b` is the highest decoder for each configuration, ε and budget.**
- **`ours_b` with 10 draws is near `b2` with 100 draws.**
  
  | Pool, assignment, ε  | `ours_b`, budget 10 | `b2`, budget 100 |
  | -------------------- | ------------------- | ---------------- |
  | random, random, 1000 | 0.73                | 0.82             |
  | random, random, 2000 | 1.58                | 1.73             |
  | hard, random, 1000   | 1.10                | 1.17             |
  | hard, random, 2000   | 2.02                | 2.15             |
  | random, label, 100   | 2.18                | 2.39             |
  | hard, label, 100     | 2.60                | 2.55             |
  
- **Fewer parameters need fewer draws.** `b2` estimates 20 means and 20 variances for
each point. With 2 draws it fails. `ours_a` uses one variance. `ours_b` also replaces
the 20 means with 7 numbers.
- **Why `ours_b` can do this for R2D:** it assumes that each deleted batch adds its own
shift to φ, and that the shifts of two batches add. The results agree with this
assumption: `ours_b` is not below `ours_a` in any cell with the full design.
- **The sparse design is not useful.** It uses half of the sign vectors, and `ours_b`
decreases to the level of `ours_a`.
- **The design gain and the decoder gain add.** The hard pool is above the random
pool 0, with the random and with the label assignment, for each decoder and each
budget.

## 7. Limits

- Each audit number is one decoder fit, with one set of calibration noise and one set
of evaluation noise.
- The global hard pool, the spread assignment, and each label and grad design have
one run.
- The comparison of the hard pool with the random pool is not conclusive.
- The maximum of the test is 4.03. For class-pure batches with ε = 1000 or more, the
test cannot show a difference between the pools or the decoders.
- The difficulty scores come from one reference model.
- The calibration sweep covers 4 designs: the random pool 0 and the hard pool, each
with the random and the label assignment.
- The explanations use measured shifts and noise of φ. They agree with the results,
but no experiment isolates each cause.