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
batch increase ε_LB at claimed ε = 100 from approximately 0.03 to approximately 2.
2. **The pool is a small effect.** Hard points are above easy points, but random pools
differ from each other by as much.
3. **The linear decoder `ours_b` needs approximately 10 times fewer calibration draws**
than the decoder of the paper for the same ε_LB.

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

The score of a training point is its per-instance privacy loss (Sepahvand et al.). For each point, calculate the weighted sum of the gradient norm at each training step from a reference model. A point with a large gradient is a "hard" point.

### Pools and assignments


| Pool        | Forget points                          |
| ----------- | -------------------------------------- |
| paper       | the split of the paper                 |
| random      | random, 45 points for each class       |
| easy        | the 45 lowest scores in each class     |
| hard        | the 45 highest scores in each class    |
| global hard | the 4,500 highest scores of all points |





| Assignment | Batch of each point                                           |
| ---------- | ------------------------------------------------------------- |
| random     | random                                                        |
| label      | samples in one superclass goes into the same batch            |
| grad       | points with a similar gradient direction go into one batch    |
| spread     | the opposite of grad: each batch gets all gradient directions |


A design is one pool with one assignment, for example "hard pool, label assignment".

## 4. Pool selection (random assignment)


| Pool        | Assignment | Runs | ε = 100 | ε = 1000            | ε = 2000            |
| ----------- | ---------- | ---- | ------- | ------------------- | ------------------- |
| paper       | random     | 1    | 0.05    | 1.04                | 1.93                |
| random      | random     | 6    | 0.03    | 0.95 \[0.82, 1.20\] | 1.79 \[1.52, 2.10\] |
| easy        | random     | 3    | 0.02    | 0.79 \[0.77, 0.82\] | 1.62 \[1.58, 1.65\] |
| hard        | random     | 3    | 0.05    | 1.16 \[1.14, 1.18\] | 2.12 \[2.03, 2.18\] |
| global hard | random     | 1    | 0.03    | 0.43                | 0.78                |


The cells give the mean [min, max] of the runs. With ε = ∞ each configuration is at the maximum 4.03.

The 6 random pools:


| Pool     | 0    | 1    | 2    | 3    | 4    | 5    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- |
| ε = 1000 | 0.82 | 0.82 | 1.16 | 1.20 | 0.84 | 0.86 |
| ε = 2000 | 1.73 | 1.78 | 2.03 | 2.10 | 1.60 | 1.52 |


### What the numbers show

- **Hard is above easy:** +0.37 (ε = 1000) and +0.50 (ε = 2000).
- **Hard is not clearly above random.** Two of the 6 random pools (2 and 3) are as high
as the hard pool.

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

**Check.** The Spearman rank correlation between d_x and the difficulty score of x.
We calculate it for the points of each class and then take the mean of the 100 classes.

**Result** (3 random pools with a random assignment, mean ± standard error):


| Random pool | ε = 100      | ε = 1000     | ε = 2000     | ε = ∞        |
| ----------- | ------------ | ------------ | ------------ | ------------ |
| 0           | −0.01 ± 0.02 | −0.04 ± 0.02 | −0.04 ± 0.02 | −0.01 ± 0.02 |
| 1           | −0.02 ± 0.02 | −0.03 ± 0.02 | −0.03 ± 0.02 | 0.00 ± 0.02  |
| 2           | −0.03 ± 0.02 | −0.01 ± 0.02 | −0.01 ± 0.02 | 0.02 ± 0.02  |


The correlation is approximately 0. A point with a high score does not have a larger audit signal.

**Why.** The score and the signal measure two different things.

- The score: how hard is it to unlearn this one point? Calculated from the size of the gradient of the point during training.
- The signal: does φ of the point change *only*  when its own batch is deleted?

A point that is hard to unlearn alone is not always a point that shows which batch of
750 was deleted. 

### Why the global hard pool is bad

The global hard pool takes the 4,500 highest scores with no limit for each class.

1. **The pool has few classes.** 32 of the 100 classes. 9 classes are 67 % of the pool.
One class gives 446 of its 450 training points.
2. **All 6 batches look the same.** The assignment is random, thus each batch is the
same mixture of the same few classes.

In one sentence: many points of few classes in each batch give the auditor the same
information many times, and that information does not identify the batch.

## 5. Batch assignment (class-pure batches)


| Pool   | Assignment | ε = 100  | ε = 1000 | ε = 2000 |
| ------ | ---------- | -------- | -------- | -------- |
| hard   | random     | 0.05     | 1.16     | 2.12     |
| hard   | label      | **2.55** | 4.03     | 4.03     |
| hard   | grad       | 2.15     | 4.03     | 4.03     |
| hard   | spread     | 0.04     | 1.41     | 2.49     |
| random | random     | 0.02     | 0.82     | 1.73     |
| random | label      | **2.39** | 4.03     | 4.03     |
| random | grad       | 1.93     | 4.03     | 4.03     |
| easy   | random     | 0.02     | 0.79     | 1.62     |
| easy   | label      | **2.03** | 4.03     | 4.03     |
| easy   | grad       | 1.78     | 4.03     | 4.03     |


The value 4.03 is the maximum of the test. Each label, grad and spread row is one run.

Lower claimed ε for the label assignment:


| Pool   | ε = 10 | ε = 20 | ε = 50 | ε = 100 |
| ------ | ------ | ------ | ------ | ------- |
| hard   | 0.07   | 0.27   | 1.26   | 2.55    |
| random | 0.05   | 0.23   | 1.05   | 2.39    |


Class-pure batches at ε = 50 give approximately the same ε_LB as mixed batches at
ε = 1000 (1.05 to 1.26 against 0.82 to 1.17).

### Why class-pure batches work

Median over the forget points:


| Quantity                                     | Random assignment | Label or grad assignment |
| -------------------------------------------- | ----------------- | ------------------------ |
| Shift of φ when the own batch is deleted     | \[0.003, 0.004\]  | \[0.044, 0.056\]         |
| Shift of φ when a different batch is deleted | \[0.003, 0.005\]  | \[0.013, 0.018\]         |


**The signal identifies the batch.** The own batch gives 3 times the shift of a
different batch. With a random assignment the two shifts are equal.

The gradient directions show the same picture before any audit:


| Assignment      | Alignment in a batch | Cosine between two batches |
| --------------- | -------------------- | -------------------------- |
| random / spread | 0.06                 | \[0.61, 0.65\]             |
| label / grad    | 0.23                 | \[0.14, 0.15\]             |


- Alignment in a batch: take the mean of the 750 normalized gradient directions in the batch. The alignment is the length of this mean.
- Cosine between two batches: cosine similarity between the mean gradient directions of the two batches.

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
- **Different:** which classes go together. Label puts the 5 classes of a superclass into one batch. Grad does not follow the rule. 
- **Result:** label is higher by 0.25 to 0.46.

Conclusion: "each class in one batch" gives most of the gain. "Similar classes in one
batch" adds a smaller gain.

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

### Random assignment


| Pool   | ε    | Decoder            | Budget 2        | Budget 10       | Budget 100 |
| ------ | ---- | ------------------ | --------------- | --------------- | ---------- |
| random | 1000 | `b2`               | 0.01 ± 0.01     | 0.30 ± 0.04     | 0.82       |
|        |      | `ours_a`           | 0.13 ± 0.03     | 0.41 ± 0.04     | 0.92       |
|        |      | `ours_b`           | **0.34 ± 0.04** | **0.73 ± 0.04** | **1.12**   |
|        |      | `ours_b` (5 pairs) | 0.15 ± 0.04     | 0.44 ± 0.05     | 0.91       |
| random | 2000 | `b2`               | 0.03 ± 0.02     | 0.71 ± 0.05     | 1.73       |
|        |      | `ours_a`           | 0.38 ± 0.05     | 0.99 ± 0.06     | 1.88       |
|        |      | `ours_b`           | **0.93 ± 0.08** | **1.58 ± 0.08** | **2.07**   |
|        |      | `ours_b` (5 pairs) | 0.36 ± 0.07     | 1.00 ± 0.11     | 1.81       |
| hard   | 1000 | `b2`               | 0.01 ± 0.01     | 0.49 ± 0.03     | 1.17       |
|        |      | `ours_a`           | 0.25 ± 0.03     | 0.67 ± 0.05     | 1.19       |
|        |      | `ours_b`           | **0.59 ± 0.06** | **1.10 ± 0.08** | **1.41**   |
|        |      | `ours_b` (5 pairs) | 0.23 ± 0.04     | 0.66 ± 0.07     | 1.29       |
| hard   | 2000 | `b2`               | 0.07 ± 0.03     | 1.13 ± 0.07     | 2.15       |
|        |      | `ours_a`           | 0.66 ± 0.08     | 1.46 ± 0.09     | 2.15       |
|        |      | `ours_b`           | **1.40 ± 0.07** | **2.02 ± 0.10** | 2.15       |
|        |      | `ours_b` (5 pairs) | 0.55 ± 0.08     | 1.44 ± 0.13     | 2.15       |




### Label assignment


| Pool   | ε   | Decoder            | Budget 2        | Budget 10       | Budget 100 |
| ------ | --- | ------------------ | --------------- | --------------- | ---------- |
| random | 50  | `b2`               | 0.00 ± 0.01     | 0.38 ± 0.03     | 1.05       |
|        |     | `ours_a`           | 0.18 ± 0.03     | 0.52 ± 0.03     | 1.11       |
|        |     | `ours_b`           | **0.46 ± 0.05** | **0.99 ± 0.09** | **1.31**   |
|        |     | `ours_b` (5 pairs) | 0.20 ± 0.04     | 0.55 ± 0.09     | 1.21       |
| random | 100 | `b2`               | 0.02 ± 0.01     | 1.07 ± 0.02     | 2.39       |
|        |     | `ours_a`           | 0.60 ± 0.06     | 1.44 ± 0.07     | 2.55       |
|        |     | `ours_b`           | **1.42 ± 0.14** | **2.18 ± 0.13** | **2.78**   |
|        |     | `ours_b` (5 pairs) | 0.52 ± 0.07     | 1.40 ± 0.17     | 2.55       |
| hard   | 50  | `b2`               | 0.00 ± 0.00     | 0.47 ± 0.03     | 1.26       |
|        |     | `ours_a`           | 0.23 ± 0.04     | 0.63 ± 0.03     | 1.37       |
|        |     | `ours_b`           | **0.58 ± 0.08** | **1.15 ± 0.11** | **1.53**   |
|        |     | `ours_b` (5 pairs) | 0.24 ± 0.04     | 0.66 ± 0.10     | 1.34       |
| hard   | 100 | `b2`               | 0.01 ± 0.01     | 1.29 ± 0.04     | 2.55       |
|        |     | `ours_a`           | 0.74 ± 0.06     | 1.69 ± 0.09     | 2.61       |
|        |     | `ours_b`           | **1.70 ± 0.17** | **2.60 ± 0.27** | **3.31**   |
|        |     | `ours_b` (5 pairs) | 0.59 ± 0.10     | 1.66 ± 0.17     | 2.78       |


### What the numbers show

- **`ours_b` is the highest decoder for each configuration, ε and budget.**
- **`ours_b` with 10 draws is near `b2` with 100 draws.**
  
  | Pool, assignment, ε  | `ours_b`, budget 10 | `b2`, budget 100 |
  | -------------------- | ------------------- | ---------------- |
  | random, random, 1000 | 0.73                | 0.82             |
  | hard, random, 1000   | 1.10                | 1.17             |
  | random, label, 100   | 2.18                | 2.39             |
  | hard, label, 100     | 2.60                | 2.55             |
  
- **Fewer parameters need fewer draws.** `b2` estimates 20 means and 20 variances for
each point. With 2 draws it fails. `ours_a` uses one variance. `ours_b` also replaces
the 20 means with 7 numbers.
- **The design gain and the decoder gain add.** Hard pool with the label assignment reports the highest ε for each decoder and each budget.