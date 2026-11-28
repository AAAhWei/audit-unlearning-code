# CIFAR-100 joint sign-vector auditing against exact accounting: experiment summary (2026/11/28)

## Objective

Measure how much of an exactly known unlearning parameter the original B2 joint sign-vector auditor can recover. The contribution is an empirical diagnosis of practical auditing bottlenecks: prediction quality, calibration budget, and the conversion from prediction results to an epsilon lower bound.

We use a ridge-regression mechanism whose output distributions permit exact accounting. This lets us measure audit shortfalls directly. The experiment uses the teammates’ auditor and calibration budgets, while adapting the learning mechanism to make that reference available.

## Main findings

At the main operating point, **K = 20,000 noisy unlearning steps**, the **exact family epsilon is 7.2628 at delta = 2 × 10⁻⁵**. B2 with 50 or 100 calibration models per assignment predicts close to chance and produces almost zero epsilon lower bounds.

An oracle with access to the complete released model and its exact distributions predicts substantially better. However, its median audited epsilon recovers only **2.4% of the exact value through fixed-delta conversion** and **19.5% through Gaussian-assisted conversion**.

Estimating B2’s calibration statistics from one million direct draws per assignment raises its mean overlap from roughly **3.1 to 3.6**, compared with roughly **4.1 for the oracle**. A second independent calibration bank reproduces the average performance. These diagnostics show that much larger calibration helps, while leaving an unresolved gap to the oracle.

A supplementary experiment reduces K to **2,000** while keeping the per-step noise scale fixed. B2 then predicts substantially better, and doubling calibration produces a clear improvement. However, exact epsilon rises to **186.8086**. The oracle makes no observed evaluation errors and reaches the converters’ ceilings, which recover only **2.53%** and **2.81%** of this larger exact value.

## CIFAR-100 setup and exact reference

Each assignment selects three of six candidate batches (750 examples per batch) to be included in training and subsequently deleted (+1); the other three batches were never trained on (−1). There are 20 such balanced sign assignments.

**Overlap** counts correctly predicted signs out of six; the reported **mean overlap** averages this count over evaluation models. Random guessing has expected overlap **3**. **Exact-match rate** is the fraction of models for which all six signs are predicted correctly; random guessing among the 20 assignments succeeds **5%** of the time.

| Item | Setting |
|---|---|
| Features and model | Frozen ImageNet-pretrained ResNet-18 features, 512 dimensions; a ridge head with 100 outputs and one-hot labels |
| Data split | 40,500 retained examples, 4,500 candidate examples, 5,000 validation examples |
| Training set per assignment | 42,750 examples: retained data plus three candidate batches |
| Deterministic training | 20,000 full-batch gradient steps from zero |
| Unlearning | K = 20,000 noisy full-batch gradient steps in the main experiment; K = 2,000 in the supplementary experiment; release only the final head |
| Regularization and step size | Ridge coefficient 0.1; step size approximately 3.35747 × 10⁻⁵ |
| Per-step noise scale | 0.2508395 at both operating points, fixed before these experiments |
| Accounting delta | 2 × 10⁻⁵ |

The main operating point comes from the earlier utility-selected mechanism and is shared by all assignments. The supplementary point changes only the number of unlearning steps to one tenth of the training steps. Neither point is recalibrated for this sign-vector experiment; epsilon and accuracy are not held fixed across K.

At each operating point, the **reference procedure** trains from zero on the retained data for 20,000 steps and then runs the same K noisy retained-data steps as the unlearning procedure. Each mechanism is compared with its matching reference, so the K = 2,000 outputs are not compared with the K = 20,000 reference.

Within each operating point, the released heads for all assignments and the matching reference have Gaussian distributions with the same covariance. Their known structure allows exact epsilon to be computed by numerically evaluating analytic expressions. The **family value** is the largest of these exact epsilons across the 20 assignments. At K = 20,000, individual values range from **7.0767 to 7.2628**; the supplementary point is accounted for separately below. This is an exact reference for the specified family and retraining procedure; it is not a uniform certificate over every possible deletion request.

The split was reconstructed from the teammates’ split script.

## Calibration and evaluation

A **calibration bank** is a collection of model releases with known assignments, used to fit the auditor. Each release is generated with fresh unlearning noise.

B2 queries each calibration model on all 4,500 candidate examples. Its score for an example is the log-odds of the softmax probability assigned to that example’s true class. For each example and each assignment, it estimates a Gaussian mean and variance from the calibration scores. To predict an evaluation model’s assignment, it sums the fitted log-densities over examples and selects the highest-scoring assignment. This likelihood treats the example scores as independent.

| Calibration budget | Models per assignment | Total calibration models per repetition |
|---|---:|---:|
| Gamma = 50 | 50 | 1,000 |
| Gamma = 100 | 100 | 2,000 |

Within a repetition, the smaller bank consists of the first 50 models per assignment from the larger bank. The budget counts **model releases**, not the number of queried examples.

Each repetition also uses **2,000 fresh evaluation models**, with assignments sampled uniformly from the 20 possibilities. Evaluation releases are independent of calibration. We run **three repetitions at each operating point**, each with new calibration and evaluation randomness on the same candidate pool. Predictors within a repetition are evaluated on the same models. The two operating points use independent audit noise, so repetitions are not paired across K.

The **overlap-optimal oracle** knows the exact distributions and observes all released model parameters. It chooses the three positive batches that maximize expected overlap. It provides a privileged prediction benchmark. A second oracle that maximizes the probability of recovering the whole assignment performs very similarly; the summary below uses the overlap-optimal one.

## What the audited bounds mean

A converter turns the observed overlap into a statistical lower bound on epsilon. We compare two routes at the same delta:

| Route | Information used | Role |
|---|---|---|
| Fixed-delta overlap bound | Observed overlap, evaluation budget, and the approximate-indistinguishability test | Main comparison against exact accounting |
| Gaussian-assisted bound | Overlap plus the established Gaussian structure of the output distributions | Additional diagnostic using mechanism-specific information |

The detailed report also records the original pure-epsilon test for protocol continuity. This mechanism has no finite pure-epsilon guarantee, so those outputs are not compared with the finite-delta accounting value.

**Recovery** means audited epsilon divided by the exact family epsilon at the same operating point: 7.2628 for K = 20,000 or 186.8086 for K = 2,000. A recovery of 2.4% describes how much of that parameter the audit detects; it is not prediction accuracy or model test accuracy. Each lower bound has nominal 95% confidence individually.

## Main audit results: K = 20,000

The following entries are **medians across the three repetitions**.

| Predictor | Mean overlap | Exact-match rate | Fixed-delta epsilon lower bound | Gaussian-assisted epsilon lower bound |
|---|---:|---:|---:|---:|
| B2, Gamma = 50 | 3.081 | 5.15% | 0 | 7.05 × 10⁻³ |
| B2, Gamma = 100 | 3.074 | 6.30% | 0 | 2.58 × 10⁻⁴ |
| Overlap-optimal oracle | 4.076 | 21.65% | 0.172 | 1.418 |

**B2 detects almost none of the known epsilon at these budgets.** Across all six baseline audits, overlap ranges from 3.030 to 3.114. The fixed-delta bound is exactly zero in four cases and at most 7.1 × 10⁻⁴; the largest Gaussian-assisted bound is 4.6 × 10⁻². A zero lower bound means the audit is inconclusive, not that distinguishability is zero.

**Doubling the calibration budget produces no clearly resolved improvement in this range.** The paired overlap change from 50 to 100 models per assignment ranges from −7 × 10⁻³ to 1.9 × 10⁻², with standard errors around 3.6 × 10⁻². The lower median at Gamma = 100 does not establish that more calibration hurts.

**The oracle predicts substantially better, but its audited bound remains loose.** Its overlap ranges from 4.066 to 4.106. Median recovery is 2.4% with fixed-delta conversion and 19.5% with Gaussian-assisted conversion. On these oracle predictions, the latter gives bounds about eight times larger, reflecting the additional distributional information used by that converter.

## What much larger calibration changes at K = 20,000

We keep B2’s prediction rule unchanged and estimate its score means and variances from **one million direct draws per assignment**: 20 million draws in total. These are sampled directly from the known output laws, without simulating every noisy gradient step. This is a diagnostic with privileged calibration access, outside the baseline budgets.

Two independent million-draw banks were evaluated on the same saved evaluation models. Each bank supplies its fitted statistics to all three repetitions.

| Million-draw calibration bank | Median overlap | Fixed-delta epsilon lower bound | Gaussian-assisted epsilon lower bound |
|---|---:|---:|---:|
| Bank 0 | 3.605 | 5.3 × 10⁻² | 0.710 |
| Bank 1, independent draws | 3.613 | 5.4 × 10⁻² | 0.722 |

For bank 0, overlap improves by **0.47–0.58** over baseline B2 on the same evaluation models. Its median bounds recover approximately **0.7%** and **9.8%** of exact epsilon. Bank 1 gives similar results.

The second bank also supports a check over increasing draw counts, from 100 to one million per assignment. Performance improves substantially over that range. At its last step, from 300,000 to one million draws, overlap changes by **−5 × 10⁻³ to 2.3 × 10⁻²**, with standard errors around 1.7 × 10⁻² to 1.8 × 10⁻². The two independent million-draw banks differ by **−8 × 10⁻³ to 2.0 × 10⁻²**, with similar standard errors.

These comparisons resolve little additional change in average performance at the largest tested budgets. However, the banks predict the same complete assignment on only **75–78%** of evaluation models. Similar average overlap does not imply stable individual predictions.

The oracle remains approximately **0.42–0.53 overlap units above** the million-draw diagnostics. The checks stop at one million draws, so they do not establish an infinite-calibration limit or identify the cause of this residual gap. Their standard errors reflect evaluation-model randomness with the banks held fixed; two banks do not characterize the full variability of calibration.

## Supplementary experiment: K = 2,000

This experiment uses the same assignments, retained data, training procedure, per-step noise scale, B2 implementation, calibration budgets, and 2,000 evaluation models per repetition. Only the number of noisy unlearning steps changes, and the reference also takes 2,000 noisy steps. No noise recalibration is performed.

The shorter unlearning run attenuates differences between assignment means less and accumulates less noise. The **exact family epsilon rises to 186.8086**; individual assignment values range from 182.3865 to 186.8086. The following results are **medians across three repetitions**.

| Predictor | Mean overlap | Exact-match rate | Fixed-delta epsilon lower bound | Gaussian-assisted epsilon lower bound |
|---|---:|---:|---:|---:|
| B2, Gamma = 50 | 5.235 | 62.25% | 0.829 | 3.429 |
| B2, Gamma = 100 | 5.557 | 77.85% | 1.183 | 4.087 |
| Overlap-optimal oracle | 6.000 | 100% observed | 4.723 | 5.242 |

**B2 predicts substantially better at this operating point.** It identifies most complete assignments, compared with near-chance performance at K = 20,000. Doubling the calibration budget improves overlap by **0.295–0.375** within each repetition, with paired standard errors around **2.2 × 10⁻²**. This is a clear budget effect in the tested range.

**The oracle makes no observed errors across all 6,000 evaluation models.** Its bounds therefore reach the two converters’ ceilings at this evaluation budget. These bounds recover only **2.53%** and **2.81%** of exact epsilon. No predictor feeding these same converters can exceed those ceilings at the fixed budget. Zero observed errors does not establish that the oracle’s population error rate is exactly zero.

**Higher audited epsilon still leaves a large accounting gap.** B2 at Gamma = 100 recovers approximately **0.63%** of exact epsilon through fixed-delta conversion and **2.19%** through Gaussian-assisted conversion. The oracle’s Gaussian-assisted bound rises from 1.418 to 5.242, but its recovery falls from 19.5% to 2.81% because exact epsilon increases much more.

The smaller oracle–B2 gap at K = 2,000 is measured in the same overlap units as at K = 20,000. However, assignment distinguishability also changes, so this comparison does not identify which B2 limitation accounts for the change. **No million-draw calibration diagnostic was run at K = 2,000**; the large-calibration findings above concern K = 20,000 only.

## Why the oracle still falls below exact accounting

The oracle is optimal for expected sign overlap. Its predictions still pass through the sign-vector audit and its statistical converter. Three distinctions help interpret the remaining gap:

- **The comparison differs.** Sign prediction distinguishes assignments from one another. Exact accounting compares each assignment with the retraining reference. A shift away from the reference shared by all assignments is invisible to the sign-prediction task.
- **Assignments remain difficult to distinguish.** At K = 20,000, a separate Monte Carlo calculation estimates the oracle’s population overlap at 4.086 out of six, and its exact-match rate at about 22%. These are estimates of prediction performance, not exact audit bounds.
- **Conversion also limits recovery.** At 2,000 evaluation models, perfect prediction gives converter outputs of only **4.723** through the fixed-delta route and **5.242** through the Gaussian-assisted route. The ceilings are the same at both K values because the audit protocol and evaluation budget are unchanged. At K = 2,000, the oracle reaches these ceilings in every repetition, even though exact epsilon is 186.8086.

The experiment does not assign a separate share of the observed gap to each effect. The oracle’s observed bound at K = 20,000 is a benchmark, not a universal ceiling on every realized audit bound. At K = 2,000, its observed bounds coincide with the separately established ceilings of these particular converters. Between B2 and the oracle, score compression, the assumed independent Gaussian likelihood, and finite calibration remain possible contributors.

## Model accuracy

Utility is evaluated on 64 independently sampled released models per repetition. The table reports the median and range of the mean test accuracies across the three repetitions.

| Operating point | Median test accuracy | Range across repetitions |
|---|---:|---:|
| K = 20,000 | 59.97% | 59.95–59.98% |
| K = 2,000 | 60.89% | 60.89–60.97% |

The standard deviation across models is about 0.30 percentage points at K = 20,000 and 0.24–0.26 percentage points at K = 2,000. Without random noise, the median accuracies of the mean released heads are 63.11% and 63.10%, respectively.

The audit gap therefore occurs at useful overall classification accuracy at both operating points. The slightly higher accuracy at K = 2,000 accompanies a much larger exact epsilon.

## What the evidence supports and its limits

The main experiment shows that B2 can return almost zero bounds despite a substantial exact epsilon, that substantially larger calibration improves B2’s prediction, and that considerable audit shortfalls remain under privileged calibration and oracle prediction. The supplementary experiment shows that B2 can predict much better at another operating point while still recovering little of exact epsilon. It also directly demonstrates the converters’ ceilings when the oracle makes no observed errors.

For the final project, this is evidence about the practical limitations of an existing auditing pipeline. Calibration-efficient methods and stronger canaries are motivated follow-up directions. They are not demonstrated improvements in this experiment. A stronger canary can also increase the true epsilon, so a larger audited value alone would not establish tighter recovery.

The results cover **one candidate pool, one family of 20 assignments, two operating points, and one evaluation budget**. Repetition ranges describe calibration and evaluation randomness on that pool. Confidence statements are pointwise, and results sharing evaluation models or nested calibration samples are dependent.

The B2 implementation and ported converters matched the teammates’ code on controlled inputs. Deterministic GD and the closed form agreed to numerical precision, and the noisy sampler passed the saved validation checks. No fixed-delta or Gaussian-assisted bound exceeded the matching exact accounting value in the 15 main audit rows, the 27 additional convergence rows, or the 12 supplementary audit rows. These are consistency checks rather than a proof of coverage.

The teammates use different learning mechanisms. Their results and ours can demonstrate how the same audit behaves in different settings, but cannot isolate an auditor improvement across mechanisms. Agreement with their actual data-split folder remains to be checked.

## Evidence and reproducibility

This summary uses results reviewed through repository commit `61d1511`. The main result-table values were checked against the saved CSVs. The previous binary-audit results are outside the scope of this summary.

| Evidence | Sources |
|---|---|
| Detailed joint sign-vector report | [Report](unlearning-auditing-accounting/joint-sign-vector-report.md) |
| Family setup and exact accounting, K = 20,000 | [Configuration](unlearning-auditing-accounting/results/cifar100_jsv/config.json), [accounting values](unlearning-auditing-accounting/results/cifar100_jsv/account.json), [report](unlearning-auditing-accounting/results/cifar100_jsv/summary.md); fingerprint `44e3dfeb59d0c2b0` |
| Main audits, oracle, and first million-draw bank, K = 20,000 | [Audit CSV](unlearning-auditing-accounting/results/cifar100_jsv_audit/audit.csv), [report](unlearning-auditing-accounting/results/cifar100_jsv_audit/summary.md); fingerprint `31d76126813b9415` |
| Calibration convergence and independent-bank comparison | [Convergence CSV](unlearning-auditing-accounting/results/cifar100_jsv_moments/convergence.csv), [bank comparison](unlearning-auditing-accounting/results/cifar100_jsv_moments/banks.csv), [report](unlearning-auditing-accounting/results/cifar100_jsv_moments/summary.md); fingerprint `142c1f3589ee5e4e` |
| Supplementary accounting, K = 2,000 | [Configuration](unlearning-auditing-accounting/results/cifar100_jsv_k2000/config.json), [accounting values](unlearning-auditing-accounting/results/cifar100_jsv_k2000/account.json), [report](unlearning-auditing-accounting/results/cifar100_jsv_k2000/summary.md); fingerprint `e677c6fddcda4815` |
| Supplementary audit and comparison | [Audit CSV](unlearning-auditing-accounting/results/cifar100_jsv_k2000_audit/audit.csv), [report](unlearning-auditing-accounting/results/cifar100_jsv_k2000_audit/summary.md), [comparison across K](unlearning-auditing-accounting/results/cifar100_jsv_k2000_audit/comparison.md); fingerprint `17c177dfdfd127a6` |
| Frozen learning mechanism | [Configuration](unlearning-auditing-accounting/results/cifar100_frozen/config.json); fingerprint `16f801c27e239697` |
