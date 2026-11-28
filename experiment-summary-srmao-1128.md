# CIFAR-100 joint sign-vector auditing against exact accounting: experiment summary (2026/11/28)

## Objective

Measure how much of an exactly known unlearning parameter the original B2 joint sign-vector auditor can recover. The contribution is an empirical diagnosis of practical auditing bottlenecks: prediction quality, calibration budget, and the conversion from prediction results to an epsilon lower bound.

We use a ridge-regression mechanism whose output distributions permit exact accounting. This lets us measure audit shortfalls directly. The experiment uses the teammates’ auditor and calibration budgets, while adapting the learning mechanism to make that reference available.

## Main findings

The **exact family epsilon is 7.2628 at delta = 2 × 10⁻⁵**. B2 with 50 or 100 calibration models per assignment predicts close to chance and produces almost zero epsilon lower bounds.

An oracle with access to the complete released model and its exact distributions predicts substantially better. However, its median audited epsilon recovers only **2.4% of the exact value through fixed-delta conversion** and **19.5% through Gaussian-assisted conversion**.

Estimating B2’s calibration statistics from one million direct draws per assignment raises its mean overlap from roughly **3.1 to 3.6**, compared with roughly **4.1 for the oracle**. A second independent calibration bank reproduces the average performance. These diagnostics show that much larger calibration helps, while leaving an unresolved gap to the oracle.

## CIFAR-100 setup and exact reference

Each assignment selects three of six candidate batches (750 examples per batch) to be included in training and subsequently deleted (+1); the other three batches were never trained on (−1). There are 20 such balanced sign assignments.

**Overlap** counts correctly predicted signs out of six; the reported **mean overlap** averages this count over evaluation models. Random guessing has expected overlap **3**. **Exact-match rate** is the fraction of models for which all six signs are predicted correctly; random guessing among the 20 assignments succeeds **5%** of the time.

| Item | Setting |
|---|---|
| Features and model | Frozen ImageNet-pretrained ResNet-18 features, 512 dimensions; a ridge head with 100 outputs and one-hot labels |
| Data split | 40,500 retained examples, 4,500 candidate examples, 5,000 validation examples |
| Training set per assignment | 42,750 examples: retained data plus three candidate batches |
| Deterministic training | 20,000 full-batch gradient steps from zero |
| Unlearning | 20,000 noisy full-batch gradient steps on the retained data; release only the final head |
| Regularization and step size | Ridge coefficient 0.1; step size approximately 3.35747 × 10⁻⁵ |
| Noise scale | 0.2508395, fixed before this experiment |
| Accounting delta | 2 × 10⁻⁵ |

The operating point comes from the earlier utility-selected mechanism and is shared by all assignments. It was not recalibrated for this sign-vector experiment.

The **reference procedure** trains from zero on the retained data for 20,000 steps and then runs the same 20,000 noisy retained-data steps. Consequently, the comparison includes noise on both sides.

For each assignment, the released head and the reference have Gaussian distributions with the same covariance. Their known structure allows exact epsilon to be computed by numerically evaluating analytic expressions. The **family value** is the largest of these exact epsilons across the 20 assignments. Individual values range from **7.0767 to 7.2628**. This is an exact reference for the specified family and retraining procedure; it is not a uniform certificate over every possible deletion request.

The split was reconstructed from the teammates’ split script.

## Calibration and evaluation

A **calibration bank** is a collection of model releases with known assignments, used to fit the auditor. Each release is generated with fresh unlearning noise.

B2 queries each calibration model on all 4,500 candidate examples. Its score for an example is the log-odds of the softmax probability assigned to that example’s true class. For each example and each assignment, it estimates a Gaussian mean and variance from the calibration scores. To predict an evaluation model’s assignment, it sums the fitted log-densities over examples and selects the highest-scoring assignment. This likelihood treats the example scores as independent.

| Calibration budget | Models per assignment | Total calibration models per repetition |
|---|---:|---:|
| Gamma = 50 | 50 | 1,000 |
| Gamma = 100 | 100 | 2,000 |

Within a repetition, the smaller bank consists of the first 50 models per assignment from the larger bank. The budget counts **model releases**, not the number of queried examples.

Each repetition also uses **2,000 fresh evaluation models**, with assignments sampled uniformly from the 20 possibilities. Evaluation releases are independent of calibration. We run **three repetitions**, each with new calibration and evaluation randomness on the same candidate pool. Predictors within a repetition are evaluated on the same models.

The **overlap-optimal oracle** knows the exact distributions and observes all released model parameters. It chooses the three positive batches that maximize expected overlap. It provides a privileged prediction benchmark. A second oracle that maximizes the probability of recovering the whole assignment performs very similarly; the summary below uses the overlap-optimal one.

## What the audited bounds mean

A converter turns the observed overlap into a statistical lower bound on epsilon. We compare two routes at the same delta:

| Route | Information used | Role |
|---|---|---|
| Fixed-delta overlap bound | Observed overlap, evaluation budget, and the approximate-indistinguishability test | Main comparison against exact accounting |
| Gaussian-assisted bound | Overlap plus the established Gaussian structure of the output distributions | Additional diagnostic using mechanism-specific information |

The detailed report also records the original pure-epsilon test for protocol continuity. This mechanism has no finite pure-epsilon guarantee, so those outputs are not compared with the finite-delta accounting value.

**Recovery** means audited epsilon divided by the exact family epsilon, 7.2628. A recovery of 2.4% describes how much of that parameter the audit detects; it is not prediction accuracy or model test accuracy. Each lower bound has nominal 95% confidence individually.

## Main audit results

The following entries are **medians across the three repetitions**.

| Predictor | Mean overlap | Fixed-delta epsilon lower bound | Gaussian-assisted epsilon lower bound |
|---|---:|---:|---:|
| B2, Gamma = 50 | 3.081 | 0 | 7.05 × 10⁻³ |
| B2, Gamma = 100 | 3.074 | 0 | 2.58 × 10⁻⁴ |
| Overlap-optimal oracle | 4.076 | 0.172 | 1.418 |

**B2 detects almost none of the known epsilon at these budgets.** Across all six baseline audits, overlap ranges from 3.030 to 3.114. The fixed-delta bound is exactly zero in four cases and at most 7.1 × 10⁻⁴; the largest Gaussian-assisted bound is 4.6 × 10⁻². A zero lower bound means the audit is inconclusive, not that distinguishability is zero.

**Doubling the calibration budget produces no clearly resolved improvement in this range.** The paired overlap change from 50 to 100 models per assignment ranges from −7 × 10⁻³ to 1.9 × 10⁻², with standard errors around 3.6 × 10⁻². The lower median at Gamma = 100 does not establish that more calibration hurts.

**The oracle predicts substantially better, but its audited bound remains loose.** Its overlap ranges from 4.066 to 4.106. Median recovery is 2.4% with fixed-delta conversion and 19.5% with Gaussian-assisted conversion. On these oracle predictions, the latter gives bounds about eight times larger, reflecting the additional distributional information used by that converter.

## What much larger calibration changes

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

## Why the oracle still falls below exact accounting

The oracle is optimal for expected sign overlap. Its predictions still pass through the sign-vector audit and its statistical converter. Three distinctions help interpret the remaining gap:

- **The comparison differs.** Sign prediction distinguishes assignments from one another. Exact accounting compares each assignment with the retraining reference. A shift away from the reference shared by all assignments is invisible to the sign-prediction task.
- **Assignments remain difficult to distinguish.** A separate Monte Carlo calculation estimates the oracle’s population overlap at 4.086 out of six, and its exact-match rate at about 22%. These are estimates of prediction performance, not exact audit bounds.
- **Conversion also limits recovery.** At 2,000 evaluation models, even perfect prediction would give converter outputs of only **4.723** through the fixed-delta route and **5.242** through the Gaussian-assisted route, below the family value of 7.263. These are converter-specific ceilings at this evaluation budget.

The experiment does not assign a separate share of the observed gap to each effect. Likewise, the oracle’s observed bound is a benchmark, not a universal ceiling on every realized audit bound. Between B2 and the oracle, score compression, the assumed independent Gaussian likelihood, and finite calibration remain possible contributors.

## Model accuracy

Released models achieve approximately **59.97% test accuracy** on CIFAR-100. The means across repetitions range from 59.95% to 59.98%, using 64 independently sampled utility models per repetition. The standard deviation across models is about 0.30 percentage points. For context, the mean released head with no random noise has about 63.11% accuracy.

The audit gap therefore occurs at useful overall classification accuracy.

## What the evidence supports and its limits

The completed work supports three empirical statements: B2 can return almost zero bounds despite a substantial exact epsilon; substantially larger calibration improves B2’s prediction; and considerable audit shortfalls remain under both privileged calibration and oracle prediction.

For the final project, this is evidence about the practical limitations of an existing auditing pipeline. Calibration-efficient methods and stronger canaries are motivated follow-up directions. They are not demonstrated improvements in this experiment. A stronger canary can also increase the true epsilon, so a larger audited value alone would not establish tighter recovery.

The results cover **one candidate pool, one family of 20 assignments, and one evaluation budget**. Repetition ranges describe calibration and evaluation randomness on that pool. Confidence statements are pointwise, and results sharing evaluation models or nested calibration samples are dependent.

The B2 implementation and ported converters matched the teammates’ code on controlled inputs. Deterministic GD and the closed form agreed to numerical precision, and the noisy sampler passed the saved validation checks. No fixed-delta or Gaussian-assisted bound exceeded exact accounting in the 15 original audit rows or the 27 additional convergence rows. These are consistency checks rather than a proof of coverage.

The teammates use different learning mechanisms. Their results and ours can demonstrate how the same audit behaves in different settings, but cannot isolate an auditor improvement across mechanisms. Agreement with their actual data-split folder remains to be checked.

## Evidence and reproducibility

This summary uses results reviewed through repository commit `f537438`. The main result-table values were checked against the saved CSVs. The previous binary-audit results are outside the scope of this summary.

| Evidence | Sources |
|---|---|
| Detailed joint sign-vector report | [Report](unlearning-auditing-accounting/joint-sign-vector-report.md) |
| Family setup and exact accounting | [Configuration](unlearning-auditing-accounting/results/cifar100_jsv/config.json), [accounting values](unlearning-auditing-accounting/results/cifar100_jsv/account.json), [report](unlearning-auditing-accounting/results/cifar100_jsv/summary.md); fingerprint `44e3dfeb59d0c2b0` |
| Baseline audits, oracle, and first million-draw bank | [Audit CSV](unlearning-auditing-accounting/results/cifar100_jsv_audit/audit.csv), [report](unlearning-auditing-accounting/results/cifar100_jsv_audit/summary.md); fingerprint `31d76126813b9415` |
| Calibration convergence and independent-bank comparison | [Convergence CSV](unlearning-auditing-accounting/results/cifar100_jsv_moments/convergence.csv), [bank comparison](unlearning-auditing-accounting/results/cifar100_jsv_moments/banks.csv), [report](unlearning-auditing-accounting/results/cifar100_jsv_moments/summary.md); fingerprint `142c1f3589ee5e4e` |
| Frozen learning mechanism | [Configuration](unlearning-auditing-accounting/results/cifar100_frozen/config.json); fingerprint `16f801c27e239697` |
