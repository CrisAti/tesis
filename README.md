# Hybrid Transfer Metric: Ruta C CLM-JMDS

This project implements the selected Ruta C architecture: a hybrid metric for
estimating whether a labeled dataset can be reused as a size-constrained
clustering problem.

The implementation combines:

- a global structure score inspired by CLM / cluster label matching ideas from
  Jeon et al., using internal validation measures adjusted against label
  permutations that preserve class proportions: https://arxiv.org/abs/2503.01097
- a local feasibility score inspired by JMDS from Lee, Jung, Yim and Yoon,
  adapting the product of data confidence and model confidence to a supervised
  labeled dataset setting: https://proceedings.mlr.press/v162/lee22c.html

This is an operational implementation, not a full reproduction of every
calibration protocol in either paper.

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or install only the direct requirements:

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `pytest` for tests

## Metric Definition

The final score is:

```text
T = sqrt(G * L)
```

where:

- `G in [0, 1]`: global CLM-inspired structure score.
- `L in [0, 1]`: local preservation factor under target size constraints.
- `C* in [0, 1]`: minimum local cost returned by the assignment solver.
- `L = 1 - C*`.

Interpretation:

- `T` close to `1`: the labels define a strong clustering structure and the
  target size constraints can be satisfied with little structural damage.
- `T` close to `0`: the labels are not globally cluster-like, or the target
  constraints require moving highly reliable/prototypical samples.

The report also includes:

- `moved_fraction`: fraction of sample mass assigned away from its original
  class.
- `assignment_matrix`: source-to-target class mass matrix.
- `observed_proportions`: empirical label proportions.
- `target_proportions`: requested feasible proportions.

## How CLM Is Approximated

The global component computes two internal validation measures:

- Silhouette score.
- Calinski-Harabasz score.

Each is adjusted against random label permutations that preserve the original
class counts. This follows the CLM motivation that a useful dataset-level
measure should capture correspondence between labels and cluster structure
instead of non-structural properties such as sample count or class balance.

Silhouette is chance-adjusted against its natural upper bound of `1`. The
Calinski-Harabasz component is log-scaled and chance-adjusted because the raw
index is unbounded.

## How JMDS Is Adapted

For each sample `i` and possible target class `k`, the reliability matrix is:

```text
R[i, k] = p_data[i, k] * p_model[i, k]
```

where:

- `p_data`: local kNN leave-one-out class probability.
- `p_model`: cross-validated logistic regression class probability.

Moving sample `i` from its observed class `y_i` to target class `k` costs:

```text
max(0, R[i, y_i] - R[i, k])
```

The size-constrained local problem is solved with `scipy.optimize.linprog`.
The solver assigns sample mass to target classes while matching the requested
target proportions.

## Iris Experiments

Run the original three-class Iris experiment:

```bash
python -m experiments.run_iris --variant original --scenario natural
python -m experiments.run_iris --variant original --scenario stress
```

Run the binary Setosa/Otra experiment:

```bash
python -m experiments.run_iris --variant binary --scenario natural
python -m experiments.run_iris --variant binary --scenario stress
```

You can also run both scenarios for one variant:

```bash
python -m experiments.run_iris --variant original --scenario all
python -m experiments.run_iris --variant binary --scenario all
```

Expected behavior:

- Original Iris, natural `q=(1/3, 1/3, 1/3)`: `L=1`; `T` is medium-high and
  mostly limited by the overlap between Versicolor and Virginica.
- Original Iris, stress `q=(0.20, 0.40, 0.40)`: `L` and `T` drop because Setosa
  must be reduced even though it is compact.
- Binary Setosa/Otra, natural `q=(1/3, 2/3)`: `G` and `T` should improve versus
  original Iris because the binary separation is cleaner.
- Binary Setosa/Otra, stress `q=(0.50, 0.50)`: `G` remains high, while `L`
  drops because mass from Otra must be moved into Setosa.

## Tests

```bash
pytest
```

The tests check:

- scores remain in `[0, 1]`;
- natural target proportions have zero local cost;
- real Iris labels score above randomized labels globally;
- the assignment solver chooses the cheapest available source samples to move;
- invalid target proportions are rejected.
# tesis
