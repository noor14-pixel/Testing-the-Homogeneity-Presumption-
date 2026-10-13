# Homogeneity test for composite household-behavior indicators

Analysis code for a study testing whether the items pooled into a composite
indicator share one coefficient, applied to the MICS Punjab 2017-18 child
discipline module and early learning (children aged 35-60 months).

## Data

The MICS Punjab 2017-18 microdata are NOT included. They are available from
UNICEF's MICS programme on registration and approval (mics.unicef.org).
Place the downloaded files in a local `data/` folder before running the code.

## Files

| File | Purpose |
| --- | --- |
| `ologit.do` | Stata: ordered logit of the Learning Index on the eleven practices, average marginal effects, Wald tests |
| `aggregation_test.py` | Python: aggregation / homogeneity tests |
| `replicate_all_tests.py` | Python: replicates the main Wald tests and robustness checks |
| `replicate_item_tests.py` | Python: tests repeated for each literacy/numeracy item separately |
| `replicate_simulation.py` | Python: size and power simulation of the Wald test |
| `Random_Forest.ipynb` | Exploratory random forests, one per practice |

## Requirements

- Stata (version: ___)
- Python (version: ___) with: numpy, pandas, scipy, statsmodels, scikit-learn
  (edit to match the imports in your scripts)

## Suggested run order

1. `ologit.do`
2. `aggregation_test.py`
3. `replicate_all_tests.py`
4. `replicate_item_tests.py`
5. `replicate_simulation.py`
6. `Random_Forest.ipynb`

## License

MIT
