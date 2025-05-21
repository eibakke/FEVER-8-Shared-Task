# Comparison Analysis: Negative vs. Baseline Prediction

## 1. Agreement Metrics

Overall agreement rate: 84.40%
Overall flip rate: 15.60%

### Agreement by negative prediction label:
- Conflicting Evidence/Cherrypicking: 37.50% (n=16)
- Not Enough Evidence: 50.00% (n=12)
- Refuted: 90.00% (n=330)
- Supported: 79.58% (n=142)

### Transition Matrix (rows=negative, columns=baseline):
- From Conflicting Evidence/Cherrypicking:
  - To Conflicting Evidence/Cherrypicking: 37.50%
  - To Not Enough Evidence: 6.20%
  - To Refuted: 50.00%
  - To Supported: 6.20%
- From Not Enough Evidence:
  - To Conflicting Evidence/Cherrypicking: 8.30%
  - To Not Enough Evidence: 50.00%
  - To Refuted: 16.70%
  - To Supported: 25.00%
- From Refuted:
  - To Conflicting Evidence/Cherrypicking: 1.20%
  - To Not Enough Evidence: 1.20%
  - To Refuted: 90.00%
  - To Supported: 7.60%
- From Supported:
  - To Conflicting Evidence/Cherrypicking: 0.70%
  - To Not Enough Evidence: 0.70%
  - To Refuted: 19.00%
  - To Supported: 79.60%

### Major label flips:
- Conflicting Evidence/Cherrypicking → Not Enough Evidence: 1 cases (1.3% of all changes)
- Conflicting Evidence/Cherrypicking → Refuted: 8 cases (10.3% of all changes)
- Conflicting Evidence/Cherrypicking → Supported: 1 cases (1.3% of all changes)
- Not Enough Evidence → Conflicting Evidence/Cherrypicking: 1 cases (1.3% of all changes)
- Not Enough Evidence → Refuted: 2 cases (2.6% of all changes)
- Not Enough Evidence → Supported: 3 cases (3.9% of all changes)
- Refuted → Conflicting Evidence/Cherrypicking: 4 cases (5.1% of all changes)
- Refuted → Not Enough Evidence: 4 cases (5.1% of all changes)
- Refuted → Supported: 25 cases (32.0% of all changes)
- Supported → Conflicting Evidence/Cherrypicking: 1 cases (1.3% of all changes)
- Supported → Not Enough Evidence: 1 cases (1.3% of all changes)
- Supported → Refuted: 27 cases (34.6% of all changes)

## 2. Correctness Impact

Negative prediction accuracy: 68.80%
Baseline accuracy: 69.20%
Accuracy delta: 0.40%

### Correction analysis:
- Correction rate: 19.23% (30 of 156 opportunities)
- Error introduction rate: 8.14% (28 of 344 opportunities)
- Net corrections: 2

### Accuracy when predictions agree vs. disagree:
- When agree (n=422): 74.88%
- When disagree (n=78):
  - Negative: 35.90%
  - Baseline: 38.46%

## 3. Evidence Analysis

Average evidence count in negative predictions: 10.00
Average evidence count in baseline predictions: 10.00

### Evidence impact on prediction changes:
- Average evidence when predictions agree (n=422): 10.00
- Average evidence when predictions disagree (n=78): 10.00
- Correlation between evidence count and disagreement: nan

## 4. Justification Analysis

Mean semantic similarity between justifications: 0.828
Median similarity: 0.891
Range: 0.146 - 1.000

- Mean similarity when predictions agree: 0.853
- Mean similarity when predictions disagree: 0.700

## 5. Label Distribution Analysis

### Label counts:
| Label | Negative | Baseline | Absolute Shift | Relative Shift |
|-------|--------|----------|----------------|---------------|
| Conflicting Evidence/Cherrypicking | 16 | 12 | -0.80% | -25.00% |
| Not Enough Evidence | 12 | 12 | 0.00% | 0.00% |
| Refuted | 330 | 334 | 0.80% | 1.21% |
| Supported | 142 | 142 | 0.00% | 0.00% |
