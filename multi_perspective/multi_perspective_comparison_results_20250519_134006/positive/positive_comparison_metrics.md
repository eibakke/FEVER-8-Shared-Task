# Comparison Analysis: Positive vs. Baseline Prediction

## 1. Agreement Metrics

Overall agreement rate: 81.80%
Overall flip rate: 18.20%

### Agreement by positive prediction label:
- Conflicting Evidence/Cherrypicking: 25.00% (n=12)
- Not Enough Evidence: 50.00% (n=12)
- Refuted: 89.81% (n=314)
- Supported: 72.84% (n=162)

### Transition Matrix (rows=positive, columns=baseline):
- From Conflicting Evidence/Cherrypicking:
  - To Conflicting Evidence/Cherrypicking: 25.00%
  - To Not Enough Evidence: 8.30%
  - To Refuted: 50.00%
  - To Supported: 16.70%
- From Not Enough Evidence:
  - To Conflicting Evidence/Cherrypicking: 8.30%
  - To Not Enough Evidence: 50.00%
  - To Refuted: 41.70%
  - To Supported: 0.00%
- From Refuted:
  - To Conflicting Evidence/Cherrypicking: 1.90%
  - To Not Enough Evidence: 1.30%
  - To Refuted: 89.80%
  - To Supported: 7.00%
- From Supported:
  - To Conflicting Evidence/Cherrypicking: 1.20%
  - To Not Enough Evidence: 0.60%
  - To Refuted: 25.30%
  - To Supported: 72.80%

### Major label flips:
- Conflicting Evidence/Cherrypicking → Not Enough Evidence: 1 cases (1.1% of all changes)
- Conflicting Evidence/Cherrypicking → Refuted: 6 cases (6.6% of all changes)
- Conflicting Evidence/Cherrypicking → Supported: 2 cases (2.2% of all changes)
- Not Enough Evidence → Conflicting Evidence/Cherrypicking: 1 cases (1.1% of all changes)
- Not Enough Evidence → Refuted: 5 cases (5.5% of all changes)
- Refuted → Conflicting Evidence/Cherrypicking: 6 cases (6.6% of all changes)
- Refuted → Not Enough Evidence: 4 cases (4.4% of all changes)
- Refuted → Supported: 22 cases (24.2% of all changes)
- Supported → Conflicting Evidence/Cherrypicking: 2 cases (2.2% of all changes)
- Supported → Not Enough Evidence: 1 cases (1.1% of all changes)
- Supported → Refuted: 41 cases (45.0% of all changes)

## 2. Correctness Impact

Positive prediction accuracy: 67.20%
Baseline accuracy: 69.20%
Accuracy delta: 2.00%

### Correction analysis:
- Correction rate: 25.61% (42 of 164 opportunities)
- Error introduction rate: 9.52% (32 of 336 opportunities)
- Net corrections: 10

### Accuracy when predictions agree vs. disagree:
- When agree (n=409): 74.33%
- When disagree (n=91):
  - Positive: 35.16%
  - Baseline: 46.15%

## 3. Evidence Analysis

Average evidence count in positive predictions: 10.00
Average evidence count in baseline predictions: 10.00

### Evidence impact on prediction changes:
- Average evidence when predictions agree (n=409): 10.00
- Average evidence when predictions disagree (n=91): 10.00
- Correlation between evidence count and disagreement: nan

## 4. Justification Analysis

Mean semantic similarity between justifications: 0.771
Median similarity: 0.825
Range: 0.188 - 1.000

- Mean similarity when predictions agree: 0.809
- Mean similarity when predictions disagree: 0.611

## 5. Label Distribution Analysis

### Label counts:
| Label | Positive | Baseline | Absolute Shift | Relative Shift |
|-------|--------|----------|----------------|---------------|
| Conflicting Evidence/Cherrypicking | 12 | 12 | 0.00% | 0.00% |
| Not Enough Evidence | 12 | 12 | 0.00% | 0.00% |
| Refuted | 314 | 334 | 4.00% | 6.37% |
| Supported | 162 | 142 | -4.00% | -12.35% |
