# Comparison Analysis: Objective vs. Baseline Prediction

## 1. Agreement Metrics

Overall agreement rate: 84.60%
Overall flip rate: 15.40%

### Agreement by objective prediction label:
- Conflicting Evidence/Cherrypicking: 33.33% (n=9)
- Not Enough Evidence: 56.25% (n=16)
- Refuted: 89.52% (n=334)
- Supported: 79.43% (n=141)

### Transition Matrix (rows=objective, columns=baseline):
- From Conflicting Evidence/Cherrypicking:
  - To Conflicting Evidence/Cherrypicking: 33.30%
  - To Not Enough Evidence: 0.00%
  - To Refuted: 44.40%
  - To Supported: 22.20%
- From Not Enough Evidence:
  - To Conflicting Evidence/Cherrypicking: 6.20%
  - To Not Enough Evidence: 56.20%
  - To Refuted: 25.00%
  - To Supported: 12.50%
- From Refuted:
  - To Conflicting Evidence/Cherrypicking: 1.80%
  - To Not Enough Evidence: 0.90%
  - To Refuted: 89.50%
  - To Supported: 7.80%
- From Supported:
  - To Conflicting Evidence/Cherrypicking: 1.40%
  - To Not Enough Evidence: 0.00%
  - To Refuted: 19.10%
  - To Supported: 79.40%

### Major label flips:
- Conflicting Evidence/Cherrypicking → Refuted: 4 cases (5.2% of all changes)
- Conflicting Evidence/Cherrypicking → Supported: 2 cases (2.6% of all changes)
- Not Enough Evidence → Conflicting Evidence/Cherrypicking: 1 cases (1.3% of all changes)
- Not Enough Evidence → Refuted: 4 cases (5.2% of all changes)
- Not Enough Evidence → Supported: 2 cases (2.6% of all changes)
- Refuted → Conflicting Evidence/Cherrypicking: 6 cases (7.8% of all changes)
- Refuted → Not Enough Evidence: 3 cases (3.9% of all changes)
- Refuted → Supported: 26 cases (33.8% of all changes)
- Supported → Conflicting Evidence/Cherrypicking: 2 cases (2.6% of all changes)
- Supported → Refuted: 27 cases (35.1% of all changes)

## 2. Correctness Impact

Objective prediction accuracy: 68.60%
Baseline accuracy: 69.20%
Accuracy delta: 0.60%

### Correction analysis:
- Correction rate: 19.75% (31 of 157 opportunities)
- Error introduction rate: 8.16% (28 of 343 opportunities)
- Net corrections: 3

### Accuracy when predictions agree vs. disagree:
- When agree (n=423): 74.47%
- When disagree (n=77):
  - Objective: 36.36%
  - Baseline: 40.26%

## 3. Evidence Analysis

Average evidence count in objective predictions: 10.00
Average evidence count in baseline predictions: 10.00

### Evidence impact on prediction changes:
- Average evidence when predictions agree (n=423): 10.00
- Average evidence when predictions disagree (n=77): 10.00
- Correlation between evidence count and disagreement: nan

## 4. Justification Analysis

Mean semantic similarity between justifications: 0.815
Median similarity: 0.842
Range: 0.271 - 1.000

- Mean similarity when predictions agree: 0.838
- Mean similarity when predictions disagree: 0.695

## 5. Label Distribution Analysis

### Label counts:
| Label | Objective | Baseline | Absolute Shift | Relative Shift |
|-------|--------|----------|----------------|---------------|
| Conflicting Evidence/Cherrypicking | 9 | 12 | 0.60% | 33.33% |
| Not Enough Evidence | 16 | 12 | -0.80% | -25.00% |
| Refuted | 334 | 334 | 0.00% | 0.00% |
| Supported | 141 | 142 | 0.20% | 0.71% |
