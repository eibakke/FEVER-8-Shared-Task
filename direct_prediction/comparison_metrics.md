# Comparison Analysis: Direct vs. Knowledge-Based Prediction

## 1. Agreement Metrics

Overall agreement rate: 31.20%
Overall flip rate: 68.80%

### Agreement by direct prediction label:
- Conflicting Evidence/Cherrypicking: 6.06% (n=33)
- Not Enough Evidence: 3.40% (n=235)
- Refuted: 81.20% (n=133)
- Supported: 38.38% (n=99)

### Transition Matrix (rows=direct, columns=knowledge-based):
- From Conflicting Evidence/Cherrypicking:
  - To Conflicting Evidence/Cherrypicking: 6.10%
  - To Not Enough Evidence: 6.10%
  - To Refuted: 66.70%
  - To Supported: 21.20%
- From Not Enough Evidence:
  - To Conflicting Evidence/Cherrypicking: 2.10%
  - To Not Enough Evidence: 3.40%
  - To Refuted: 62.10%
  - To Supported: 32.30%
- From Refuted:
  - To Conflicting Evidence/Cherrypicking: 1.50%
  - To Not Enough Evidence: 1.50%
  - To Refuted: 81.20%
  - To Supported: 15.80%
- From Supported:
  - To Conflicting Evidence/Cherrypicking: 3.00%
  - To Not Enough Evidence: 0.00%
  - To Refuted: 58.60%
  - To Supported: 38.40%

### Major label flips:
- Conflicting Evidence/Cherrypicking → Not Enough Evidence: 2 cases (0.6% of all changes)
- Conflicting Evidence/Cherrypicking → Refuted: 22 cases (6.4% of all changes)
- Conflicting Evidence/Cherrypicking → Supported: 7 cases (2.0% of all changes)
- Not Enough Evidence → Conflicting Evidence/Cherrypicking: 5 cases (1.4% of all changes)
- Not Enough Evidence → Refuted: 146 cases (42.4% of all changes)
- Not Enough Evidence → Supported: 76 cases (22.1% of all changes)
- Refuted → Conflicting Evidence/Cherrypicking: 2 cases (0.6% of all changes)
- Refuted → Not Enough Evidence: 2 cases (0.6% of all changes)
- Refuted → Supported: 21 cases (6.1% of all changes)
- Supported → Conflicting Evidence/Cherrypicking: 3 cases (0.9% of all changes)
- Supported → Refuted: 58 cases (16.9% of all changes)

## 2. Correctness Impact

Direct prediction accuracy: 32.00%
Knowledge-based accuracy: 69.20%
Accuracy delta: 37.20%

### Correction analysis:
- Correction rate: 66.76% (227 of 340 opportunities)
- Error introduction rate: 25.62% (41 of 160 opportunities)
- Net corrections: 186

### Accuracy when predictions agree vs. disagree:
- When agree (n=156): 76.28%
- When disagree (n=344):
  - Direct: 11.92%
  - Knowledge-based: 65.99%

## 3. Evidence Analysis

Average evidence count in direct predictions: 2.98
Average evidence count in knowledge-based predictions: 10.00

### Evidence impact on prediction changes:
- Average evidence when predictions agree (n=156): 10.00
- Average evidence when predictions disagree (n=344): 10.00
- Correlation between evidence count and disagreement: nan

## 4. Justification Analysis

Mean semantic similarity between justifications: 0.589
Median similarity: 0.617
Range: -0.000 - 0.911

- Mean similarity when predictions agree: 0.606
- Mean similarity when predictions disagree: 0.580

## 5. Label Distribution Analysis

### Label counts:
| Label | Direct | Knowledge-Based | Absolute Shift | Relative Shift |
|-------|--------|----------------|----------------|---------------|
| Conflicting Evidence/Cherrypicking | 33 | 12 | -4.20% | -63.64% |
| Not Enough Evidence | 235 | 12 | -44.60% | -94.89% |
| Refuted | 133 | 334 | 40.20% | 151.13% |
| Supported | 99 | 142 | 8.60% | 43.43% |
