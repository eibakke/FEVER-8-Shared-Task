# Comparison Analysis: Direct vs. Knowledge-Based Prediction

## 1. Agreement Metrics

Overall agreement rate: 28.40%
Overall flip rate: 71.60%

### Agreement by direct prediction label:
- Conflicting Evidence/Cherrypicking: 4.17% (n=48)
- Not Enough Evidence: 4.69% (n=512)
- Refuted: 79.18% (n=245)
- Supported: 32.82% (n=195)

### Transition Matrix (rows=direct, columns=knowledge-based):
- From Conflicting Evidence/Cherrypicking:
  - To Conflicting Evidence/Cherrypicking: 4.20%
  - To Not Enough Evidence: 0.00%
  - To Refuted: 60.40%
  - To Supported: 35.40%
- From Not Enough Evidence:
  - To Conflicting Evidence/Cherrypicking: 2.70%
  - To Not Enough Evidence: 4.70%
  - To Refuted: 64.50%
  - To Supported: 28.10%
- From Refuted:
  - To Conflicting Evidence/Cherrypicking: 2.00%
  - To Not Enough Evidence: 1.60%
  - To Refuted: 79.20%
  - To Supported: 17.10%
- From Supported:
  - To Conflicting Evidence/Cherrypicking: 4.60%
  - To Not Enough Evidence: 4.10%
  - To Refuted: 58.50%
  - To Supported: 32.80%

### Major label flips:
- Conflicting Evidence/Cherrypicking → Refuted: 29 cases (4.0% of all changes)
- Conflicting Evidence/Cherrypicking → Supported: 17 cases (2.4% of all changes)
- Not Enough Evidence → Conflicting Evidence/Cherrypicking: 14 cases (2.0% of all changes)
- Not Enough Evidence → Refuted: 330 cases (46.1% of all changes)
- Not Enough Evidence → Supported: 144 cases (20.1% of all changes)
- Refuted → Conflicting Evidence/Cherrypicking: 5 cases (0.7% of all changes)
- Refuted → Not Enough Evidence: 4 cases (0.6% of all changes)
- Refuted → Supported: 42 cases (5.9% of all changes)
- Supported → Conflicting Evidence/Cherrypicking: 9 cases (1.3% of all changes)
- Supported → Not Enough Evidence: 8 cases (1.1% of all changes)
- Supported → Refuted: 114 cases (15.9% of all changes)

## 2. Correctness Impact

Direct prediction accuracy: 30.70%
Knowledge-based accuracy: 71.30%
Accuracy delta: 40.60%

### Correction analysis:
- Correction rate: 71.28% (494 of 693 opportunities)
- Error introduction rate: 28.66% (88 of 307 opportunities)
- Net corrections: 406

### Accuracy when predictions agree vs. disagree:
- When agree (n=284): 77.11%
- When disagree (n=716):
  - Direct: 12.29%
  - Knowledge-based: 68.99%

## 3. Evidence Analysis

Average evidence count in direct predictions: 3.00
Average evidence count in knowledge-based predictions: 10.00

### Evidence impact on prediction changes:
- Average evidence when predictions agree (n=284): 10.00
- Average evidence when predictions disagree (n=716): 10.00
- Correlation between evidence count and disagreement: nan

## 4. Justification Analysis

Mean semantic similarity between justifications: 0.633
Median similarity: 0.680
Range: 0.043 - 0.919

- Mean similarity when predictions agree: 0.685
- Mean similarity when predictions disagree: 0.610

## 5. Label Distribution Analysis

### Label counts:
| Label | Direct | Knowledge-Based | Absolute Shift | Relative Shift |
|-------|--------|----------------|----------------|---------------|
| Conflicting Evidence/Cherrypicking | 48 | 30 | -1.80% | -37.50% |
| Not Enough Evidence | 512 | 36 | -47.60% | -92.97% |
| Refuted | 245 | 667 | 42.20% | 172.24% |
| Supported | 195 | 267 | 7.20% | 36.92% |
