# Analysis by Verification Category

## Overall Results

- Baseline overall accuracy: 0.6940 (347/500)
- Multi-perspective overall accuracy: 0.6620 (331/500)

### Baseline Classification Report

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|--------|
| Supported | 0.6240 | 0.6393 | 0.6316 | 122.0 |
| Refuted | 0.7715 | 0.8525 | 0.8100 | 305.0 |
| Not Enough Evidence | 0.2381 | 0.1429 | 0.1786 | 35.0 |
| Conflicting Evidence/Cherrypicking | 0.2353 | 0.1053 | 0.1455 | 38.0 |

### Multi-perspective Classification Report

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|--------|
| Supported | 0.5696 | 0.7377 | 0.6429 | 122.0 |
| Refuted | 0.7852 | 0.7672 | 0.7761 | 305.0 |
| Not Enough Evidence | 0.1333 | 0.0571 | 0.0800 | 35.0 |
| Conflicting Evidence/Cherrypicking | 0.1724 | 0.1316 | 0.1493 | 38.0 |

### Improvement Analysis

| Category | Baseline F1 | Multi F1 | Abs. Improvement | Rel. Improvement |
|----------|-------------|----------|------------------|------------------|
| Supported | 0.6316 | 0.6429 | **+0.0113** | **+1.79%** |
| Refuted | 0.8100 | 0.7761 | -0.0338 | -4.18% |
| Not Enough Evidence | 0.1786 | 0.0800 | -0.0986 | -55.20% |
| Conflicting Evidence/Cherrypicking | 0.1455 | 0.1493 | **+0.0038** | **+2.61%** |

## Examples from Challenging Categories


### Not Enough Evidence Examples

No improved cases for this category.

#### Degraded Cases (3 examples)

**Example 1**: Nigeria has seen a 60% drop in government revenue

- True label: Not Enough Evidence
- Baseline prediction: Not Enough Evidence (correct)
- Multi-perspective prediction: Refuted (incorrect)

Question counts by perspective:

- Positive: 10
- Negative: 10
- Objective: 10

**Example 2**: The South African Police Service kills three times more people per capita than the United States police force

- True label: Not Enough Evidence
- Baseline prediction: Not Enough Evidence (correct)
- Multi-perspective prediction: Refuted (incorrect)

Question counts by perspective:

- Positive: 10
- Negative: 10
- Objective: 10

**Example 3**: As we speak the US are developing a growing number of treatments, including convalescent plasma, that are saving lives all across the country.

- True label: Not Enough Evidence
- Baseline prediction: Not Enough Evidence (correct)
- Multi-perspective prediction: Conflicting Evidence/Cherrypicking (incorrect)

Question counts by perspective:

- Positive: 10
- Negative: 10
- Objective: 10


### Conflicting Evidence/Cherrypicking Examples

#### Improved Cases (1 examples)

**Example 1**: Face masks cause hypoxia.

- True label: Conflicting Evidence/Cherrypicking
- Baseline prediction: Refuted (incorrect)
- Multi-perspective prediction: Conflicting Evidence/Cherrypicking (correct)

Question counts by perspective:

- Positive: 10
- Negative: 10
- Objective: 10

No degraded cases for this category.

