# Retrieval Comparison Report

## Overview

Total claims analyzed: 500
Total documents retrieved: 15000

## Document Overlap

Average Jaccard similarity between bias conditions:

| Bias 1 | Bias 2 | Jaccard Similarity |
|--------|--------|--------------------|
| negative | objective | 0.6159 |
| positive | negative | 0.4209 |
| positive | objective | 0.4608 |

## Rank Correlation

Average rank correlation between bias conditions:

| Bias 1 | Bias 2 | Spearman | Kendall |
|--------|--------|----------|----------|
| negative | objective | 0.7908 | 0.6814 |
| positive | negative | 0.5754 | 0.4856 |
| positive | objective | 0.5663 | 0.4733 |

## Domain Analysis

Top domains by bias type:

### Positive

| Domain | Count |
|--------|-------|
| nbcnews.com | 92 |
| factcheck.afp.com | 77 |
| usatoday.com | 75 |
| cnbc.com | 74 |
| en.wikipedia.org | 69 |
| cnn.com | 66 |
| snopes.com | 66 |
| cbsnews.com | 65 |
| businessinsider.com | 60 |
| abcnews.go.com | 59 |

### Negative

| Domain | Count |
|--------|-------|
| nbcnews.com | 102 |
| usatoday.com | 100 |
| cnn.com | 75 |
| snopes.com | 75 |
| en.wikipedia.org | 72 |
| cnbc.com | 70 |
| abcnews.go.com | 69 |
| politifact.com | 69 |
| cbsnews.com | 68 |
| factcheck.afp.com | 63 |

### Objective

| Domain | Count |
|--------|-------|
| nbcnews.com | 101 |
| usatoday.com | 97 |
| cnn.com | 82 |
| politifact.com | 82 |
| cnbc.com | 78 |
| factcheck.afp.com | 71 |
| en.wikipedia.org | 69 |
| snopes.com | 69 |
| businessinsider.com | 65 |
| cbsnews.com | 62 |


## Conclusion

This report provides a basic analysis of how different bias conditions in hypothetical document generation affect document retrieval. The key findings include differences in document overlap, ranking patterns, and domain distributions across bias conditions.