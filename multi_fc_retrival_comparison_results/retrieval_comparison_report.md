# Retrieval Comparison Report

## Overview

Total claims analyzed: 500
Total documents retrieved: 15000

## Document Overlap

Average Jaccard similarity between bias conditions:

| Bias 1 | Bias 2 | Jaccard Similarity |
|--------|--------|--------------------|
| negative | objective | 0.5582 |
| positive | negative | 0.4209 |
| positive | objective | 0.4796 |

## Rank Correlation

Average rank correlation between bias conditions:

| Bias 1 | Bias 2 | Spearman | Kendall |
|--------|--------|----------|----------|
| negative | objective | 0.7284 | 0.6185 |
| positive | negative | 0.5754 | 0.4856 |
| positive | objective | 0.6161 | 0.5209 |

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
| nbcnews.com | 109 |
| usatoday.com | 91 |
| cnn.com | 76 |
| cnbc.com | 75 |
| cbsnews.com | 73 |
| politifact.com | 70 |
| snopes.com | 70 |
| en.wikipedia.org | 67 |
| factcheck.afp.com | 65 |
| businessinsider.com | 59 |


## Conclusion

This report provides a basic analysis of how different bias conditions in hypothetical document generation affect document retrieval. The key findings include differences in document overlap, ranking patterns, and domain distributions across bias conditions.