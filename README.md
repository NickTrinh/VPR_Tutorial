# VPR_Tutorial

## Overview
VPR Research repository. These are primarily written in Python scripts. FRCV Lab

## Prerequisites
Before you begin, ensure you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

## Installation

### Step 1: Clone the repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/NickTrinh/VPR_Tutorial.git
cd VPR_Tutorial
```

### Step 2: Activate virtual environment
Activate the VPR_tutorial environment using the following commands:
```bash
source activate base
conda activate vprtutorial
```

This is what each file does
- get_n_scores.py: Getting scores for n times. This produces n .csv files of the calculated scores.
- average_test_results.py: This averages the n .csv files above to get our threshold numbers used in later testing.
- places_data.py: This stores our averaged numbers from average_test_results.py to be used in testing.
- test_with_threshold.py: This uses the averaged numbers in places_data.py to run tests for TP, FP, TN, FN and calculates our metrics.

Currently, we are using the average of 30 runs (get_n_scores's n = 30) as threshold. Metrics are in my VPVH folder. These are scores and metrics for the old DB.
Our new database (FordhamPlaces) is in images/FordhamPlaces.
