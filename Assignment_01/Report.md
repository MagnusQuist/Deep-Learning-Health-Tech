# Assignment 01

### 1. Brief Problem description
**Problem**: multi-class image classification of single blood-cell microscope images (BloodMNIST).
**Task**: Given a small 28×28 color image of a single blood cell, predict which of 8 blood-cell classes it belongs to (e.g., neutrophil, lymphocyte, platelet, …). This is a standard benchmark in the MedMNIST collection.

**Why relevant**: automatic classification of blood-cell types has clinical/diagnostic utility (hematology workflows, automated screening) and is a representative medical-image classification benchmark that tests models under limited image size and class imbalance.

### 2. The Dataset - Facts & Counts
**Image type**: color microscope images of single blood cells, provided at 28×28 (preprocessed / standardized). Images originally larger, center-cropped and resized to 28×28 RGB.

**Total size & splits**

* Total images: 17092
* Training / Validation / Test splits (Ratio 7:1:12):
    * Training: 11959
    * Validation: 1712
    * Test: 3421

