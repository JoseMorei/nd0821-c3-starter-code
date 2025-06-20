# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Type**: Random Forest Classifier
- **Framework**: scikit-learn
- **Task**: Binary classification – Predict if income >50K

## Intended Use
Exploring census data and predicting income levels based on demographic and employment attributes.

## Training Data
Based on UCI Adult Census Dataset cleaned in `census_clean.csv`.

## Evaluation Data
Model evaluated for fairness across:
- `education`
- `sex`
- `race`
- (see `slice_output.txt` for details)
- 
## Metrics
- **Precision**: 0.740
- **Recall**:    0.629
- **F1 Score**:  0.68

## Ethical Considerations
- Dataset may include historical biases.
- 
## Caveats and Recommendations
- Not intended for real-world decision making