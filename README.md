# Auto Email / Ticket Categorizer

## Project Overview

A lightweight NLP classification system that automatically categorizes incoming support tickets into four departments:

- Billing
- Technical
- HR
- General

The project demonstrates an end-to-end text classification workflow using TF-IDF and Logistic Regression.

## Workflow

Raw Ticket
→ Text Preprocessing
→ TF-IDF
→ Logistic Regression
→ Category Prediction
→ Confidence Score
→ Priority Tagging
→ Department Routing

## Features

- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Accuracy, precision, recall and F1 evaluation
- Confusion matrix
- Unseen ticket prediction
- Confidence score
- 60% human-review threshold
- Urgent/Normal priority tagging
- Manual-review fallback
- CLI live demonstration

## Model

TF-IDF + Logistic Regression

Logistic Regression was selected because it is fast, interpretable, effective for sparse text features, and provides class probabilities that can be used for confidence-based routing.

## Dataset

A synthetic support-ticket dataset was created because no dataset was provided in the assessment portal.

The dataset contains labeled examples for Billing, Technical, HR and General categories.

## Future Improvements

With more real-world data, the system could be improved using a larger and more diverse dataset, confidence calibration, model monitoring, drift detection, and periodic retraining.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
