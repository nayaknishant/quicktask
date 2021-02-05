# Predicting whether candidate is machine or human translation

Used scikit-learn's logistic regression model to predict whether a Chinese to English translation was done by a machine or by a human. The independent variable used in the model is a BLEU score signifying the quality of the translation. The dependent variable is the label indicating whether the candidate comes from a machine (M) or human (H).

Once in repo, run:
```
python3 classifier.py
```

Output:
```
F1 score using logistic regression classifier: 0.7675675675675675
```
