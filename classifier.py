import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

# Pre-process and training logistic regression model on training data
train = open('data/train.txt', 'r')
lines = train.readlines()
count = 0
bleu_score = 0
bleu_to_label = []

for line in lines:
    count += 1
    if line.strip() == 'H' or line.strip() == 'M':
    	bleu_to_label.append(tuple((bleu_score, line.strip())))
    if is_float(line.strip()):
    	bleu_score = float(line.strip())
bleu_lists = list(map(list, zip(*bleu_to_label)))
bleu_scores = np.asarray(bleu_lists[0]).reshape(-1, 1)
bleu_labels = bleu_lists[1]
labels_to_binary = np.asarray([1 if label == 'H' else 0 for label in bleu_labels])

# Uncomment to train with validation data
'''
x_train, x_val, y_train, y_val = train_test_split(bleu_scores, labels_to_binary, test_size=0.2, random_state=0)
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = clf.predict(x_val)
f1_score = f1_score(y_pred, y_val))
print(f1_score)
'''

clf = LogisticRegression(random_state=0).fit(bleu_scores, labels_to_binary)
train.close()


# Pre-process and predicting wether test candidate is machine or human translation
test = open('data/test.txt', 'r')
lines = test.readlines()

count = 0
bleu_score = 0
bleu_to_label = []
for line in lines:
    count += 1
    if line.strip() == 'H' or line.strip() == 'M':
        bleu_to_label.append(tuple((bleu_score, line.strip())))
    if is_float(line.strip()):
        bleu_score = float(line.strip())
bleu_lists = list(map(list, zip(*bleu_to_label)))
bleu_scores = np.asarray(bleu_lists[0]).reshape(-1, 1)
bleu_labels = bleu_lists[1]
labels_to_binary = np.asarray([1 if label == 'H' else 0 for label in bleu_labels])


y_pred = clf.predict(bleu_scores)

f1_score = f1_score(labels_to_binary, y_pred)
print("F1 score using logistic regression classifier: {score}".format(score = str(f1_score)))
test.close()