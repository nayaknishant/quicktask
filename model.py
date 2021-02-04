import numpy as np

f = open('data/train.txt', 'r')
lines = f.readlines()

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

count = 0
bleu_score = 0
bleu_to_label = []
# Strips the newline character
for line in lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    if line.strip() == 'H' or line.strip() == 'M':
    	bleu_to_label.append(tuple((bleu_score, line.strip())))
    if is_float(line.strip()):
    	bleu_score = float(line.strip())
bleu_lists = list(map(list, zip(*bleu_to_label)))
bleu_scores = np.asarray(bleu_lists[0])
bleu_labels = bleu_lists[1]
labels_to_binary = np.asarray([1 if label == 'H' else 0 for label in bleu_labels])
print(bleu_scores)
print(labels_to_binary)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(type(X))

f.close()