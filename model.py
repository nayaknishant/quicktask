import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
bleu_scores = np.asarray(bleu_lists[0]).reshape(-1, 1)
bleu_labels = bleu_lists[1]
labels_to_binary = np.asarray([1 if label == 'H' else 0 for label in bleu_labels])
print(bleu_scores)
print(labels_to_binary)

x_train, x_val, y_train, y_val = train_test_split(bleu_scores, labels_to_binary, test_size=0.05, random_state=0)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
print(clf.score(x_val, y_val))
print(clf.score(x_train, y_train))

f.close()