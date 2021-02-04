f = open('data/train.txt', 'r')
Lines = f.readlines()

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False


count = 0
# Strips the newline character
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    if line.strip() == 'H' or line.strip() == 'M':
    	print("AYO")
    if is_float(line.strip()):
    	print(float(line.strip()))

f.close()