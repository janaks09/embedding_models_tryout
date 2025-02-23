import random
data = [
  [i for i in range(2000)],
  [str(i) for i in range(2000)],
  [i for i in range(10000, 12000)],
  [[random.random() for _ in range(2)] for _ in range(2000)],
  # use `default_value` for a field
  [], 
  # or
  None,
  # or just omit the field
]

data.append([str("dy"*i) for i in range(2000)])

for d in data:
    print(d)