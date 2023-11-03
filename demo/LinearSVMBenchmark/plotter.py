import matplotlib.pyplot as plt
import csv

name = 'result.csv'
with open(name, 'r') as file:
    reader = csv.reader(file)
    a, b, c = map(float, next(reader))
    points = [tuple(map(float, row)) for row in reader]

x = [-1, 1]
y = [(-c+a)/b, (-c-a)/b]
for i in range(2):
    if y[i] < -1 or y[i] > 1:
        y[i] = min(max(y[i], -1), 1)
        x[i] = (-c-y[i]*b)/a

plt.figure(figsize=(12, 12))

plt.plot(x, y, linewidth=2.0)
plt.scatter([p[0] for p in points], [p[1] for p in points], c=[
            p[2] for p in points], linewidths=[p[3] for p in points], cmap='bwr')

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('axes', 0.5))

plt.savefig('plot.pdf')
plt.show()
