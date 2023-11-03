import matplotlib.pyplot as plt
import csv
import numpy as np

name = 'result.csv'
with open(name, 'r') as file:
    reader = csv.reader(file)
    raw_points = [tuple(map(float, row)) for row in reader]
    points = raw_points[:500]
    pred_mesh = raw_points[500:]

plt.figure(figsize=(24, 24))

plt.scatter([p[0] for p in points], [p[1] for p in points], c=[
            p[2] for p in points], linewidths=[p[3] for p in points], cmap='bwr')
x = np.linspace(-4.5, 4.5, num=1000)
y = np.linspace(-4.5, 4.5, num=1000)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, np.array([p[0]
             for p in pred_mesh]).reshape(1000, 1000).transpose(1, 0), alpha=0.2)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('SVM result')

plt.show()
