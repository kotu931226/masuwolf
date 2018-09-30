import pickle
import matplotlib.pyplot as plt

with open('log_wolf_vec.bf', 'rb') as f:
    wolf_vec = pickle.load(f)

counter = []
counter2 = 0
for one_file in wolf_vec:
    counter.append(len(one_file))
    if 0 < len(one_file) < 10:
        counter2 += 1

print(counter2/len(wolf_vec))

counter.sort()
plt.scatter(range(len(counter)), counter)
plt.show()
