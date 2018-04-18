
import numpy as np

file = "S://a//a10.txt"
list = []
with open(file, 'r') as f:
    for line in f:
        list.extend([float(i) for i in line.split()])

count = 0
df = []
k = 1
for i in range(120000):
    df.append(list[i])
    count += 1
    if(count%600==0):
        np.savetxt("S://a//a10//" + str(k) + ".txt", df)
        df = []
        k+=1