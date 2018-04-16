import numpy as np
import pprint

# L1  = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L1 = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
L2 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# L3  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
L3 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
L4 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L5 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
L6 = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
L7 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
L8 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
L9 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
L10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

L = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10])

# L1 = [0,1,1,0]
# L2 = [0,0,0,1]
# L3 = [0,1,0,1]
# L4 = [0,0,0,1]
#
# L = np.array([L1,L2,L3,L4])

ITERATIONS = 100


def printResult(rank):
    result = list()
    for i in range(1, len(rank) + 1):
        row = list()
        row.append(i)
        row.append(rank[i - 1])
        result.append(row)
    result.sort(key=lambda x: x[1], reverse=True)
    for r in result:
        print(r[0], "\t", r[1])


### TODO 1: Compute stochastic matrix M
def getM(L):
    M = np.zeros([len(L[0]), len(L[0])], dtype=float)
    # number of outgoing links
    for i in range(len(L[0])):
        sum = 0
        for j in range(len(M[0])):
            sum += L[i, j]
        for j in range(len(L[0])):
            if sum != 0:
                M[i, j] = L[i, j] / sum

    return M.transpose()


print("Matrix L (indices)")
print(L)

M = getM(L)

print("Matrix M (stochastic matrix)")
print(M)
### TODO 2: compute pagerank with damping factor q = 0.15
### Then, sort and print: (page index (first index = 1 add +1) : pagerank)
### (use regular array + sort method + lambda function)
print("PAGERANK")

q = 0.15

pr = np.zeros([len(M[0])], dtype=float)

for i in range(len(pr)):
    pr[i] = 1 / len(pr)

# print(pr)

for k in range(ITERATIONS):
    tmp = pr.copy()
    for i in range(len(pr)):
        sum = 0
        for j in range(len(pr)):
            sum += pr[j] * M[j, i]
        tmp[i] = q + (1 - q) * sum
    pr = tmp.copy()

# print(pr)
# print(sorted(pr))
printResult(pr)

### TODO 3: compute trustrank with damping factor q = 0.15
### Documents that are good = 1, 2 (indexes = 0, 1)
### Then, sort and print: (page index (first index = 1, add +1) : trustrank)
### (use regular array + sort method + lambda function)
print("TRUSTRANK (DOCUMENTS 1 AND 2 ARE GOOD)")

q = 0.15

d = np.zeros([len(M[0])], dtype=float)
d[1] = 0.5
d[2] = 0.5

tr = [v for v in d]

for k in range(ITERATIONS):
    tmp = tr
    for i in range(len(tr)):
        sum = 0
        for j in range(len(tr)):
            sum += M[j, i] * tr[j]
        tmp[i] = q * d[i] + (1 - q) * sum
    tr = tmp

# print(tr)
# print(sorted(tr))
printResult(tr)

### TODO 4: Repeat TODO 3 but remove the connections 3->7 and 1->5 (indexes: 2->6, 0->4) 
### before computing trustrank
