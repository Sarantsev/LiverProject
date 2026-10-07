nnMatrix = list(map(int,input().split()))
matrix = []
moves = []
rows = nnMatrix[0]
cols = nnMatrix[1]
for i in range(rows):
    row = list(map(int, input().split()))
    matrix.append(row)
sum = matrix[0][0]
h = 0
w = 0

for i in range(rows-1):
    for j in range(cols-1):
        if(i == rows-1 and j == cols-1):
            break
        if(matrix[i+1][j] > matrix[i][j+1]):
            sum += matrix[i+1][j]
            moves.append("D")
        else:
            sum += matrix[i][j+1]
            moves.append("R")
print(sum)
print(" ".join(moves))