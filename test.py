

num = [[1,2], [2,3]]
count = 0

list = []
list2 = []
for index in range(0, 1000, 20):
    list.extend([[index]])
    list2.append([index])

print(list)
print(list2)