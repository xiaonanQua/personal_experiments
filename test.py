import numpy as np


# def a_1():
#     global j
#     j = 0
#     for i in range(j*128, (j+1)*128):
#         print(i)


if __name__ == '__main__':
    # for i in range(0, 6):
    #     a_1()
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = [[1], [2], [3]]
    np.random.shuffle(a)
    np.random.shuffle(b)
    print(a)
    print(b)
    data = np.array(range(0, 128))
    np.random.shuffle(data)
    for i in data:
        print(i)