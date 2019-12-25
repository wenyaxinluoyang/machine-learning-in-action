from numpy import *

if __name__ == '__main__':
    var = random.rand(4,4)
    print(type(var))
    print(var)
    matrix = mat(var)
    print(type(matrix))
    print(matrix)
    print('逆矩阵:', matrix.I)
    # 矩阵乘
    print(matrix*matrix.I)
    print(matrix*matrix.I - eye(4))
