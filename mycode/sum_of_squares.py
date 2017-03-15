import re, math, random # regexes, math functions, random numbers

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    l = list(zip(v, w))
    return sum(v_i * w_i for v_i, w_i in l)

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

print(dot([1,2,3],[4,5,6]))

print(squared_distance([1,2],[2,3]))

for i in list(range(5)):
    print(i)


def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in list(range(num_cols))]
            for i in list(range(num_rows))]

def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

print(make_matrix(5,5,is_diagonal))