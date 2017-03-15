from functools import partial, reduce

def vector_sum(vectors):
    return reduce(vector_add, vectors)


def vector_add(v, w):
    a = v
    b = w
    if isinstance(v,list):
        a = v
    else:
        a = [v]
    if isinstance(w, list):
        b = w
    else:
        b = [w]

    c = list(zip(a,b))
    print(c)
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in c]

vectors = [0,1,2]

vector_sum(vectors)