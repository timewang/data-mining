b = [9, 5, 68, 94, 3, 51, 23, 46, 0, 90, 901]


def quicksort(a):
    print(a)
    if len(a) <= 1:
        return a
    '''如果a為一位數則直接傳回a'''
    return quicksort([x for x in a[1:] if x < a[0]]) + [a[0]] + quicksort([x for x in a[1:] if x > a[0]])


print(quicksort(b))
