def max_count_class(x):
    z = list(map(lambda a: a[0], x))
    return max(z, key=lambda a: z.count(a))