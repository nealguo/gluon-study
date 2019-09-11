def add(a, b):
    return a + b


def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g


def add_str():
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''


def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''


if __name__ == '__main__':
    print(fancy_func(1, 2, 3, 4))

    prog = evoke_str()
    print(prog)
    y = compile(prog, '', 'exec')
    exec(y)
