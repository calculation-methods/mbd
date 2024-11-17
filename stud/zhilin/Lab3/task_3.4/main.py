if __name__ == "__main__":
    # x_marked = 1.0
    # x = [i / 10 for i in range(-1, 4)]
    # y = [2.3562, 1.5708, 0.7854, 0.46635, 0.32175]

    x_marked = 0.2
    x = [i/10 for i in range(5)]
    y = [1.0, 1.1052, 1.2214, 1.3499, 1.4918]

    n = len(x)
    derivative_first = [(y[i+1] - y[i])/(x[i+1] - x[i]) for i in range(n-1)]
    derivative_second = [2*((y[i+2]-y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i])) / (x[i + 2] - x[i]) for i in range(n - 2)]

    for i in range(n-1):
        if x[i] == x_marked:
            print(f'The left-hand first derivative {derivative_first[i-1]}')
            print(f'The left-hand first derivative {derivative_first[i]}')
            break
        elif x[i] < x_marked < x[i+1]:
            print(f'First derivative {derivative_first[i]}')

    print()

    for i in range(n-2):
        if x[i] <= x_marked <= x[i+1]:
            print(f'First derivative {derivative_second[i]}')
            break
