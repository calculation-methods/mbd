import matplotlib.pyplot as plt
import numpy as np

file = open ("./py_args.txt", "r")

x = [float(x) for x in file.readline().split()]
y = [float(x) for x in file.readline().split()]
poly_1 = [float(x) for x in file.readline().split()]
poly_2 = [float(x) for x in file.readline().split()]

x_1 = np.linspace(x[0], x[-1], 100)
y_1 = poly_1[0] + poly_1[1]*x_1

x_2 = np.linspace(x[0], x[-1], 100)
y_2 = poly_2[0] + poly_2[1]*x_1 + poly_2[2]*x_1*x_1
plt.plot(x, y, 'ro')
plt.plot(x_1, y_1)
plt.plot(x_2, y_2)

plt.xlabel('x')  
plt.ylabel('y')  
    
plt.title('My graph')  

plt.show()
file.close()