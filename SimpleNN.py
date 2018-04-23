# simple neural network implementation, 

# based on tutorial from the link

# https://proglib.io/p/neural-nets-guide/

# =======================================



# neural network properties:

# input - 3 values

# 1 middle layer with 3 values



import matplotlib.pylab as plt

import numpy as np





# activation function plot

'''

x = np.arange(-8, 8, 0.1)

f = 1 / (1 + np.exp(-x))

plt.plot(x,f)

plt.xlabel ('x')

plt.ylabel ('f(x)')

plt.show()

'''

#lets create weigth matrix for first connection

w1 = np.array ([

     [0.2, 0.2, 0.2],

     [0.4, 0.4, 0.4],

     [0.6, 0.6, 0.6]

     ])

#shift value

b1 = np.array ([0.8, 0.8, 0.8])  



#and second connection

w2 = np.zeros((1,3))

w2[0, : ] = np.array([0.5, 0.5, 0.5])



#shift value

b2 = np.array([0.2])



#activation function

def f(x):

    return 1 / (1 + np.exp(-x))



# simple implementation by direct method

def simple_looped_nn_calc (n_layers, x, w, b):

    for l in range(n_layers - 1):

        if l == 0:

            node_in = x

        else:

            node_in = h # array of output values 

        

        h = np.zeros(([w[l].shape[0], ]))

        

        for i in range (w[l].shape[0]):

            f_sum = 0

            for j in range (w[l].shape[1]):

                f_sum += w[l][i][j] * node_in[j]

            f_sum += b[l][i]

            h[i] = f(f_sum)

        

    return h



w = [w1,w2]

b = [b1,b2]

x = [1.5, 2.0, 3.0]



print (simple_looped_nn_calc(3, x, w, b))



# vector implementation by direct method

def matrix_feed_forward_calc( n_layers, x, w, b):

    for l in range (n_layers - 1):

        if (l == 0):

            node_in = x

        else:

            node_in = h

        z = w[l].dot(node_in) + b[l]

        h = f(z)

    return h
