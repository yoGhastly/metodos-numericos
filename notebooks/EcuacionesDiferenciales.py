# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Ecuaciones Diferenciales Ordinarias
# 
# ## ¿Qué son?
# 
# ## ¿Cuál es su solución?

# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
from numpy.linalg import norm
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpmath import findroot
from IPython.display import Math,display
sp.init_printing(use_latex='mathjax')
import warnings
import math
import ipywidgets as widgets
#warnings.filterwarnings('error')
np.seterr(all='print')

# %% [markdown]
# ## Euler hacia Adelante
# 
# ### ¿Cómo funciona?
# 
# ### ¿Cómo lo puedo usar?

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
resultado_y1 = fun_np(y0,t0)
y1 = y0 + (h * resultado_y1)
print("El valor de y1 es: ", y1)
t1 = t0 + h
resultado_y2 = fun_np(y1,t1)
y2 = y1 + (h * resultado_y2)
print("El valor de y2 es: ",y2)

# %% [markdown]
# ## Euler hacia Atrás

# %%
x, y, t = sp.symbols("x y t")
fun1 = input('Ingrese la ecuación despejada para y\': ')
fun_sim1= sp.sympify(fun1)
display(Math(sp.latex(fun_sim1)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
yn_1 = y0

fun_np = sp.lambdify([y,t], fun_sim1, 'numpy')
resultado_y1 = fun_np(yn_1,t0)
y1 = yn_1 + (h * resultado_y1)
print("El valor de y1 es: ", y1)
t1 = t0 + h
resultado_y2 = fun_np(y1,t1)
y2 = y1 + (h * resultado_y2)
print("El valor de y2 es: ",y2)

# %% [markdown]
# ## Euler Modificado

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
y1 = float(input("Ingrese el valor de y1: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
t1 = t0 + h
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
resultado_yn = fun_np(y0,t0)
resultado_yn1 = fun_np(y1,t1)
y1prima = y0 + ((h/2) * (resultado_yn + resultado_yn1))
print("El valor de y\'1 es: ", y1prima)
t0 = t0 + h
t1 = t0 + h
y0 = y1
y1 = y1prima
resultado_yn = fun_np(y0,t0)
resultado_yn1 = fun_np(y1,t1)
y2prima = y0 + ((h/2) * (resultado_yn + resultado_yn1))
print("El valor de y\'2 es: ",y2prima)

# %% [markdown]
# ## Runge Kutta $2^o$ orden

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
k1 = h * (fun_np(y0,t0))
yn_k1 = y0 + k1
tn_h = t0 + h
print(yn_k1)
print(tn_h)
resultado_f = fun_np(yn_k1, tn_h)
print(resultado_f)
k2 = h * resultado_f
y_n1 = y0 + (0.5*(k1+k2))
print("k1 para y1 : ",k1)
print("k2 para y1: ",k2)
print('El valor de y1 es: ',y_n1)
y0 = y_n1
t0 = t0 + h
k1 = h * (fun_np(y0,t0))
yn_k1 = y0 + k1
tn_h = t0 + h
k2 = h * (fun_np(yn_k1, tn_h))
y_n2 = y0 + (0.5*(k1+k2))
print("k1 para y2 : ",k1)
print("k2 para y2: ",k2)
print('El valor de y2 es: ',y_n1)

# %% [markdown]
# ## Runge Kutta de $3^{er}$ orden

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
k1 = h * (fun_np(y0,t0))
yn_k1 = y0 + (k1/2)
tn_h2 = t0 + (h/2)
resultado_f2 = fun_np(yn_k1, tn_h2)
k2 = h * resultado_f2
yn_k1_2k2 = y0 -k1 + (2*k2)
tn_h = t0 + h
resultado_f3 = fun_np(yn_k1_2k2, tn_h)
k3 = h * resultado_f3
y_n1 = y0 +((1/6)  * (k1 + (4*k2) + k3))
print("k1 para y1 : ",k1)
print("k2 para y1: ",k2)
print("k3 para y1: ",k3)
print('El valor de y1 es: ',y_n1)
y0 = y_n1
t0 = t0 + h
k1 = h * (fun_np(y0,t0))
yn_k1 = y0 + (k1/2)
tn_h2 = t0 + (h/2)
resultado_f2 = fun_np(yn_k1, tn_h2)
k2 = h * resultado_f2
yn_k1_2k2 = y0 -k1 + (2*k2)
tn_h = t0 + h
resultado_f3 = fun_np(yn_k1_2k2, tn_h)
k3 = h * resultado_f3
y_n2 = y0 +((1/6)  * (k1 + (4*k2) + k3))
print("k1 para y2 : ",k1)
print("k2 para y2: ",k2)
print("k3 para y2: ",k3)
print('El valor de y2 es: ',y_n2)

# %% [markdown]
# ## Runge Kutta de $4^{to}$ orden por $\dfrac{1}{3}$ de Simpson

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
k1 = h * (fun_np(y0,t0))
yn_k2 = y0 + (k1/2)
tn_h2 = t0 + (h/2)
resultado_f2 = fun_np(yn_k2, tn_h2)
k2 = h * resultado_f2
yn_k3 = y0 +(k2/2)
resultado_f3 = fun_np(yn_k3, tn_h2)
k3 = h * resultado_f3
yn_k4 = y0 + k3
tn_h = t0 + h
resultado_f4 = fun_np(yn_k4, tn_h)
k4 = h * resultado_f4
y_n1 = y0 +((1/6)  * (k1 + (2*k2) + (2*k3) + k4))
print("k1 para y1 : ",k1)
print("k2 para y1: ",k2)
print("k3 para y1: ",k3)
print("k4 para y1: ",k4)
print('El valor de y1 es: ',y_n1)
y0 = y_n1
t0 = t0 + h
k1 = h * (fun_np(y0,t0))
yn_k2 = y0 + (k1/2)
tn_h2 = t0 + (h/2)
resultado_f2 = fun_np(yn_k2, tn_h2)
k2 = h * resultado_f2
yn_k3 = y0 +(k2/2)
resultado_f3 = fun_np(yn_k3, tn_h2)
k3 = h * resultado_f3
yn_k4 = y0 + k3
tn_h = t0 + h
resultado_f4 = fun_np(yn_k4, tn_h)
k4 = h * resultado_f4
y_n2 = y0 +((1/6)  * (k1 + (2*k2) + (2*k3) + k4))
print("k1 para y2 : ",k1)
print("k2 para y2: ",k2)
print("k3 para y2: ",k3)
print("k4 para y2: ",k4)
print('El valor de y2 es: ',y_n2)

# %% [markdown]
# ## Runge Kutta de $4^{to}$ orden por $\dfrac{3}{8}$ de Simpson

# %%
x, y, t = sp.symbols("x y t")
fun = input('Ingrese la ecuación despejada para y\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
y0 = float(input("Ingrese el valor de y0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,t], fun_sim, 'numpy')
k1 = h * (fun_np(y0,t0))
yn_k2 = y0 + (k1/3)
tn_h3 = t0 + (h/3)
resultado_f2 = fun_np(yn_k2, tn_h3)
k2 = h * resultado_f2
yn_k3 = y0 +(k1/3) + (k2/3)
tn_23h = (2/3) * h
resultado_f3 = fun_np(yn_k3, tn_23h)
k3 = h * resultado_f3
yn_k4 = y0 + k1-k2+k3
tn_h = t0 + h
resultado_f4 = fun_np(yn_k4, tn_h)
k4 = h * resultado_f4
y_n1 = y0 +((1/8)  * (k1 + (3*k2) + (3*k3) + k4))
print("k1 para y1 : ",k1)
print("k2 para y1: ",k2)
print("k3 para y1: ",k3)
print("k4 para y1: ",k4)
print('El valor de y1 es: ',y_n1)
y0 = y_n1
t0 = t0 + h
k1 = h * (fun_np(y0,t0))
yn_k2 = y0 + (k1/3)
tn_h3 = t0 + (h/3)
resultado_f2 = fun_np(yn_k2, tn_h3)
k2 = h * resultado_f2
yn_k3 = y0 +(k1/3) + (k2/3)
tn_23h = (2/3) * h
resultado_f3 = fun_np(yn_k3, tn_23h)
k3 = h * resultado_f3
yn_k4 = y0 + k1-k2+k3
tn_h = t0 + h
resultado_f4 = fun_np(yn_k4, tn_h)
k4 = h * resultado_f4
y_n2 = y0 +((1/8)  * (k1 + (3*k2) + (3*k3) + k4))
print("k1 para y2 : ",k1)
print("k2 para y2: ",k2)
print("k3 para y2: ",k3)
print("k4 para y2: ",k4)
print('El valor de y2 es: ',y_n2)

# %% [markdown]
# ## Runge Kutta de Orden superior
# 
# Para ingresar y', hay que usar v en lugar de y'

# %%
v, x, y, t = sp.symbols("v x y t")
fun = input('Ingrese la ecuación despejada para y\'\': ')
fun_sim= sp.sympify(fun)
display(Math(sp.latex(fun_sim)))
coeficientes = fun_sim.as_coefficients_dict()
a = coeficientes.get(v)
b = coeficientes.get(y)
if a is None:
    coef_v = fun_sim.coeff(v,1)
    coef_v_bueno = coef_v.coeff(t,1)
    a = coef_v_bueno
if b is None:
    coef_y = fun_sim.coeff(y,1)
    coef_y_bueno = coef_y.coeff(t,1)
    b = coef_y_bueno
a = abs(a)
b = abs(b)
print(a)
print(b)
y0 = float(input("Ingrese el valor de y0: "))
y0prima = float(input("Ingrese el valor de y\'0: "))
h = float(input("Ingrese el valor de h: "))
t0 = 0
fun_np = sp.lambdify([y,v,t], fun_sim, 'numpy')
k1 = h * y0prima
m1 = h * (fun_np(y0,y0prima,t0))
k2 = h * (y0prima + m1)
m2 = h * (a*((y0prima+m1) * (t0 + h)) -b*(y0+k1))
y1 = y0 + (0.5 * (k1+k2))
y1prima = y0prima + (0.5 * (m1+m2))
print('k1 para y1: ',k1)
print('k2 para y1: ',k2)
print('m1 para y1: ',m1)
print('m2 para y1: ',m2)
print('y1 es: ',y1)
print('y\'1 es: ', y1prima)
y0 = y1
y0prima = y1prima
t0 = t0 + h
k1 = h * y0prima
m1 = h * (fun_np(y0,y0prima,t0))
k2 = h * (y0prima + m1)
m2 = h * (a*((y0prima+m1) * (t0 + h)) -b*(y0+k1))
y2 = y0 + (0.5 * (k1+k2))
y2prima = y0prima + (0.5 * (m1+m2))
print('k1 para y2: ',k1)
print('k2 para y2: ',k2)
print('m1 para y2: ',m1)
print('m2 para y2: ',m2)
print('y2 es: ',y2)
print('y\'2 es: ', y2prima)


