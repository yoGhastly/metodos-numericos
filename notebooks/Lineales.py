# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Ecuaciones Lineales
# 
# ## ¿Qué son?
# 
# Una ecuación lineal es una ecuación en la que cada uno de los términos que la componen tiene una variable de primer grado, es decir, que el exponente de su potencia es igual a 1. Por ejemplo $3x + y - 4z = 23$
# 
# Un sistema de ecuaciones lineales es un conjunto de ecuaciones lineales definidas y relacionadas entre sí. Por ejemplo $\begin{cases} 3x + 5y + z = 3\\ 7x – 2y + 4z = 7\\ -6x + 3y + 2z = -9\end{cases}$
# 
# ## ¿Cuál es su solución?
# 
# Un sistema de ecuaciones lineales se considera solucionado cuando se encuentran los valores de las incógnitas que satisfagan todas y cada una de las ecuaciones individuales que componen al sistema. Por ejemplo, para $\begin{cases} 3x + 5y + z = 3\\ 7x – 2y + 4z = 7\\ -6x + 3y + 2z = -9\end{cases}$    
# La solución sería: $\begin{array}{rcl} x & = & \dfrac{295}{229}\\ y & = & \dfrac{-15}{229}\\ z & = & \dfrac{-123}{229} \end{array}$
# 
# Para poder solucionar a los sistemas de ecuaciones existen distintos procedimientos algebraicos, gráficos o matriciales. En este capítulo se abordarán 5 métodos matriciales:
#    * Montante
#    * Gauss Jordan
#    * Eliminación Gaussiana
#    * Gauss Seidel
#    * Jacobi
#   
# 

# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
import sympy as sp
from IPython.display import Math,display
sp.init_printing(use_latex='mathjax')
import warnings
import math
import ipywidgets as widgets
#warnings.filterwarnings('error')
np.seterr(all='print')

# %% [markdown]
# ## Montante
# 
# ### ¿Cómo funciona?
# 
# Dentro de la matriz tomaremos a un elemento y lo llamaremos pivote, este pivote tiene que ser distinto a cero y considerar el signo que le corresponde por su ubicación en la determinante de la matriz.
# 
# La fórmula del Método Montante es:
# 
# $N.E. = \dfrac{(E.P.)(E.A.)-(E.C.F.P.)(E.C.C.P.)}{P.A.}$
# 
# Donde:
#    * $N.E.$ es el nuevo elemento
#    * $E.P.$ es el elemento pivote
#    * $E.A.$ es el elemento actual
#    * $E.C.F.P.$ es el elemento correspondiente a la fila del pivote
#    * $E.C.C.P.$ es el elemento correspondiente a la columna del pivote
#    * $P.A.$ es el pivote anterior
#    
# ## ¿Cómo lo puedo usar?
# 
# Para usar este método, hay que ejecutar la **celda** de código siguiente e ingresar los coeficientes de las variables del sistema de ecuaciones y su solución. Un ejemplo sería: 

# %%
#https://stackoverflow.com/questions/41251911/check-if-an-array-is-a-multiple-of-another-array
#el algoritmo de montante funciona, hay que revisar como se validan todas las entradas
def coincident(one, two):
    if one[0]%two[0] == 0:
        return True
    else:
        rango = len(one)
        for v in range(1,rango):
            if one[v] % two[v] == 0:
                return True
        return False

n = int(input("Ingrese el número de variables del sistema"))
m = []
for i in range(0,n):
    m.append([int(j) for j in input("Ingrese los coeficientes de cada ecuación y su solución separados con un espacio").split()])
matriz_np = np.array(m, dtype=object)
print(matriz_np)
x = matriz_np[0][0]
if x == 0:
    matriz_np[[0, 1]] = matriz_np[[1, 0]] #swapea las filas si es 0
    x = matriz_np[0][0]
    if x == 0:
        matriz_np[[0, 2]] = matriz_np[[2, 0]]
        x = matriz_np[0][0]
        if x == 0:
            print("La matriz no se puede resolver, todos los coeficientes de la primer variable son 0")
            exit
contenido_x = matriz_np[:3, :1].copy() #copia cada columna a un array de arrays
contenido_y = matriz_np[:3, 1:2].copy()
contenido_z = matriz_np[:3, 2:3].copy()
soluciones = matriz_np[:3, 3:4].copy()
flatten_x = np.reshape(contenido_x,-1) #flattea la copia a un array unidimensional
flatten_y = np.reshape(contenido_y,-1)
flatten_z = np.reshape(contenido_z,-1)
flatten_sol = np.reshape(soluciones,-1)

is_all_zero = np.all((flatten_y == 0)) #no checamos por x porque por eso se hace el reacomodo de filas
if is_all_zero:
    print("La matriz no se puede resolver, todos los coeficientes de la segunda variable son 0")
    exit
is_all_zero = np.all((flatten_z == 0))
if is_all_zero:
    print("La matriz no se puede resolver, todos los coeficientes de la tercer variable son 0")
    exit
is_all_zero = np.all((flatten_sol == 0))
if is_all_zero:
    print("La matriz no se puede resolver, todas las soluciones son 0")
    exit

'''
multiplo_xy = coincident(flatten_x, flatten_y) #si las filas son múltiplos tampoco se puede resolver
if multiplo_xy:
    print("Las ecuaciones son múltiplos completos, no se puede resolver")
    print("xy")
    exit
else:
    multiplo_xz = coincident(flatten_x, flatten_z)
    if multiplo_xz:
        print("Las ecuaciones son múltiplos completos, no se puede resolver")
        print("xz")
        print(contenido_x)
        print(contenido_z)
        exit
    else:    
        multiplo_yz = coincident(flatten_y, flatten_z)
        if multiplo_yz:
            print("Las ecuaciones son múltiplos completos, no se puede resolver")
            print("yz")
            print(contenido_y)
            print(contenido_z)
            exit
        else:
            print("ok")
'''


pivAnt = 1
for pivN in range(0,n):
	for r in range(0,n):
		for c in range(0,n+1):
			if c != pivN and r!= pivN:
				matriz_np[r][c]=(matriz_np[pivN][pivN] * matriz_np[r][c] -matriz_np[r][pivN]*matriz_np[pivN][c])/pivAnt
		if r != pivN:
			matriz_np[r][pivN]	= 0
	print ("\n" + str(matriz_np))
	pivAnt=matriz_np[pivN][pivN]

print ("Matriz final:\n" + str(matriz_np))
x=matriz_np[0][n]/matriz_np[0][0]
y=matriz_np[1][n]/matriz_np[1][1]
z=matriz_np[2][n]/matriz_np[2][2]


print ("\nValores de x,y,z:") 
print ("X = " + str(x))
print ("Y = " + str(y))
print ("Z = " + str(z))




# %% [markdown]
# ## Gauss Jordan

# %%
#falta validar
n = int(input("Ingrese el número de incógnitas"))
m_coef = []
for i in range(0,n):
    m_coef.append([int(j) for j in input("Ingrese los coeficientes de cada ecuación separados con un espacio").split()])
matriz_coef_np = np.array(m_coef, dtype=object)
m_sol = []
for i in range(0,n):
    m_sol.append([int(j) for j in input("Ingrese una por una las soluciones de cada ecuación").split()])
matriz_sol_np = np.array(m_sol, dtype=object)
#matriz de coeficientes A
#vector de soluciones
A = matriz_coef_np

B = matriz_sol_np

# PROCEDIMIENTO
casicero = 1e-15 # Considerar como 0
# Evitar truncamiento en operaciones
A = np.array(A,dtype=float) 

# Matriz aumentada
AB = np.concatenate((A,B),axis=1)
AB0 = np.copy(AB)

# Pivoteo parcial por filas
tamano = np.shape(AB)
n = tamano[0]
m = tamano[1]

# Para cada fila en AB
for i in range(0,n-1,1):
    # columna desde diagonal i en adelante
    columna = abs(AB[i:,i])
    dondemax = np.argmax(columna)
    
    # dondemax no está en diagonal
    if (dondemax !=0):
        # intercambia filas
        temporal = np.copy(AB[i,:])
        AB[i,:] = AB[dondemax+i,:]
        AB[dondemax+i,:] = temporal
AB1 = np.copy(AB)

# eliminacion hacia adelante
for i in range(0,n-1,1):
    pivote = AB[i,i]
    adelante = i+1
    for k in range(adelante,n,1):
        factor = AB[k,i]/pivote
        AB[k,:] = AB[k,:] - AB[i,:]*factor
AB2 = np.copy(AB)

# elimina hacia atras
ultfila = n-1
ultcolumna = m-1
for i in range(ultfila,0-1,-1):
    pivote = AB[i,i]
    atras = i-1 
    for k in range(atras,0-1,-1):
        factor = AB[k,i]/pivote
        AB[k,:] = AB[k,:] - AB[i,:]*factor
    # diagonal a unos
    AB[i,:] = AB[i,:]/AB[i,i]
X = np.copy(AB[:,ultcolumna])
X = np.transpose([X])


# SALIDA
print('Matriz aumentada:')
print(AB0)
print('Pivoteo parcial por filas')
print(AB1)
print('Eliminacion hacia adelante')
print(AB2)
print('Eliminación hacia atrás')
print(AB)
print('Solución del sistema: ')
print(X)

# %% [markdown]
# ## Gauss

# %%
#falta validar
n = int(input("Ingrese el número de incógnitas"))
m_coef = []
for i in range(0,n):
    m_coef.append([int(j) for j in input("Ingrese los coeficientes de cada ecuación separados con un espacio").split()])
matriz_coef_np = np.array(m_coef, dtype=object)
m_sol = []
for i in range(0,n):
    m_sol.append([int(j) for j in input("Ingrese una por una las soluciones de cada ecuación").split()])
matriz_sol_np = np.array(m_sol, dtype=object)
#matriz de coeficientes A
#vector de soluciones
A = matriz_coef_np

B = matriz_sol_np


# PROCEDIMIENTO
casicero = 1e-15 # Considerar como 0
# Evitar truncamiento en operaciones
A = np.array(A,dtype=float) 

# Matriz aumentada
AB = np.concatenate((A,B),axis=1)
AB0 = np.copy(AB)

# Pivoteo parcial por filas
tamano = np.shape(AB)
n = tamano[0]
m = tamano[1]

# Para cada fila en AB
for i in range(0,n-1,1):
    # columna desde diagonal i en adelante
    columna = abs(AB[i:,i])
    dondemax = np.argmax(columna)
    
    # dondemax no está en diagonal
    if (dondemax !=0):
        # intercambia filas
        temporal = np.copy(AB[i,:])
        AB[i,:] = AB[dondemax+i,:]
        AB[dondemax+i,:] = temporal
AB1 = np.copy(AB)

# eliminación hacia adelante
for i in range(0,n-1,1):
    pivote = AB[i,i]
    adelante = i+1
    for k in range(adelante,n,1):
        factor = AB[k,i]/pivote
        AB[k,:] = AB[k,:] - AB[i,:]*factor

# sustitución hacia atrás
ultfila = n-1
ultcolumna = m-1
X = np.zeros(n,dtype=float)

for i in range(ultfila,0-1,-1):
    suma = 0
    for j in range(i+1,ultcolumna,1):
        suma = suma +AB[i,j]*X[j]
    b = AB[i,ultcolumna]
    X[i] = (b-suma)/AB[i,i]

X = np.transpose([X])


# SALIDA
print('Matriz aumentada:')
print(AB0)
print('Pivoteo parcial por filas')
print(AB1)
print('Eliminación hacia adelante')
print(AB)
print('Solución del sistema: ')
print(X)

# %% [markdown]
# ## Gauss Seidel

# %%
def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all( 2*np.diag(abs_x) >= np.sum(abs_x, axis=1) )


#falta validar
n = int(input("Ingrese el número de incógnitas"))
m_coef = []
for i in range(0,n):
    m_coef.append([int(j) for j in input("Ingrese los coeficientes de cada ecuación separados con un espacio").split()])
matriz_coef_np = np.array(m_coef, dtype=object)
m_sol = []
for i in range(0,n):
    m_sol.append([int(j) for j in input("Ingrese una por una las soluciones de cada ecuación").split()])
matriz_sol_np = np.array(m_sol, dtype=object)
#matriz de coeficientes A
#vector de soluciones
A = matriz_coef_np

B = matriz_sol_np

if(is_diagonally_dominant(A)):
    X0  = np.zeros(n)
    tolera = 0.001
    iteramax = 100


    # PROCEDIMIENTO

    # Gauss-Seidel
    tamano = np.shape(A)
    n = tamano[0]
    m = tamano[1]
    #  valores iniciales
    X = np.copy(X0)
    diferencia = np.ones(n, dtype=float)
    errado = 2*tolera

    itera = 0
    while not(errado<=tolera or itera>iteramax):
        # por fila
        for i in range(0,n,1):
            # por columna
            suma = 0 
            for j in range(0,m,1):
                # excepto diagonal de A
                if (i!=j): 
                    suma = suma-A[i,j]*X[j]

            nuevo = (B[i]+suma)/A[i,i]
            diferencia[i] = np.abs(nuevo-X[i])
            X[i] = nuevo
        errado = np.max(diferencia)
        itera = itera + 1
        print("Iteracion: " + str(itera))
        print(X)


    # revisa si NO converge
    if (itera>iteramax):
        X=0
    # revisa respuesta
    verifica = np.dot(A,X)

    # SALIDA
    print('Las respuestas del sistema son: ')
    print(X)
    
else:
    print("La matriz no tiene diagonal dominante, prueba a reacomodarla")
    exit()

# %% [markdown]
# ## Jacobi

# %%
def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all( 2*np.diag(abs_x) >= np.sum(abs_x, axis=1) )

def jacobi(A, b, x_init, epsilon=0.001, max_iterations=500):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    D_inv = np.diag(1 / np.diag(D))
    for i in range(1,max_iterations):
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
        x = x_new
        val_x = x[0][0]
        val_y = x[1][1]
        try:
            val_z = x[2][2]
        except:
            val_z = None
        if val_z is not None:
            str_respuesta = "La solución actual es: " + str(val_x) + " " + str(val_y) + " " +str(val_z)
        else:
            str_respuesta = "La solución actual es: " + str(val_x) + " " + str(val_y)
        print("Iteracion: " + str(i))
        print(str_respuesta)
    return x




#falta validar
n = int(input("Ingrese el número de incógnitas"))
m_coef = []
for i in range(0,n):
    m_coef.append([int(j) for j in input("Ingrese los coeficientes de cada ecuación separados con un espacio").split()])
matriz_coef_np = np.array(m_coef, dtype=object)
m_sol = []
for i in range(0,n):
    m_sol.append([int(j) for j in input("Ingrese una por una las soluciones de cada ecuación").split()])
matriz_sol_np = np.array(m_sol, dtype=object)
#matriz de coeficientes A
#vector de soluciones
A = matriz_coef_np

b = matriz_sol_np
x_init = np.ones(len(b))
if is_diagonally_dominant(A):
    x = jacobi(A, b, x_init)
    val_x = x[0][0]
    val_y = x[1][1]
    try:
        val_z = x[2][2]
    except:
        val_z = None
    if val_z is not None:
        str_respuesta = "La solución es: " + str(val_x) + " " + str(val_y) + " " +str(val_z)
    else:
        str_respuesta = "La solución es: " + str(val_x) + " " + str(val_y)
    print(str_respuesta)
else:
    print("La matriz no tiene diagonal dominante, prueba a reacomodarla")
    exit()


# %%



