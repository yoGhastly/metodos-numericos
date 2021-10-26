# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Integración numérica:
# 
# ## ¿Qué es?
# 
# La integración numérica consiste en el uso de diversos algoritmos creados para calcular el valor numérico de una integral definida, existen múlitples razones para preferir el uso de la integración numérica a una solución analítica. La principal puede ser la imposibilidad de realizar la integración de forma analítica. Es decir, integrales que requerirían de un gran conocimiento y manejo de matemática avanzada pueden ser resueltas de una manera más sencilla mediante métodos numéricos. Incluso existen funciones integrables pero cuya primitiva no puede ser calculada, siendo la integración numérica de vital importancia. La solución analítica de una integral nos arrojaría una solución exacta, mientras que la solución numérica nos daría una solución aproximada.
# 
# ## ¿Cuál es su solución?
# 
# Para que un problema de integración numérica se considere resuelto, se tiene que obtener una aproximación al resultado de la integración definida de una función de un límite superior a un límite inferior. Esta aproximación se obtiene aplicando distintos métodos numéricos, en este caso se usarán los siguientes:
#    * Regla trapezoidal
#    * Newton Cotes Cerradas
#    * Newton Cotes Abiertas
#    * Regla de $\dfrac{1}{3}$ de Simpson 
#    * Regla de $\dfrac{3}{8}$ de Simpson 
# 

# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy as sp
sp.init_printing(use_latex='mathjax')
from IPython.display import Math,display

# %% [markdown]
# ## Regla trapezoidal:
# 
# ### ¿Cómo funciona?
# La regla se basa en aproximar el valor de la integral de $f(x)$ por el de la función lineal, que pasa a través de los puntos $(a,f(a))$ y $(b,f(b))$. La integral de ésta es igual al área del trapecio bajo la gráfica de la función lineal. Se tienen que usar polinomios de primer grado.
# 
# Su fórmula es:
# 
# $I = \dfrac{h}{2} \left(f(a)+2\sum_{i=1}^{n-1}f(a+ih)+f(b)\right)$
# 
# Donde la integral a integrar se define como:
# 
# $\int_{a}^{b} f(x) \,dx$
# 
# Para calcular el valor de $h$, se usa la fórmula $h = \dfrac{b-a}{n}$
# 
# ### ¿Cómo la puedo usar?
# 
# Para usar este método, primero hay que ingresar la función a integrar.
# El formato de funciones aceptado en esta libreta es el siguiente:
#    * Para representar productos $3x$ hay que ingresar ```3*x```
#    * Para representar funciones trigonométricas $sen(x)$ hay que usar ```sin(x)```, las demás permanecen igual
#    * Para representar exponenciales $e^x$ hay que usar ```exp(x)``` salvo un caso en específico que se detalla adelante.
#    * Para representar potencias $x^2$ hay que usar ```x^2```
#    
# Un ejemplo de una ecuación ingresada sería $3x-sin(x)+e^x$ es: ```3*x -sin(x)+exp(x)```
# 
# Después de ingresar la función, hay que ingresar el límite superior a evaluar que se representa como $a$ y el límite inferior a evaluar que se representa como $b$. Posterior a eso, se ingresa el número de subintervalos del trapezio, representado con la variable $n$. Finalmente, el método mostrará el resultado de la aproximación

# %%
import numpy as np
#la función de la integral "buena"
#TODO: MOSTRAR CALCULOS INTERNOS
#definimos la funcion de la integral a la que le pasaremos la n
def integral_iterar(integral,a,b,n):
    #calculamos h
    h = (b-a)/n
    #calculamos el valor de afuera, es decir 3/8 por h
    afuera = h/2
    primer_termino = integral(a) #el primer termino de 3/8 siempre es f(a)
    #el segundo termino es 3 multiplicado por la suma de f de i hasta n-1
    #guardaremos cada resultado de f de i hasta n - 1 en el array de terminos_sumatoria
    terminos_sumatoria = []
    #desde i = 1 hasta n-1
    #es decir 1,2,3,4 si n vale 5
    for i in range (1,n):
        #obtener el valor de ih
        i_h = i*h
        #obtener el valor de a +ih
        a_sumada = a + i_h
        #aplicar a la integral el valor de a_sumada
        termino_suma = integral(a_sumada)
        #guardar en el array el resultado de la integral
        terminos_sumatoria.append(termino_suma)
    #una vez que llega hasta n-1, multiplica por 3 todos los resultados
    segundo_termino_sin_sumar = np.multiply(2, terminos_sumatoria)
    #luego suma todos los resultados y los guarda en el segundo término
    segundo_termino_sumado = np.sum(segundo_termino_sin_sumar)
    #el tercer término es siempre f(b)
    tercer_termino = integral(b)
    #suma el primero mas el segundo mas el tercero
    suma_interna = primer_termino + segundo_termino_sumado + tercer_termino
    #mulitplica por lo de afuera a la suma
    resultado = afuera * suma_interna
    #retorna el resultado
    return resultado
x, y, z = sp.symbols("x y z")
funcion_string = input("Ingrese la función a integrar")
funcion_sp = sp.sympify(funcion_string)
display(Math(sp.latex(funcion_sp)))
funcion_integrar_np = sp.lambdify(x,funcion_sp, "numpy")
a = float(input("Ingrese el límite inferior del rango de evaluación"))
b = float(input("Ingrese el límite superior del rango de evaluación"))
n = int(input("Ingrese el número de subintervalos de la integral"))
resultado = integral_iterar(funcion_integrar_np,a,b,n )
#cambiar el nombre de la función
print("El resultado de la integral es: ",resultado)

# %% [markdown]
# ## Newton Cotes Cerradas:
# 

# %%
def newton_cerradas(funcion_integrar, a,b,n):
    dic_alfas = {1:1/2, 2:1/3, 3:3/8, 4:2/45, 5:2/288, 6:1/140,7:7/17280,8:14/14175,9:9/89600,10:5/299376}
    alfa = dic_alfas.get(n)
    dic_n1 = {0:1,1:1}
    dic_n2 = {0:1,1:4,2:1}
    dic_n3 = {0:1,1:3,2:3,3:1}
    dic_n4 = {0:7,1:32,2:12,3:32,4:7}
    dic_n5 = {0:19,1:75,2:50,3:50,4:75,5:19}
    dic_n6 = {0:41,1:216,2:27,3:272,4:27,5:216,6:41}
    dic_n7 = {0:751,1:3577,2:1323,3:2989,4:2989,5:1323,6:3577,7:751}
    dic_n8 = {0:989,1:5888,2:-928,3:10946,4:-4540,5:10946,6:-928,7:5888,8:989}
    dic_n9 = {0:2857,1:15741,2:1080,3:19344,4:5788,5:5788,6:19344,7:1080,8:15741,9:2857}
    dic_n10 = {0:16067,1:106300,2:-48525,3:272400,4:-260550,5:427368,6:-260550,7:272400,8:-48525,9:106300,10:16067}
    dic_constantes = {1:dic_n1,2:dic_n2,3:dic_n3,4:dic_n4,5:dic_n5,6:dic_n6,7:dic_n7,8:dic_n8,9:dic_n9,10:dic_n10}
    constantes = dic_constantes.get(n)
    
    h = (b-a)/n
    afuera = alfa * h
    print("h: ",h)
    print("alfa: ",alfa)
    sumatoria_terminos = []
    
    for i in range(0,n+1):
        wi = constantes.get(i)
        ih = i*h
        a_ih = a+ih
        resultado_parcial = funcion_integrar(a_ih)
        completo = wi * resultado_parcial
        print("Término de i = " + str(i) + ": ", completo)
        sumatoria_terminos.append(completo)
    
    suma_interna = np.sum(sumatoria_terminos)
    print("Suma interna: ",suma_interna)
    resultado = afuera * suma_interna
    return resultado


x, y, z = sp.symbols("x y z")
funcion_string = input("Ingrese la función a integrar")
funcion_sp = sp.sympify(funcion_string)
display(Math(sp.latex(funcion_sp)))
funcion_integrar_np = sp.lambdify(x,funcion_sp, "numpy")
a = float(input("Ingrese el límite inferior del rango de evaluación"))
b = float(input("Ingrese el límite superior del rango de evaluación"))
n = int(input("Ingrese el número de subintervalos de la integral"))

resultado = newton_cerradas(funcion_integrar_np, a,b,n)
print("El resultado de la integral es: ", resultado)

# %% [markdown]
# ## Newton Cotes Abiertas

# %%
def newton_abiertas(funcion_integrar, a,b,n):
    dic_alfas = {1:3/2, 2:4/3, 3:5/24, 4:6/20, 5:7/1440, 6:8/945}
    alfa = dic_alfas.get(n)
    dic_n1 = {0:0,1:1,2:1,3:0}
    dic_n2 = {0:0,1:2,2:-1,3:2,4:0}
    dic_n3 = {0:0,1:11,2:1,3:1,4:11,5:0}
    dic_n4 = {0:0,1:11,2:-14,3:26,4:-14,5:11,6:0}
    dic_n5 = {0:0,1:611,2:-453,3:562,4:562,5:-453,6:611,7:0}
    dic_n6 = {0:0,1:460,2:-954,3:2196,4:-2459,5:2196,6:-954,7:460,8:0}
    dic_constantes = {1:dic_n1,2:dic_n2,3:dic_n3,4:dic_n4,5:dic_n5,6:dic_n6}
    constantes = dic_constantes.get(n)
    
    h = (b-a)/(n+2)
    afuera = alfa * h
    print("h: ",h)
    print("alfa: ",alfa)
    sumatoria_terminos = []
    
    for i in range(0,n+3):
        wi = constantes.get(i)
        ih = i*h
        a_ih = a+ih
        resultado_parcial = funcion_integrar(a_ih)
        completo = wi * resultado_parcial
        print("Término de i = " + str(i) + ": ", completo)
        sumatoria_terminos.append(completo)
    
    suma_interna = np.sum(sumatoria_terminos)
    print("Suma interna: ", suma_interna)
    resultado = afuera * suma_interna
    return resultado


x, y, z = sp.symbols("x y z")
funcion_string = input("Ingrese la función a integrar")
funcion_sp = sp.sympify(funcion_string)
display(Math(sp.latex(funcion_sp)))
funcion_integrar_np = sp.lambdify(x,funcion_sp, "numpy")
a = float(input("Ingrese el límite inferior del rango de evaluación"))
b = float(input("Ingrese el límite superior del rango de evaluación"))
n = int(input("Ingrese el número de subintervalos de la integral"))

resultado = newton_abiertas(funcion_integrar_np, a,b,n)
print("El resultado de la integral es: ", resultado)

# %% [markdown]
# ## Regla de $\dfrac{1}{3}$ de Simpson

# %%
import numpy as np
#la función de la integral "buena"

#definimos la funcion de la integral a la que le pasaremos la n
def simpson_13(integral,a,b,n):
    #calculamos h
    h = (b-a)/n
    print("h: ",h)
    #calculamos el valor de afuera, es decir 3/8 por h
    afuera = h/3
    primer_termino = integral(a) #el primer termino de 3/8 siempre es f(a)
    print("Primer término f(a) : ",primer_termino)
    #el segundo termino es 3 multiplicado por la suma de f de i hasta n-1
    #guardaremos cada resultado de f de i hasta n - 1 en el array de terminos_sumatoria
    terminos_sumatoria_4 = []
    #desde i = 1 hasta n-1
    #es decir 1,2,3,4 si n vale 5
    print("Calculando los terminos de la primera sumatoria: ")
    for i in range (1,n):
        if i%2 !=0:
            #obtener el valor de ih
            i_h = i*h
            #obtener el valor de a +ih
            a_sumada = a + i_h
            #aplicar a la integral el valor de a_sumada
            termino_suma = integral(a_sumada)
            print("Término de i = " + str(i) + "para el valor de a+ih: ",a_sumada," : ",termino_suma)
            #guardar en el array el resultado de la integral
            terminos_sumatoria_4.append(termino_suma)
    #una vez que llega hasta n-1, multiplica por 3 todos los resultados
    suma_de4 = np.sum(terminos_sumatoria_4)
    #luego suma todos los resultados y los guarda en el segundo término
    suma_por4 = np.multiply(suma_de4,4)
    #TODO: Darle vista bonita con latex
    print("El resultado de 4sigma es: ", suma_por4)
    
    terminos_sumatoria_2 = []
    #desde i = 1 hasta n-1
    #es decir 1,2,3,4 si n vale 5
    print("Calculando los terminos de la primera sumatoria: ")
    for i in range (2,n-1):
        if i%2==0:
            #obtener el valor de ih
            i_h = i*h
            #obtener el valor de a +ih
            a_sumada = a + i_h
            #aplicar a la integral el valor de a_sumada
            termino_suma = integral(a_sumada)
            print("Término de i = " + str(i) + "para el valor de a+ih: ",a_sumada," : ",termino_suma)
            #guardar en el array el resultado de la integral
            terminos_sumatoria_2.append(termino_suma)
    #una vez que llega hasta n-1, multiplica por 3 todos los resultados
    suma_de2 = np.sum(terminos_sumatoria_2)
    #luego suma todos los resultados y los guarda en el segundo término
    suma_por2 = np.multiply(suma_de2, 2)
    #TODO: Darle vista bonita con latex
    print("El resultado de 2sigma es: ", suma_por2)
    
    segundo_termino= suma_por4 + suma_por2
    
    tercer_termino = integral(b)
    print("Tercer término f(b) : ",tercer_termino)

    #suma el primero mas el segundo mas el tercero
    suma_interna = primer_termino + segundo_termino + tercer_termino
    #mulitplica por lo de afuera a la suma
    print("Suma interna: ", suma_interna)
    resultado = afuera * suma_interna
    #retorna el resultado
    return resultado
x, y, z = sp.symbols("x y z")
funcion_string = input("Ingrese la función a integrar")
funcion_sp = sp.sympify(funcion_string)
display(Math(sp.latex(funcion_sp)))
funcion_integrar_np = sp.lambdify(x,funcion_sp, "numpy")
a = float(input("Ingrese el límite inferior del rango de evaluación"))
b = float(input("Ingrese el límite superior del rango de evaluación"))
n = int(input("Ingrese el número de subintervalos de la integral"))
resultado = simpson_13(funcion_integrar_np,a,b,n )
#cambiar el nombre de la función
print("El resultado de la integral es: ",resultado)

# %% [markdown]
# ## Regla de $\dfrac{3}{8}$ de Simpson:
# 

# %%
import numpy as np
#la función de la integral "buena"

#definimos la funcion de la integral a la que le pasaremos la n
def simpson_38(integral,a,b,n):
    #calculamos h
    h = (b-a)/n
    print("h: ",h)
    #calculamos el valor de afuera, es decir 3/8 por h
    afuera = h * (3/8)
    primer_termino = integral(a) #el primer termino de 3/8 siempre es f(a)
    print("Primer término f(a) : ",primer_termino)
    #el segundo termino es 3 multiplicado por la su
    
    terminos_sumatoria = []
    #desde i = 1 hasta n-1
    #es decir 1,2,3,4 si n vale 5
    print("Calculando los terminos de la primera sumatoria: ")
    for i in range (1,n):
        #obtener el valor de ih
        i_h = i*h
        #obtener el valor de a +ih
        a_sumada = a + i_h
        #aplicar a la integral el valor de a_sumada
        termino_suma = integral(a_sumada)
        print("Término de i = " + str(i) + "para el valor de a+ih: ",a_sumada," : ",termino_suma)
        #guardar en el array el resultado de la integral
        terminos_sumatoria.append(termino_suma)
    #una vez que llega hasta n-1, multiplica por 3 todos los resultados
    suma = np.sum(terminos_sumatoria)
    #luego suma todos los resultados y los guarda en el segundo término
    suma_por3 = np.multiply(suma,3)
    #TODO: Darle vista bonita con latex
    print("El resultado de 3sigma es: ", suma_por3)
    tercer_termino = integral(b)
    print("Tercer término f(b) : ",tercer_termino)

    #suma el primero mas el segundo mas el tercero
    suma_interna = primer_termino + suma_por3 + tercer_termino
    #mulitplica por lo de afuera a la suma
    print("Suma interna: ", suma_interna)
    resultado = afuera * suma_interna
    #retorna el resultado
    return resultado
x, y, z = sp.symbols("x y z")
funcion_string = input("Ingrese la función a integrar")
funcion_sp = sp.sympify(funcion_string)
display(Math(sp.latex(funcion_sp)))
funcion_integrar_np = sp.lambdify(x,funcion_sp, "numpy")
a = float(input("Ingrese el límite inferior del rango de evaluación"))
b = float(input("Ingrese el límite superior del rango de evaluación"))
n = int(input("Ingrese el número de subintervalos de la integral"))
resultado = simpson_38(funcion_integrar_np,a,b,n )
#cambiar el nombre de la función
print("El resultado de la integral es: ",resultado)


