import random
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.stats as stats


def poblacion_inicial(t,m):#t: numero de individuos en la pablacion inicial(n par); m: numero total de conjuntos
    result = []
    for i in range(t):
        v = []
        for j in range(m):
            p = random.randint(1,2)
            v.append(p - 1)
        result.append(v)
    return result


def copiar_lista(l):#copia la lista l
    result = []
    for i in range(len(l)):
        result.append(l[i])
    return result


def hacer_parejas(l): #parte de precondicion de que n es par. Recibe una lista con todos los progenitores y los hijos
    lista_aux = copiar_lista(l) #para no borrar la lista que tenia
    result = []
    while len(lista_aux) > 0:
        v = lista_aux[0]
        lista_aux.remove(v)
        p = random.randint(0, (len(lista_aux) - 1))
        w = lista_aux[p]
        result.append((v,w))
        lista_aux.remove(w)
    return result


def siguiente_generacion(r): #no permitimos que un hijo sea igual a uno de los progenitores
    result = []
    for i in range(len(r)):
        progenitor1, progenitor2 = r[i]
        p = random.randint(1,(len(progenitor1) - 1))
        result.append(progenitor1[:p] + progenitor2[p:])
        result.append(progenitor2[:p] + progenitor1[p:])
    return result


def mutacion(l): #aplica una mutacion a los hijos. La probabilidad es del 1%
    for hijo in l:
        for j in range(len(hijo)):
            p = random.randint(1,100)
            if p == 1:
                hijo[j] = 1 - hijo[j]
    return l


def enfrenta_torneo(l, lista_objetos, c, i, n): #recibe una lista de parejas. Devuelve una lista con los ganadadores de los respectivos enfrentamientos. Gana el de mas fitness.
    result = []
    for i in range(len(l)):
        part1,part2 = l[i]
        punt1 = fitness(part1, lista_objetos, c, i, n)
        punt2 = fitness(part2, lista_objetos, c, i, n)
        if punt1 == punt2: #resuelvo el empate con una moneda
            p = random.randint(1,2)
            if p == 1:
                result.append(part1)
            else:
                result.append(part2)
        elif punt1 > punt2:
            result.append(part1)
        else:
            result.append(part2)
    return result


def fitness(sol,l,c,i,n): #l: una lista de tuplas (peso,valor); i: iteracion
    result = 0
    ptotal = 0
    for j in range(len(sol)):
        (peso,beneficio) = l[j]
        if (sol[j] == 1):
            result = result + beneficio
            ptotal = ptotal + peso
    diferencia = ptotal - c
    if diferencia > 0:
        result = 0
    return result


def alg_gen_MochilaP(n,t,l,c): #n: numero de iteraciones, t: tamaño de la poblacion inicial (numero par),l: lista de conjuntos
    m = len(l)
    i = 1
    #startTime = time.time()
    poblacion = poblacion_inicial(t,m)
    mejor, mejor_fit = mejor_individuo(poblacion, l, c, i, n)
    iteraciones = 1
    #print(f"Los individuos de la generacion 1 son: {poblacion}")
    while i < n: #tambien se podria añadir el criterio de parar si el 90% de los individuos son iguales
        i = i + 1
        parejas_repro = hacer_parejas(poblacion)
        sig = siguiente_generacion(parejas_repro)
        sig_mutada = mutacion(sig)
        indiv_total = poblacion + sig_mutada
        parejas_torneo = hacer_parejas(indiv_total)
        poblacion = enfrenta_torneo(parejas_torneo,l,c,i,n)
        candidato, candidato_fit = mejor_individuo(poblacion, l, c, i, n)
        if candidato_fit > mejor_fit:
            mejor = candidato
            iteraciones = i
            mejor_fit = candidato_fit
        #print(f"Los individuos de la generacion {i} son: {poblacion}")
    #print(f"Los individuos de la ultima generacion son: {poblacion}")
    #executionTime = (time.time() - startTime)
    #print(f'Execution time in seconds: {executionTime}')
    #dist = distancia_hamming(mejor,sol_optima)
    #print(f"La distancia con el optimo es de {dist}")
    return mejor, mejor_fit
        
        
def distancia_hamming(sol1,sol2): #distancia de Hamming
    result = 0
    for i in range(len(sol1)):
        if sol1[i] != sol2[i]:
            result = result + 1
    return result


def info_distancia_fitness(sol,lista_soluciones,lista_conjuntos): #se compara tambien consigo misma
    result = []
    for elem in lista_soluciones:
        result.append((elem, distancia_hamming(sol,elem), fitness(elem,lista_conjuntos), abs(fitness(elem,lista_conjuntos) - fitness(sol,lista_conjuntos))))
    return result

def mejor_individuo(pob, l, c, i ,n):#mejor individuo de una poblacion no vacia
    result = pob[0]
    fit = fitness(pob[0], l, c, i, n)
    for i in range(1, len(pob)):
        sig_fit = fitness(pob[i], l, c, i, n)
        if sig_fit > fit:
            result = pob[i]
            fit = sig_fit
    return result,fit

############
#RESULTADOS#
############
def fitness_media(pob, l, c, i ,n): #pob es una lista de individuos y l los conjuntos del enunciado. La funcion da la media del fitness de una poblacion
    den = len(pob)
    num = sum(list(map((lambda x: fitness(x, l, c, i ,n)), pob)))
    return num/den

def fitness_max(pob,l, c, i ,n): #cuidado que no funciona con conjuntos vacios
    result = fitness(pob[0],l, c, i ,n)
    for i in range(1,len(pob)):
        sig = fitness(pob[i],l, c, i ,n)
        if sig > result:
            result = sig
    return result

def fitness_min(pob,l, c, i ,n):
    result = fitness(pob[0],l, c, i ,n)
    for i in range(1,len(pob)):
        sig = fitness(pob[i],l, c, i ,n)
        if sig < result:
            result = sig
    return result

def siguiente_mutacion(pob):
    return mutacion(pob)

def siguiente_algoritmo(pob,l,c, i ,n):
    poblacion = pob
    parejas_repro = hacer_parejas(poblacion)
    sig = siguiente_generacion(parejas_repro)
    sig_mutada = mutacion(sig)
    indiv_total = poblacion + sig_mutada
    parejas_torneo = hacer_parejas(indiv_total)
    poblacion = enfrenta_torneo(parejas_torneo,l,c, i ,n)
    return poblacion
    
    
def grafica_medias_mut(tamano, l, num, c, i ,n): #tamano: tamano de la poblacion; num es el numero de generaciones que queremos estudiar
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_media(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_media(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_media(sig, l, c, i ,n))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness medio de la generación')
    plt.title('Fitness medio')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness medio primera generación')
    plt.show()
    
def grafica_min_mut(tamano, l ,num, c, i ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_min(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_min(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_min(sig, l, c, i ,n))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness minimo de la generación')
    plt.title('Fitness minimo')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness minimo primera generación')
    plt.show()
    

def grafica_max_mut(tamano, l ,num, c, i ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_max(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_max(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_max(sig, l, c, i ,n))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness maximo de la generación')
    plt.title('Fitness maximo')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness maximo primera generación')
    plt.show()

def grafica_medias_alg(tamano, l ,num, c, i ,n): #n es el numero de generaciones
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_media(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_media(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_media(sig, l, c, i ,n))
        sig = siguiente_algoritmo(sig, l, c, i ,n)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness medio de la generación')
    plt.title('Fitness medio')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness medio primera generación')
    plt.show()
    
def grafica_min_alg(tamano, l ,num, c, i ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_min(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_min(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_min(sig, l, c, i ,n))
        sig = siguiente_algoritmo(sig,l, c, i ,n)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness minimo de la generación')
    plt.title('Fitness minimo')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness minimo primera generación')
    plt.show()
    

def grafica_max_alg(tamano, l ,num, c, i ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    lista_dif = []
    fitness_piv = fitness_max(sig, l, c, i ,n)
    for i in range(num):
        generaciones.append(i)
        lista_fit.append(fitness_max(sig, l, c, i ,n))
        lista_dif.append(fitness_piv - fitness_max(sig, l, c, i ,n))
        sig = siguiente_algoritmo(sig, l, c, i ,n)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera generacion')
    plt.ylabel('Fitness maximo de la generación')
    plt.title('Fitness maximo')
    plt.show()
    plt.plot(generaciones,lista_dif)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness maximo primera generación')
    plt.show()
    
def grafica_solopt_itbit(n,t,l,c,pruebas): #El AG siempre alcanza la solucion optima.los parametros son los del algoritmo y pruebas es el n de veces que queremos que lo haga
    veces = [i for i in range(1,(pruebas + 1))]
    gen = []
    for j in range(pruebas):
        sol, it = alg_gen_MochilaP(n,t,l,c)
        gen.append(it)
    plt.plot(veces,gen)
    plt.xlabel('Casos de prueba')
    plt.ylabel('Generacion en la que aparece el optimo')
    plt.title('Cuanto tarda en aparecer la solucion optima')
    plt.show()
    
#ESTUDIO DISTANCIA DE HAMMING

def estudiom_hamming(terna): #terna es las instancias
    parejas = [i for i in range(1,101)]
    distancias = []
    diferencias_fitness = []
    for i in range(len(terna)):
        capacidad, pesos, beneficios = terna[i]
        l = list(zip(pesos,beneficios))
        for j in range(20):
            sol1 = []
            sol2 = []
            for k in range(len(l)):
                sol1.append(random.randint(0,1))
                sol2.append(random.randint(0,1))
            distancia = distancia_hamming(sol1, sol2)
            f1, f2 = fitness(sol1, l, capacidad, 0, 0), fitness(sol2, l, capacidad, 0, 0)
            distancias.append(distancia)
            diferencias_fitness.append(abs(f1 - f2))
            if j == 0:
              print(sol1,f1)
              print(sol2,f2)
              print(abs(f1 - f2))
    plt.plot(parejas,distancias)
    plt.xlabel('Parejas')
    plt.ylabel('Distancia entre los individuos de la pareja')
    plt.title('Distancias')
    plt.show()
    plt.plot(parejas,diferencias_fitness)
    plt.xlabel('Parejas')
    plt.ylabel('Diferencia de fitness entre los individuos')
    plt.title('Diferencia entre los fitness de los individuos')
    plt.show()
    r, p = stats.pearsonr(distancias, diferencias_fitness)
    print(f"Correlacion Pearson: r={r}, p-value={p}")
    r, p = stats.spearmanr(distancias, diferencias_fitness)
    print(f"Correlacion Spearman: r={r}, p-value={p}")
    r, p = stats.kendalltau(distancias, diferencias_fitness)
    print(f"Correlacion Kendall: r={r}, p-value={p}")
    
def rendimiento_bitmochila(instancias, it, pob): #instancias es la lista con las instancias (terna2)
    #it,iteraciones, pob, poblacion inicial
    medias = []
    casos = [k for k in range(1,9)]
    for i in range(len(instancias)):
        media = 0
        c, pesos, beneficios = instancias[i]
        l = list(zip(pesos,beneficios))
        print(c,l)
        for j in range(25): #para que haga 200 ejecuciones ya que len(instancias) es 8
            mejor, fit_mejor = alg_gen_MochilaP(it, pob, l, c)
            media = media + fit_mejor
        media = media/25
        medias.append(media)
    print(medias)
    plt.plot(casos,medias)
    plt.xlabel('Instancias')
    plt.ylabel('Media del fitness de la mejor solución en las ejecuciones')
    plt.title('Rendimiento')
    plt.show()
    
def fitness1(sol,l,c,i,n): #l: una lista de tuplas (peso,valor); i: iteracion
    result = 0
    ptotal = 0
    ben = 0
    for j in range(len(sol)):
        (peso,beneficio) = l[j]
        ben = ben + beneficio
        if (sol[j] == 1):
            result = result + beneficio
            ptotal = ptotal + peso
    diferencia = ptotal - c
    #v = 5/10#esto se corresponde con cuanto queremos penalizar en 0 iteraciones
    if diferencia > 0:
        #if (i > (n//32)) and (i <= (n//16)):
            #result = result - ((1/4)*(ben + 1)*diferencia)
        #elif (i > (n//16)) and (i <= (n//8)):
            #result = result - ((1/2)*(ben + 1)*diferencia)
        #elif (i > (n//8)):
            #result = result - ((ben + 1)*diferencia)
        result = result - ((ben + 1)*diferencia)
        #result = result - (((((1 - v)/n)*i)+v)*(ben + 1)*diferencia)
    return result
