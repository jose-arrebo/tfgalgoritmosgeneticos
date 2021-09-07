#Pasar de un vector de naturales a un vector de bits
def trans_bits(v,l):
    conj = set()
    c_total = set()
    for j in l:
         c_total = c_total.union(j)
    w = quicksort_prim_comp(list(zip(v,[i for i in range(len(v))])))
    result = [0]*len(v)
    indices = []
    i = 0
    while conj != c_total:
        prior,ind = w[i]
        indices.append(ind)
        conj = conj.union(l[ind])
        i = i + 1
    for j in indices:
        result[j] = 1
    return result
        
    
def quicksort_prim_comp(l): #l lista de tuplas. Resultado de mayor a menor
    result = []
    if len(l) != 0:
        piv,ind = l[0]
        l.pop(0)
        result = quicksort_prim_comp(mayores(piv,l)) + [(piv,ind)] + quicksort_prim_comp(menores_ig(piv,l))
    return result

def menores_ig(e,l):
    result = []
    for elem,pos in l:
        if elem <= e:
            result.append((elem,pos))
    return result

def mayores(e,l):
    result = []
    for elem,pos in l:
        if elem > e:
            result.append((elem,pos))
    return result

"""
TFG ALGORITMOS GENETICOS SET COVER
"""
import random
import time
import matplotlib.pyplot as plt
import math
import scipy.stats as stats


def poblacion_inicial(t,m):#t: numero de individuos en la pablacion inicial(n par); m: numero total de conjuntos
    result = []
    for i in range(t):
        v = []
        for j in range(m):
            p = random.randint(0,100)
            v.append(p)
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
                moneda = random.randint(1,2)
                hijo[j] = hijo[j] + (((-1)**moneda)*5)
                if hijo[j] > 100:
                    hijo[j] = 100
                elif hijo[j] < 0:
                    hijo[j] = 0
    return l


def enfrenta_torneo(l, lista_conjuntos): #recibe una lista de parejas. Devuelve una lista con los ganadadores de los respectivos enfrentamientos. Gana el de menos fitness.
    result = []
    for i in range(len(l)):
        part1,part2 = l[i]
        punt1 = fitness1(part1, lista_conjuntos)
        punt2 = fitness1(part2, lista_conjuntos)
        if punt1 == punt2: #resuelvo el empate con una moneda
            p = random.randint(1,2)
            if p == 1:
                result.append(part1)
            else:
                result.append(part2)
        elif punt1 < punt2:
            result.append(part1)
        else:
            result.append(part2)
    return result


def fitness(sol, l): #l: una lista de conjuntos
    t = trans_bits(sol,l)
    result = t.count(1)
    c1 = set()
    c2 = set()
    for conj in l:
         c1 = c1.union(conj)
    for i in range(len(sol)):
        if t[i] == 1:
            c2 = c2.union(l[i])
    c_aux = c1.difference(c2)
    if len(c_aux) > 0:
        result = result + ((len(l) + 1)*len(c_aux))
    return result


def alg_gen_setCoverNat(n,t,l): #n: numero de iteraciones, t: tamaño de la poblacion inicial (numero par),l: lista de conjuntos
    m = len(l)
    i = 1
    startTime = time.time()
    poblacion = poblacion_inicial(t,m)
    mejor, mejor_fit = mejor_individuo(poblacion, l)
    iteraciones = 1
    #print(f"Los individuos de la generacion 1 son: {poblacion}")
    while i < n:
        i = i + 1
        parejas_repro = hacer_parejas(poblacion)
        sig = siguiente_generacion(parejas_repro)
        sig_mutada = mutacion(sig)
        indiv_total = poblacion + sig_mutada
        parejas_torneo = hacer_parejas(indiv_total)
        poblacion = enfrenta_torneo(parejas_torneo,l)
        candidato, candidato_fit = mejor_individuo(poblacion, l)
        if candidato_fit < mejor_fit:
            mejor = candidato
            iteraciones = i
            mejor_fit = candidato_fit
        #print(f"Los individuos de la generacion {i} son: {poblacion}")
    #print(f"Los individuos de la ultima generacion son: {poblacion}")
    executionTime = (time.time() - startTime)
    #print(f'Execution time in seconds: {executionTime}')
    #print(mejor_fit)
    #print(f'La solucion optima {mejor} aparecio en la generacion {iteraciones}')
    #print(f"Con una transformacion a vector de bits: {trans_bits(mejor,l)}")
    return mejor, mejor_fit

def instancia_prueba(n,m): #n: numero de conjuntos; m:número maximo de elementos que pueden aparecer. 65/100 de que tenga primer cuarto. 29/100, 4/100, 2/100
    l = []
    for j in range(n):
        s = set()
        p = random.randint(1,100)
        if p <= 65:
            for i in range(m//4): #el conjunto tendra <= elementos que m//4
                k = random.randint(1,m)
                s.add(k)
        elif (p > 65) and (p <= 94):
            for i in range(m//2): #el conjunto tendra <= elementos
                k = random.randint(1,m)
                s.add(k)
        elif (p > 94) and (p <= 98):
            for i in range((3*m)//4): #el conjunto tendra <= elementos
                k = random.randint(1,m)
                s.add(k)
        else:
            for i in range(m): #el conjunto tendra <= elementos
                k = random.randint(1,m)
                s.add(k)
        l.append(s)
    return l

############
#RESULTADOS#
############
def fitness_media(pob, l): #pob es una lista de individuos y l los conjuntos del enunciado. La funcion da la media del fitness de una poblacion
    den = len(pob)
    num = sum(list(map((lambda x: fitness(x, l)), pob)))
    return num/den

def fitness_max(pob,l): #cuidado que no funciona con conjuntos vacios
    result = fitness(pob[0],l)
    for i in range(1,len(pob)):
        sig = fitness(pob[i],l)
        if sig > result:
            result = sig
    return result

def fitness_min(pob,l):
    result = fitness(pob[0],l)
    for i in range(1,len(pob)):
        sig = fitness(pob[i],l)
        if sig < result:
            result = sig
    return result

def siguiente_mutacion(pob):
    return mutacion(pob)

def siguiente_algoritmo(pob,l):
    poblacion = pob
    parejas_repro = hacer_parejas(poblacion)
    sig = siguiente_generacion(parejas_repro)
    sig_mutada = mutacion(sig)
    indiv_total = poblacion + sig_mutada
    parejas_torneo = hacer_parejas(indiv_total)
    poblacion = enfrenta_torneo(parejas_torneo,l)
    return poblacion
    
    
def grafica_medias_mut(tamano, l, n): #tamano: tamano de la poblacion; n es el numero de generaciones que queremos estudiar
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_media(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness medio de la generación')
    plt.title('Fitness medio')
    plt.show()
    
def grafica_min_mut(tamano, l ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_min(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness minimo de la generación')
    plt.title('Fitness minimo')
    plt.show()
    

def grafica_max_mut(tamano, l ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_max(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness maximo de la generación')
    plt.title('Fitness maximo')
    plt.show()

def grafica_medias_alg(tamano, l ,n): #n es el numero de generaciones
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_media(sig, l))
        sig = siguiente_algoritmo(sig, l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness medio de la generación')
    plt.title('Fitness medio')
    plt.show()
    
def grafica_min_alg(tamano, l ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_min(sig, l))
        sig = siguiente_algoritmo(sig,l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Fitness minimo de la generación')
    plt.title('Fitness minimo')
    plt.show()
    

def grafica_max_alg(tamano, l ,n):
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_max(sig, l))
        sig = siguiente_algoritmo(sig, l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera generacion')
    plt.ylabel('Fitness maximo de la generación')
    plt.title('Fitness maximo')
    plt.show()
    
#VOY A HACER OTRO TIPO DE GRAFICAS QUE REPRESENTAN LA DIFERNECIA
def grafica_difmedias_mut(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LAS MEDIAS CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_media(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_media(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness medio primera generación')
    plt.show()
    
def grafica_difmin_mut(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LOS MIN CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_min(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_min(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness minimo primera generación')
    plt.show()
    
def grafica_difmax_mut(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LOS MAX CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_max(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_max(sig, l))
        sig = siguiente_mutacion(sig)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness maximo primera generación')
    plt.show()
    
def grafica_difmedias_alg(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LAS MEDIAS CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_media(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_media(sig, l))
        sig = siguiente_algoritmo(sig, l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness medio primera generación')
    plt.show()
    
def grafica_difmin_alg(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LOS MIN CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_min(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_min(sig, l))
        sig = siguiente_algoritmo(sig, l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness minimo primera generación')
    plt.show()
    
def grafica_difmax_alg(tamano, l, n): #REPRESENTA LA DIFERENCIA DE LOS MAX CON RESPECTO A LA 1ERA GEN
    sig = poblacion_inicial(tamano, len(l))
    generaciones = []
    lista_fit = []
    fitness_piv = fitness_max(sig, l)
    for i in range(n):
        generaciones.append(i)
        lista_fit.append(fitness_piv - fitness_max(sig, l))
        sig = siguiente_algoritmo(sig, l)
    plt.plot(generaciones,lista_fit)
    plt.xlabel('Distancia con respecto a la primera gerenacion')
    plt.ylabel('Variación del fitness')
    plt.title('Diferencia con el fitness maximo primera generación')
    plt.show()
    
def mejor_individuo(pob, l):#mejor individuo de una poblacion no vacia
    result = pob[0]
    fit = fitness(pob[0], l)
    for i in range(1, len(pob)):
        sig_fit = fitness(pob[i], l)
        if sig_fit < fit:
            result = pob[i]
            fit = sig_fit
    return result,fit

def grafica_solopt_it(n,t,l,pruebas): #los parametros son los del algoritmo y pruebas es el n de veces que queremos que lo haga
    veces = [i for i in range(1,(pruebas + 1))]
    gen = []
    for j in range(pruebas):
        sol, it = alg_gen_setCoverNat(n,t,l)
        gen.append(it)
    plt.plot(veces,gen)
    plt.xlabel('Casos de prueba')
    plt.ylabel('Generacion en la que aparece el optimo')
    plt.title('Cuanto tarda en aparecer la solucion optima')
    plt.show()
    
#DISTANCIA VALOR ABS
def distancia_abs(sol1, sol2):
    result = 0
    for i in range(len(sol1)):
        result = result + abs(sol1[i] - sol2[i])
    return result

def estudio_valorabs(par1,par2): #lo llamo con 10, 10
    parejas = [i for i in range(1,101)]
    distancias = []
    diferencias_fitness = []
    param1, param2 = par1, par2
    for i in range(10):
        l = instancia_prueba(param1,param2)
        for j in range(10):
            sol1 = []
            sol2 = []
            for k in range(param1):
                sol1.append(random.randint(0,100))
                sol2.append(random.randint(0,100))
            distancia = distancia_abs(sol1, sol2)
            f1, f2 = fitness1(sol1, l), fitness1(sol2, l)
            distancias.append(distancia)
            #diferencias_fitness.append(max(f1/f2, f2/f1) - 1)
            #diferencias_fitness.append(abs(f1 - f2)/((len(l) + 1)*param2))
            diferencias_fitness.append(abs(f1 - f2))
        param1 = param1 + 2
        param2 = param2 + 2
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

def gen_instancias(enunciado): #enunciado = 50
    result = []
    for i in range(10):
        result.append(instancia_prueba(enunciado, enunciado))
        enunciado = enunciado + 2
    return result
    
def rendimiento_natsetcover(instancias): #instancias lista de instancias
    #empiezo a probar con un rango de instancias de 50 a 70 conjuntos. Por empezar por algun sitio
    #100 iteraciones. 50 individuos de poblacion inicial
    medias = []
    casos = [k for k in range(1,11)]
    for i in instancias:
        media = 0
        for j in range(20):
            mejor, fit_mejor = alg_gen_setCoverNat(100, 50, i)
            media = media + fit_mejor
        media = media/20
        medias.append(media)
    plt.plot(casos,medias)
    plt.xlabel('Instancias')
    plt.ylabel('Media del fitness de la mejor solución en las ejecuciones')
    plt.title('Rendimiento')
    plt.show()

def fitness1(sol, l): #l: una lista de conjuntos
    sol_bits = trans_bits(sol, l)
    result = sol_bits.count(1)
    c1 = set()
    c2 = set()
    for conj in l:
         c1 = c1.union(conj)
    for i in range(len(sol_bits)):
        if sol_bits[i] == 1:
            c2 = c2.union(l[i])
    c_aux = c1.difference(c2)
    return result + len(c_aux)
