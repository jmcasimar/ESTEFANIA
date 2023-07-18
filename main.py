# Import dependencies
import os
import sys
import numpy as np
import pandas as pd
from scipy import optimize

## Ingreso de variables
arglist = sys.argv

## Funciones
def iones():
    nombre = 'iones'
    path = os.getcwd() + '/templates/'
    dict_csv = { os.path.basename(x).split('.')[0]: path + '/' + x for x in os.listdir(path) if x.endswith('.csv') }
    ion_df = pd.read_csv(dict_csv[nombre]).fillna(0)

    indice = ion_df['Nombre']

    copy = ion_df.copy()
    del copy['Nombre']
    del copy['Ingredientes']
    del copy['Proveedor']

    A = copy.to_numpy()
    AT = A.transpose()

    return ion_df, indice, A, AT


def objetivo(nombre):
    path = os.getcwd() + '/inputs/'
    dict_csv = { os.path.basename(x).split('.')[0]: path + '/' + x for x in os.listdir(path) if x.endswith('.csv') }
    df = pd.read_csv(dict_csv[nombre]).fillna(0)

    target = df.iloc[0].to_numpy()[2:]
    const = df['VALOR'].to_numpy()[1:]

    return target, const


def corrida(indice, constraints, matriz):
    lista = []
    index = []

    for i in range(len(indice)):
        if constraints[i] == 1:
            print('Añadiendo: ' + indice[i])
            lista.append(matriz[i])
            index.append(indice[i])
        if constraints[i] == 0:
            print('No se utilizará: ' + indice[i])
    A = np.array(lista)

    AT = A.transpose()

    return A, AT, index

def loss(x0, k, A):
    lista = [sum((k - np.dot(x0, A))**2)/k.shape[0] for i in range(x0.shape[0])]
    return sum(lista)

def estefania(variedad, guardar=False):
    # Read objective
    df, indice, matriz, trans = iones()

    # Read the objective file
    target, const = objetivo(variedad)

    # Adjust our run as a x vector to optimize
    A, AT, nuevoindice = corrida(indice, const, matriz)
    x = np.random.rand(A.shape[0])

    # Optimize
    res = optimize.minimize(loss, x0=x, args=(target, A), options={'disp': False}, 
                        method='L-BFGS-B', bounds=[(0,1)]*x.shape[0])

    # resultados
    d = pd.DataFrame({'Fertilizante': nuevoindice, 'g/l': res.x})

    # % de error
    error = (target - np.dot(res.x, A))*100/1
    err = [round(num, 2) for num in error]
    e = pd.DataFrame({'error': err})

    resultados = d.join(e)
    if guardar == 'True' or guardar == 'true' or guardar == '1':
        destino = os.getcwd() + '/outputs/'
        resultados.to_csv(destino + variedad + '.csv', index=False)
    return resultados


#Run on the command line with the arguments: name_of_input, save(true or false)
if(len(arglist)>2): estefania(arglist[1], arglist[2])
elif(len(arglist)>1): estefania(arglist[1])
else: print('Faltan especificar el nombre del archivo y si se desea guardar el archivo de salida')

