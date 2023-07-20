# Importar dependencias
import os
import sys
    
## Ingreso de variables
arglist = sys.argv

# Corre el comando en una línea con los sigueintes parámetros: foler_path, filename, save_file(true or false)
if(len(arglist)<=2):
    print('Falta especificar la ruta de las carpetas y el nombre del archivo')
else:
    # Obtener parametros
    folder_path = arglist[1]
    filename = arglist[2]
    save_file = False
    if(len(arglist)>3): save_file = arglist[3]

    # Revisar si el folder existe
    path = os.path.join(folder_path, 'inputs')
    dict_csv = { os.path.basename(x).split('.')[0]: path + '/' + x for x in os.listdir(path) if x.endswith('.csv') }
    if(dict_csv.get(filename)==None):
        error = 'No se encontró el archivo con el nombre ' + filename
        print(error)
    else:
        import numpy as np
        import pandas as pd
        from scipy import optimize

        # Convert save_file a boleano
        if save_file == 'True' or save_file == 'true' or save_file == '1': save_file = True
        else: save_file = False

        ## Funciones
        def iones():
            ion_path = os.path.join(folder_path, 'templates')
            dict_csv = { os.path.basename(x).split('.')[0]: os.path.join(ion_path, x) for x in os.listdir(ion_path) if x.endswith('.csv') }
            ion_df = pd.read_csv(dict_csv['iones']).fillna(0)

            indice = ion_df['Nombre']

            copy = ion_df.copy()
            del copy['Nombre']
            del copy['Ingredientes']
            del copy['Proveedor']

            A = copy.to_numpy()
            AT = A.transpose()

            return ion_df, indice, A, AT

        def objetivo():
            input_path = os.path.join(folder_path, 'inputs')
            dict_csv = { os.path.basename(x).split('.')[0]: os.path.join(input_path, x) for x in os.listdir(input_path) if x.endswith('.csv') }
            df = pd.read_csv(dict_csv[filename]).fillna(0)

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

        def estefania():
            # Read objective
            df, indice, matriz, trans = iones()

            # Read the objective file
            target, const = objetivo()

            # Adjust our run as a x vector to optimize
            A, AT, nuevoindice = corrida(indice, const, matriz)
            x = np.random.rand(A.shape[0])

            # Optimize
            res = optimize.minimize(loss, x0=x, args=(target, A), options={'disp': False}, 
                                method='L-BFGS-B', bounds=[(0,1)]*x.shape[0])

            # resultados
            d = pd.DataFrame({'Fertilizante': nuevoindice, 'g/l': res.x})

            # % de error
            error = np.divide((target - np.dot(res.x, A))*100, target, out=np.zeros_like(target), where=target!=0)
            err = [round(num, 2) for num in error]
            e = pd.DataFrame({'error': err})

            resultados = d.join(e)
            if save_file:
                destino = os.path.join(folder_path, 'outputs', filename)            
                resultados.to_csv(destino + '.csv', index=False)
            return resultados

        estefania()