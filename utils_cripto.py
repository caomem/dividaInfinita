import numpy as np
from numba import njit, int32, int64, float64
from numba.core.types.containers import ListType
from numba.core.types import Array

int_var = int64
int_vect_var = int64[:]
float_var = float64
np_int_var = np.int64
np_float_var = np.float64

def preparar_x_y_cripto(dados, tam_janela_x=100, dist_y=1):
    pos_atual_min_x = 0

    dados = np.array(dados)
    len_dados = len(dados)

    x_vals = [0] * (len(dados) - tam_janela_x - dist_y + 1)
    #x_vals = {i : 0 for i in range( (len(dados) - tam_janela_x - dist_y + 1) )}
    y_vals = np.zeros(len(dados) - tam_janela_x - dist_y + 1)
    j = 0
    while(pos_atual_min_x + tam_janela_x + dist_y - 1 < len_dados):
        fim_janela = pos_atual_min_x + tam_janela_x
        x_vals[j] = dados[pos_atual_min_x: fim_janela].copy()
        y_vals[j] = np.sign(dados[fim_janela + dist_y - 1] - dados[fim_janela - 1])
        #y_vals[j] = np.sign(dados[fim_janela + dist_y - 1] - x_vals[j][-1])[0]
        pos_atual_min_x += 1
        j += 1
    
    return x_vals, y_vals

def calcular_Sigma(x_vals):
    n = len(x_vals)

    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            #Sigma[i, j] = (x_vals[i] * x_vals[j]).sum()
            Sigma[i, j] = x_vals[i].dot(x_vals[j])
            if(j != i):
                Sigma[j, i] = Sigma[i, j]

    return Sigma

@njit(float_var[:, :](ListType(Array(float_var, 1, 'C'))), parallel=True, cache=True)
def numba_calcular_Sigma(x_vals):
    n = len(x_vals)

    Sigma = np.zeros((n, n), dtype=float_var)
    for i in range(n):
        for j in range(i, n):
            #Sigma[i, j] = (x_vals[i] * x_vals[j]).sum()
            Sigma[i, j] = x_vals[i].dot(x_vals[j])
            if(j != i):
                Sigma[j, i] = Sigma[i, j]

    return Sigma
