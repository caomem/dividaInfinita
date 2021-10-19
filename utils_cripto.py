import numpy as np

def preparar_x_y_cripto(dados, tam_janela_x=100, dist_y=1):
    pos_atual_min_x = 0

    dados = np.array(dados)
    len_dados = len(dados)

    x_vals = [0] * (len(dados) - tam_janela_x)
    y_vals = np.zeros(len(dados) - tam_janela_x)
    j = 0
    while(pos_atual_min_x + tam_janela_x + dist_y - 1 < len_dados):
        fim_janela = pos_atual_min_x + tam_janela_x
        x_vals[j] = dados[pos_atual_min_x: fim_janela].copy()
        y_vals[j] = np.sign(dados[fim_janela + dist_y - 1] - dados[fim_janela + dist_y - 2])[0]
        pos_atual_min_x += 1
        j += 1
    
    return x_vals, y_vals

def calcular_Sigma(x_vals):
    n = len(x_vals)

    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            #Sigma[i, j] = (x_vals[i] * x_vals[j]).sum()
            Sigma[i, j] = (x_vals[i].T @ x_vals[j])[0][0]
            if(j != i):
                Sigma[j, i] = Sigma[i, j]

    return Sigma