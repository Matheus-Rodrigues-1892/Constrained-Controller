from scipy.optimize import linprog
import numpy as np
import time

import numpy as np

Gv = np.array([
    [ 1.0000,  0,       0,       0],
    [-1.0000,  0,       0,       0],
    [ 0,       1.0000,  0,       0],
    [ 0,      -1.0000,  0,       0],
    [ 0,       0,       1.0000,  0],
    [ 0,       0,      -1.0000,  0],
    [ 0,       0,       0,       1.0000],
    [ 0,       0,       0,      -1.0000],
    [ 0.9435,  0,       0.1067,  0],
    [-0.9435,  0,      -0.1067,  0],
    [ 0,       0.9435,  0,       0.1067],
    [ 0,      -0.9435,  0,      -0.1067],
    [ 0.8901,  0,       0.1956,  0],
    [ 0,       0.8901,  0,       0.1956],
    [ 0.8398,  0,       0.2691,  0],
    [ 0,       0.8398,  0,       0.2691],
    [ 0.7923,  0,       0.3291,  0],
    [ 0,       0.7923,  0,       0.3291],
    [ 0.7475,  0,       0.3775,  0],
    [ 0,       0.7475,  0,       0.3775],
    [ 0.7053,  0,       0.4157,  0],
    [ 0,       0.7053,  0,       0.4157],
    [ 0.6654,  0,       0.4453,  0],
    [ 0,       0.6654,  0,       0.4453],
    [ 0.6278,  0,       0.4673,  0],
    [ 0,       0.6278,  0,       0.4673],
    [ 0.5923,  0,       0.4830,  0],
    [ 0,       0.5923,  0,       0.4830],
    [ 0.5588,  0,       0.4931,  0],
    [ 0,       0.5588,  0,       0.4931],
    [ 0.5272,  0,       0.4985,  0],
    [ 0,       0.5272,  0,       0.4985],
    [ 0.4974,  0,       0.5000,  0],
    [ 0,       0.4974,  0,       0.5000],
    [ 0.4693,  0,       0.4981,  0],
    [ 0,       0.4693,  0,       0.4981],
    [ 0.4428,  0,       0.4934,  0],
    [ 0,       0.4428,  0,       0.4934],
    [ 0.4177,  0,       0.4865,  0],
    [ 0,       0.4177,  0,       0.4865],
    [ 0.3941,  0,       0.4776,  0],
    [ 0,       0.3941,  0,       0.4776],
    [ 0.3718,  0,       0.4671,  0],
    [ 0,       0.3718,  0,       0.4671],
    [ 0.3508,  0,       0.4555,  0],
    [ 0,       0.3508,  0,       0.4555],
    [ 0.3310,  0,       0.4429,  0],
    [ 0,       0.3310,  0,       0.4429],
    [ 0.3123,  0,       0.4295,  0],
    [ 0,       0.3123,  0,       0.4295],
    [ 0.2946,  0,       0.4156,  0],
    [ 0,       0.2946,  0,       0.4156],
    [ 0.2780,  0,       0.4014,  0],
    [ 0,       0.2780,  0,       0.4014],
    [ 0.2622,  0,       0.3869,  0],
    [ 0,       0.2622,  0,       0.3869],
    [ 0.2474,  0,       0.3724,  0],
    [ 0,       0.2474,  0,       0.3724],
    [ 0.2334,  0,       0.3579,  0],
    [ 0,       0.2334,  0,       0.3579],
    [ 0.2202,  0,       0.3435,  0],
    [ 0,       0.2202,  0,       0.3435]
])

rhov = np.array([
    10.9537, 19.0463, 10.9537, 19.0463, 25.2384, 4.7616, 25.2384, 4.7616,
    10.6251, 18.4749, 10.6251, 18.4749, 10.3063, 10.3063, 9.9972, 9.9972,
    9.6972, 9.6972, 9.4063, 9.4063, 9.1241, 9.1241, 8.8504, 8.8504,
    8.5849, 8.5849, 8.3274, 8.3274, 8.0775, 8.0775, 7.8352, 7.8352,
    7.6002, 7.6002, 7.3721, 7.3721, 7.1510, 7.1510, 6.9365, 6.9365,
    6.7284, 6.7284, 6.5265, 6.5265, 6.3307, 6.3307, 6.1408, 6.1408,
    5.9566, 5.9566, 5.7779, 5.7779, 5.6045, 5.6045, 5.4364, 5.4364,
    5.2733, 5.2733, 5.1151, 5.1151, 4.9617, 4.9617
]).reshape(-1, 1)

A = np.array([
    [0.9435, 0,      0.1067, 0],
    [0,      0.9435, 0,      0.1067],
    [0,      0,      0.8901, 0],
    [0,      0,      0,      0.8901]
])

B = np.array([
    [0.1346, 0.0076],
    [0.0076, 0.1346],
    [0,      0.1308],
    [0.1308, 0]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

U = np.array([
    [1,  0],
    [-1, 0],
    [0,  1],
    [0, -1]
])


phi = np.array([
    4,
    8,
    4,
    8
]).reshape(-1, 1)

N = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])

nla = A.shape[0]
lambda_v = 0.97


# nlgv_ = Gv_.shape[0]
nlgv = Gv.shape[0]
nlu = U.shape[0]


GvB = Gv @ B
Apo = np.hstack((GvB, -rhov))
Apo = np.vstack((Apo, np.hstack((U, np.zeros((nlu, 1))))))

##############  | Gv_B -rhov |
##############  |  U  0 |
##############  |--------|

########### Servidor ###################
import socket
import struct  

server_port = 3000
input_buffer_size = 32

# Cria o socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Permite reutilização do endereço
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Liga o socket ao endereço local e porta
server_socket.bind(('localhost', server_port))

# Habilita o servidor para aceitar conexões
server_socket.listen(1)

#####################################

tempo_inicial = time.time()

print(f'Servidor TCP/IP em execução na porta {server_port}...')
k = 0

try:
    while True:
        # Aguarda por uma nova conexão
        client_socket, client_address = server_socket.accept()
        print(f'Conexão recebida de {client_address}')
        
        try:
            while True:
                # Recebe os dados do cliente
                data = client_socket.recv(input_buffer_size)
                if not data:
                    print('Cliente desconectado.')
                    break
                
                start_time = time.time()
                # Decodifica os bytes recebidos em 4 valores double (big-endian)
                received_data = struct.unpack('!dddd', data)

                xk = np.array([
                    [received_data[0]],
                    [received_data[1]],
                    [received_data[2]],
                    [received_data[3]]
                ])

                y = C @ xk

                Aa = np.vstack((Gv, C, -C))
                Bb = np.vstack(((lambda_v ** k) * rhov, y, -y))
                psi = np.zeros((nlgv, 1))

                for j in range(0, nlgv):
                    fobj_psi= -Gv[j, :] @ A
                    res_psi= linprog(fobj_psi, A_ub=Aa, b_ub=Bb, bounds=[None, None], method='highs')
                    
                    if res_psi.success and res_psi.fun is not None:
                        psi[j] = -res_psi.fun
                    else:
                        print(f'Otimização falhou para j={j}: {res_psi.message}')
                        psi[j] = 0  # Valor padrão em caso de falha


                Bpo = np.vstack((
                    -psi, 
                    phi,           
                )).flatten()

                print(f'Estado atual: {xk.T}')
                print(f'Bpo shape: {Bpo.shape}, Apo shape: {Apo.shape}')

                fobj = np.array([0, 0, 1])
                res_u = linprog(fobj, A_ub=Apo, b_ub=Bpo, bounds=[None, None], method='highs')
                
                if res_u.success and res_u.x is not None:
                    uk = struct.pack('!dd', res_u.x[0], res_u.x[1])
                    print(f'Controle: {res_u.x}')
                    print(k)
                    k += 1
                    client_socket.sendall(uk)
                else:
                    print(f'Otimização de controle falhou: {res_u.message}')
                    # Enviar controle zero em caso de falha
                    uk = struct.pack('!dd', 0.0, 0.0)
                    client_socket.sendall(uk)
                end_time = time.time()
            
        except Exception as e:
            print(f'Erro durante a comunicação com o cliente: {e}')
        finally:
            
            client_socket.close()

except Exception as e:
    print(f'Erro durante a execução do servidor: {e}')
finally:
    # Fecha o socket do servidor
    server_socket.close()
    print('Servidor TCP/IP encerrado.')
