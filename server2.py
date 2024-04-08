
from scipy.optimize import linprog
import numpy as np

#server_port = 3000
#input_buffer_size = 1024

Gv = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [0.0317, 0.9657],
    [-0.0317, -0.9657],
    [0.0609, 0.9288],
    [-0.0609, -0.9288],
    [0.08756, 0.8901],
    [-0.0876, -0.8901],
    [0.1115, 0.8501],
    [-0.1115, -0.8501],
    [-0.1327, -0.8095]
])

rhov = np.array([
    [15],
    [15],
    [15],
    [15],
    [14.8343],
    [14.8282],
    [14.6417],
    [14.6177],
    [14.4263],
    [14.3747],
    [14.1925],
    [14.1043],
    [13.8124]
])

A = np.array([
    [0.9677, 0],
    [0.0317, 0.9677]
])

B = np.array([
    [0.1300],
    [0.0021]
])

U = np.array([
    [1, 0],
    [-1, 0]
])

phi = np.array([
    [4.5541],
    [7.4459]
])
                # ncu = 1
                # nlu =2

Apo_left = Gv.dot(B)
Apo = np.concatenate((Apo_left, -rhov), axis=1)
Apo = np.concatenate((Apo, U), axis = 0)

fobj = np.array([
    [0, 1]
])

import socket
import struct  

server_port = 3000
input_buffer_size = 1024
import time

# Cria o socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Permite reutilização do endereço
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Liga o socket ao endereço local e porta
server_socket.bind(('localhost', server_port))

# Habilita o servidor para aceitar conexões
server_socket.listen(1)

print(f'Servidor TCP/IP em execução na porta {server_port}...')

try:
    while True:
        # Aguarda por uma nova conexão
        client_socket, client_address = server_socket.accept()
        print(f'Conexão recebida de {client_address}')
        
        try:
            while True:
                
            
                data = client_socket.recv(input_buffer_size)
                if not data:
                    print('Cliente desconectado.')
                    break
                
                # Decodifica os bytes recebidos em dois valores double (big-endian)
                received_data = struct.unpack('!dd', data)

                #print(data)
                
                xk = np.array([
                    [received_data[0]],
                    [received_data[1]]
                ])
                
                Bpo_left = -Gv.dot(A)
                #print(Bpo_left)
                Bpo = Bpo_left.dot(xk)
                #print(Bpo)

                Bpo = np.concatenate((Bpo, phi), axis = 0)
                #print(Bpo)

                res = linprog(fobj, A_ub=Apo, b_ub=Bpo, bounds=[None, None], method='highs')
                uk = struct.pack('!d', res.x[0])
                #input()
                
                # Envia o resultado de volta para o cliente
                client_socket.sendall(uk)
    
                print('Dados recebidos:')
                print(received_data)
                print('Resultado:')
                print(res.x[0])
                #input()
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
