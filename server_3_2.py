
from scipy.optimize import linprog
import numpy as np
import time

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

C = np.array([0, 1])

U = np.array([
    [1],
    [-1]
])

phi = np.array([
    [4.5541],
    [7.4459]
])

H = np.vstack((C.dot(A), -C.dot(A)))
h = np.ones((2*A.shape[0], 1))
# Definir a referência r
r = 5
             
########################################
########### Servidor ###################
import socket
import struct  

server_port = 3000
input_buffer_size = 16


# Cria o socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Permite reutilização do endereço
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Liga o socket ao endereço local e porta
server_socket.bind(('localhost', server_port))

# Habilita o servidor para aceitar conexões
server_socket.listen(1)

###### Calcular a média móvel ######
exponential_moving_average = 0
ema_smoothing = 0.5
#####################################

tempo_inicial = time.time()

print(f'Servidor TCP/IP em execução na porta {server_port}...')

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
                # Decodifica os bytes recebidos em dois valores double (big-endian)
                received_data = struct.unpack('!dd', data)
                
                # xk é o estado do sistema (tanque 1 e tanque 2)

                xk = np.array([
                    [received_data[0]],
                    [received_data[1]]
                ])
                
                # y é a saída do sistema (tanque 2)
                y = C.dot(xk)

                # Número de linhas da matriz Gv
                n = Gv.shape[0]
                
                
                #######################################################################
                ############ COMENTE ESSE BLOCO PARA USAR A ESTRATÉGIA DE STATE FEEDBACK #############
                # Inicializa o vetor psi com zeros
                psi = np.zeros((n, 1))

                # Matriz A
                Aa = np.vstack((Gv, C, -C))

                # Queremos somente a informação do tanque 2
                #B_psi = np.vstack((rhov, 1+y, 1-y)) # Estimar o ruído

                Bb = np.vstack((rhov, y, -y))
                
                # Função para calcular o vetor psi (com 13 elementos (0, 1, 2, ..., 12))
                
                for i in range(0, n):

                    # Queremos maximixar, então invertemos o sinal
                
                    fobj_psi= -Gv[i, :].dot(A)
                    
                    res_psi= linprog(fobj_psi, A_ub=Aa, b_ub=Bb, bounds=[None, None], method='highs')
                    
                    # Guarda o valor ótimo do linprog no vetor psi
                    # Invertemos o sinal de novo para obter o valor correto
                    #print(res_psi.fun)
                    psi[i] = -res_psi.fun
                

                # Número de linhas da matriz H
                nlh = H.shape[0]

                #Incializa o vetor psi com 0's
                gamma = np.zeros((nlh, 1))

                for i in range(0, nlh):

                    # Queremos maximixar, então invertemos o sinal
                
                    fobj_gamma = -H[i, :]
                    
                    res_gamma = linprog(fobj_gamma, A_ub=Aa, b_ub=Bb, bounds=[None, None], method='highs')
                    
                    # Guarda o valor ótimo do linprog no vetor gamma
                    # Invertemos o sinal de novo para obter o valor correto
                    #print(res_gamma.fun)
                    gamma[i] = -res_gamma.fun

                

                #######################################################################
                
                            

                # Multiplica Gv por B e concatena com -rhov
                Apo = np.hstack((Gv.dot(B), np.zeros((n, 1))))

                # Concatena a matriz Gv com a matriz U e zeros
                Apo = np.vstack((Apo, np.hstack((U, np.zeros((U.shape[0], 1))))))

                # Multiplica C por B
                CB = C.dot(B)

                Apo = np.vstack((Apo, np.hstack((CB, np.array([[-1]]))), np.hstack((-CB, np.array([[-1]])))))

                tempo_atual = time.time()

                if tempo_atual - tempo_inicial >= 300: 
                    r = 8
                # Concatena o vetor gamma com o vetor psi para formar o vetor Bpo
                R = np.vstack((r, -r))

                #Bpo = np.vstack((rhov - Gv.dot(A).dot(xk), psi, R - H.dot(xk))) # State feedback
                Bpo = np.vstack(((rhov - psi), psi, R - gamma)) # Output feedback


                # A função objetivo é [0, 1]
                fobj = np.array([0, 1])
                
                
                res = linprog(fobj, A_ub=Apo, b_ub=Bpo, bounds=[None, None], method='highs')
                
                uk = struct.pack('!d', res.x[0])
                
                # Envia o resultado de volta para o cliente
                client_socket.sendall(uk)
                end_time = time.time()

                diff_time = end_time-start_time
                exponential_moving_average = (1-ema_smoothing)*exponential_moving_average + ema_smoothing*diff_time

                print(f'Avg: {exponential_moving_average}, Current: {diff_time}')
                print('Dados recebidos:')
                print(received_data)
                print('Resultado:')
                print(res.x)
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
