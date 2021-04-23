# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:07:54 2021

@author: Heez
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt 

################################################################################################################################################################
#########################################################################   Classes   ##########################################################################
################################################################################################################################################################

class Client:

    def __init__(self, name, x, y, quantity, start, stop):
        self.name = name
        self.x = x
        self.y = y
        self.quantity = quantity
        self.start = start
        self.stop = stop
        self.delivered = False

class Truck:
    
    def __init__(self, name, quantity_max, start, stop):
        self.name = name
        self.x = 0
        self.y = 0
        self.quantity_max = quantity_max
        self.start = start
        self.stop = stop
        self.remaining_quantity=quantity_max
        self.vitesse = 25

    def delivery(self, client):
        self.x=client.x
        self.y=client.y
        self.remaining_quantity -= client.quantity

################################################################################################################################################################
#########################################################################   Entries   ##########################################################################
################################################################################################################################################################


truck_capacity = 25
trucks_disponibility = 6

client_0 = Client(0,0,0,0,0,40)    
client_1 = Client(1,-2,4,1,7,12)
client_2 = Client(2,4,4,1,10,15)
client_3 = Client(3,-4,3,3,16,18)
client_4 = Client(4,-3,3,4,10,13)
client_5 = Client(5,1,2,2,0,5)
client_6 = Client(6,3,2,4,5,10)
client_7 = Client(7,-1,1,8 ,0,4)
client_8 = Client(8,2,1,8,5,10)
client_9 = Client(9,1,-1,1,0,3)
client_10 = Client(10,4,-1,2,10,16)
client_11 = Client(11,-3,-2,1,10,15)
client_12 = Client(12,-2,-2,2,0,5)
client_13 = Client(13,-1,-3,14,5,10)
client_14 = Client(14,2,-3,4,7,8)
client_15 = Client(15,-4,-4,8,10,15)
client_16 = Client(16,3,-4,8,11,15)

C = [client_1,client_2,client_3,client_4,client_5,client_6,client_7,client_8,client_9,client_10,client_11,client_12,
     client_13,client_14,client_15,client_16]


time_matrix = [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
    ]


################################################################################################################################################################
####################################################################   Utility functions   #####################################################################
################################################################################################################################################################

def find_client_by_name(name,C):
    i = 0
    n = len(C)
    find = False
    client = Client(-1,0,0,0,0,0)
    while i < n and find == False : 
        if C[i].name == name :
            client = C[i]
            find = True
        i +=1
    return client
    
        
def distance(C1, C2):
    return(np.sqrt((C1.y-C2.y)**2 + (C1.x - C2.x)**2))
    
 
    
def cost_matrix(C):
    n = len(C)
    i=0
    j=0
    M = np.zeros((n+1,n+1))
    for i in range(1,n+1):
        M[i][0] = np.sqrt(C[i-1].x**2+C[i-1].y**2)
        M[0][i] = np.sqrt(C[i-1].x**2+C[i-1].y**2)
        for j in range(1,n+1):
            M[i][j] = distance(C[i-1],C[j-1])
    return M
            
def quantity(C):
    q = 0
    for i in range(len(C)):
        q += C[i].quantity
    return q


def copy(F):
    L = []
    for x in F:
        V = []
        for i in range(len(x)):
            V.append(x[i])
        L.append(V)
    return L


def closing_tour(itineraire): 
    for x in itineraire:
        x.insert(0,0)
        if x[-1]!=0:
            x.append(0)
        del(x[1])

def convert_solution(solution, trucks_list):
    if solution[0][0]==0:
        for i in range(len(solution)):
            solution[i][0]=trucks_list[i]
            del(solution[i][-1])
#         for i in range(len(solution),len(trucks_list)+1):
#             solution.append(trucks_list[i])
    return solution

################################################################################################################################################################
#######################################################################   RS Algorithm   #######################################################################
################################################################################################################################################################


    
    
def random_solution(C):
    C2 = [client.name for client in C]
    q = quantity(C)
    t = (q // truck_capacity) + 2
    T = [Truck(i,truck_capacity,0,0) for i in range(trucks_disponibility)]
    R = []
    C_bis = C2.copy()
    for i in range(t-1):
        s = rd.sample(C_bis,rd.randint(1,len(C_bis)-(t-i+1)))
        R.append([T[i]]+s)
        for x in s :
            C_bis.remove(x)
    rd.shuffle(C_bis)
    L  = [T[t-1]]+C_bis
    R.append(L)
    for i in range(t,len(T)):
        R.append([T[i]])
    return R
     
def neighbouring_solution(R):
     n = len(R)
     S = copy(R)
     i = rd.randint(0,n-1)
     j = rd.randint(0,n-1)
     l = rd.randint(1,len(S[i]))
     k = rd.randint(1,len(S[j]))
     
     if l == len(S[i]) and k == len(S[j]):
         return S
     
     elif k == len(S[j]) :
         client = S[i][l]
         S[j].append(client)
         S[i].remove(client)
             
     elif l == len(S[i]) : 
        client = S[j][k]
        S[i].append(client)
        S[j].remove(client)
    
    
     else:         
         client1 = S[i][l]
         client2 = S[j][k]
         S[i][l] = client2
         S[j][k] = client1
    
     
     return S



def cost_function(R,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c):
    T = 0
    n = len(R)
    K = 0
    for x in R :
        if len(x)> 1 :
            K+=1
    t = 0
    d = 0
    c = 0
    q = 0
    dispo = [R[i][0].start for i in range(n) ]
    
    for i in range(n):
        if len(R[i])> 1 :
            truck = R[i][0]
            truck.remaining_quantity = truck.quantity_max
            client = find_client_by_name(R[i][1],C)
            dispo[i] = dispo[i] + M_time[client.name][0]
            c+= M[client.name][0]
            d += max(0,dispo[i] - client.stop)
            t -= min(0, dispo[i] - client.start)
            truck.delivery(client)
            
            for j in range(2,len(R[i])):
                client_prec = find_client_by_name(R[i][j-1],C)
                client = find_client_by_name(R[i][j],C)
                c+= M[client_prec.name][client.name]
                dispo[i] = dispo[i] + M_time[client_prec.name][client.name]
                d += max(0,dispo[i] - client.stop)
                t -= min(0, dispo[i] - client.start)
                truck.delivery(client)
            c += M[0][client.name]
            q -= min(0,truck.remaining_quantity)
    T = weight_K * K + weight_q * q + weight_t * t + weight_d * d + weight_c * c
    return T

  
    
def algo_RS(C,M_time,n, weight_K=10_000, weight_q=10_000, weight_t=2, weight_d=15, weight_c=10):
    M = cost_matrix(C)
    R = random_solution(C)
    T = cost_function(R,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c)
    i = 0
    while  i < n :
        S = neighbouring_solution(R)
        T_bis = cost_function(S,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c)
        p = (T_bis - T)*1000000
        random = rd.random()
        if random < np.exp(-p/T):
            R = copy(S)
            T = T_bis
        i+= 1
    return (R,T_bis)  

################################################################################################################################################################
########################################################################   Indicators   ########################################################################
################################################################################################################################################################
    


def delay_indicator(R,C,M_time):
    dispo = [0]* len(R)
    delay_by_truck = [0]* len(R)
    clients_name = []
    cumulative_delay = 0
    for x in R:
        if len(x)>1:
            c= find_client_by_name(x[1],C)
            dispo[x[0].name] += M_time[c.name][0]
            if dispo[x[0].name]-c.stop > 0 :
                delay_by_truck[x[0].name] += dispo[x[0].name]-c.stop
                clients_name.append(c.name)
                cumulative_delay += dispo[x[0].name]-c.stop
            for i in range(2,len(x)) :
                c= find_client_by_name(x[i],C)
                dispo[x[0].name] += M_time[c.name][x[i-1]]
                if dispo[x[0].name]-c.stop > 0 :
                    clients_name.append(c.name)
                    delay_by_truck[x[0].name] += dispo[x[0].name]-c.stop
                    cumulative_delay +=dispo[x[0].name]-c.stop
    return delay_by_truck, clients_name, cumulative_delay


def advance_indicator(R,C,M_time):
    dispo = [0]* len(R)
    advance_by_truck = [0] * len(R)
    cumulative_advance = 0
    for x in R:
        if len(x)>1:
            c= find_client_by_name(x[1],C)
            dispo[x[0].name] += M_time[c.name][0]
            if c.start - dispo[x[0].name] > 0 :
                advance_by_truck[x[0].name] += c.start - dispo[x[0].name]
                cumulative_advance += c.start - dispo[x[0].name]
            for i in range(2,len(x)) :
                c= find_client_by_name(x[i],C)
                dispo[x[0].name] += M_time[c.name][x[i-1]]
                if dispo[x[0].name]-c.stop > 0 :
                    advance_by_truck[x[0].name] +=c.start - dispo[x[0].name]
                    cumulative_advance +=c.start - dispo[x[0].name]
    return advance_by_truck, cumulative_advance


################################################################################################################################################################
########################################################################   Simulation   ########################################################################
################################################################################################################################################################

L = algo_RS(C,time_matrix,100_000)
R = L[0]
T = L[1]
d = delay_indicator(R,C,time_matrix)
t = advance_indicator(R,C,time_matrix)


################################################################################################################################################################
#########################################################################   Display   ##########################################################################
################################################################################################################################################################

closing_tour(R)
liste_clients = [[0, 0, 0, 0, 0, 40],
 [1, -2, 4, 1, 7, 12],
 [2, 4, 4, 1, 10, 15],
 [3, -4, 3, 3, 16, 18],
 [4, -3, 3, 4, 10, 13],
 [5, 1, 2, 2, 0, 5],
 [6, 3, 2, 4, 5, 10],
 [7, -1, 1, 8, 0, 4],
 [8, 2, 1, 8, 5, 10],
 [9, 1, -1, 1, 0, 3],
 [10, 4, -1, 2, 10, 16],
 [11, -3, -2, 1, 10, 15],
 [12, -2, -2, 2, 0, 5],
 [13, -1, -3, 14, 5, 10],
 [14, 2, -3, 4, 7, 8],
 [15, -4, -4, 8, 10, 15],
 [16, 3, -4, 8, 11, 15]]

y = [0]
z = [0]
X=[]
Y=[]
infos = ["Entrepot"]
for i in range(1,len(liste_clients)):

    X.append(liste_clients[i][1])
    Y.append(liste_clients[i][2])
    txt = "Client n°"+str(liste_clients[i][0])+"\n"+str(liste_clients[i][3])+" - ["+str(liste_clients[i][4])+", "+str(liste_clients[i][5])+"]"
    infos.append(txt)


n = infos



plt.figure(figsize=(20,20))
plt.grid()
for i in range(len(R)):
    X2=[0]
    Y2=[0]
    for client in R[i][1:]:
        X2.append(liste_clients[client][1])
        Y2.append(liste_clients[client][2])
        
    X.append(X2)
    Y.append(Y2)
    plt.plot(X2,Y2)
    
    
#fig, ax = plt.subplots()
#ax.scatter(X, Y)
#ax.plot(X, Y)

for i, txt in enumerate(n):
    print(txt)
    print(i)
    plt.annotate(txt, (liste_clients[i][1], liste_clients[i][2]))


plt.savefig('chemin.png', format='png')


### graphes intéressants
"""
L_d = []
L_d_all = []
for d in range(10,21):
    count = 0
    count_all = 0
    for i in range(100):
        L = algo_RS(C,time_matrix, 100_000, weight_d=d)
        R = L[0]
        T = L[1]
        delay = delay_indicator(R, C, time_matrix)
        if delay!= ([0, 0, 0, 0, 0, 0], [], 0):
            count+=1
            count_all += delay[-1]
    L_d.append(count)
    L_d_all.append(count_all)
plt.figure()
plt.plot([i for i in range(10,21)], L_c)
plt.ylabel('number of delays')
plt.xlabel('weight')
plt.title('effect of the weight on the number of delay')
plt.show()
"""