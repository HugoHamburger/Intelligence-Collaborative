# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:18:36 2021

@author: Julien
"""


import matplotlib.pyplot as plt
import random as rd



###############################################################################
######################   Enregistrement des constantes   ######################
###############################################################################

#Équipements        
n_trucks = 4 #Nombre de camions
truck_capacity = 25 #Capacité unitaire pouvant être emmenée par chaque camion

#Paramètres AGs
nb_pop = 200 #taille de notre population
nb_generations = 5000 #nombre de générations étudiées par l'AGs
elitism = True #Volonté de ne sélectionner que les meilleurs éléments
best_pop = 20 #Taille de la population élite : les N-meilleurs membres de la population
mutation_rate = 0.3 #probabilité de mutation lors du passage à la génération suivante

#Pénalités
time_penalty = 100 #Pénalité si un client n'est pas livré dans les temps
quantity_penalty = 100 #Pénalité si on ne livre pas intégralement un client
truck_cost = 0 #Coût d'utilisation d'un camion

###############################################################################
########################   Paramètres du problème      ########################
###############################################################################

#On définit d'abord la matrice des temps depuis https://developers.google.com/optimization/routing/vrptw?fbclid=IwAR1Cy40SLDbDJqmIlqQclEQVtuHwwQdPEJ7G0ufurS0fYV-KIemjkwc3gDM
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


class Client:
    def __init__(self, name, x, y, quantity, start, stop):
        self.name = name
        self.x = x
        self.y = y
        self.quantity = quantity
        self.start = start
        self.stop = stop
        self.delivered = False


#On définit les clients à livrer selon la syntaxe suivante : 
#Client(nom du client, coordonnée x dans le plan, coordonnée y dans le plan,
#       quantité à livrer, début de la fenêtre de livraison, fin de la fenêtre de livraison)

list_clients = []
list_clients.append(Client(0,0,0,0,0,1000))
list_clients.append(Client(1,-2,4,1,7,12))
list_clients.append(Client(2,4,4,1,10,15))
list_clients.append(Client(3,-4,3,3,16,18))
list_clients.append(Client(4,-3,3,4,10,13))
list_clients.append(Client(5,1,2,2,0,5))
list_clients.append(Client(6,3,2,4,5,10))
list_clients.append(Client(7,-1,1,8 ,0,4))
list_clients.append(Client(8,2,1,8,5,10))
list_clients.append(Client(9,1,-1,1,0,3))
list_clients.append(Client(10,4,-1,2,10,16))
list_clients.append(Client(11,-3,-2,1,10,15))
list_clients.append(Client(12,-2,-2,2,0,5))
list_clients.append(Client(13,-1,-3,14,5,10))
list_clients.append(Client(14,2,-3,4,7,8))
list_clients.append(Client(15,-4,-4,8,10,15))
list_clients.append(Client(16,3,-4,8,11,15))

#On récupère ici une liste tronquée de nos clients avec les paramètres suivants :
#   quantité à livrer, début de la fenêtre de livraison, fin de la fenêtre de livraison

clients= [[list_clients[i].quantity, list_clients[i].start, list_clients[i].stop] for i in range(len(list_clients))]


###############################################################################
##########################   Fonctions d'évaluation  ##########################
###############################################################################


def track_evaluation(truck_track):
    t3=0
    trucks_time = []
    for track in truck_track:
        t=0
        q=truck_capacity
        if len(track) >2 :
            t3+= truck_cost #Si le camion est utilisé, ajout du coût d'utilisation dans le score du circuit
        for i in range(1,len(track)):
            
            # Evaluation fenêtre
            t2= time_matrix[track[i-1]][track[i]]
            if t+t2 <= clients[track[i]][2]: #Si arrivé du livreur avant la fin de la fenêtre
                t+=t2
                t+=+max(0,clients[track[i]][1]-t)  #ajout d'un temps d'attente si le livreur arrive avant le début de la fenêtre
            else :
                t3+= (t+t2 - clients[track[i]][2])*time_penalty #ajout d'une pénalité si une fenêtre n'est pas respectée
                t+=t2
                               
            # Evaluation quantité            
            if q >= clients[track[i]][0]:
                q-=clients[track[i]][0]
            else : 
                t3+=(clients[track[i]][0]-q)*quantity_penalty   #ajout d'une pénalité si une quantité n'est pas respectée
                q=0                
        trucks_time.append(t)
        t3+=t
    return t3,trucks_time
    
                
def truck_track_constructor(member):
    track=member[0]   


    cgt=[0]
    truck_track=[]
    
    for j in range(1,len(track)):
        if track[j] <= 0:           
            cgt.append(j)
            truck_track.append([0] + track[cgt[-2]+1:cgt[-1]] + [0])
            
    cgt.append(j)
    truck_track.append([0] + track[cgt[-2]+1:cgt[-1]+1] + [0])  

    return truck_track

def population_evaluation(member): 
    t3,trucks_time=track_evaluation(truck_track_constructor(member))        
    member[1]=t3
    return member

#Nous avons reprogrammé ci-dessous un tri fusion
def merge(left,right): 
    result = []
    index_left, index_right = 0, 0
    while index_left < len(left) and index_right < len(right):        
        if left[index_left][1] <= right[index_right][1]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
    if left:
        result.extend(left[index_left:])
    if right:
        result.extend(right[index_right:])
    return result
 
def merge_sort(m):
    if len(m) <= 1:
        return m
    middle = len(m) // 2
    left = m[:middle]
    right = m[middle:]
    left = merge_sort(left)
    right = merge_sort(right)
    return list(merge(left, right))

###############################################################################
##########################   fonctions d'évolution   ##########################
###############################################################################



rand=rd.random

Phenon=[i for i in range(len(clients))] +[-i for i in range(1,n_trucks)]
back_to_depot=[-i for i in range(n_trucks)]

nv=len(Phenon)

def generate():
    a=Phenon[1:]
    rd.shuffle(a)
    return [0]+a

#def taille_pop2():
#    pop = []
#    for i in range(taille_pop):
#        circuit2=generate()
#        distance=0
#        for j in range(1,nv):
#            distance+=dist[circuit2[j-1],circuit2[j]]
#        pop.append([circuit2,distance])
#    return pop
        
    

def crossover(parent1,parent2):
    """
    Utilisation d'un two points crossover
    Les points de départ et de fin sont choisis aléatoirement
    """
    
    parent1 = parent1[1:]
    parent2 = parent2[1:]
    nv=len(parent1)
    # On récupère une partie des critères du premier parent
    start_point=rd.randint(0,nv-1)
    end_point=rd.randint(0,nv-1)
    
    if start_point > end_point :
        start_point,end_point=end_point,start_point
    
    child=["False" for i in range(nv)]
    heritage_parent1=parent1[start_point:end_point+1]
    child[start_point:end_point+1]=heritage_parent1
    
    
    # On récupère ensuite le maximum de critères possibles du parent2 tout en conservant un enfant sans doublons
    m1=[]
    for i in range(nv):
        if i > end_point or i < start_point: 
            criteria=parent2[i]
            if criteria not in heritage_parent1:
                child[i]=criteria
            else:
                m1.append(i)
                
                
    # On complète par les critères manquants            
    m2=[]
#    print(enfant)
    for i in Phenon[1:]:

        
        if i not in child:
            m2.append(i)

    rd.shuffle(m2)
#    print(patrimoine_parent1)
#    print(parent2)
#    print(m2)
#    print(m1)
#    print(enfant)
    for i in range(len(m1)):
        child[m1[i]]=m2[i]
    
    #mutation

    
    if rd.random() < mutation_rate:
        a=rd.randint(0,len(child)-1)
        b=rd.randint(0,len(child)-1)   
        child[a], child[b] = child[b], child[a]
            
    return [0]+child 

def uniform_cross(p1,p2):
    
    p1 = p1[1:]
    p2 = p2[1:]
    
    child1 = []
    child2 = []
    
    for i in range(len(p1)):
        if(p1[i] in child1 and p2[i] in child1):
            child2.append(p1[i])
            child2.append(p2[i])
        elif(p1[i] in child2 and p2[i] in child2):
            child1.append(p1[i])
            child1.append(p2[i])
        elif(p1[i] in child2 or p2[i] in child1):
            child1.append(p1[i])
            child2.append(p2[i])
            
        elif(p1[i] in child1 or p2[i] in child2):
            child1.append(p2[i])
            child2.append(p1[i])
            
        else:
            a = rd.randint(1,2)
            if(a == 1):
                child1.append(p1[i])
                child2.append(p2[i])
            else:
                child1.append(p2[i])
                child2.append(p1[i])
    child1=mutation(child1, mutation_rate)
    child2=mutation(child2, mutation_rate)
    
    return([0]+child1, [0] + child2)

def mutation(child, mutation_rate):
    if(rd.random()-mutation_rate < 0):
        point1 = rd.randint(0, len(child)-1)
        point2 = rd.randint(0, len(child)-1)
        child[point1], child[point2] = child[point2], child[point1]
    return(child)
        
def init_pop(n):
    pop=[]
    for i in range(n):
        track=generate()
        pop.append(population_evaluation([track,0]))
    pop=merge_sort(pop)
    return pop



def next_gen(population):
  
    new_gen=[]
    
    if elitism == True:
        elite=population[:best_pop]
    else:
        elite=[]
        

    nb_child=(nb_pop//2)
    
    couples=[i for i in range(nb_child*2)]

    rd.shuffle(couples)


    for i in range(nb_child//2):
        
        # child=crossover(population[couples[i]][0],population[couples[nb_pop//2+i]][0])        
        # new_gen.append(population_evaluation([child,0]))
        
        child1=uniform_cross(population[couples[i]][0],population[couples[nb_pop//2+i]][0])[0]
        child2=uniform_cross(population[couples[i]][0],population[couples[nb_pop//2+i]][0])[1]
        new_gen.append(population_evaluation([child1,0]))
        new_gen.append(population_evaluation([child2,0]))
        
    new_gen=new_gen+elite
    
    new_gen= new_gen + init_pop(nb_pop-len(new_gen))
    
    population=merge_sort(new_gen[:])
    
    return(population)
    
    

###############################################################################
#######################   génération de la population  ########################
###############################################################################
population=init_pop(nb_pop)

X=[1]
Y=[population[0][1]]

for i in range(2,nb_generations+1):
    population = next_gen(population)
    X.append(i)
    Y.append(population[0][1])
plt.plot(X,Y)
plt.xlabel('génération')
plt.ylabel("Score")
plt.title("Score : "+str(population[0][1])+" | "+str(n_trucks)+" camions d'une capacité de " + str(truck_capacity)+"\nTaux de mutation : "+str(mutation_rate)+" | Taile de la population : "+str(nb_pop) )    
#    if i%100 == 0: 
#        print(population[0][1])
    
plt.savefig(str(population[0][1])+'_distance.png', format='png')    


track=truck_track_constructor(population[0])

 #on genere une carte représentant notre solution
y = [0]
z = [0]
infos = ["Entrepot"]
for i in range(1,len(list_clients)):

    X.append(list_clients[i].x)
    Y.append(list_clients[i].y)
    txt = "Client n°"+str(list_clients[i].name)+"\n"+str(list_clients[i].quantity)+" - ["+str(list_clients[i].start)+", "+str(list_clients[i].stop)+"]"
    infos.append(txt)


n = infos

X=[]
Y=[]

plt.figure(figsize=(20,20))
plt.grid()
for i in range(n_trucks):
    X2=[]
    Y2=[]
    for j in track[i]:
        X2.append(list_clients[j].x)
        Y2.append(list_clients[j].y)
        
    X.append(X2)
    Y.append(Y2)
    plt.plot(X2,Y2)
    
    

for i, txt in enumerate(n):

    plt.annotate(txt, (list_clients[i].x, list_clients[i].y))
plt.title("Score : "+str(population[0][1])+" | "+str(n_trucks)+" camions d'une capacité de " + str(truck_capacity) + "\nOrdonnancement : " + str(population[0][0]))
plt.savefig(str(population[0][1])+'_chemin.png', format='png') 
