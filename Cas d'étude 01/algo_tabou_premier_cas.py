import random as rd

class Depot:
    
    def __init__(self, num_truck):
        self.x = 0
        self.y = 0
        self.num_truck = num_truck
        self.T = []
        for i in range (num_truck):
            truck = Truck (i, 15)
            self.T.append(truck)

class Truck:
    
    def __init__(self, name, quantity_max):
        self.name = name
        self.quantity_max = quantity_max
        self.remaining_quantity=quantity_max
        self.P = [0]
        self.cost = 0
        self.time = 0
            
   
    def calculate_cost(self, data):
        demande = 0
        weight_K = 10000
        weight_q = 10000
        weight_t = 3
        weight_d = 13
        weight_c = 10   
        for m in range (len(self.P)):
            demande += data['demands'][m]
        self.remaining_quantity-=demande
        if(self.remaining_quantity<0):
            self.cost+=self.remaining_quantity*(-1)*weight_q
        for i in range (len(self.P)-1):
            self.cost += distance (self.P[i],self.P[i+1],data)*weight_c
            self.time += data['time_matrix'][self.P[i]][self.P[i+1]]
            if self.time <= data['time_windows'][self.P[i+1]][0]:
                self.cost = data['time_windows'][self.P[i+1]][0]*weight_t
            if self.time >= data['time_windows'][self.P[i+1]][1]:
                self.cost += self.time - data['time_windows'][self.P[i+1]][1]*weight_d
        self.cost += weight_K
        return self.cost
                
import numpy as np

def distance(C1, C2, data):
    return(np.sqrt((data['coordinates'][C1][1]-data['coordinates'][C2][1])**2 + (data['coordinates'][C1][0] - data['coordinates'][C2][0])**2))
    
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = [
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
    data['time_windows'] = [
        (0, 5),  # depot
        (7, 12),  # 1
        (10, 15),  # 2
        (16, 18),  # 3
        (10, 13),  # 4
        (0, 5),  # 5
        (5, 10),  # 6
        (0, 4),  # 7
        (5, 10),  # 8
        (0, 3),  # 9
        (10, 16),  # 10
        (10, 15),  # 11
        (0, 5),  # 12
        (5, 10),  # 13
        (7, 8),  # 14
        (10, 15),  # 15
        (11, 15),  # 16
    ]
    data['coordinates'] = [
        (0, 0),  # depot
        (-2, 4),  # 1
        (4, 4),  # 2
        (-4, 3),  # 3
        (-3, 3),  # 4
        (1, 2),  # 5
        (3, 2),  # 6
        (-1, 1),  # 7
        (2, 1),  # 8
        (1, -1),  # 9
        (4, -1),  # 10
        (-3, -2),  # 11
        (-2, -2),  # 12
        (-1, -3),  # 13
        (2, -3),  # 14
        (-4, -4),  # 15
        (3, -4),  # 16
    ]
    data['demands'] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    return data

def solution_initiale (num_truck):
    S=[]
    depot = Depot(num_truck)
    data = create_data_model()
    clients = [i for i in range (1,len(data['time_windows']))]
    while len (clients) != 0:
        for k in range (len (depot.T)):
            if len (clients) != 0:
                j = rd.randint (0,len(clients)-1)
                depot.T[k].P.append(clients[j])
                clients.remove(clients[j])

    for k in range (len(depot.T)):
        depot.T[k].P.append(0)
    for i in range (num_truck):
        S.append(depot.T[i].P)
    return (total_cost(depot,data),S)
    
                    
def total_cost (depot, data):
    total_cost = 0
    for i in range (depot.num_truck):
        (depot.T[i]).calculate_cost(data)
        total_cost += depot.T[i].cost
    return total_cost

def simple_permut (P, old, new):
    temp = P[old]
    P.remove(P[old])
    P.insert(new,temp)
    return P

def voisinage_simple (parcours, num_truck, data):
    copie_parcours=parcours+[]
    vs=[parcours]
    truck= Truck(num_truck+1,15)
    truck.P = parcours
    cost=[truck.calculate_cost(data)]
    for i in range (1,len(copie_parcours)-1):
        for j in range (1,len(copie_parcours)-1):
            if i < j:
                truck = Truck(num_truck+2,15)
                copie_parcours=simple_permut(copie_parcours,i,j)
                vs.append(copie_parcours)
                truck.P = copie_parcours
                cost.append(truck.calculate_cost(data))
                copie_parcours=parcours+[]
    mini=cost[0]
    index_opti=0
    for k in range (1,len(cost)):
        if cost[k]<mini:
            mini=cost[k]
            index_opti=k
    return vs[index_opti]

def exchange (P1, P2, old_pos, new_pos):
    copy_P1=P1+[]
    copy_P2=P2+[]
    temp1 = copy_P1[old_pos]
    temp2 = copy_P2[new_pos]
    copy_P1.remove(copy_P1[old_pos])
    copy_P2.remove(copy_P2[new_pos])
    copy_P1.insert(old_pos,temp2)
    copy_P2.insert(new_pos,temp1)
    return copy_P1,copy_P2
    
def voisinage_complexe (ens_parcours, num_truck, data,tabou):
    copie_ens_parcours=ens_parcours+[]
    vc=[]
    list_tot_cost=[]
    for i in range (len(ens_parcours)):
        for j in range (len(ens_parcours)):
            if i != j:
                for k in range (1,len(copie_ens_parcours[i])-1):
                    for l in range (1,len(copie_ens_parcours[j])-1):
                        (a,b) = transfert(copie_ens_parcours[i],copie_ens_parcours[j],k,l)
                        ens_parcours_to_add=[]
                        depot0 = Depot (num_truck)
                        for w in range (len(copie_ens_parcours)):
                            if w == i :
                                ens_parcours_to_add.append(a)
                                depot0.T[w].P=a
                            elif w == j :
                                ens_parcours_to_add.append(b)
                                depot0.T[w].P=b
                            else:
                                ens_parcours_to_add.append(copie_ens_parcours[w])
                                depot0.T[w].P=copie_ens_parcours[w]
                        vc.append(ens_parcours_to_add)
                        list_tot_cost.append(total_cost(depot0,data))
    return (list_tot_cost, vc)                                      
                    
def best_possible_exchange (P1,  P2, data):
    pos=[(P1,P2)]
    truck1 = Truck(1,15)
    truck2 = Truck(2,15)
    truck1.P = P1
    truck2.P = P2
    cost1 = truck1.calculate_cost(data)
    cost2 = truck2.calculate_cost(data)
    cost = cost1+cost2
    index_min = 0
    for i in range (1,len(P1)-1):
        demand1 = 0
        demand2 = 0
        for m in range (len(P1)):
            demand1 += data['demands'][m]
        for j in range (len(P2)):
            demand2 += data['demands'][j]
        for k in range (1,len(P2)-1):
            if (demand1 - data['demands'][i] + data['demands'][k] <= 15) and (demand2 - data['demands'][k] + data['demands'][i] <= 15):
                pos.append(exchange(P1,P2,i,k))
    for l in range (len(pos)):
        truck1 = Truck(1,15)
        truck2 = Truck(2,15)
        truck1.P = pos[l][0]
        truck2.P = pos[l][1]
        cost1 = truck1.calculate_cost(data)
        cost2 = truck2.calculate_cost(data)
        if cost1 + cost2 < cost :
            cost = cost1 + cost2
            index_min = l
    return pos[index_min]
            
def best_voisinage (num_truck, data,sol_actuelle,tabou,best_saved_cost):
    all_ens_vs=[]
    all_cost=[]
    for i in range (len(sol_actuelle)):
        ens_vs=[]
        cost=0
        for j in range (len(sol_actuelle)):
            if i==j:
                truck = Truck(num_truck+1, 15)
                truck.P = voisinage_simple(sol_actuelle[i], num_truck, data)
                ens_vs.append(truck.P)
                cost+= truck.calculate_cost(data)
            else:
                truck = Truck(num_truck+1, 15)
                truck.P = sol_actuelle[j]
                cost+= truck.calculate_cost(data)
                ens_vs.append(sol_actuelle[j])
        all_ens_vs.append(ens_vs)
        all_cost.append(cost)
    
    (cost_vc,vc)=voisinage_complexe(sol_actuelle,num_truck,data,tabou)
    all_ens_vs = all_ens_vs+vc
    all_cost = all_cost+cost_vc
    (all_ens_vs,all_cost) = sol_filter(data,sol_actuelle,tabou,all_ens_vs,all_cost,best_saved_cost)
    if(len(all_ens_vs)>0):
        min_cost=all_cost[0]
        index_best_cost=0
        for i in range (len(all_cost)):
            if (all_cost[i] < min_cost):
                min_cost = all_cost[i]
                index_best_cost = i
        return (all_cost[index_best_cost],all_ens_vs[index_best_cost])
    else:
        return (0,[])
    
def sol_filter(data,sol_actuelle,tabou,all_ens_vs,all_cost,best_saved_cost):
    new_sol =[]
    new_sol_cost=[]
    for i in range (len(all_ens_vs)):
        if((sol_actuelle,all_ens_vs[i])not in tabou or all_cost[i]<best_saved_cost):
            new_sol.append(all_ens_vs[i])
            new_sol_cost.append(all_cost[i])
    return (new_sol,new_sol_cost)
    
def transfert (P1, P2, old_pos, new_pos):
    copy_P1=P1+[]
    copy_P2=P2+[]
    temp = copy_P1[old_pos]
    copy_P1.remove(copy_P1[old_pos])
    copy_P2.insert(new_pos,temp)
    return (copy_P1,copy_P2)

def best_possible_transfert (P1,  P2, data):
    pos=[(P1,P2)]
    truck1 = Truck(1,15)
    truck2 = Truck(2,15)
    truck1.P = P1
    truck2.P = P2
    cost1 = truck1.calculate_cost(data)
    cost2 = truck2.calculate_cost(data)
    cost = cost1+cost2
    index_min = 0
    for i in range (1,len(P1)-1):
        for k in range (1,len(P2)-1):
            pos.append(transfert(P1,P2,i,k))        
    for l in range (len(pos)):
        truck1 = Truck(1,15)
        truck2 = Truck(2,15)
        truck1.P = pos[l][0]
        truck2.P = pos[l][1]
        cost1 = truck1.calculate_cost(data)
        cost2 = truck2.calculate_cost(data)
        if cost1 + cost2 < cost :
            cost = cost1 + cost2
            index_min = l
    return pos[index_min]

def algo_tabou (nb_iter, max_tabou_size,number_trucks):
    curr_cost =0
    curr_sol=[]
    tabou =[]
    data = create_data_model()
    (curr_cost,curr_sol) = solution_initiale(number_trucks)
    best_sol=curr_sol
    best_cost=curr_cost
    for i in range(nb_iter):
        (curr_cost,best_curr_neigh) = best_voisinage(number_trucks,data,curr_sol,tabou,best_cost)
        if(curr_cost==0):
            break
        if(len(tabou)>=max_tabou_size):
            tabou.pop(0)
        tabou.append((best_curr_neigh,curr_sol))
        curr_sol=best_curr_neigh
        if(curr_cost<best_cost):
            best_cost=curr_cost
            best_sol=curr_sol
    return best_sol,best_cost

for i in range (1,31):
    print(algo_tabou(i,1000,20))
        
        

    
    

    
            
        
                
    
    
    
    
    
    

        