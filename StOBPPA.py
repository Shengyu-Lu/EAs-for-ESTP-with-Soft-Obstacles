import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, Point, LineString, Polygon, MultiPolygon
from scipy.spatial import distance as ds
from scipy.spatial import Delaunay 
import random
import math


class Graph:
 
    def __init__(self, vertices):
        self.V = vertices  
        self.graph = []  
       
    # adding an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    # finding set of an element i
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # union of two sets
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
       
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
    def Kruskal_MST(self):
 
        result = []  
         
        i = 0
        e = 0
 
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        while e < self.V - 1 and i < len(self.graph):

            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
 
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
             
        minimumCost = 0
        # print ("Edges in MST")
        for u, v, weight in result:
            minimumCost += weight
            # print("%d - %d = %d" % (u, v, weight))
        # print("MST cost" , minimumCost)
        return minimumCost, result



def extract_terminals(instance):
    # extract terminal coords from csv file -> global var terminals[]
    terminals_csv =  'TestCases/Type' + str(TYPE) + '/terminals' + instance + '.csv'
    terminals_df = pd.read_csv(terminals_csv,float_precision = 'round_trip')
    
    terminals_df['Xcoord'] = terminals_df['Xcoord'].astype(float)
    terminals_df['Ycoord'] = terminals_df['Ycoord'].astype(float)

    terminals_num = len(terminals_df)
    global terminals

    for i in range(terminals_num):
        
        terminals.append((float(format(terminals_df['Xcoord'][i],'.3f')),float(format(terminals_df['Ycoord'][i],'.3f')))) 

def extract_obstacles(instance):
    # extract cornerpoint coords from csv file -> global var cornerpoints[]
    global cornerpoints
    
    obstacles_temp = []
    
    obstacles_csv = 'TestCases/Type' + str(TYPE) + '/obstacles' + instance + '.csv'
    obstacles_df = pd.read_csv(obstacles_csv, header = None, float_precision = 'round_trip')

    row_count = obstacles_df.shape[0]
    # count the number of obstacles
    obstacle_count = 0



    # extract penalty values and obstacle corner points from df
    for row in range(row_count):
        
        if str(obstacles_df.iloc[row,][0]) != 'nan' and str(obstacles_df.iloc[row,][1]) == 'nan':
            tem = []
            penalty.append(obstacles_df.iloc[row,][0])

        if str(obstacles_df.iloc[row,][0]) != 'nan' and str(obstacles_df.iloc[row,][1]) != 'nan':
            tem.append((float(format(obstacles_df.iloc[row,][0],'.3f')),float(format(obstacles_df.iloc[row,][1],'.3f'))))

        if str(obstacles_df.iloc[row,][0]) == 'nan' and str(obstacles_df.iloc[row,][1]) == 'nan':
            obstacle_count += 1
            if tem != []:
                obstacles_temp.append(tem)
        
        if row == row_count - 1 :

            obstacles_temp.append(tem)
    
    # print(obstacles_temp)
    
    for i in range(len(obstacles_temp)):

        if penalty[i] != 999.0:

            soft_obstacles.append(Polygon(obstacles_temp[i]))
            
        else:

            solid_obstacles.append(Polygon(obstacles_temp[i]))
    
    for obstacle in obstacles_temp:
        
        obstacles.append(Polygon(obstacle))
    
    for obstacle in obstacles:
        
        for coord in list(obstacle.exterior.coords)[:-1]:

            cornerpoints.append(coord)


def Kruskal_Mst_No_Obs(points):
  
    tri = Delaunay(points)
    list_of_edges = []

    for triangle in tri.simplices:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

    array_of_edges_index = np.unique(list_of_edges, axis=0) # remove duplicates
    # array_of_edges = []

    list_of_lengths = []
    g = Graph(len(tri.points))
    graph_edge = []
    for p1,p2 in array_of_edges_index:
        edge = LineString([tri.points[p1],tri.points[p2]])
        # array_of_edges.append([[tri.points[p1],tri.points[p2]]])
        length = edge.length
        
        list_of_lengths.append(length)
        graph_edge.append((p1,p2,length))

    for i in range(len(graph_edge)):
        g.addEdge(graph_edge[i][0],graph_edge[i][1],graph_edge[i][2])

    total_distance,kruskal_results = g.Kruskal_MST()

   
    return total_distance
            
            
            
def init_instance(instance):
#  given instance number in str, e.g "1"
    global max_distance
    global min_distance
    extract_terminals(instance)
    min_distance = 0.74309*Kruskal_Mst_No_Obs(terminals)
    extract_obstacles(instance)

    global border_edges
    
    for obstacle in obstacles:
        for i in range(len(list(obstacle.exterior.coords))-1):
            border_edges.append(LineString([list(obstacle.exterior.coords)[i],list(obstacle.exterior.coords)[i+1]]))
            
            border_edges.append(LineString([list(obstacle.exterior.coords)[i+1],list(obstacle.exterior.coords)[i]]))
    #  calculate the max distance
    # individual_max = []
    # chromosome1 = []
    # chromosome2 = []
    
    # chromosome1 = np.array([])
    # chromosome2 = np.ones((len(cornerpoints),), dtype=int)
    # individual_max.append(chromosome1)
    # individual_max.append(chromosome2)
    # max_distance = cal_fitness(individual_max)
    max_distance = min_distance + 2
    
    

def init_population():
    global population, fitness, distance,max_distance

    
   
    population = []
    fitness = []
    distance = []
    
    possible_steiner_list = []
  
#   add individual(s) of type_1
    
    chromosome1 = []
    chromosome2 = []
    terminals_and_cornerpoints = np.array(terminals + cornerpoints)
#     print(terminals_and_cornerpoints)
    tri = Delaunay(terminals_and_cornerpoints)
    for triangle in tri.simplices:
        x = 1/3 * (terminals_and_cornerpoints[triangle[0]][0] + terminals_and_cornerpoints[triangle[1]][0] + terminals_and_cornerpoints[triangle[2]][0])
        y = 1/3 * (terminals_and_cornerpoints[triangle[0]][1] + terminals_and_cornerpoints[triangle[1]][1] + terminals_and_cornerpoints[triangle[2]][1])
        steiner = (x,y)
        # check if steiner inside of a solid obstacle
        flag = 0
        for solid_obstalce in solid_obstacles:
            if Point(steiner).within(solid_obstalce):
                flag = 1
        if flag == 0:
            possible_steiner_list.append(steiner)
    cornerindex_list = []
    
    
    for i in range(len(cornerpoints)):
        cornerindex_list.append(i)


    for i in range(30):
        
        indi = []
        num = np.random.randint(len(terminals) - 1) + 1
        chromosome1 = np.array(random.sample(possible_steiner_list,num))
        # chromosome1 = np.array(random.choices(possible_steiner_list,k=num))
        if i % 2 == 0:
            chromosome2 = np.zeros((len(cornerpoints),), dtype=int)
            corner_num = np.random.randint(len(cornerpoints) - 1) + 1
            flipindex = np.array(random.choices(cornerindex_list,k=corner_num))
            for j in flipindex:
                chromosome2[j] = 1
        else:
            chromosome2 = np.zeros((len(cornerpoints),), dtype=int)
        indi.append(chromosome1)
        indi.append(chromosome2)
        population.append(indi)
        mstweight = cal_fitness(indi)
        distance.append(mstweight)
        fitness.append(new_fitness(mstweight))

    # max_distance = max(distance)
  
    
#   show the mst of individual_type1
#     plt.triplot(terminals_and_cornerpoints[:,0], terminals_and_cornerpoints[:,1], tri.simplices)
#     plt.plot(terminals_and_cornerpoints[:,0], terminals_and_cornerpoints[:,1], 'o')
#     plt.show()

  





        
def cal_fitness(individual):
#  calculate fitness value of an individual
#  change global var steinerpoints -> steinerpoints of current individual
#  change global var selected_cornerpoints -> selected cornerpoints of current individual
#  change global var all_points -> all points of current individual = terminals + selectedcornerpoints + steinerpoints

    global steinerpoints
    global selected_cornerpoints
    global all_points
    
# extract two parts of the chromosome
    chromosome1 = individual[0]
    chromosome2 = individual[1]
    
#   extract steinerpoints from chromosome1
    steinerpoints = chromosome1.tolist()

    
#     extract selected corner points from chromosome2
    temp = []
    for index in range(len(chromosome2)):

        if chromosome2[index] == 1:

            temp.append(cornerpoints[index])  
            
    selected_cornerpoints = temp

#   calculate all_points of this individual    
    all_points = terminals + selected_cornerpoints + steinerpoints

    # return Prim_Mst(all_points) #will return total-distance of mst of current individual
    return Kruskal_Mst(all_points)

def Kruskal_Mst(points):
    global mst_edgepoints
    global fitness_cal_time

    mst_edgepoints = []
    tri = Delaunay(points)
    list_of_edges = []

    for triangle in tri.simplices:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

    array_of_edges_index = np.unique(list_of_edges, axis=0) # remove duplicates
    # array_of_edges = []

    list_of_lengths = []
    g = Graph(len(tri.points))
    graph_edge = []
    for p1,p2 in array_of_edges_index:
        edge = LineString([tri.points[p1],tri.points[p2]])
        # array_of_edges.append([[tri.points[p1],tri.points[p2]]])
        length = edge.length
        for p in range(len(penalty)):
            intersection = edge.intersection(obstacles[p])
            if intersection not in border_edges:
                length += intersection.length*(penalty[p]-1)
        list_of_lengths.append(length)
        graph_edge.append((p1,p2,length))

    for i in range(len(graph_edge)):
        g.addEdge(graph_edge[i][0],graph_edge[i][1],graph_edge[i][2])

    total_distance,kruskal_results = g.Kruskal_MST()

    for i in kruskal_results:
        mst_edgepoints.append((tri.points[i[0]].tolist(),tri.points[i[1]].tolist()))
    fitness_cal_time += 1
    return total_distance

def less_first(a, b):
    return [a,b] if a < b else [b,a]

def new_fitness(old_fitness):
    if old_fitness >= max_distance:
        fitness = 0.01
    elif old_fitness <= min_distance:
        fitness = 0.99
    else:

        objective = ((old_fitness - min_distance) / (max_distance - min_distance))
        if objective > 1:
            objective = 1.0
        fitness = 0.5*(np.tanh(8*(1-objective)-4)+1)
    return fitness

def add_steiner(individual):
    global angles_less_than_120
    angles_less_than_120 = []
    cal_fitness(individual)
    reverse_mst_edgepoints = []
    for i in range(len(mst_edgepoints)):    
        temp = mst_edgepoints[i]
        temp1 = []
        temp1.append(temp[1])
        temp1.append(temp[0])
        reverse_mst_edgepoints.append(temp1)
    double_mst_edgepoints = reverse_mst_edgepoints + mst_edgepoints
    for point in all_points:
        count = 0
        connect_edges = []
        for i in range(len(double_mst_edgepoints)):
            if point == double_mst_edgepoints[i][0]:
                count += 1
                connect_edges.append(double_mst_edgepoints[i])
        if count > 1:
            check_angle_less_than_120(connect_edges)
    
    length = len(angles_less_than_120)
    
    flag = 0
    
    if length > 0:
        random_index = np.random.randint(length)
        
        picked_three_points = [angles_less_than_120[random_index][0][0]]+\
        [angles_less_than_120[random_index][0][1]]+\
        [angles_less_than_120[random_index][1][1]]
        
        new_steiner = ((picked_three_points[0][0]+picked_three_points[1][0]+picked_three_points[2][0])/3.0,(picked_three_points[0][1]+picked_three_points[1][1]+picked_three_points[2][1])/3.0)
        
        
        
        for solid_obstacle in solid_obstacles:
            
            if Point(new_steiner).within(solid_obstacle):
                
                flag = 1

    else:
        new_steiner = np.random.rand(2) 
        
    while flag == 1:
       
        new_steiner = np.random.rand(2)
        
        flag = 0
        
        for solid_obstacle in solid_obstacles:
            
            if Point(new_steiner).within(solid_obstacle):
                
                flag = 1
        
        #todo: 1. check if its in solid obstacle, 
        ##     2. when the size of plane > 1...
    return new_steiner
            
def check_angle_less_than_120(edges):
    global angles_less_than_120
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            degree = get_angle(edges[i],edges[j])
            if degree < 120 :
                angles_less_than_120.append([edges[i],edges[j]])
                
# p0p1p2 corner
def get_angle(edge1,edge2):
# edge1&2 must be edges extracted from edges_check_angle
    p0 = edge1[1]
    p1 = edge1[0]
    p2 = edge2[1]
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    if np.degrees(angle) < 0:

        degree = -np.degrees(angle)
    else:
        degree = np.degrees(angle)
    return degree

def remove_steiner_less_than_2(individual):
    
    cal_fitness(individual)
    reverse_mst_edgepoints = []
    
    
    new_steiner = individual[0].tolist() 
    # print('newsteiner=')
    # print(new_steiner)

    for i in range(len(mst_edgepoints)):    
        temp = mst_edgepoints[i]
        temp1 = []
        temp1.append(temp[1])
        temp1.append(temp[0])
        reverse_mst_edgepoints.append(temp1)
    double_mst_edgepoints = reverse_mst_edgepoints + mst_edgepoints
    
    flag = 0
    # if a steiner <3 is found ,set flag to 1, else, remove a random steiner

    for index in range(len(new_steiner)):
        count = 0
        for i in range(len(double_mst_edgepoints)):
            if new_steiner[index] == double_mst_edgepoints[i][0]:
                count += 1
        if count < 3:
            new_steiner.pop(index)
            flag = 1
            break
    # print('newsteiner=')
    # print(new_steiner)
    # print("flag=")
    # print(flag)   
    if flag == 1:
        new_steiner = np.array(new_steiner)
    else:
        remove_index = np.random.randint(low = 0, high = len(new_steiner), size = 1)[0]
        new_steiner.pop(remove_index)
        new_steiner = np.array(new_steiner)

    return new_steiner





def mutation(indi,fitval,gen):
    
    global population, fitness, distance
    
    gene_total = len(indi[0])+len(indi[1])
    p_flipmove = max(p_flipmove_max*(1-0.5*fitval),p_flipmove_min)
    p_addsteiner = 1-0.5*p_flipmove
    p_removesteiner = p_addsteiner
    move_range = average_terminal_distance * max(1-fitval,0.01)
    
#     20 need to be discussed
    offspring_num = math.ceil(4 * fitval)
   
    gene_mut_num = math.ceil(5 * (1 - fitval))
    
    for i in range(int(offspring_num)):
        
        mu = random.choices(Mutations, weights=(p_flipmove, p_addsteiner, p_removesteiner), k=1)
        if mu == ["FlipMove"]:
            temp1 = indi[0].tolist()
            temp2 = indi[1].tolist()
            offspring = [np.array(temp1),np.array(temp2)]
            gene_list = np.random.randint(low = 0, high = len(offspring[0])+len(offspring[1]), size = gene_mut_num).tolist()
            for j in gene_list:
                gene_index = j
                if gene_index < len(offspring[0]):
                    flag = 0
                    newx = offspring[0][gene_index][0] + random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
                    newy = offspring[0][gene_index][1] + random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
                    for solid_obstacle in solid_obstacles:
                        if Point(newx,newy).within(solid_obstacle):
                            flag = 1
                    if flag == 0:
                        if 0 < newx < 1 and 0 < newy < 1:
                            offspring[0][gene_index][0] = newx
                            offspring[0][gene_index][1] = newy
                else:                   
                    if offspring[1][gene_index-len(offspring[0])] == 0: 
                        offspring[1][gene_index-len(offspring[0])] = 1    
                    else:
                        offspring[1][gene_index-len(offspring[0])] = 0
         
            population.append(offspring)
            mstweight = cal_fitness(offspring)
            distance.append(mstweight)
            fitness.append(new_fitness(mstweight))
           
        
        if mu == ["RemoveSteiner"]:
            
            temp1 = indi[0].tolist()
            temp2 = indi[1].tolist()
            offspring = [np.array(temp1),np.array(temp2)]
            
            if len(offspring[0]) < 1:
                
                mu = ["AddSteiner"]
            
            else:
                
                for i in range(gene_mut_num):
                    
                    offspring[0] = remove_steiner_less_than_2(offspring)

                    if len(offspring[0]) < 1:
                        break

                population.append(offspring)
                mstweight = cal_fitness(offspring)
                distance.append(mstweight)
                fitness.append(new_fitness(mstweight))
                    
        if mu == ["AddSteiner"]:
            
            temp1 = indi[0].tolist()
            temp2 = indi[1].tolist()
            offspring = [np.array(temp1),np.array(temp2)]
            
            new_steiner = add_steiner(offspring)

            temp = offspring[0].tolist()

            temp.append(new_steiner)

            offspring[0] = np.array(temp)
             
            population.append(offspring)
            mstweight = cal_fitness(offspring)
            distance.append(mstweight)
            fitness.append(new_fitness(mstweight))
            
def select_best_30():
    
    global population, fitness, distance
    
    new_population = []
    new_fitness = []
    new_distance = []
    

    count = 0
    val = 100
    for i in range(len(population)):
        
        index = np.argmax(fitness)
        if fitness[index] == val:
            population.pop(index)
            distance.pop(index)
            fitness.pop(index)
        else:
            val = fitness[index]
            count += 1
            new_population.append(population[index])
            new_distance.append(distance[index])
            new_fitness.append(fitness[index])
        
            population.pop(index)
            distance.pop(index)
            fitness.pop(index)
        if count == 30:
            break
        
    population = new_population
    distance = new_distance
    fitness = new_fitness

def elitest_tournament_selection():
    
    global population, fitness, distance

    newpopulation = []
    newfitness = []
    newdistance = []
    real_index_list = []

    # best 1 survive
    max_index = np.argmin(distance)
    newpopulation.append(population[max_index])
    newfitness.append(fitness[max_index])
    newdistance.append(distance[max_index])
    real_index_list.append(max_index)


    while len(newpopulation) != 30:

        random_index = np.random.choice(len(population), size=tournament_size,replace = False)

        fitness_tournament = [distance[index] for index in random_index]

        index_max = np.argmin(fitness_tournament)

        real_index = random_index[index_max]

        if real_index not in real_index_list:
            real_index_list.append(real_index)
            newpopulation.append(population[real_index])
            newfitness.append(fitness[real_index])
            newdistance.append(distance[real_index])

    population = newpopulation
    fitness = newfitness
    distance = newdistance




INF = 999
population_size = 30
tournament_size = 3
p_flipmove_max = 0.99
p_flipmove_min = 0.6
Mutations = ["FlipMove", "AddSteiner", "RemoveSteiner"]
generation_num = 300



fitness_cal_time = 0
fitness_cal_time_list = []
time_list = []


TYPE = 1
instance_list = [str(i) for i in range(1,67)]



# main loop
for repeat in range(1,6):
    for instance in instance_list:

        fitness_cal_time = 0
        fitness_cal_time_list = []

        # instance vars
        terminals = []
        obstacles = []
        soft_obstacles = []
        solid_obstacles = []
        penalty = []
        cornerpoints = []
        border_edges = []
        average_terminal_distance = 0
        max_distance = 0
        ub = 0
        lb = 0
        
        # changeable var per generation
        population = []
        distance =[] 
        fitness = []
        
        # store the results
        mindistance = []
        mostfitness = []
        best_individual = []

        output_file = "PPA/Type" + str(TYPE) + "/instance" + str(instance) + "_trial_" + str(repeat) + ".pkl"
        # output_file = "PPA/ParameterTest/" + str(repeat) + ".pkl"
        # output_file = "PPA/ParameterTest/test.pkl"

        init_instance(instance)

        for i in range(len(terminals)):
            for j in range(len(terminals)):
                if i < j:
                    average_terminal_distance += round(ds.euclidean(terminals[i],terminals[j]),4)
        average_terminal_distance = average_terminal_distance/(len(terminals) * (len(terminals) - 1) * 0.5)
        average_terminal_distance = round(average_terminal_distance,4)
        
        init_population()
        print("population initialization finished！")
        
        count_for_mindist = 0
        count_for_maxdist = 0

        stopping = 0
        current_best = 999.0

        for generation in range(generation_num):
            
            fitness_cal_time_this_generation = fitness_cal_time

            for i in range(30):
            

                # print("individual "+ str(i) + " fitness = %.3f" %fitness[i] + ' distance = %.3f' %distance[i])
                
                mutation(population[i],fitness[i],generation)
            
            
            # dynamic check for lowerbound and upperbound
           
            if max(fitness) < 0.8:
                count_for_mindist += 1
            if min(fitness) > 0.2:
                count_for_maxdist += 1
            if count_for_mindist == 10:
                min_distance += 0.05
                count_for_mindist = 0
            if count_for_maxdist == 10:
                max_distance -= 0.05
                count_for_maxdist = 0 



            if min_distance >= min(distance):
                min_distance = min(distance) - 0.1
                print("mindistance->min(distance)-0.1")


            # if max_distance <= max(distance):

            #     max_distance = max(distance) + 0.1
            #     print("maxdistance->max(distance)+0.1")
            
            
            
            for i in range(len(population)):
                fitness[i] = new_fitness(distance[i])

            print(distance)
            elitest_tournament_selection()

            for i in range(30):

                print("individual "+ str(i) + " fitness = %.3f" %fitness[i] + ' distance = %.3f' %distance[i])
            print("LB = %.4f"%min_distance + " UB = %.4f "%max_distance)




            index_max = np.argmin(distance)
            mindistance.append(distance[index_max])
            mostfitness.append(fitness[index_max])
            best_individual.append(population[index_max])
            fitness_cal_time_list.append(fitness_cal_time)
            print("generation " + str(generation) + "  fitness = %.3f "%fitness[index_max] + 'distance =  %.3f'%distance[index_max])
        




            if max_distance <= min(distance):

                max_distance = min(distance) + 0.3
                print("maxdistance->max(distance)+0.1")

            if np.mean(fitness) >= 0.75:
                max_distance -= max(0.1 - 0.001 * generation,0.01)
            
            if np.mean(fitness) <= 0.25:
                min_distance += max(0.1 - 0.001 * generation,0.01)
                # if np.mean(distance) <= 0.5 * (max_distance + min_distance):
                #     max_distance -= max(0.05 - 0.001 * generation,0.01)
                # else:
                #     min_distance -= max(0.05 - 0.001 * generation,0.01)

            if mindistance[generation] != current_best:
                stopping = 0
            
            else:
                temp = fitness_cal_time - fitness_cal_time_this_generation
                stopping += temp

            print("stopping count:   " + str(stopping))
            current_best = mindistance[generation]
            if stopping > 5000:

                break
       
        d = {'min_distance': mindistance, 'best_individual': best_individual,'fitness': mostfitness, 'fitness_cal_time':fitness_cal_time_list}
        df = pd.DataFrame(data=d)
        df.to_pickle(output_file) 
                
            
            
        
        
    
        


        
    
        

