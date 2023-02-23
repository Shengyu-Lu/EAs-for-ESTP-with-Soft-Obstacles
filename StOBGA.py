from os import terminal_size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance,  Delaunay
from shapely.geometry import MultiPoint, Point, LineString, Polygon, MultiPolygon
import random
import time




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
    terminals_csv =  'TestCases/Type1/terminals' + instance + '.csv'
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
    
    obstacles_csv = 'TestCases/Type1/obstacles' + instance + '.csv'
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

#Prim Algorithm 

# def Prim_Mst(points):
# #   calculate the total distance of an individual, should input all_points of the current individual
# #   change the value of global var mst_edgepoints -> mst edge points of the current individual e.g. [[(x1,y1),(x2,y2)],[],[]]
# #    create border edges
#     global mst_edgepoints
#     border_edges = []
#     for obstacle in obstacles:
#         for i in range(len(list(obstacle.exterior.coords))-1):
#             border_edges.append(LineString([list(obstacle.exterior.coords)[i],list(obstacle.exterior.coords)[i+1]]))
#             border_edges.append(LineString([list(obstacle.exterior.coords)[i+1],list(obstacle.exterior.coords)[i]]))
    
#     mst_edgepoints = []

# #     create adj_matrix
#     adj_matrix = [[0 for x in range(len(points))] for y in range(len(points))] 
    
#     for i in range(len(points)):
#         for j in range(len(points)):
#             if i < j:

#                 edge = LineString([points[i],points[j]])
#                 length = edge.length
#                 for p in range(len(penalty)):
#                     intersection = edge.intersection(obstacles[p])
#                     if intersection not in border_edges:
#                         length += intersection.length*(penalty[p]-1)
#                 adj_matrix[i][j] = length
#             if i > j:
#                 adj_matrix[i][j] = adj_matrix[j][i]
    
    
#     total_distance = 0

#     selected_node = [0 for i in range(len(points))]

#     no_edge = 0

#     selected_node[0] = True

#     # printing for edge and weight
#     # print("Edge : Weight\n")
#     while (no_edge < len(points) - 1):
        
#         minimum = INF
#         a = 0
#         b = 0
#         for m in range(len(points)):
#             if selected_node[m]:
#                 for n in range(len(points)):
#                     if ((not selected_node[n]) and adj_matrix[m][n]):  
#                         # not in selected and there is an edge
#                         if minimum > adj_matrix[m][n]:
#                             minimum = adj_matrix[m][n]
#                             a = m
#                             b = n
#         # print(str(a) + "-" + str(b) + ":" + str(adj_matrix[a][b]))
#         mst_edgepoints.append((points[a],points[b]))
#         # mst_edgepoints.append(all_points[a])
#         # mst_edgepoints.append(all_points[b])
#         selected_node[b] = True
#         no_edge += 1
#         total_distance += adj_matrix[a][b]

#     return total_distance


def less_first(a, b):
    return [a,b] if a < b else [b,a]


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






def create_figs(individual,figname):

    cal_fitness(individual)
    # create a list for polygons(obstacles)
    polygons = []

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # draw obstacles
    for i in range(len(penalty)):
        polygons.append(Polygon(obstacles[i]))
    

    for polygon in polygons:
        ax.fill(*polygon.exterior.xy,'#B4F0B4')


    # draw terminals
    terminals_shp = MultiPoint(terminals)
    terminal_xs = [point.x for point in terminals_shp]
    terminal_ys = [point.y for point in terminals_shp]

    ax.scatter(terminal_xs,terminal_ys, s = 10, c = '#000000')

    # draw steiner points
    steinerpoints_shp = MultiPoint(steinerpoints)
    steinerpoint_xs = [point.x for point in steinerpoints_shp]
    steinerpoint_ys = [point.y for point in steinerpoints_shp]

    ax.scatter(steinerpoint_xs,steinerpoint_ys, s = 10, c = 'r')

    # draw corner points
    cornerpoints_shp = MultiPoint(selected_cornerpoints)
    cornerpoint_xs = [point.x for point in cornerpoints_shp]
    cornerpoint_ys = [point.y for point in cornerpoints_shp]

    ax.scatter(cornerpoint_xs,cornerpoint_ys, s = 10, c = 'b')

    # draw edges of MST
    
    for i in range(len(mst_edgepoints)):
        x = (mst_edgepoints[i][0][0],mst_edgepoints[i][1][0])
        y = (mst_edgepoints[i][0][1],mst_edgepoints[i][1][1])
        ax.plot(x, y, c = '#000000', linewidth = 0.3)    

    
    plt.savefig("%s"%figname)

def init_instance(instance):
#  given instance number in str, e.g "1"
    extract_terminals(instance)
    extract_obstacles(instance)
    
    global border_edges
    for obstacle in obstacles:
        for i in range(len(list(obstacle.exterior.coords))-1):
            border_edges.append(LineString([list(obstacle.exterior.coords)[i],list(obstacle.exterior.coords)[i+1]]))
            border_edges.append(LineString([list(obstacle.exterior.coords)[i+1],list(obstacle.exterior.coords)[i]]))
   
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
    if length > 0:
        random_index = np.random.randint(length)
        
        picked_three_points = [angles_less_than_120[random_index][0][0]]+\
        [angles_less_than_120[random_index][0][1]]+\
        [angles_less_than_120[random_index][1][1]]
        
        new_steiner = ((picked_three_points[0][0]+picked_three_points[1][0]+picked_three_points[2][0])/3.0,(picked_three_points[0][1]+picked_three_points[1][1]+picked_three_points[2][1])/3.0)
    else:
        new_steiner = np.random.rand(2) 
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
    steiner_index = []
    steiner_larger_than_2 = []

    for i in range(len(mst_edgepoints)):    
        temp = mst_edgepoints[i]
        temp1 = []
        temp1.append(temp[1])
        temp1.append(temp[0])
        reverse_mst_edgepoints.append(temp1)
    double_mst_edgepoints = reverse_mst_edgepoints + mst_edgepoints

    for index in range(len(steinerpoints)):
        count = 0
        for i in range(len(double_mst_edgepoints)):
            if steinerpoints[index] == double_mst_edgepoints[i][0]:
                count += 1
        if count > 2:
            steiner_index.append(index)

    for index in steiner_index:
        steiner_larger_than_2.append(steinerpoints[index])
    steiner_larger_than_2 = np.array(steiner_larger_than_2)

    return steiner_larger_than_2


def init_population():
    global population
    global fitness
    
   
    population = []
    fitness = []
#   add individual(s) of type_1
    individual_type1 = []
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
            chromosome1.append(steiner)
    chromosome1 = np.array(chromosome1)
    chromosome2 = np.zeros((len(cornerpoints),), dtype=int)
    individual_type1.append(chromosome1)
    individual_type1.append(chromosome2)
    
#   show the mst of individual_type1
#     plt.triplot(terminals_and_cornerpoints[:,0], terminals_and_cornerpoints[:,1], tri.simplices)
#     plt.plot(terminals_and_cornerpoints[:,0], terminals_and_cornerpoints[:,1], 'o')
#     plt.show()

    for i in range(type1_num):
        population.append(individual_type1)
        fitness.append(cal_fitness(individual_type1))
  

# def Kruskal


#   add individuals of type_2

    steiner_num = len(terminals) + len(cornerpoints)
    for i in range(type2_num):
        individual_type2 = []
        chromosome1 = np.random.rand(steiner_num, 2)
        chromosome2 = np.zeros((len(cornerpoints),), dtype=int)
        individual_type2.append(chromosome1)
        individual_type2.append(chromosome2)
        population.append(individual_type2)
        fitness.append(cal_fitness(individual_type2))
        
#   add individuals of type_3

    for i in range(type3_num):
        individual_type3 = []
        chromosome1 = np.array([])
        chromosome2 = np.random.randint(2,size=len(cornerpoints))
        individual_type3.append(chromosome1)
        individual_type3.append(chromosome2)
        population.append(individual_type3)
        fitness.append(cal_fitness(individual_type3))
        
def tournament_selection_best():
#     return the array of index, population[index] = parents
    parents = [] # two parents
    no_repeat = [] # avoid picking same parents
    while len(no_repeat) < parents_size:

        random_index = np.random.choice(len(population), size=tournament_size,replace = False)

        fitness_tournament = [fitness[index] for index in random_index]

        index_min = np.argmin(fitness_tournament)

        if random_index[index_min] not in no_repeat:
            no_repeat.append(random_index[index_min])
#             parents.append(population[random_index[index_min]])
            
    return no_repeat

def tournament_selection_worst():

    global population
    global fitness

    random_index = np.random.choice(len(population), size=tournament_size,replace = False)

    fitness_tournament = [fitness[index] for index in random_index]

    index_max = np.argmax(fitness_tournament)

    population.pop(random_index[index_max])
    fitness.pop(random_index[index_max])

def crossover(parents):

#  return two offspring(individuals)

    parent1 = parents[0]
    parent2 = parents[1]

    # get the scope of x of all points of two individuals (parents)
    all_points_indi = []
    for i in range(len(terminals)):
        all_points_indi.append(terminals[i])
    for i in range(len(cornerpoints)):
        all_points_indi.append(cornerpoints[i])
    for i in range(len(parent1[0])):
        all_points_indi.append(parent1[0][i])
    for i in range(len(parent2[0])):
        all_points_indi.append(parent2[0][i])
    all_xs = []
    for i in range(len(all_points_indi)):
        all_xs.append(all_points_indi[i][0])

    x_min = min(all_xs)
    x_max = max(all_xs)

    # randomize x split value
    xaxis_split = np.random.uniform(low = x_min, high = x_max)

    # initialize two offspring
    offspring1 = []
    offspring1.append(np.array([]))
    offspring1.append(np.zeros((len(cornerpoints),), dtype=int))

    offspring2 = []
    offspring2.append(np.array([]))
    offspring2.append(np.zeros((len(cornerpoints),), dtype=int))

    tem1 = []
    tem2 = []

    for i in range(len(parent1[0])):
        if parent1[0][i][0] <= xaxis_split:
            tem1.append(parent1[0].tolist()[i])
            # offspring1[0].append(individual1[0][i])
        else:
            tem2.append(parent1[0].tolist()[i])
            # offspring2[0].append(individual1[0][i])

    for i in range(len(parent1[1])):
        if parent1[1][i] == 1:
            if cornerpoints[i][0] <= xaxis_split:
                offspring1[1][i] = 1
            else:
                offspring2[1][i] = 1

    for i in range(len(parent2[0])):
        if parent2[0][i][0] >= xaxis_split:
            tem1.append(parent2[0].tolist()[i])
            # offspring1[0].append(individual2[0][i])
        else:
            tem2.append(parent2[0].tolist()[i])
            # offspring2[0].append(individual2[0][i])

    for i in range(len(parent2[1])):
        if parent2[1][i] == 1:
            if cornerpoints[i][0] >= xaxis_split:
                offspring1[1][i] = 1
            else:
                offspring2[1][i] = 1

    offspring1[0] = np.array(tem1)
    offspring2[0] = np.array(tem2)
    
    return offspring1, offspring2 

def mutation(individual1, individual2):
    
    offspring1 = individual1
    offspring2 = individual2
    
    p_flipmove = max(p_flipmove_max*(1-generation/1000),p_flipmove_min)
    p_addsteiner = 1 - p_flipmove/2
    p_removesteiner = p_addsteiner
    move_range = average_terminal_distance * max(1-generation/1000,0.01)


    # choose one mutation from ["FlipMove", "AddSteiner", "RemoveSteiner"] for each offspring
    mutation1 = random.choices(Mutations, weights=(p_flipmove, p_addsteiner, p_removesteiner), k=1)
    mutation2 = random.choices(Mutations, weights=(p_flipmove, p_addsteiner, p_removesteiner), k=1)

# SEPERATE MUTATION for offspring1 and offspring2

    # FlipMove Mutation
    if mutation1 == ['FlipMove']:
        gene_num_1 = len(offspring1[0]) + len(offspring1[1])
        p = 1/gene_num_1
        gene_index = np.random.randint(low = 0, high = gene_num_1, size = 1).tolist()[0]
        # if it's steinerpoint
        if gene_index < len(offspring1[0]):
            offspring1[0][gene_index][0] += random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
            offspring1[0][gene_index][1] += random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
        else:
            if offspring1[1][gene_index-len(offspring1[0])] == 0:
                offspring1[1][gene_index-len(offspring1[0])] = 1
            else:
                offspring1[1][gene_index-len(offspring1[0])] = 0

    if mutation2 == ['FlipMove']:

        gene_num_2 = len(offspring2[0]) + len(offspring2[1])
        p = 1/gene_num_2
        gene_index = np.random.randint(low = 0, high = gene_num_2, size = 1).tolist()[0]
        # if it's steinerpoint
        if gene_index < len(offspring2[0]):
            offspring2[0][gene_index][0] += random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
            offspring2[0][gene_index][1] += random.choices([1,-1], k=1)[0] * np.random.uniform(low = -move_range, high = 0, size = 1).tolist()[0] 
        else:
            if offspring2[1][gene_index-len(offspring2[0])] == 0:
                offspring2[1][gene_index-len(offspring2[0])] = 1
            else:
                offspring2[1][gene_index-len(offspring2[0])] = 0

    # AddSteiner Mutation
    if mutation1 == ['AddSteiner']:
        new_steiner = add_steiner(offspring1)
        temp = offspring1[0].tolist()
        temp.append(new_steiner)
        offspring1[0]= np.array(temp)

    if mutation2 == ['AddSteiner']:
        new_steiner = add_steiner(offspring2)
        temp = offspring2[0].tolist()
        temp.append(new_steiner)
        offspring2[0]= np.array(temp)

    # RemoveSteiner Mutation
    if mutation1 == ['RemoveSteiner']:
        offspring1[0] = remove_steiner_less_than_2(offspring1)

    if mutation2 == ['RemoveSteiner']:
        offspring2[0] = remove_steiner_less_than_2(offspring2)



    # print("offspring1",offspring1)
    # print("offspring2",offspring2) 

    # add new offspring to the population
    population.append(offspring1)
    population.append(offspring2)

    # add fitness 
    fitness.append(cal_fitness(offspring1))
    fitness.append(cal_fitness(offspring2))    



# global stable var for all instances
INF = 999
population_size = 500
p_flipmove_max = 0.99
p_flipmove_min = 0.6
Mutations = ["FlipMove", "AddSteiner", "RemoveSteiner"]
generation_num = 500
tournament_size = 5
parents_size = 2
offspring_size = 2
type1_num = 2
type2_num = 50
type3_num = 50

fitness_cal_time = 0
fitness_cal_time_list = []
time_list = []


instance_list = ['6','7','8','9','10','12','16','19','20','21','22']
instance_list2 = ['1','2','3','4','5','12','13','14','15','16','27','29','30','32','33']
instance_list3 = ['5','12','13','14','15','16','27','29','30','32','33']
instance_list4 = ['1','2','3']
instance_list5 = ['10','20','30','40','50','60','70','80','90','100']
instance_list6 = ['60','70','80','90','100']
instance_list7 = ['21']
instance_list8 = ['61','101','201','301','401','501','1001']
instance_list9 = ['4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
instance_list10 = ['24','25','26','29','30','31','32','33','34','35']
# initialize the instance
for repeat in range(1,2):
    time_list = []
    fitness_cal_time_list = []

    for instance in instance_list10:

        
        start_time = time.time()

        fitness_cal_time = 0
        # store the results
        min_distance = []
        best_individual = []



        # fixed values per instance
        terminals = []
        penalty = []
        obstacles = [] 
        soft_obstacles =[]
        solid_obstacles = []
        cornerpoints = []
        border_edges = []
        average_terminal_distance = 0  # will be calculated after extracting instance

        # changeable var per individual
        steinerpoints = []
        selected_cornerpoints = []
        mst_edgepoints = []
        all_points = []
        angles_less_than_120 = []

        # changeable var per generation
        population = []
        fitness = []

        # used to control the generation loop
        total_length = 0
        count = 0

        output_file = "Results_data/Type1/instance" + instance + "_trial_" + str(repeat) + ".pkl"
        init_instance(instance)
        print("finish")
        # calculate the average distance of initial terminals
        for i in range(len(terminals)):
            for j in range(len(terminals)):
                if i < j:
                    average_terminal_distance += round(distance.euclidean(terminals[i],terminals[j]),4)
        average_terminal_distance = average_terminal_distance/(len(terminals) * (len(terminals) - 1) * 0.5)
        average_terminal_distance = round(average_terminal_distance,4)


        print("Instance Initialization Finished")
        print("average_terminal_distance:%f"%average_terminal_distance)

        # initialize population and its fitness list
        init_population()

        print("population initialization finished 102/500")


        for i in range(199): #for each loop produce two offspring into population
            generation = 0
            parents_index = tournament_selection_best()
            offspring1, offspring2 = crossover([population[parents_index[0]],population[parents_index[1]]])
            mutation(offspring1,offspring2)
            print("population initialization finished %i/199"%i)
        # create_figs(population[fitness.index(min(fitness))],"initial_best_solution")


        for generation in range(generation_num):
            # produce 166 offspring into population (500 + 166)
           
            for i in range(83):
                parents_index = tournament_selection_best()
                offspring1, offspring2 = crossover([population[parents_index[0]],population[parents_index[1]]])
                mutation(offspring1,offspring2)
            # drop 166 individuals from population (500 + 166 - 166)
            for i in range(166):
                tournament_selection_worst()

            index_min = np.argmin(fitness)
            min_distance.append(fitness[index_min])
            best_individual.append(population[index_min])
            # create_figs(population[fitness.index(min(fitness))],"generation_%i_best_solution"%generation)
            print("instance:",instance,"generation:",generation,"best solution:",round(min(fitness),4))
            
            # count number of generations of the same best solution 
            if round(min(fitness),4) == total_length:
                count += 1
            else:
                count = 0
            total_length = round(min(fitness),4)

            # control converge
            if count == 50:
                break
        
        
        # save result data
        d = {'min_distance': min_distance, 'best_individual': best_individual}
        df = pd.DataFrame(data=d)
        df.to_pickle(output_file) 

        # for save processing time
        fitness_cal_time_list.append(fitness_cal_time)
        time_list.append(time.time() - start_time)
        print('total time',(time.time() - start_time))

        if count != 50:

            print('Not Converge!!')
        # df_time = pd.DataFrame(data={"col1": time_list})
        # df_time.to_csv("superbig_" + str(instance) + "_time.csv", sep=',',index=False)
        # df_fitness_cal_time = pd.DataFrame(data={"fitnesscaltime":fitness_cal_time_list})
        # df_fitness_cal_time.to_csv("superbig_" + str(instance) +" _fitnesscal.csv", sep=',',index=False)
    # save processing time 
    # only for big size instances
    # print(time_list)
    # print(fitness_cal_time)
    # df_time = pd.DataFrame(data={"col1": time_list})
    # df_time.to_csv("instances3-4to3-20_trial_" + str(repeat) +"_time.csv", sep=',',index=False)
    # df_fitness_cal_time = pd.DataFrame(data={"fitnesscaltime":fitness_cal_time_list})
    # df_fitness_cal_time.to_csv("instances3-4to3-20_trial_" + str(repeat) +"_fiteval.csv", sep=',',index=False)






