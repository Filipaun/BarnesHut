import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# Phsycial constants
# --- Gravitational constant
G = 1


def make_random_bodies(N=10,xrange = (0,100000), yrange = (0,100000),seed = 42):
    """
    Description: 
        Generate specific number of bodies with random initial conditions
    Inputs:
        N: int, number of bodies
        xrange: 2 int tuple, lower and upper bound for x domain
        yrange: 2 int tuple, lower and upper bound for y domain
        seed:  int, seed for rng
    Output:
        tuple of ndarray,  position vector, velocity vector and mass of bodies
    """

    # Max velocity
    velocity_scale = 20

    rng = np.random.default_rng(42)
    positions = rng.normal(size = (N,2))*np.array([(xrange[1]-xrange[0]),yrange[1]-yrange[0]])+np.array([xrange[0],yrange[0]])
    velocity = rng.normal(size = (N,2))*velocity_scale
    mass = np.full(N,1.0)


    return positions, velocity, mass 


class Cell :
    """
    Object representing cell in the quadtree
    """
    def __init__(self,x_domain,y_domain,is_root = False) -> None:
        
        self.is_root = is_root
        # True if cell contains only one particle, leaf in quadtree
        self.is_edge = False
        # True if cell contains zero particles
        self.is_empty = True
        # Center of mass for the cell
        self.com = None
        # Mass of particles in cell
        self.mass = 0

        # child cells of the current cell
        self.child_cells = [None,None,None,None]

        # x value range of the cell domain : size 2 ndarray
        self.x_domain = x_domain
        # y value range of the cell domain : size 2 ndarray
        self.y_domain = y_domain

    def find_quadrant(self,pos_vec):
        """
        Description: 
            Find quadrant index for poisition
        Inputs:
            pos_vec: poisition vector of particle, (2,) ndarray
        Output:
            target_quadrant index (0 : sw, 1 : se, 2 : nw, 3 : ne) or None if not in cell domain
        """
        target_quadrant = 0
        if (pos_vec[0] >= self.x_domain[0] and pos_vec[0] <= self.x_domain[1]) and (pos_vec[1] >= self.y_domain[0] and pos_vec[1] <= self.y_domain[1]) :
            if pos_vec[0] < np.mean(self.x_domain):
                target_quadrant = 0
            else:
                target_quadrant = 1

            if pos_vec[1] >= np.mean(self.y_domain):
                target_quadrant += 2
            else:
                pass
            return target_quadrant
        else:
            return None
        
    def get_quadrant_domain(self,subquadrant):
        """
        Description: 
            Calculate domain of quadrant
        Inputs:
            subquadrant: int, index of subquadrant
        Output:
            domain of quadrant
        """
        if subquadrant == 0 or subquadrant == 1:
            y_subdomain = np.array([self.y_domain[0],np.mean(self.y_domain)])
        else:
            y_subdomain = np.array([np.mean(self.y_domain),self.y_domain[1]])

        if subquadrant == 0 or subquadrant == 2:
            x_subdomain = np.array([self.x_domain[0],np.mean(self.x_domain)])
        else:
            x_subdomain = np.array([np.mean(self.x_domain),self.x_domain[1]])

        return x_subdomain,y_subdomain
            

    def insert(self,pos_vec,point_mass):
        """
        Description: 
            Insert particle into cell/tree
        Inputs:
            pos_vec: (2,) ndarray, position vector
            point_mass: float64, mass of particle
        Output:
            int
        """
        if self.is_empty:
            self.is_empty = False
            self.is_edge = True
            self.com = pos_vec
            self.mass = point_mass

            return 1
        
        elif self.is_edge:
            initial_value_quadrant = self.find_quadrant(self.com)
            child_x_domain, child_y_domain = self.get_quadrant_domain(initial_value_quadrant)
            initial_child_cell = Cell(child_x_domain, child_y_domain)
            initial_child_cell.insert(self.com,self.mass)

            self.child_cells[initial_value_quadrant] = initial_child_cell
            self.is_edge = False

        target_quadrant = self.find_quadrant(pos_vec)
        if self.child_cells[target_quadrant] == None:
            child_x_domain, child_y_domain = self.get_quadrant_domain(target_quadrant)
            child_cell = Cell(child_x_domain, child_y_domain)
            child_cell.insert(pos_vec,point_mass)

            self.child_cells[target_quadrant] = child_cell
        else:
            self.child_cells[target_quadrant].insert(pos_vec, point_mass)


        return 1
    def update_com_mass(self):
        """
        Description: 
            Updates com and mass of cells recursively.
        Output:
            int
        """
        if self.is_edge:
            return 1
        else:
            temp_com = 0
            temp_mass = 0
            for child in self.child_cells:
                if child != None:
                    child.update_com_mass()
                    temp_com += child.com*child.mass
                    temp_mass += child.mass
            self.com = temp_com/temp_mass
            self.mass = temp_mass
            
            return 1
        
def plot_bodies_and_cells(root, com_list, domain_list, plot_domain):
    """
    Description: 
        Gives list of center of mass for all leafs of the quadtree
    Input:
        com_list : empty list
    Output:
        (pass by reference) list of floats
    """

    if root.is_empty:
        return
    if root.is_edge:
        if plot_domain:
            domain_list.append((np.array([root.x_domain,root.y_domain]),1))
        com_list.append(root.com)
    else:
        if plot_domain:
            domain_list.append((np.array([root.x_domain,root.y_domain]),0))
        for cell in root.child_cells:
            if cell is not None:
                plot_bodies_and_cells(cell, com_list, domain_list, plot_domain)

    if root.is_root:
        for (domain, domain_type) in domain_list:
            for j in range(2):
                if domain_type == 1:
                    plt.plot([domain[0,0],domain[0,1]],[domain[1,j],domain[1,j]], "--c")
                    plt.plot([domain[0,j],domain[0,j]],[domain[1,0],domain[1,1]], "--c")
                else:
                    plt.plot([domain[0,0],domain[0,1]],[domain[1,j],domain[1,j]], "--r")
                    plt.plot([domain[0,j],domain[0,j]],[domain[1,0],domain[1,1]], "--r")

        for com in com_list:
            plt.plot(com[0],com[1], ".w")
        plt.show()

def min_square_bounds(positions):
    x_bounds = np.array([np.amin(positions[:,0]),np.amax(positions[:,0])])
    y_bounds = np.array([np.amin(positions[:,1]),np.amax(positions[:,1])])

    x_span = np.linalg.norm(x_bounds)
    y_span = np.linalg.norm(y_bounds)

    if x_span <= y_span:
        x_bounds += np.array([-y_span/2,y_span/2])
    else:
        y_bounds += np.array([-y_span/2,y_span/2])
    
    return x_bounds, y_bounds

positions,velocites, mass = make_random_bodies(20)
x_bounds, y_bands = min_square_bounds(positions)
quadtree = Cell(x_bounds,y_bands,True)

for i in range(len(mass)):
    quadtree.insert(positions[i],mass[i])

quadtree.update_com_mass()

plot_bodies_and_cells(quadtree,[],[], True)




