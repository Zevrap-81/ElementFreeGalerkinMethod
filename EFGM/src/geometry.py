from __future__ import  annotations
from abc import ABC, abstractmethod
from EFGM.src.pygmsh_extension import MyGeometry
import numpy as np
from typing import Tuple, Optional 
import gmsh

class GeometricModel(ABC):
    def __init__(self):
        self.P = None
        self.u_bar= None
    @abstractmethod
    def create(self, env:MyGeometry):
        pass
    
    @abstractmethod
    def mesh_size(self):
        pass

    @abstractmethod
    def traction_force(self):
        pass

    # # @abstractmethod
    # # def displacement(self):
    # #     pass

class RectangularBeam(GeometricModel):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        n_div_len: int = None, 
        n_div_width: int = None):

        self.x0 = np.array([0, -b/2, 0]) #Bottom Left point
        self.length= a
        self.width = b
        self.n_div_len= n_div_len
        self.n_div_width= n_div_width
    
    def create(self, env:MyGeometry):
        
        self.rect= env.add_polygon([
            self.x0,
            self.x0 + [0,self.width,0], 
            self.x0 + [self.length,self.width,0], 
            self.x0 + [self.length,0,0]
        ])
        
        # bisection_line = env.add_line(self.x0 + [self.length/2, 0,0], self.x0 + [self.length/2, self.width, 0])
        
        env.set_transfinite_curve([*self.rect.lines[0::2]], self.n_div_width+1)  #bisection_line
        env.set_transfinite_curve(self.rect.lines[1::2], self.n_div_len+1)
        env.set_transfinite_surface(self.rect)
        
        env.add_physical(self.rect.lines[0], "Displacement")
        env.add_physical(self.rect.lines[2], "Traction")
        # env.add_physical(bisection_line, "Midsection")
    
    @property
    def mesh_size(self):
        return np.array([self.length/self.n_div_len, self.width/self.n_div_width])
    @property
    def I(self):
        return (self.width**3)/12

    def traction_force(self, x):
        '''
            Parabolic traction force
        '''
        x, y = x[0], x[1]
        c= -self.P/(2*self.I)

        t = np.empty(2)
        t[0] = 0
        t[1] = c*((self.width/2)**2 - y**2)
        return t
    
    def essential_boundary(self, x, material_params):
        # fix me implement this interms of get_disp_exact
        x, y = x[0], x[1]
        c1= self.P*(2+material_params.poi)/(6*self.I* material_params.E)
        c2 = -self.P*material_params.poi*self.length/(2*self.I* material_params.E)

        u_bar = np.empty(2)
        u_bar[0] = c1*( y**2 - (self.width/2)**2 )*y
        u_bar[1] = c2* y**2
        return u_bar
    
    def get_disp_exact(self, x, material_params):
        x, y = x[0], x[1]
        c = self.P/(6*self.I* material_params.E)

        u = np.empty(2)
        temp = (6*self.length-3*x)*x
        temp+= (2+material_params.poi)*(y**2 -(self.width/2)**2)
        u[0]= c*y*temp
        
        temp = 3*material_params.poi*(self.length-x)*y**2
        temp+= (4 + 5*material_params.poi)*x*(self.width/2)**2
        temp+= (3*self.length - x)*x**2
        u[1]= -c*temp  

        return u   

    def get_stress_exact(self, x):
        c= self.P/self.I 
        x, y = x[0], x[1]
        sigma = np.empty(3)
        sigma[0]= c*(self.length - x)*y
        sigma[1]= 0
        sigma[2]= -0.5*c*((self.width/2)**2 - y**2)

        return sigma 
    # def get_cell_data_dict(self):

    #     #mesh.cells_dict = {"celltype/elementtype": array([elementboundaries])}
    #     #mesh.cell_sets_dict = {"physical_groupname": {"elemnttype" : array(elementindices)} }
    #     #cell_data_dict= {"physical_groupname": {"elemnttype" : array([elementboundaries])} }

    #     assert self.mesh is not None
    #     cell_data_dict = {}
    #     for key, data in self.mesh.cell_sets_dict.items():
    #         temp = {}
    #         for element_type, indices in data.items():
    #             temp[element_type]= self.mesh.cells_dict.get(element_type)[indices]

    #         if len(temp)==1: temp = temp[element_type]
    #         cell_data_dict[key]= temp

    #     return cell_data_dict


    







