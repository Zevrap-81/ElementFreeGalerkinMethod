from abc import ABC
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from EFGM.src.geometry import GeometricModel

class ParametersBase(ABC):
    def __init__(self):
        pass

class ParametersQuadrature(ParametersBase):
    def __init__(self):
        self.integration_type= "CompositeGauss7"
        self.gauss_internal = None
        self.gauss_traction_boundary = None
        self.gauss_displacement_boundary = None
        self.lagrange_shape_func= None


class GeometryParameters(ParametersBase):
    def __init__(self):
        """
        Contains all parameters related to geometry
        """
        self.geometry_type= '3D'
        self.load_geometry= False
        self.load_geometry_path= None

        self.model : GeometricModel= None

        self.visualise= False
        self.save_geometry= False
        self.save_geometry_path= None
class BoundaryParameters(ParametersBase):
    def __init__(self):
        self.P = None
        
class MeshParameters(ParametersBase):
    def __init__(self):
        self.num_nodes:int= None
        self.num_dims:int= 3
        self.mesh= None
        self.mesh_size:Tuple[int,int,int]= None #avg mesh size in different dimensions
        
class MaterialParameters(ParametersBase):
    def __init__(self):
        self.E = None
        self.poi = None
        self.Dmat = None
        self.Cmat = None
        self.type:str= None #= 'plane_stress', 'plane_strain'

    def _init_plane_stress(self):
        self.Dmat = np.dot((self.E/(1-self.poi**2)),
                           np.array([[1, self.poi, 0], [self.poi, 1, 0], [0, 0, (1-self.poi)/2]]))

        self.Cmat = np.array(
            [[1, -self.poi, 0], [-self.poi, 1, 0], [0, 0, 1+self.poi]])/self.E

    def _init_plane_strain(self):
        pass

class DomainParameters(ParametersBase):
    def __init__(self):
        self.dmax = 0
        self.dm = None
        self.weight_function= 'exp' #'exp' and 'cubic' are supported

@dataclass
class DescreteEquations(ParametersBase):
    K_stiff: np.ndarray= None
    G : np.ndarray = None
    f : np.ndarray = None
    q : np.ndarray = None

class PostProcessingParameters(ParametersBase):
    def __init__(self):
        self.disp = None
        self.stress = None
        self.strain = None
        self.energy_norm = None
        self.calculate_stress = False
        self.calculate_strain = False
        self.calculate_energy_norm = False

class Parameters:
    def __init__(self):
        self.geometry= GeometryParameters()
        self.mesh= MeshParameters()
        self.material= MaterialParameters()
        self.quadrature= ParametersQuadrature()
        self.domain = DomainParameters()
        self.descrete_equations= DescreteEquations()
        self.post_processing = PostProcessingParameters()
        
