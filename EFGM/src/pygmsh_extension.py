from __future__ import  annotations
from collections.abc import  Iterable
import gmsh
import pygmsh
import meshio
import numpy as np
import typing
from pygmsh.occ import Geometry
from pygmsh.common.point import Point 

class MyGeometry(Geometry):
    def __init__(self):
        super().__init__()
    
    def open(self, filename:str):
        gmsh.open(filename)
    
    def add_line(self, p0: Point | np.ndarray, p1:Point| np.ndarray):
        if isinstance(p0, Point) and isinstance(p1, Point):
            return super().add_line(p0, p1)
        else: 
            p0= self.add_point(p0)
            p1= self.add_point(p1)
            return super().add_line(p0, p1)

    def generate_quad_mesh(self):
        self.set_opt("Mesh.RecombineAll", 1)
        self.set_opt("Mesh.RecombinationAlgorithm", 3)
        mesh = self.generate_mesh()
        gmsh.model.mesh.remove_duplicate_nodes()

        self.synchronize()
        return mesh
    
    def set_transfinite_curve(self, curves: list, num_nodes: int, mesh_type: str = 'Progression', coeff: float=1.0):
        if not isinstance(curves, list): curves= list(curves)
        for curve in curves:
            super().set_transfinite_curve(curve, num_nodes, mesh_type, coeff)
    
    def visualise(self):
        gmsh.fltk.run()
    
    def get_entities_for_physicalgroup_name(self, physicalgroup_name:str=None):
        """If physicalgroup_name is none return all the physical groups present"""

        assert isinstance(physicalgroup_name, str) or physicalgroup_name is None

        physical_groups= gmsh.model.getPhysicalGroups() #dimtags of physical groups
        entity_data_dict = {}

        for dimtag in physical_groups:
            name= gmsh.model.getPhysicalName(*dimtag)
            entity_data_dict[name]= gmsh.model.getEntitiesForPhysicalGroup(*dimtag)
        if physicalgroup_name:
            try:
                return entity_data_dict[physicalgroup_name]
            except KeyError:
                print(f"There is no physical group by name {physicalgroup_name}.")
        else:
            return entity_data_dict
    
    def get_quadrature(self, element_type: int, integration_type:str="CompositeGauss7", tags:list[int]= [-1], geometry_type: str= "3D"):
        #element_type: int= 1(line), 2(triangle), 3(quadrilateral)
        
        
        uvw, weights= gmsh.model.mesh.getIntegrationPoints(element_type, integration_type)
        weights = weights.tolist()

        points= []
        dets= []
        num_elems= 0
        for tag in tags:
            jac, det, point = gmsh.model.mesh.getJacobians(element_type, uvw, tag)
            points+=point.tolist()
            dets+= det.tolist()
        
        points= np.array(points).reshape(-1,3)
        dets= np.array(dets).reshape(-1,1)

        quado= len(weights) #number of quadrature points
        num_elems = int(len(points)/quado)

        weights*=num_elems
        weights = np.array(weights).reshape(-1,1)

        if geometry_type=='1D':
            points= points[:,0]
        elif geometry_type=='2D':
            points= points[:,0:2]

        gauss_quad = np.hstack((points, weights, dets))
        return gauss_quad.reshape(num_elems, -1, *gauss_quad.shape[1:])
    
    def get_nodes(self, element_type: int = 1, tags: list = [-1]):
        '''
            get node_tags on specific entity tags
        '''
        if not isinstance(tags, Iterable):
            tags = list(tags)

        node_tags= []
        for tag in tags:    
            t, _, nodes= gmsh.model.mesh.getElements(1, tag)
            idx= np.where(t==element_type)[0].item()
            node_tags+= nodes[idx].tolist()
        return node_tags
    
    def get_basis(self, element_type: int, integration_type:str="CompositeGauss7", function_space_type= "Lagrange"):
        uvw, weights= gmsh.model.mesh.getIntegrationPoints(element_type, integration_type)
        num_comp, shape_func, _ = gmsh.model.mesh.getBasisFunctions(element_type, uvw, function_space_type)
        return shape_func.reshape(len(weights),-1)

    def save_geometry(self, filename: str):
        self.set_opt("Mesh.SaveAll", 1)
        return super().save_geometry(filename)

    # taken from https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_8_4/tutorial/python/t8.py#L89
    @staticmethod
    def set_opt(name, val):
        # if it's a string, call the string method
        val_type = type(val)
        if val_type == type("str"):
            gmsh.option.setString(name, val)
            # some examples include "Geometry.CurveLabels" , "Mesh.NodeLabels", "General.RotationX" etc

        # otherwise call the number method
        elif val_type == type(0.5) or val_type == type(1):
            gmsh.option.setNumber(name, val)
        else:
            print("error: bad input to set_opt: " + name + " = " + str(val))
            print("error: set_opt is only meant for numbers and strings, aborting")
            quit(1)
 