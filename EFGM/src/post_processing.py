from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from EFGM.common.parameters import Parameters
from EFGM.src.efgm import EFGM_Method


class PostProcessing(ABC):
    def __init__(self):
        pass 


class PostProcessing_RectangularBeam(PostProcessing):
    def __init__(self, params:Parameters,  solver:EFGM_Method):
        self.params= params
        self.solver = solver
    
    def disp_error(self):
        model = self.params.geometry.model

        disp = self.params.post_processing.disp
        disp_exact = model.get_disp_exact()

    def plot_stresses(self, stress_type= "nominal"):
        #fix me
        node2 = np.linspace(-self.params.geometry.model.width/2, self.params.geometry.model.width/2, 10)     
        node1 = np.ones_like(node2)*self.params.geometry.model.length*0
        
        nodes = np.vstack((node1, node2)).T

        stresses= np.empty((len(nodes), 3))
        stresses_ex = np.empty((len(nodes),3))
        model = self.params.geometry.model
        
        for i, node in enumerate(nodes):
            print(node)
            stresses[i]= self.solver.get_stress(node)
            stresses_ex[i]= model.get_stress_exact(node)

        if stress_type=='nominal':
            i= 0
        elif stress_type=='shear':
            i= 2

        fig, ax = plt.subplots()
        ax.plot(nodes[:,1], -stresses[:,i], color='b', label='Numerical')
        ax.scatter(nodes[:,1], stresses_ex[:,i], color='r', marker='o', label='Analytical')
        ax.legend()
        plt.xlabel("nodes along width")
        plt.ylabel("τ (psi)")
        plt.title(f"{stress_type} stress along cross-section")
        plt.show()

        # entities= self.solver.geom.get_entities_for_physicalgroup_name(entity_name)
        # node_tags, nodes = self.solver.geom.get_nodes(tags= entities)
        
