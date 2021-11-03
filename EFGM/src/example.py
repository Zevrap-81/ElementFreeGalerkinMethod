import numpy as np

from efgm import EFGM_Method
from geometry import RectangularBeam    
from EFGM.common.parameters import Parameters      

beam = RectangularBeam(x0=[-1,-1,0], a=48, b= 12, n_div_len=23, n_div_width=11)

#initialise parameters
params= Parameters()

#parameters related to geometry
params.geometry.geometry_type= '2D'
params.geometry.model= beam 

params.geometry.model.P= 1000 #traction force

params.save_geometry= True
params.save_geometry_path= "/Users/parvezmohammed/Downloads/Downloads/Summer21_22/ResearchProject/ElementFreeGalerkinMethod/EFGM/geometries"
params.geometry.visualise= False

#parameters related to material
params.material.E = 30e6 
params.material.poi = 0.3
params.material.type = 'plane_stress'

#paramters related to quadrature
params.quadrature.integration_type = "CompositeGauss7" 

#parameters related to domain
params.domain.dmax = 4


solver = EFGM_Method(params)
solver.initialize()
solver.create_geometry()
solver.create_mesh()
solver.init_material_properties()
solver.set_nodes_domain()
solver.assemble_descrete_equations()
solver.solve()
solver.visualize()
