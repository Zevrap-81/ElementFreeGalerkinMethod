import numpy as np
from EFGM.src.post_processing import PostProcessing_RectangularBeam

from efgm import EFGM_Method
from geometry import RectangularBeam    
from EFGM.common.parameters import Parameters      

beam = RectangularBeam(a=48, b= 12, n_div_len=48, n_div_width=24)

#initialise parameters
params= Parameters()

#parameters related to geometry
params.geometry.geometry_type= '2D'
params.geometry.model= beam 

params.geometry.model.P= 1000 #traction force

params.geometry.visualise= False

#parameters related to material
params.material.E = 30e6 
params.material.poi = 0.3
params.material.type = 'plane_stress'

#paramters related to quadrature
params.quadrature.integration_type = "CompositeGauss2" 

#parameters related to domain
params.domain.dmax = 2
params.domain.weight_function= 'cubic'
#parameters related to postprocessing
params.post_processing.calculate_energy_norm= True


solver = EFGM_Method(params)
# solver.initialize()
# solver.create_geometry()
# solver.create_mesh()
# solver.init_material_properties()
# solver.set_nodes_domain()
# solver.assemble_descrete_equations()
# solver.solve()
# solver.visualize()

#or do 
solver.simulation_automatic()

post = PostProcessing_RectangularBeam(params, solver)

post.plot_stresses("shear") #"Midsection"
