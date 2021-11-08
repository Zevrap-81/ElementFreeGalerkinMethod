from matplotlib import pyplot as plt
import numpy as np
from EFGM.common.parameters import Parameters
from EFGM.src.efgm import EFGM_Method
from EFGM.src.geometry import RectangularBeam 

#initialise parameters
params= Parameters()

#parameters related to geometry
params.geometry.geometry_type= '2D'

params.geometry.visualise= False

#parameters related to material
params.material.E = 30e6 
params.material.poi = 0.3
params.material.type = 'plane_stress'

#paramters related to quadrature
params.quadrature.integration_type = "CompositeGauss7" 

#parameters related to domain
params.domain.dmax = 4
params.domain.weight_function = 'cubic'
#parameters related to postprocessing
params.post_processing.calculate_energy_norm= True


length, width = 48, 12 

h= np.logspace(-0.2, 1.0, base= 10, num=10)
energy_norm= []
for element_size in h:
    n_div_len, n_div_width = int(length/element_size), int(width/element_size)
    model = RectangularBeam(length, width, n_div_len, n_div_width)
    params.geometry.model= model 
    params.geometry.model.P= 1000 #traction force

    solver= EFGM_Method(params)
    e_norm= solver.simulation_automatic() 
    solver.finalize()
    print(energy_norm)
    energy_norm.append(e_norm) 

energy_norm = np.array(energy_norm)

slope, intercept= np.polyfit(h,energy_norm, 1)

data = np.vstack((h, energy_norm))
np.savetxt('data.csv', data, delimiter=',')

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

fig, ax = plt.subplots()
ax.plot(np.log10(h), np.log10(energy_norm), color='b', label=f'Slope= {slope:.2f}')
ax.legend()
plt.xlabel("$Log_{10}(h)$")
plt.ylabel("$Log_{10}(energy_norm)")
plt.title("shear stress along cross-section")
plt.show()
