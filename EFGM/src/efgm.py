from __future__ import  annotations
from dataclasses import  dataclass
import numpy as np 
from meshio import Mesh
import pytest
import gmsh
from EFGM.common.parameters import Parameters
from EFGM.src.gauss import get_gauss
from EFGM.src.support import support
from EFGM.src.pygmsh_extension import MyGeometry
from EFGM.src.shape import shape

    

class EFGM_Method:
    def __init__(self, params:Parameters):
        self.geom= MyGeometry()
        self.params= params
        self.geometry_params= params.geometry

    def initialize(self):
        self.geom.__enter__()
    def finalize(self):
        self.geom.__exit__()
    
    def create_geometry(self):
        if self.geometry_params.model is not None:
            self.geometry_params.model.create(self.geom)
        else:
            if self.geometry_params.load_geometry:
                self.geom.open(self.geometry_params.load_geometry_path)     
            else:
                raise Exception("No geometric model found. Either load a model or provide one")  

    def create_mesh(self):
        self.params.mesh.mesh = self.geom.generate_quad_mesh()
        self.params.mesh.num_nodes = len(self.params.mesh.mesh.points)
        self.params.mesh.mesh_size = self.geometry_params.model.mesh_size

        if self.geometry_params.geometry_type == "1D":
            self.params.mesh.num_dims = 1 
        elif  self.geometry_params.geometry_type == "2D":
            self.params.mesh.num_dims = 2 

    def init_material_properties(self):
        if self.params.material.type == 'plane_stress':
            self.params.material._init_plane_stress()
        elif self.params.material.type == 'plane_strain':
            self.params.material._init_plane_strain()
        else: 
            raise Exception(f"This type: {self.params.material.type}is not supported at the moment")


    def assemble_descrete_equations(self): 
        self.set_nodes_domain()
        self.init_boundary_n_internal()
        self.set_LHS()
        self.set_RHS()
    
    def set_nodes_domain(self):
        num_dims= self.params.mesh.num_dims
        num_nodes = self.params.mesh.num_nodes
        mesh_size = self.params.mesh.mesh_size

        dmax = self.params.domain.dmax
        self.params.domain.dm = np.empty((num_nodes, num_dims))
        self.params.domain.dm[:] = dmax*mesh_size
    
    def init_boundary_n_internal(self):
        num_nodes= self.params.mesh.num_nodes
        num_dims= self.params.mesh.num_dims

        integration_type = self.params.quadrature.integration_type
        geometry_type = self.params.geometry.geometry_type

        #internal
        element_type = 3 #(quadrilateral)

        self.params.quadrature.gauss_internal= self.geom.get_quadrature(element_type, integration_type, geometry_type= geometry_type)
        self.params.descrete_equations.K_stiff = np.zeros((num_nodes*num_dims, num_nodes*num_dims))
        
        #boundary
        element_type = 1 #(line)
        self.params.quadrature.lagrange_shape_func = self.geom.get_basis(element_type, integration_type)

        #Displacement Boundary
        disp_tags= self.geom.get_entities_for_physicalgroup_name("Displacement") # entity tags (lines) on the displacement boundary
        disp_node_tags, _ = self.geom.get_nodes(element_type, disp_tags)
        self.params.quadrature.gauss_displacement_boundary = self.geom.get_quadrature(element_type, integration_type, disp_tags, geometry_type)
        self.params.descrete_equations.G = np.zeros((num_nodes*num_dims, len(disp_node_tags)*num_dims))
        self.params.descrete_equations.q= np.zeros(len(disp_node_tags)*num_dims)

        #Traction Boundary
        traction_tags= self.geom.get_entities_for_physicalgroup_name("Traction") #entity tags (lines) on the traction boundary
        traction_node_tags, _ = self.geom.get_nodes(element_type, traction_tags)
        self.params.quadrature.gauss_traction_boundary = self.geom.get_quadrature(element_type, integration_type, traction_tags, geometry_type)
        self.params.descrete_equations.f = np.zeros(num_nodes*num_dims)


    def set_LHS(self):
        
        #Get the stifness matrix
        size = self.params.quadrature.gauss_internal.shape
        #reshape because of not wanting to loop over elements and directly loop over gauss points
        for quad in self.params.quadrature.gauss_internal.reshape(-1, *size[2:]):
            g_point = get_gauss(quad, self.params)
            phi, dphidx, dphidy= shape(g_point, self.params)
            assert pytest.approx(sum(phi), 1*np.finfo(np.float64).eps) == 1
            
            Bmat = self.get_Bmat(g_point, dphidx, dphidy)
            
            ids= self.get_indices_voigt(g_point)
            id_x, id_y = np.meshgrid(ids, ids)
            K = g_point.jac_det* g_point.weight * np.dot(Bmat.T, np.dot(self.params.material.Dmat,Bmat))
           
            self.params.descrete_equations.K_stiff[id_x, id_y] +=  K

        #Get the G matrix (lagrange multiplier approach)
        for e_id, element in enumerate(self.params.quadrature.gauss_displacement_boundary):
            for i, quad in enumerate(element):
                g_point = get_gauss(quad, self.params)
                phi, dphidx, dphidy = shape(g_point, self.params)
                assert pytest.approx(sum(phi), 1*np.finfo(np.float64).eps) == 1
                assert pytest.approx(sum(phi*self.params.mesh.mesh.points[g_point.support_nodes, 0]), 10**(-4)) == g_point.coord[0]

                lsf = self.params.quadrature.lagrange_shape_func[i]
                Nmat = self.get_Nmat(lsf)

                G= Nmat[None, :,:]*phi[:, None, None]
                G= - g_point.jac_det* g_point.weight*G.reshape(-1, len(lsf)*2)
                
                id_x= self.get_indices_voigt(g_point)

                id_y= np.arange(0, len(lsf)*2)
                offset= e_id*2
                id_y+= offset

                id_x, id_y = np.meshgrid(id_x, id_y)
                
                self.params.descrete_equations.G[id_x,id_y]+=  G.T
        # print(self.params.descrete_equations.G)

   
    def set_RHS(self):
        #integrating force along the traction boundary
        size = self.params.quadrature.gauss_traction_boundary.shape
        for quad in self.params.quadrature.gauss_traction_boundary.reshape(-1, *size[2:]):
            g_point = get_gauss(quad, self.params)
            phi, dphidx, dphidy = shape(g_point, self.params)
            assert pytest.approx(sum(phi), 10*np.finfo(np.float64).eps) == 1
            
            t = self.params.geometry.model.traction_force(g_point.coord)
            f= t[None, :] * phi[:, None]
            f= g_point.weight * f.flatten()
            
            id_x= self.get_indices_voigt(g_point)

            self.params.descrete_equations.f[id_x]+= g_point.jac_det * f #no meshgrid so no transpose
        
        for e_id, element in enumerate(self.params.quadrature.gauss_displacement_boundary):
            for i, quad in enumerate(element):
                g_point = get_gauss(quad, self.params)
                phi, dphidx, dphidy = shape(g_point, self.params)
                assert pytest.approx(sum(phi), 10*np.finfo(np.float64).eps) == 1

                u_bar = self.params.geometry.model.essential_boundary(g_point.coord, self.params.material)
                lsf = self.params.quadrature.lagrange_shape_func[i]
                Nmat = self.get_Nmat(lsf)

                q= Nmat.T.dot(u_bar)
                q*= -g_point.jac_det * g_point.weight* q

                id= np.arange(0, len(lsf)*2)
                offset= e_id*2
                id+= offset

                self.params.descrete_equations.q[id]+= q

    def solve(self):
        f= np.concatenate((self.params.descrete_equations.f, self.params.descrete_equations.q))
        
        size = self.params.descrete_equations.G.shape
        a = np.hstack((self.params.descrete_equations.K_stiff, self.params.descrete_equations.G))
        b= np.hstack((self.params.descrete_equations.G.T, np.zeros((size[1], size[1]))))
        K = np.vstack((a,b))

        d = np.linalg.solve(K,f)
        u = d[0:self.params.mesh.num_nodes*self.params.mesh.num_dims]
        self.params.post_processing.disp = u 

    def get_displacement(self, coord):
        num_dims= self.params.mesh.num_dims

        g_point= get_gauss(coord, self.params)
        phi, dphidx, dphidy = shape(g_point, self.params)
        id_x = self.get_indices_voigt(g_point)

        disp = np.empty(num_dims)
        for i in range(num_dims):
            disp[i]= np.sum(phi[:, None]*self.params.post_processing.disp[id_x][i::num_dims])
        return disp

    def get_stress(self, coord):
        g_point= get_gauss(coord, self.params)
        phi, dphidx, dphidy = shape(g_point, self.params)
        Bmat = self.get_Bmat(g_point, dphidx, dphidy)
        id_x= self.get_indices_voigt(g_point)
        stress = np.matmul(self.params.material.Dmat, Bmat.dot(self.params.post_processing.disp[id_x]))
        return stress
        

    def get_indices_voigt(self, g_point):
        id= np.repeat(g_point.support_nodes, self.params.mesh.num_dims)
        id*= self.params.mesh.num_dims
        offset= np.tile(range(self.params.mesh.num_dims), g_point.len)  # offset= [0,1,2,0,1,2...]
        id+= offset
        return id

    @staticmethod
    def get_Bmat(g_point, dphidx, dphidy):
        Bmat = np.zeros((3, 2*g_point.len))
        Bmat[0, 0::2]= dphidx
        Bmat[1, 1::2]= dphidy
        Bmat[2, 0::2] = dphidy
        Bmat[2, 1::2] = dphidx     
        return Bmat
    
    @staticmethod
    def get_Nmat(lsf):
        Nmat = np.zeros((2, len(lsf)*2))
        Nmat[0, 0::2] = lsf 
        Nmat[1, 1::2] = lsf
        return Nmat

    def calculate_disp_exact(self):
        mesh = self.params.mesh.mesh
        model = self.params.geometry.model

        disp_ex= np.empty((len(mesh.points),2))
        for i, point in enumerate(mesh.points):
            disp_ex[i]= model.get_disp_exact(point, self.params.material)
        return disp_ex.flatten()

    def calculate_energy_norm(self):
        size = self.params.quadrature.gauss_internal.shape

        inv_Dmat= np.linalg.inv(self.params.material.Dmat)
        e_norm= 0.0
        for quad in self.params.quadrature.gauss_internal.reshape(-1, *size[2:]):
            g_point= get_gauss(quad, self.params)
            sigma= self.get_stress(quad)
            sigma_ex = self.params.geometry.model.get_stress_exact(quad[0:self.params.mesh.num_dims])
            
            error= sigma - sigma_ex
            norm= np.dot(inv_Dmat, error)
            norm= 0.5* g_point.jac_det* g_point.weight* np.inner(norm, error)
         
            e_norm+= norm
        e_norm = np.sqrt(e_norm)
        return e_norm

    def simulation_automatic(self):
        self.initialize()
        self.create_geometry()
        self.create_mesh()
        self.init_material_properties()
        self.set_nodes_domain()
        self.assemble_descrete_equations()
        self.solve()
        self.visualize()
        if self.params.post_processing.calculate_energy_norm:
            return self.calculate_energy_norm()


    def visualize(self):
        if self.geometry_params.visualise: self.geom.visualise()

    def save(self):
        if self.params.save_mesh: 
            if self.geometry_params.save_geometry_path is None: self.geometry_params.save_geometry_path= "./newGeometry.msh"
            self.geom.save_geometry(self.geometry_params.save_geometry_path)




        # tags= self.geom.get_entities_for_physicalgroup_name("Displacement")
        # gg= self.geom.get_quadrature(1,"CompositeGauss5", tags, '2D')
        # gg1= self.geom.get_quadrature(3,"CompositeGauss5", geomtype='2D')
        # self.geom.set_opt("Mesh.NodeLabels", 1) #set node labels to be visible
        # self.geom.set_opt("Geometry.CurveLabels", 1) #set curve labels to be visible

