#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:16:55 2017

@author: bgris
"""
import odl
import numpy as np

##%% Create data from lddmm registration
import matplotlib.pyplot as plt

from deform import Kernel
from deform import DeformationModuleAbstract
from deform import SumTranslations
from deform import TemporalAttachmentModulesGeom

##%% Generate data

I0name = '/home/bgris/Downloads/pictures/i_highres.png'
I1name = '/home/bgris/Downloads/pictures/c_highres.png'


# Get digital images
I0 = plt.imread(I0name)
I1 =plt.imread(I1name)

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
    dtype='float32', interp='linear')

I0=space.element(I0)
I1=space.element(I1)
# Give the number of directions
num_angles = 2

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')


# Create projection data by calling the ray transform on the phantom
proj_data = forward_op(I1)




##%% Define Module

dim=2
Ntrans=25

kernel=Kernel.GaussianKernel(2)
translation=SumTranslations.SumTranslations(space, Ntrans, kernel)

Module=translation

#Module=DeformationModuleAbstract.Compound([translation,translation])


##%% Define functional
lam=0.001
nb_time_point_int=10
template=I0


##data_time_points=np.array([0,0.5,0.8,1])
#data_time_points=np.array([0,0.2,0.4,0.6,0.8,1])
#data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
#data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[10]),
#              forward_op(image_N0[0]),forward_op(image_N0[10]),
#              forward_op(image_N0[0]),forward_op(image_N0[10])])
##data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[5]),
##              forward_op(image_N0[8]),forward_op(image_N0[10])])
#
#forward_operators=[forward_op,forward_op,forward_op,forward_op,
#                   forward_op, forward_op, forward_op]
#data_image=[(image_N0[0]),(image_N0[10]),
#              (image_N0[0]),(image_N0[10]),
#              (image_N0[0]),(image_N0[10])]

data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([proj_data])
forward_operators=[forward_op]
data_image=[I1]


Norm=odl.solvers.L2NormSquared(forward_op.range)



functional = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lam, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)

GD_init=Module.GDspace.element([[-4,0], [-2,0], [0,0], [2,0], [4,0],[-4,2], [-2,2], [0,2], [2,2], [4,2],[-4,4], [-2,4], [0,4], [2,4], [4,4],[-4,-2], [-2,-2], [0,-2], [2,-2], [4,-2],[-4,-4], [-2,-4], [0,-4], [2,-4], [4,-4]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

#%%
energy=functional([GD_init,Cont_init])

#%%

grad=functional.gradient([GD_init,Cont_init])


#%%




#%% Gradient descente
niter=100
eps = 0.001
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
print(" Initial ,  attachment term : {}".format(attachment_term))

energy=functional(X)+1
for k in range(niter):
    grad=functional.gradient(X)
    #X[1]= (X[1]- eps *grad[1]).copy()
    energytemp=functional(X)
    if energytemp< energy:
        X= (X- eps *grad).copy()
        energy = energytemp
        print(" iter : {}  ,  energy : {}".format(k,energy))
    else:
        eps = eps/2
        print(" iter : {}  ,  epsilon : {}".format(k,eps))
#
#%%  see result
I_t=template
GD_t=functional.ComputetrajectoryGD(X)
I_t.show('time {}'.format(0))
for t in range(nb_time_point_int+1):
    vect_field=-Module.ComputeField(GD_t[t],X[1][t]).copy()
    deform_op = odl.deform.LinDeformFixedTempl(I_t)
    I_t = deform_op(vect_field)
    I_t.show('time {}'.format(t+1))
#



#%%
data_image[0].show()

((template-data_image[0])**2).show('Initial difference')
((I_t-data_image[0])**2).show('Final difference')


((forward_op(template-data_image[0]))**2).show('Initial difference')
((forward_op(I_t-data_image[0]))**2).show('Final difference')


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#DomainField=space
#
#GDspace=odl.ProductSpace(odl.space.rn(dim),Ntrans)
#Contspace=odl.ProductSpace(odl.space.rn(dim),Ntrans)
#
#vector_field=translation.ComputeField([[0,0],[0,1]],[[0,1],[0,2]])
#eva=translation.ComputeFieldevaluate([[0,0],[0,1]],[[0,1],[0,2]])
#eva(odl.space.rn(dim).element([0,0]))
#
#vector_field=translation.ComputeFieldDer([[-10,-10],[10,10]],[[0,1],[0,1]])([[1,0],[1,0]])
##vector_field.show()
#
#speed=translation.ComputeFieldDerevaluate([[-1,-1],[1,1]],[[0,1],[0,1]])([[[0,1],[0,1]],[0,1]])
#
#X=odl.ProductSpace(GDspace,DomainField.tangent_bundle).element([[[0,0],[0,1]],vector_field])
#
#speed=translation.ApplyVectorField(X[0],X[1])
#
#GD=GDspace.element([[-10,-10],[10,10]])
#Cont=Contspace.element([[0,1],[0,1]])
#appli=translation.ApplyModule(translation,GD,Cont)
#appli(GD)
#
#energy=translation.Cost(GD,Cont)
#energy=translation.CostDerGD(GD,Cont)(GD)
#energy=translation.CostDerCont(GD,Cont)(Cont)
#
#
#mod=[translation,translation,translation]
#GDspace=odl.ProductSpace(*[mod[i].GDspace for i in range(len(mod))])
#mod_new=Compound(mod)
#vector_field=mod_new.ComputeField(GD,Cont)
#eva=mod_new.ComputeFieldEvaluate(GD,Cont)([0,0])
#vector_field=mod_new.ComputeFieldDer(GD,Cont)(GD)
#eva=mod_new.ComputeFieldDerEvaluate(GD,Cont)([GD,[0,0]])
#appli=mod_new.ApplyVectorField(GD,vector_field)
#
#energy=mod_new.Cost(GD,Cont)
#energy=mod_new.CostDerGD(GD,Cont)(GD)
#energy=mod_new.CostDerCont(GD,Cont)(Cont)
#
#
#o=GDspace.element([[0,0]])
#h=Contspace.element([[0,5]])
#vector_field=DomainField.tangent_bundle.zero()
#Kernel=kernel
#
#
#mg = DomainField.meshgrid
#for i in range(Ntrans):
#    kern = Kernel([mgu - ou for mgu, ou in zip(mg, o[i])])
#    vector_field += DomainField.tangent_bundle.element([kern * hu for hu in h[i]])
#
#
#
#x=odl.space.rn(dim).element([0,0])
#speed=odl.space.rn(dim).zero()
#for i in range(Ntrans):
#    a=Kernel(o[i]-x)
#    speed+=a*h[i]
#
#
#
#
#class GaussianKernel(object):
#    def __init__(self,scale):
#        self.scale=scale
#
#    def Eval(self,x):
#        scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x]
#        return np.exp(-sum(scaled))
#
#    @property
#    def derivative(self):
#        ker=self
#        class ComputeDer(object):
#            def __init__(self,x0):
#                self.x0=x0
#            def Eval(self,dx):
#                a=ker.Eval(self.x0)
#                b=[-xi*dxi/( ker.scale ** 2) for xi, dxi in zip(self.x0,dx) ]
#                return a*sum(b)
#        return ComputeDer
#
#
#Kernel=GaussianKernel(2)
#Kernel.Eval([0,0])
#Kernel.derivative([0,0]).Eval([0,0])
#
#KernelEval=Kernel.Eval
#
#
#
#
#
#
#%%

template=I0
# Create a product space for displacement field
disp_field_space = space.tangent_bundle

# Define a displacement field that bends the template a bit towards the
# upper left. We use a list of 2 functions and discretize it using the
# disp_field_space.element() method.
sigma = 2
h=[10,0]
disp_func = [
    lambda x: h[0]* np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2)),
    lambda x: h[1] * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2))]

disp_field = disp_field_space.element(disp_func)


# Initialize the deformation operator with fixed template
deform_op = odl.deform.LinDeformFixedTempl(template)

# Apply the deformation operator to get the deformed template.
deformed_template = deform_op(disp_field)

template.show()
deformed_template.show()
proj_data = forward_op(deformed_template)
data_image=[deformed_template]