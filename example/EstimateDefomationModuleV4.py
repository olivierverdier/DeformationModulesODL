#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:52:24 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:54:57 2017

@author: bgris
"""


import odl
from odl.deform.linearized import _linear_deform
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
import numpy as np
import scipy

def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='numpy')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel

space=odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')
#
## For ellipse mouvement
#a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
#b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
#c0_list=0.1*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
#c1_list=0.1*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
#theta=10*np.array([np.pi, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0, 0, 0,0.1*np.pi,-0.2*np.pi,0])
#fac=0.3
#nb_ellipses=len(a_list)
#images_ellipses_source=[]
#images_ellipses_target=[]
#for i in range(nb_ellipses):
#    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[0], c1_list[0], theta[i]]]
#    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
#    ellipses=[[1,fac* (a_list[i]+0.2), fac*(b_list[i]), c0_list[0], c1_list[0], theta[i]]]
#    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())

#for i in range(nb_ellipses):
#    images_ellipses_source[i].show('source {}'.format(i))
#    images_ellipses_target[i].show('target {}'.format(i))
#

# For rotation mouvement
a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
c0_list=0.0*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
c1_list=0.0*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
theta_init=50*np.array([0, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0,  -0.25*np.pi,  0.5*np.pi,0.1*np.pi,-0.2*np.pi,0])
fac=0.3
nb_ellipses=len(a_list)
images_ellipses_source=[]
images_ellipses_target=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]+5]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())




# For rotation mouvement, with a fixed surrounding ellipse to force locality
a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
c0_list=0.0*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
c1_list=0.0*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
theta_init=50*np.array([0, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0,  -0.25*np.pi,  0.5*np.pi,0.1*np.pi,-0.2*np.pi,0])
fac=0.3
nb_ellipses=len(a_list)
images_ellipses_source=[]
images_ellipses_target=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[i]+5],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())





def fun_u_theta(theta):

    return np.array([np.cos(theta), np.sin(theta)]).copy()

def fun_v_theta(theta):

    return np.array([-np.sin(theta) , np.cos(theta)]).copy()


def fun_f(z):
    no=sum(z[i]**2 for i in range(len(z)))
    return (z/no).copy()

def fun_f_diff (z, deltaz):
    no=sum(z[i]**2 for i in range(len(z)))
    u=deltaz.copy()
    u-=sum([z[i]*deltaz[i] for i in range(len(z))])*(2/np.sqrt(no))*z
    return (u/no).copy()

def fun_g(z):
    no=sum(z[i]**2 for i in range(len(z)))
    return (z/np.sqrt(no)).copy()

def fun_g_diff(z,i):
    no=sum(z[i]**2 for i in range(len(z)))
    u=np.zeros_like(z).copy()
    u[i]=(1/np.sqrt(no))
    u= (u- (z[i] /(np.sqrt(no)**3))*z).copy()

    return u.copy()

def fun_g_perp(z):
    u=fun_g(z).copy()
    v=np.empty_like(u)
    v[0]=-u[1]
    v[1]=u[0]

    return v.copy()

def fun_g_perp_diff(z,i):
    u=fun_g_diff(z,i).copy()
    v=np.empty_like(u)
    v[0]=-u[1]
    v[1]=u[0]

    return v.copy()



def fun_alpha(b,theta,points):
    u_theta=fun_u_theta(theta).copy()

    points_dec=(points-b).copy()
    I=space.element(sum(points_dec.T[i]*u_theta[i] for i in range(len(u_theta))))

    return I

def fun_beta(b,theta,points):
    v_theta=fun_v_theta(theta).copy()

    points_dec=(points-b).copy()
    I=space.element(sum(points_dec.T[i]*v_theta[i] for i in range(len(v_theta))))

    return I



def fun_alpha_diff(b,theta,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt b or theta
    #diff_ind[1] determins wrt which component we differentiate(not important for theta)
    I=space.zero()
    if (diff_ind[0]==0):
        # derivate wrt b
        I=space.zero()
        if(diff_ind[0]==0):
            I=(I- np.cos(theta)).copy()
        else:
            I=(I- np.sin(theta)).copy()
    else:
       theta_rot=theta + 0.5*np.pi
       I=fun_alpha(b,theta_rot,points).copy()

    return I.copy()



def fun_beta_diff(b,theta,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt b or theta
    #diff_ind[1] determins wrt which component we differentiate(not important for theta)
    I=space.zero()
    if (diff_ind[0]==0):
        # derivate wrt b
        I=space.zero()
        if(diff_ind[0]==0):
            I=(I+ np.sin(theta)).copy()
        else:
            I=(I- np.cos(theta)).copy()
    else:
       theta_rot=theta+0.5*np.pi
       I=fun_beta(b,theta_rot,points).copy()

    return I.copy()
#



b=np.array([0.0,1.0])
theta=0

alph=fun_alpha(b,theta,space.points())
alph_diff=fun_alpha_diff(b,theta,space.points(),[0,0])
#alph.show('alpha')
#alph_diff.show('alpha diff')

v=[]
v.append(space.zero())
v.append(-0.5*space.one())
h=1
#
#def Compute_4_Vectors(theta):
#    vectors=[]
#    vectors.append(fun_g(a-c).copy())
#    vectors.append(fun_g_perp(a-c).copy())
#    vectors.append(fun_g(b-d).copy())
#    vectors.append(fun_g_perp(b-d).copy())
#
#    return vectors.copy()
##

def ComputeLocTrans(c,alpha,kernel):
    #Computes the vector fields equal to the local translations centred at c, of vector alpha
    points=space.points()
    local=space.element([kernel([points[k][u]-c[u] for u in range(len(c))]) for k in range(len(points))])

    trans=space.tangent_bundle.zero()
    for u in range(len(alpha)):
        trans[u]=alpha[u]
    trans*=local.copy()

    return trans.copy()
#





#%% Here we constrain the reference vector field to be generated by k0 translations
# Besides we impose that the sum of centre is 0 and sum of vectors is (1,0)


def ComputeVectorFieldV4(c,theta,h,xlist,alphalist,kernel):
    # (h,theta,c) is in DSE (c is a point, h and theta are scalars)
    # xlist is a list of k0-1 points (lists)
    # alphalist is a list of k0-1 vectors (lists)

    numtrans=len(xlist)+1

    # we define the used centre and vectors by adding the k0-th one
    xlist_tot=xlist.copy()
    alphalist_tot=alphalist.copy()

    alphasum=[1,0]
    xlist_tot.append([sum([-xlist[u][v] for u in range(numtrans-1)]) for v in range(len(c))])
    alphalist_tot.append([alphasum[v]-sum([alphalist[u][v] for u in range(numtrans-1)]) for v in range(len(c))])

    # deform xlist and alphalist by the deformation of (h,theta,c)
    xlist_def=[]
    alphalist_def=[]

    utheta=fun_u_theta(theta)
    vtheta=fun_v_theta(theta)

    for u in range(numtrans):
        xlist_def.append([c[v]+xlist_tot[u][0]*utheta[v]+xlist_tot[u][1]*vtheta[v] for v in range(len(c))])
        alphalist_def.append([h*alphalist_tot[u][0]*utheta[v]+h*alphalist_tot[u][1]*vtheta[v] for v in range(len(c))])


    vect_field=space.tangent_bundle.element(sum([
            ComputeLocTrans(xlist_def[u],alphalist_def[u],kernel) for u in range(numtrans)]))


    return vect_field.copy()
#

def diff_vectfield(vectfield,x,deltax):
    deltah=vectfield.space[0].cell_sides[0]
    x_depl=np.array(x+deltah*deltax)#[u] for u in range(len(x))])

    diff=np.array([vectfield[u].interpolation(x_depl)-vectfield[u].interpolation(x) for u in range(len(x))])

    return diff.copy()/deltah



def DvectfieldT(vectfield,x):
    # x is a list of points
    # output is a list of np.array of size 2x2 with the transposed differential matrices
    dim=len(vectfield)
    nbpts=len(x)
    deltah0=vectfield.space[0].cell_sides[0]
    deltah1=vectfield.space[0].cell_sides[1]
    x_depl0=np.array(x).copy()+np.array([deltah0,0])
    x_depl1=np.array(x).copy()+np.array([0,deltah1])

    interp_x=np.array([vectfield[u].interpolation(np.array(x).T) for u in range(dim)]).T
    interp_depl0=np.array([vectfield[u].interpolation(x_depl0.T) for u in range(dim)]).T
    interp_depl1=np.array([vectfield[u].interpolation(x_depl1.T) for u in range(dim)]).T

    DmatT=[]

    for u in range(nbpts):
        DmatT.append(np.array([(interp_depl0[u]-interp_x[u])/deltah0,(interp_depl1[u]-interp_x[u])/deltah1]).T.copy())



    return DmatT.copy()
#



#%%
def energyVectField(source_list, target_list,kernel, forward_op,norm, X, lamv):
    # X is a list of vector fields (same size as source_list and target_list)


    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)



    energy1=0
    energy2=0
    nb_data=len(source_list)
    for i in range(nb_data):


        if (i==0):
            for j in range(len(X)):
                temp=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(X[j]) * ft_kernel_fitting).copy()
                energy1+=temp.inner(X[j])

        temp=_linear_deform(source_list[i],-X[i]).copy()
        energy2+=norm(forward_op(temp)-target_list[i])

    print(" energy V = {},  energy attach = {} ".format(lamv*energy1, energy2))
    energy= lamv*energy1+ energy2

    return energy

def energyVectField_gradient(source_list, target_list,kernel, forward_op,norm, X, lamv):
    # X is a list of vector fields (same size as source_list and target_list)

    grad_X=[]

    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    grad_op = Gradient(domain=space, method='forward', pad_mode='symmetric')

    nb_data=len(source_list)

    for i in range(nb_data):
        # Gradient of function : I \mapsto norm(T(I) -target[i] )  taken in source[i] transforme by vect_field_data[i]
        grad_S_i=(norm*(forward_op - target_list[i])).gradient(_linear_deform(source_list[i],-X[i]).copy()).copy()
        grad_source_i=grad_op(source_list[i]).copy()
        grad_source_i_depl=space.tangent_bundle.element([
                _linear_deform(grad_source_i[d],-X[i]).copy() for d in range(dim)].copy())

        tmp=grad_source_i_depl.copy()
        for u in range(dim):
            tmp[u]*= grad_S_i

        tmp3=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp) * ft_kernel_fitting).copy()

        grad_X.append(2*lamv*X[i].copy() - tmp3.copy())

    return grad_X.copy()
#
#%%


forward_op = odl.IdentityOperator(space)

nb_data=10

source_list=[]
target_list=[]

for i in range(nb_data):
    source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_source[i].copy(),1.5)))
    target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_target[i].copy(),1.5)))

#source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[3],3)))
#target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[4],3)))
#%%
norm = odl.solvers.L2NormSquared(forward_op.range)
#import random
X_init=[[],[],[],[],[],[]]
nb_vect_fields=1

X_init[0]=[space.zero() for uu in range(2)]
temph=[]
tempb=[]
temptheta=[]
tempd=[]
for k in range(nb_data):
    temph.append(1)
    tempb.append(np.array([0.0,0.0]))
    temptheta.append(0.0)

X_init[1]=temph.copy()
X_init[2]=tempb.copy()
X_init[3]=temptheta.copy()
#X_init[3]=[0.5*theta_init[u]*np.pi/180 for u in range(len(theta_init))].copy()

#energy(source_list, target_list,kernel, forward_op,norm, X)
#grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X)


# The parameter for kernel function
sigma = 0.5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))
#



#%% Gradient descent
lamh=1e-5
lamv=1*1e-1
lamo=1e-5
X=[space.tangent_bundle.zero() for u in range(len(source_list))]
ener=energyVectField(source_list, target_list,kernel, forward_op,norm, X, lamv)
print('Initial energy = {}'.format(ener))
niter=200
eps=0.02
eps0=eps
eps1=eps
eps2=eps
eps3=eps
#%%
cont=0
for i in range(niter):
    if (cont==0):
        grad=energyVectField_gradient(source_list, target_list,kernel, forward_op,norm, X, lamv)

    X_temp=[X[u].copy()-eps*grad[u].copy() for u in range(len(source_list))]


    ener_temp=energyVectField(source_list, target_list,kernel, forward_op,norm, X_temp, lamv)

    if (ener_temp<ener):
        X=X_temp.copy()
        ener=ener_temp
        print('Iter = {},  energy = {},  eps={}  '.format(i,ener,eps))
        eps*=1.2
        cont=0
    else:
        cont=1
        eps*=0.8

vectorfield_list=X.copy()
#
#%%
#source_list=source_list[1:3]
#target_list=target_list[1:3]
#lamh=1
#lamo=1
#lamv=1
#h_list=[1,1]
#b_list=[b,b]
#theta_list=[0,0]
#v=[]
#v.append(space.zero())
#v.append(space.zero())
#X=[v,h_list,b_list, theta_list]
#
##vectorfield_list=[]
#for i in range(nb_data):
#    vect=space.tangent_bundle.zero()
#    vectorfield_list.append(vect.copy())
tempx=[]
tempc=[]
tempalpha=[]
temptheta=[]
temph=[]
for k in range(nb_data):
    temph.append(1)
    tempc.append(np.array([0.0,0.0]))
    temptheta.append(0.0)

k0=4
tempx=np.array([[0.0,-3.0],[0.0,0.0],[0.0,3.0]])
for k in range(k0-1):
    tempalpha.append(np.array([0.0,0.0]))

X_init=[]
X_init.append(tempx.copy())
X_init.append(tempalpha.copy())
X_init.append(temph.copy())
X_init.append(temptheta.copy())
X_init.append(tempc.copy())
X=X_init.copy()
#%%

def energyV4(vectorfield_list,kernel,X):
    # X is a list containing (it this order) xlist which is a list of k0-1 points,
    # alphalist which is a list of k0-1 vectors,
    # h_list which is a list of scalars
    # theta_list which is a list of angles, c_list which is a list of points

    xlist=np.array(X[0]).tolist()
    alphalist=np.array(X[1]).tolist()
    hlist=np.array(X[2]).tolist()
    thetalist=np.array(X[3]).tolist()
    clist=np.array(X[4]).tolist()
    nbdata=len(vectorfield_list)
    energy=0

    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)



    for i in range(nbdata):
        vect_i=ComputeVectorFieldV4(clist[i],thetalist[i],hlist[i],xlist,alphalist,kernel).copy()
        diff=(vectorfield_list[i] - vect_i).copy()
        diff_V=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(diff) * ft_kernel_fitting).copy()
        energy+=diff.inner(diff_V)

    return energy



def energyV4_gradient(vectorfield_list,kernel,X):
    # X is a list containing (it this order) xlist which is a list of k0-1 points,
    # alphalist which is a list of k0-1 vectors,
    # h_list which is a list of scalars
    # theta_list which is a list of angles, c_list which is a list of points

    xlist=np.array(X[0]).tolist()
    alphalist=np.array(X[1]).tolist()
    hlist=np.array(X[2]).tolist()
    thetalist=np.array(X[3]).tolist()
    clist=np.array(X[4]).tolist()

    dim=2
    numtrans=len(xlist)+1
    # we define the used centre and vectors by adding the k0-th one
    xlist_tot=np.array(xlist.copy()).tolist()
    alphalist_tot=np.array(alphalist.copy()).tolist()

    alphasum=[1,0]
    xlist_tot.append([sum([-xlist[u][v] for u in range(numtrans-1)]) for v in range(dim)])
    alphalist_tot.append([alphasum[v]-sum([alphalist[u][v] for u in range(numtrans-1)]) for v in range(dim)])


    gradx=np.zeros_like(xlist).tolist()
    gradalpha=np.zeros_like(alphalist).tolist()
    gradh=[]
    gradtheta=np.zeros_like(thetalist).tolist()
    gradc=np.zeros_like(clist).tolist()

    k0=len(xlist)+1

    nb_data=len(vectorfield_list)

    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    list_vect_field_data=[]

    for i in range(nb_data):

        vect_field_i=ComputeVectorFieldV4(clist[i],thetalist[i],hlist[i],xlist,alphalist,kernel).copy()

        list_vect_field_data.append(vect_field_i.copy())



    for i in range(nb_data):
        tmp=ComputeVectorFieldV4(clist[i],thetalist[i],1.0,xlist,alphalist,kernel).copy()
        tmp2=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp) * ft_kernel_fitting).copy()

        gradh.append(2*tmp2.inner(list_vect_field_data[i]))
        thetai=thetalist[i]
        ci=clist[i]
        diff_vect_i=(list_vect_field_data[i]-vectorfield_list[i]).copy()

        # points in xlist transformed by (theta,c)
        xtransf=np.array([ [np.cos(thetai)*xlist_tot[k][0] - np.sin(thetai)*xlist_tot[k][1] +ci[0],  np.sin(thetai)*xlist_tot[k][0] + np.cos(thetai)*xlist_tot[k][1] +ci[1]] for k in range(k0) ])
        alphatransf=hlist[i]*np.array([ [np.cos(thetai)*alphalist_tot[k][0] - np.sin(thetai)*alphalist_tot[k][1],  np.sin(thetai)*alphalist_tot[k][0] + np.cos(thetai)*alphalist_tot[k][1]] for k in range(k0) ])

        vect_x=np.array([diff_vect_i[u].interpolation(xtransf.T) for u in range(dim)]).T
        DvectiT=DvectfieldT(diff_vect_i,xlist_tot)

        for k in range(k0-1):
            gradx[k]+=2*np.dot(DvectiT[k],alphatransf[k]).copy()
            gradx[k]-=2*np.dot(DvectiT[k0-1],alphatransf[k0-1]).copy()
            vect_appl_k=(vect_x[k]-vect_x[k0-1]).copy()
            gradalpha[k]+=2*hlist[i]*np.array([np.cos(thetai)*vect_appl_k[0] +  np.sin(thetai)*vect_appl_k[1] , -np.sin(thetai)*vect_appl_k[0] +  np.cos(thetai)*vect_appl_k[1] ]).copy()
            gradc[i]+=2*np.dot(DvectiT[k],alphatransf[k]).copy()
            gradtheta[i]+=2*np.dot(np.dot(DvectiT[k],alphatransf[k]),alphatransf[k]).copy()
            alphakdeplinv=hlist[i]*np.array( [np.cos(thetai)*alphalist_tot[k][0] + np.sin(thetai)*alphalist_tot[k][1],  -np.sin(thetai)*alphalist_tot[k][0] + np.cos(thetai)*alphalist_tot[k][1]]).copy()
            gradtheta[i]+=2*np.dot(vect_x[k],alphakdeplinv)

        gradc[i]+=2*np.dot(DvectiT[k0-1],alphatransf[k0-1])
        alphakdeplinv=hlist[i]*np.array( [np.cos(thetai)*alphalist_tot[k0-1][0] + np.sin(thetai)*alphalist_tot[k0-1][1],  -np.sin(thetai)*alphalist_tot[k0-1][0] + np.cos(thetai)*alphalist_tot[k0-1][1]]).copy()
        gradtheta[i]+=2*np.dot(vect_x[k0-1],alphakdeplinv)

    return [gradx,gradalpha,gradh,gradtheta,gradc]

#
#%%
evpoints=np.array([[-16.0, -16.0], [16.0, 16.0]])

discr=odl.uniform_discr([0, 0], [2, 5], evpoints.shape)
grad = Gradient(discr)
#%% Gradient descent
lamh=1e-5
lamv=1*1e-1
lamo=1e-5
X=X_init.copy()
ener=energyV4(vectorfield_list,kernel)
print('Initial energy = {}'.format(ener))
niter=200
eps=0.02
eps0=eps
eps1=eps
eps2=eps
eps3=eps
eps4=eps
#%%

cont=0
for i in range(niter):
    if (cont==0):
        grad=energyV4_gradient(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo)

    X_temp=X.copy()
    X_temp[0]=[np.array(X[0][uu]) - eps0*np.array(grad[0][uu]) for uu in range(len(X[0]))].copy()
    X_temp[1]=[np.array(X[1][uu]) - eps1*np.array(grad[1][uu]) for uu in range(len(X[1]))].copy()
    X_temp[2]=[X[2][uu]-grad[2][uu] for uu in range(len(X[2]))]
    X_temp[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(len(X[3]))]
    X_temp[4]=[np.array(X[4][uu]) - eps4*np.array(grad[4][uu]) for uu in range(len(X[4]))]

    ener_temp=energyV4(vectorfield_list,kernel,X_temp)

    if (ener_temp<ener):
        X=X_temp.copy()
        ener=ener_temp
        print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={}  '.format(i,ener,eps0,eps1,eps2))
        eps0*=1.2
        eps1*=1.2
        eps2*=1.2
        eps3*=1.2
        cont=0
    else:
        X_temp0=X.copy()
        X_temp[0]=[np.array(X[0][uu]) - eps0*np.array(grad[0][uu]) for uu in range(len(X[0]))].copy()
        X_temp[1]=[np.array(X[1][uu]) - eps1*np.array(grad[1][uu]) for uu in range(len(X[1]))].copy()
        X_temp[2]=[X[2][uu]-grad[2][uu] for uu in range(len(X[2]))]
        X_temp[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(len(X[3]))]
        X_temp[4]=[np.array(X[4][uu]) - eps4*np.array(grad[4][uu]) for uu in range(len(X[4]))]
        ener_temp0=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp0, lamh, lamv, lamo)

        X_temp1=X.copy()
        X_temp1[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp1[1]=[X[1][uu] - 0.5*eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp1[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp1[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp1=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp1, lamh, lamv, lamo)

        X_temp2=X.copy()
        X_temp2[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp2[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp2[2]=[np.array(X[2][uu])- 0.5*eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp2[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp2=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp2, lamh, lamv, lamo)

        X_temp3=X.copy()
        X_temp3[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp3[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp3[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp3[3]=[X[3][uu] - 0.5*eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp3=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp3, lamh, lamv, lamo)

        if (ener_temp0 < ener_temp1 and ener_temp0 < ener_temp2 and ener_temp0 < ener_temp3):
            X_temp=X_temp0.copy()
            eps0*=0.5
            ener_temp=ener_temp0
        else:
            if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2 and ener_temp1 < ener_temp3):
                X_temp=X_temp1.copy()
                eps1*=0.5
                ener_temp=ener_temp1
            else:
                if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2 and ener_temp1 < ener_temp3):
                    X_temp=X_temp2.copy()
                    eps2*=0.5
                    ener_temp=ener_temp2
                else:
                    X_temp=X_temp3.copy()
                    eps3*=0.5
                    ener_temp=ener_temp3

        if (ener_temp<ener):
            X=X_temp.copy()
            ener=ener_temp
            print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={} , eps3={}'.format(i,ener,eps0,eps1,eps2,eps3))
            eps0*=1.2
            eps1*=1.2
            eps2*=1.2
            eps3*=1.2
            cont=0

        else:
            cont=1
            eps0*=0.5
            eps1*=0.5
            eps2*=0.5
            eps3*=0.5


        print('Iter = {},  eps = {}'.format(i,eps))
#
#%%
nb_datamax=nb_data

import matplotlib.pyplot as plt
for n in range(nb_datamax):
    space.element(source_list[n]).show('Source {}'.format(n))
    #space.element( source_list[n] - target_list[n]).show('Initial difference {}'.format(n))
    vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
    temp=_linear_deform(source_list[n],-vect_field_n).copy()
    (space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
    #(space.element(temp)).show('Transported source {}'.format(n))
    points=space.points()
    v=vect_field_n.copy()
#    plt.figure()
#    plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
#    plt.axis('equal')
#    plt.title('v n {}'.format(n))
    #vect_field_n.show('Vector field {}'.format(n))
#
#%%
import matplotlib.pyplot as plt
n=1
source_list[n] .show('source {}'.format(n))
vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
temp=_linear_deform(source_list[n],-vect_field_n).copy()
#(space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
(space.element(temp)).show('Transported source {}'.format(n))
points=space.points()
v=vect_field_n.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v n {}'.format(n))

#%% plot reference
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],X[0][0][::20],X[0][1][::20])
plt.axis('equal')
plt.title('Reference')


#%%

for i in range(2):
    X[0][i].show('{}'.format(i))
#
#%% Save vector field estimated
#np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid',X[0])

np.savetxt('/home/bgris/DeformationModulesODL/deform/vect_field_rotation_mvt_V3_sigma_2_lamv_1e__3_lamh_1e__5_lamo_1e__5',space.tangent_bundle.element(X[0]))

vect_field=space.tangent_bundle.element(np.loadtxt('/home/bgris/DeformationModulesODL/deform/vect_field_V3_sigma_2_lamv_1e__3_lamh_1e__5_lamo_1e__5')).copy()

import matplotlib.pyplot as plt
points=space.points()
#v=X[0]
v=vect_field_n.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v n')

#%%
for u in range(2):
    X[0][u].show('{}'.format(u))
#

#%%

n=0
ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[n]],
          [1,0.1, 0.1, -0.3, 0.3, theta_init[i]],
          [1,0.1, 0.1, -0.5, -0.5, theta_init[i]],
          [1,0.1, 0.1, 0.5, 0.5, theta_init[i]],
          [1,0.1, 0.1, 0.5, -0.5, theta_init[i]]]

ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[n]+5],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[n]]]

source_test=odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy()
source_test.show()
source_list[n] .show('source {}'.format(n))
vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
temp=_linear_deform(source_test,-vect_field_n).copy()
space.element(temp).show()


