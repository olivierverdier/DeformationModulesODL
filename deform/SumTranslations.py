#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:47:09 2017

@author: bgris
"""


# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np

from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.deform.linearized import _linear_deform
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space.fspace import FunctionSpace
import odl
from odl.set import LinearSpace, LinearSpaceElement, Set, Field
from odl.deform import TemporalAttachmentLDDMMGeom
from odl.deform import ShootTemplateFromVectorFields
from odl.solvers.functional.functional import Functional
from deform.DeformationModuleAbstract import DeformationModule
__all__ = ('SumTranslations', 'SumTranslationsFourier')



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
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel



class SumTranslations(DeformationModule):
    def __init__(self,DomainField, Ntrans, Kernel):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Ntrans : number of translations
        Kernel : kernel, class that has at least methods Eval and derivative
        """

        self.Ntrans=Ntrans
        self.KernelClass=Kernel
        self.Kernel=Kernel.Eval
        self.dim=DomainField.ndim

        GDspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)
        Contspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)

        basis=[]
        for i in range(self.Ntrans):
            for d in range(self.dim):
                a=GDspace.zero()
                a[i][d]=1
                basis.append(a.copy())

        basisGD=basis.copy()
        basisCont=basis.copy()

        super().__init__(GDspace,Contspace,basisGD,basisCont,DomainField)



    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """
        if o not in self.GDspace:
            try:
                o = self.GDspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' o is not in `GDspace` instance'
                            '')

        if h not in self.Contspace:
            try:
                o = self.Contspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' h is not in `Contspace` instance'
                            '')

        vector_field=self.DomainField.tangent_bundle.zero()

        mg = self.DomainField.meshgrid
        for i in range(self.Ntrans):
            kern = self.Kernel([mgu - ou for mgu, ou in zip(mg, o[i])])
            vector_field += self.DomainField.tangent_bundle.element([kern * hu for hu in h[i]])

        return vector_field

    @property
    def ComputeFieldEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.space.rn(ope.dim), odl.space.rn(ope.dim),
                                 linear=False)


            def _call(self,x):
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Ntrans):
                    a=ope.Kernel(self.GD[i]-x)
                    speed+=a*self.Cont[i]

                return speed

        return Eval

    @property
    def ComputeFieldDer(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                vector_field=ope.DomainField.tangent_bundle.zero()

                mg = ope.DomainField.meshgrid
                for i in range(ope.Ntrans):
                    kern = ope.KernelClass.derivative([mgu - ou for mgu, ou in zip(mg, self.GD[i])])
                    vector_field += ope.DomainField.tangent_bundle.element([kern.Eval(dGD[i]) * hu for hu in self.Cont[i]])

                return vector_field

        return Eval



    @property
    def ComputeFieldDerEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.ProductSpace(ope.GDspace,odl.space.rn(ope.dim)), odl.space.rn(ope.dim),
                                 linear=True)


            def _call(self,X):
                dGD=ope.GDspace.element(X[0]).copy()
                x=odl.space.rn(ope.dim).element(X[1])
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Ntrans):
                    kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(x, self.GD[i])])
                    speed+=kern.Eval(dGD[i])*self.Cont[i]
                return speed

        return Eval

    def ApplyVectorField(self,GD,vect_field):
            #GD=self.GDspace.element(X[0])
            #vect_field=self.DomainField.tangent_bundle.element(X[1])
            speed=self.GDspace.element()
            for u in range(self.dim):
                for i in range(len(GD)):
                    speed[i][u]=vect_field[u].interpolation(GD[i])

            return speed

    @property
    def ApplyModule(self):
        ope = self
        class apply(Operator):
            def __init__(self,Module,GDmod,Contmod):
                self.apply_op=Module.ComputeFieldEvaluate(GDmod,Contmod)
                super().__init__(ope.GDspace, ope.GDspace,
                                 linear=False)

            def _call(self,GD):
                speed=ope.GDspace.element()
                for i in range(len(GD)):
                    speed[i]=self.apply_op(GD[i])
                return speed
        return apply

    def Cost(self,GD,Cont):
        energy=0
        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                prod=[hi*hj for hi, hj in zip(Cont[i],Cont[j])]
                energy+=self.Kernel(GD[i]-GD[j])*sum(prod)
        return energy


    @property
    def CostDerGD(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace,
                                 linear=True)
           def _call(self,dDG):
                dGD=ope.GDspace.element(dDG).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],self.Cont[j])]
                        kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(self.GD[j], self.GD[i])])
                        energy+=kern.Eval(dGD[j]-dGD[i])*sum(prod)

                return energy
        return compute



    @property
    def CostDerCont(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.Contspace,
                                 linear=True)
           def _call(self,dCont):
                dCont=ope.Contspace.element(dCont).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],dCont[j])]
                        energy+=ope.Kernel(self.GD[i]-self.GD[j])*2*sum(prod)

                return energy
        return compute




    def CostGradGD(self,GD,Cont):
        grad=self.GDspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                prod=[hi*hj for hi, hj in zip(Cont[i],Cont[j])]
                kern = self.KernelClass.gradient(odl.space.rn(self.dim).element([xd - ou for xd, ou in zip(GD[j], GD[i])]))
                grad[j]+=sum(prod)*kern.copy()

        return grad



    def CostGradCont(self,GD,Cont):
        grad=self.Contspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                grad[j]+=self.Kernel(odl.space.rn(self.dim).element(GD[i]-GD[j]))*2*Cont[i].copy()

        return grad



class SumTranslationsFourier(DeformationModule):
    def __init__(self,DomainField, Ntrans, Kernel):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Ntrans : number of translations
        Kernel : kernel, class that has at least methods Eval and derivative
        """

        self.Ntrans=Ntrans
        self.KernelClass=Kernel
        self.Kernel=Kernel.Eval
        self.dim=DomainField.ndim

        GDspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)
        Contspace=odl.ProductSpace(odl.space.rn(self.dim),self.Ntrans)

        basis=[]
        for i in range(self.Ntrans):
            for d in range(self.dim):
                a=GDspace.zero()
                a[i][d]=1
                basis.append(a.copy())

        basisGD=basis.copy()
        basisCont=basis.copy()
        super().__init__(GDspace,Contspace,basisGD,basisCont,DomainField)

    def ProjectGD(self,o,h):
        """Return a vector field equal to
        zero except on points of the grid the closest to GDs
        where it is equal to the sum of the corresponding controls
        """

        vect_field=[]
        for u in range(self.dim):
            vect_field.append(np.zeros(self.DomainField.shape))

        for i in range(self.Ntrans):
            indi = self.DomainField.partition.index(o[i])

            for u in range(self.dim):
                vect_field[u][indi]+= h[i][u]/self.DomainField.cell_volume

        return self.DomainField.tangent_bundle.element(vect_field)



    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """
        if o not in self.GDspace:
            try:
                o = self.GDspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' o is not in `GDspace` instance'
                            '')

        if h not in self.Contspace:
            try:
                o = self.Contspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' h is not in `Contspace` instance'
                            '')

        #vector_field=self.DomainField.tangent_bundle.zero()
        vect_fieldGDCont=self.ProjectGD(o,h)


        # FFT setting for data matching term, 1 means 100% padding
        padded_size = 2 * self.DomainField.shape[0]
        padded_ft_fit_op = padded_ft_op(self.DomainField, padded_size)
        vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * self.dim))


        # Compute the FT of kernel in fitting term
        discretized_kernel = fitting_kernel(self.DomainField, self.Kernel)
        ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)




        vector_field= (2 * np.pi) ** (self.dim / 2.0) * vectorial_ft_fit_op.inverse(
        vectorial_ft_fit_op(vect_fieldGDCont) *ft_kernel_fitting)

        return vector_field

    @property
    def ComputeFieldEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.space.rn(ope.dim), odl.space.rn(ope.dim),
                                 linear=False)


            def _call(self,x):
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Ntrans):
                    a=ope.Kernel(self.GD[i]-x)
                    speed+=a*self.Cont[i]

                return speed

        return Eval

    @property
    def ComputeFieldDer(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                vector_field=ope.DomainField.tangent_bundle.zero()

                mg = ope.DomainField.meshgrid
                for i in range(ope.Ntrans):
                    kern = ope.KernelClass.derivative([mgu - ou for mgu, ou in zip(mg, self.GD[i])])
                    vector_field += ope.DomainField.tangent_bundle.element([kern.Eval(dGD[i]) * hu for hu in self.Cont[i]])

                return vector_field

        return Eval



    @property
    def ComputeFieldDerEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.ProductSpace(ope.GDspace,odl.space.rn(ope.dim)), odl.space.rn(ope.dim),
                                 linear=True)


            def _call(self,X):
                dGD=ope.GDspace.element(X[0]).copy()
                x=odl.space.rn(ope.dim).element(X[1])
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Ntrans):
                    kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(x, self.GD[i])])
                    speed+=kern.Eval(dGD[i])*self.Cont[i]
                return speed

        return Eval

    def ApplyVectorField(self,GD,vect_field):
            #GD=self.GDspace.element(X[0])
            #vect_field=self.DomainField.tangent_bundle.element(X[1])
            speed=self.GDspace.element()
            for u in range(self.dim):
                for i in range(len(GD)):
                    speed[i][u]=vect_field[u].interpolation(GD[i])

            return speed

    @property
    def ApplyModule(self):
        ope = self
        class apply(Operator):
            def __init__(self,Module,GDmod,Contmod):
                self.apply_op=Module.ComputeFieldEvaluate(GDmod,Contmod)
                super().__init__(ope.GDspace, ope.GDspace,
                                 linear=False)

            def _call(self,GD):
                speed=ope.GDspace.element()
                for i in range(len(GD)):
                    speed[i]=self.apply_op(GD[i])
                return speed
        return apply

    def Cost(self,GD,Cont):
        energy=0
        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                prod=[hi*hj for hi, hj in zip(Cont[i],Cont[j])]
                energy+=self.Kernel(GD[i]-GD[j])*sum(prod)
        return energy


    @property
    def CostDerGD(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace,
                                 linear=True)
           def _call(self,dDG):
                dGD=ope.GDspace.element(dDG).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],self.Cont[j])]
                        kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(self.GD[j], self.GD[i])])
                        energy+=kern.Eval(dGD[j]-dGD[i])*sum(prod)

                return energy
        return compute



    @property
    def CostDerCont(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.Contspace,
                                 linear=True)
           def _call(self,dCont):
                dCont=ope.Contspace.element(dCont).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],dCont[j])]
                        energy+=ope.Kernel(self.GD[i]-self.GD[j])*2*sum(prod)

                return energy
        return compute




    def CostGradGD(self,GD,Cont):
        grad=self.GDspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                prod=[hi*hj for hi, hj in zip(Cont[i],Cont[j])]
                kern = self.KernelClass.gradient(odl.space.rn(self.dim).element([xd - ou for xd, ou in zip(GD[j], GD[i])]))
                grad[j]+=sum(prod)*kern.copy()

        return grad



    def CostGradCont(self,GD,Cont):
        grad=self.Contspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                grad[j]+=self.Kernel(odl.space.rn(self.dim).element(GD[i]-GD[j]))*2*Cont[i].copy()

        return grad

























