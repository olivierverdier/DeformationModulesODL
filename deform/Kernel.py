#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:54:53 2017

@author: bgris
"""


import numpy as np
__all__ = ('GaussianKernel', )

class GaussianKernel(object):
    def __init__(self,scale):
        self.scale=scale

    def Eval(self,x):
        scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x]
        return np.exp(-sum(scaled))

    @property
    def derivative(self):
        ker=self
        class ComputeDer(object):
            def __init__(self,x0):
                self.x0=x0
            def Eval(self,dx):
                a=ker.Eval(self.x0)
                b=[-xi*dxi/( ker.scale ** 2) for xi, dxi in zip(self.x0,dx) ]
                return a*sum(b)
        return ComputeDer


    def gradient(self,x0):
        a=self.Eval(x0)
        return -(1/(self.scale ** 2))*a*x0