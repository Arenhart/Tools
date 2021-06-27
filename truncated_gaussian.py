# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:37:03 2017

@author: Arenhart
"""

import math
import random as rnd

import numpy as np
from scipy import optimize
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft

class TruncatedGaussian():
    
    def __init__(self, size = (256,256,256), lam = 5, porosity = 0.35):
        self.size = size
        self.lam = lam
        self.porosity = porosity
        self.N = math.ceil(math.log2(max(self.size)))
        self.truncated_field = None
    
    def adjust_lam(self, data_series):
        opt, _ = optimize.curve_fit(self.C, 
                                          range(len(data_series)), data_series)
        self.lam = opt[0]

    def C(self,u):
        return np.exp(-(u/self.lam))
    
    def generate_gaussian_field(self):
        '''
        Funcao principal que inicia o campo gaussiano na variavel self.field
        '''
        self.corr_array = np.zeros(self.size)
        self.adjusted_corr_array = np.zeros(self.size)
        half_x = self.size[0]//2
        half_y = self.size[1]//2
        half_z = self.size[2]//2
        for i in [(a,b,c) for a in range(self.size[0]) 
                        for b in range(self.size[1])
                        for c in range(self.size[2])]:
            j = math.sqrt((i[0]-half_x+1)**2 + 
                          (i[1]-half_y+1)**2 +
                          (i[2]-half_z+1)**2 )
            self.corr_array[i[0],i[1],i[2]] = self.C(j)
            self.adjusted_corr_array[i[0],i[1],i[2]] = self.C(j*0.6)
    
        self.W = np.random.rand(self.size[0],self.size[1],self.size[2])
        self.gamma = ifft(self.adjusted_corr_array)
        self.gamma = np.sqrt(self.gamma)
        self.phi = fft(self.W)
        self.phi = self.gamma * self.phi
        self.phi = ifft(self.phi)
        self.field = self.phi.real
        
    '''
    Codigo com algoritmo improrpio, nao utilizar
    
    def generate_field(self):
        
        old_array = np.zeros((1,1))
        
        for i in range(self.N):
            new_array = np.repeat(np.repeat(old_array,2,0),2,1)
            disp = self.C(2**(self.N-i)) - self.C(2**(self.N-i-1))
            print(i, self.N, disp)
            for x in np.nditer(new_array,
                               op_flags = ['readwrite',]):
                x += disp * (rnd.random()*2 - 1)
            old_array = new_array
            
        self.field = np.resize(old_array, self.size)
    '''
    
    def truncate_field(self):
        '''
        truncates 0 as pore and 1 as solid
        '''
        self.values = self.field.flatten()
        self.values.sort()
        threshold = self.values[int(self.values.size * self.porosity)]
        self.truncated_field = np.copy(self.field)
        self.truncated_field = self.truncated_field >= threshold
        self.truncated_field = self.truncated_field.astype('uint8')
        
    def test(self):
        '''
        generates correlation graphs for the gaussian field and the truncated
        gaussian field
        '''
        max_y = self.size[0]//2
        self.y = np.array(range(max_y))
        self.exact_correlation = np.array([self.C(u) for u in self.y])
        #flat_field = self.field.flatten()
        #flat_truncated_field = self.truncated_field.flatten()
        self.correlation = np.zeros(max_y)
        self.truncated_correlation = np.zeros(max_y)
        self.truncated_correlation[0] = (self.truncated_field.sum()/
                                          self.truncated_field.size)
        for i in range(1,max_y):
            self.correlation[i] = abs(self.field[i:,:,:]-
                                        self.field[:-i,:,:]).sum()
            self.truncated_correlation[i] = abs(
                    (self.truncated_field[:-i,:,:]*
                     self.truncated_field[i:,:,:]).sum()
                     /((self.size[0]-i)*self.size[1]*self.size[2]))
        self.correlation = abs(self.correlation/max(self.correlation) - 1)
        p = self.truncated_correlation[0]
        self.normalized_correlation = ((self.truncated_correlation-p**2)/
                                       (p - p**2))
    
    def cube_iterator(self, length):
        l = length
        it1 = [(i,j,k) for i in range(-l,(l+1))
                       for j in range(-l,(l+1))
                       for k in (-l,l)]
        it2 = [(i,j,k) for i in range(-l,(l+1))
                       for j in (-l,l)
                       for k in range(-(l-1),l)]
        it3 = [(i,j,k) for i in (-l,l)
                       for j in range(-(l-1),l)
                       for k in range(-(l-1),l)]
        
        return it1 + it2 + it3
    
    def coord_in_domain(self,coord):
        
        x,y,z = coord
        
        if x < 0: return False
        if y < 0: return False
        if z < 0: return False
        if x >= self.size[0]: return False
        if y >= self.size[1]: return False
        if z >= self.size[2]: return False
        
        return True
    
    def add_cube_layer(self,coord,r):
        '''
        returns 1 if added without touching an invalid location
        returns 0 otherwise
        '''
        for disp in self.cube_iterator(r):
            
            target = tuple([a[0]+a[1] for a in zip(coord,disp)])
            if not self.coord_in_domain(target): return 0
            if self.truncated_field[target] != 0: return 0
            self.truncated_field[target] = 2
            self.clay += 1
            if self.clay >= self.target_clay: return 0
            
        return 1
        
    
    def add_clay(self, fraction = 0.25, mode = "truncated range"):
        '''
        available methods: "truncated range", "compound maps", "added cubes"
                            "truncated solid"
        uses "2" for clay voxels
        '''
        self.clay_fraction = fraction
        
        if mode == 'truncated range':
            if type(self.truncated_field) == type(None):
                print('Instance must have a truncated field')
                return
            t_low = self.values[int(self.values.size * self.porosity)]
            t_high = self.values[int(self.values.size * self.porosity+
                                     self.values.size * self.clay_fraction)]
            for i in [(a,b,c) for a in range(self.size[0]) 
                            for b in range(self.size[1])
                            for c in range(self.size[2])]:
                if self.field[i] > t_low and self.field[i] <= t_high:
                    self.truncated_field[i] = 2
        
        elif mode == 'compound maps':
            clay_map = TruncatedGaussian(self.size, self.lam, fraction)
            clay_map.generate_gaussian_field()
            clay_map.truncate_field()
            clay_field = clay_map.truncated_field
            for i in [(a,b,c) for a in range(self.size[0]) 
                            for b in range(self.size[1])
                            for c in range(self.size[2])]:
                if clay_field[i] == 0:
                    self.truncated_field[i] = 2
        
        elif mode == 'added cubes':
            self.target_clay = self.field.size * fraction
            rnd_coords = lambda : (rnd.randrange(self.size[0]),
                                   rnd.randrange(self.size[1]),
                                   rnd.randrange(self.size[2]))
            coord = rnd_coords()
            while self.truncated_field[coord] != 0:
                coord = rnd_coords()
            
            self.clay = 0
            
            while True:
                self.truncated_field[coord] = 2
                self.clay += 1
                if self.clay >= self.target_clay: break
                r = 0
                
                while True:
                    r+=1
                    if self.add_cube_layer(coord,r) == 0: break
                if self.clay >= self.target_clay: break
                
                while self.truncated_field[coord] != 0:
                    coord = rnd_coords()   
                    
        if mode == 'truncated solid':
            if type(self.truncated_field) == type(None):
                print('Instance must have a truncated field')
                return
            t_low = self.values[int(self.values.size * 
                                    (self.porosity-self.clay_fraction))]
            t_high = self.values[int(self.values.size * self.porosity)]
            for i in [(a,b,c) for a in range(self.size[0]) 
                            for b in range(self.size[1])
                            for c in range(self.size[2])]:
                if self.field[i] > t_low and self.field[i] <= t_high:
                    self.truncated_field[i] = 2
                

def correlacao(matriz_binaria):
	if not matriz_binaria.dtype == 'bool':
		matriz_binaria = (matriz_binaria / matriz_binaria.max()).astype('uint8')
	comprimento = min(matriz_binaria.shape)//2
	correlacao_x = []
	correlacao_y = []
	correlacao_x.append(matriz_binaria.mean())
	for i in range(1,comprimento):
		correlacao_x.append(
				( (matriz_binaria[0:-i,:] * matriz_binaria[i:,:]).sum() )
				 / matriz_binaria[i:,:].size )
		
	correlacao_y.append(matriz_binaria.mean())
	for i in range(1,comprimento):
		correlacao_y.append(
				( (matriz_binaria[:,0:-i] * matriz_binaria[:,i:]).sum() )
				 / matriz_binaria[:,i:].size )
	correlacao_x = np.array(correlacao_x)
	correlacao_y = np.array(correlacao_y)
	correlacao = (correlacao_x + correlacao_y)/2
	return (correlacao_x, correlacao_y, correlacao) 
        
def corr(x, lam, corr_curve):
	

	if x < len(corr_curve):
		return corr_curve[x]
	else:
		return 0
		
	return math.exp(-(x/lam))
        
def create_gaussian_field(shape = (101,101), lam = 10, corr_curve = None):
    C = np.zeros(shape)
    for i in [(a,b) for a in range(shape[0]) for b in range(shape[2])]:
        j = int(math.sqrt((i[0]-shape[0]//2)**2 + (i[1]-shape[1]//2)**2))
        C[i[0],i[1]] = corr(j, lam, corr_curve)
    
    W = np.random.rand(shape)    
    gamma = ifft(C)
    B = ifft(gamma)
    B = B.real
    phi = fft(W)
    phi = gamma * phi
    phi = ifft(phi)
    phi = phi.real
    return phi

def truncate_field(gauss_field, porosity):
	'''
	truncates 0 as pore and 1 as solid
	'''
	values = gauss_field.flatten()
	values.sort()
	threshold = values[int(values.size * porosity)]
	truncated_field = np.copy(gauss_field)
	truncated_field = truncated_field >= threshold
	truncated_field = truncated_field.astype('uint8')
	return truncated_field


          
'''
Testing
'''
'''
TGF = TruncatedGaussian()
TGF.generate_gaussian_field()
TGF.truncate_field()
TGF.test()
field = TGF.field
truncated_field = TGF.truncated_field
y = TGF.y
exact_correlation = TGF.exact_correlation
correlation = TGF.correlation
truncated_correlation = TGF.truncated_correlation  
normalized_correlation = TGF.normalized_correlation
'''      
            
'''
import PIL.Image as pil
import tkinter.filedialog as filedialog
import skimage.filters as filters
import matplotlib.pyplot as plt
img = pil.open(filedialog.askopenfilename()).convert('L')
mat = np.array(img)
limiar = filters.threshold_otsu(mat)
mat = ((mat>= limiar) *255).astype('uint8')
'''
       
