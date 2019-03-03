import sys
sys.path.insert(0, '../ELINA/python_interface/')
import pdb

import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def mult(w,x,b):
    return [min(s * x[0], s* x[1]), max(s*x[0],s*x[1])]

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res

def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0

def analyze(nn, LB_N0, UB_N0, label):
    verify_flag=(LB_N0[0]==UB_N0[0])
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer

    man = elina_box_manager_alloc() #manager
    itv = elina_interval_array_alloc(num_pixels) #interval
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        #print("Layer ",layerno)
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           #if(layerno == 1):
               #print(weights)

           #optimize one layer
           #opt_onelayer(lowbound,upbound,weight,biases)
           biases = nn.biases[nn.ffn_counter]
           #if(layerno==1):
               #print('bias',biases)

           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)

           dimadd = elina_dimchange_alloc(0,num_out_pixels)
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           # handle affine layer
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)


           #Affine Output Fadhil
           #print("Affine Output")
           bounds = elina_abstract0_to_box(man,element)
           output_size = len(weights)
           for i in range(output_size):
                inf = bounds[i].contents.inf.contents.val.dbl
                sup = bounds[i].contents.sup.contents.val.dbl
                #print(i,', [',inf,',',sup,']')

           # handle ReLU layer
           if(nn.layertypes[layerno]=='ReLU'):
               # TODO just put this here to not break stuff while testing
               element = relu_box_layerwise(man,True,element,0, num_out_pixels)

               # Run Linear Solver up until heuristic layer # cutoff
               if (layerno >= 1) and not verify_flag:
                   #print("Running solver")
                   # 2 layer combined Relu
                   dims = elina_abstract0_dimension(man,element)
                   output_size = dims.intdim + dims.realdim
                   affine_upbounds=[]
                   affine_lowbounds=[]

                   for i in range(len(weights)):  # Loop over all Relu outputs of this layer
                           m = Model('relu')
                           # set the bounds of each input value
                           dim_prev_layer = len(weights[i])

                           # TODO replace all these comprehensions with single loop that cuts out relus below 0
                           objective = biases[i]
                           for j in range(dim_prev_layer):
                               prev_affine_lowbound = prev_affine_lowbounds[j]
                               prev_affine_upbound = prev_affine_upbounds[j]
                               if prev_affine_upbound==prev_affine_lowbound:
                                   objective += prev_affine_upbound
                               elif prev_affine_upbound > 0:
                                   # Normal expected bounds that cross over x=0
                                   lam = prev_affine_upbound/(prev_affine_upbound-prev_affine_lowbound)
                                   mu  = -prev_affine_upbound*prev_affine_lowbound/(prev_affine_upbound-prev_affine_lowbound)
                                   prev_affine = m.addVar(lb=prev_affine_lowbound,ub=prev_affine_upbound,
                                                          vtype=GRB.CONTINUOUS, name="prev_affine{}".format(j))
                                   prev_relu = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="relu{}".format(j))
                                   m.addConstr(prev_relu>=0)
                                   m.addConstr(prev_relu>=prev_affine)
                                   m.addConstr(prev_relu<=lam*prev_affine+mu)
                                   objective += weights[i][j]*prev_relu
                               if prev_affine_upbound <=0:
                                   # Skip this relu and don't add it to the optimization
                                   continue

                           m.update()
                           if isinstance(objective,float):
                               #print('No previous ReLU\'s added. Something is wrong')
                               #pdb.set_trace()
                               if biases[i] !=0:
                                   affine_lowbounds.append(biases[i])
                                   affine_upbounds.append(biases[i])
                               else:
                                   affine_lowbounds.append(0)
                                   affine_upbounds.append(0)

                           else:
                               m.addConstr(objective>=0)
                               m.setObjective(objective, GRB.MINIMIZE)
                               m.setParam( 'OutputFlag', False )
                               #Optimize and create affine layer
                               m.optimize()
                               try:
                                   affine_lowbounds.append(m.objVal)
                               except AttributeError:
                                   #print('Optimisation neuron ',i,'unbounded')
                                   affine_lowbounds.append(0)
                               m.setObjective(objective, GRB.MAXIMIZE)
                               m.optimize()
                               try:
                                   affine_upbounds.append(m.objVal)
                               except AttributeError:
                                   #print('Optimisation neuron ',i,'unbounded')
                                   affine_upbounds.append(0)
                               inf = bounds[i].contents.inf.contents.val.dbl
                               sup = bounds[i].contents.sup.contents.val.dbl
                               #print('Box ',i,', [',inf,',',sup,']')
                               #print('LinSolver',i,'[',affine_lowbounds[i],',',affine_upbounds[i],']')

                   prev_affine_lowbounds=affine_lowbounds
                   prev_affine_upbounds =affine_upbounds

                   #Pass interval to abstract domain
                   man = elina_box_manager_alloc() #manager
                   itv = elina_interval_array_alloc(len(weights)) #interval
                   for i in range(len(weights)):
                       elina_interval_set_double(itv[i],affine_lowbounds[i],affine_upbounds[i])
				   ## construct input abstraction
                   element = elina_abstract0_of_box(man, 0, len(weights), itv)
                   elina_interval_array_free(itv,len(weights))
                   #pdb.set_trace()
                   if(layerno==numlayer-1):
                       element = relu_box_layerwise(man,True,element,0, num_out_pixels)

               else:
                   #TODO
                   #print("Chickened out of solver")
                   # Just run box constraints
                   element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                   # retain the previous layer bounds
                   prev_bounds = bounds
                   dim_prev_layer = len(weights)
                   prev_affine_lowbounds = [prev_bounds[j].contents.inf.contents.val.dbl
                                                        for j in range(dim_prev_layer)]
                   prev_affine_upbounds  = [prev_bounds[j].contents.sup.contents.val.dbl
													    for j in range(dim_prev_layer)]

           #print("Relu Output")
           bounds = elina_abstract0_to_box(man,element)
           output_size = len(weights)
           for i in range(output_size):
                inf = bounds[i].contents.inf.contents.val.dbl
                sup = bounds[i].contents.sup.contents.val.dbl
                #print(i,', [',inf,',',sup,']')

           nn.ffn_counter+=1

        else:
           print(' net type not supported')

    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)

    # if epsilon is zero, try to classify else verify robustness

    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break
    else:

        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)
    return predicted_label, verified_flag



if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    label, _ = analyze(nn,LB_N0,UB_N0,0)
    start = time.time()
    #print("Verifying...")
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
