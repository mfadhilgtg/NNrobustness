from gurobipy import *
import pdb

def mult(s,x):
    return [min(s * x[0], s* x[1]), max(s*x[0],s*x[1])]
def add(x,y):
    return [x[0]+y[0],x[1]+y[1]]

try:

    # Create a new model
    m = Model("relu")

    # Create variables
    # Fixed
    w11 = 0.1
    w12 = 0.2
    w21 = 0.1
    w22 = 0.2
    w31 = 0.1
    w32 = 0.2
    i1 = [0.0, 0.1]
    i2 = [0.0, 0.1]
    h1_itv = add(mult(w11,i1),mult(w12,i2))
    h2_itv = add(mult(w21,i1),mult(w22,i2))
    #Line eq
    m1=h1_itv[1]/(h1_itv[1]+h1_itv[0])
    c1=-1*h1_itv[1]*h1_itv[0]/(h1_itv[1]-h1_itv[0])
    m2=h2_itv[1]/(h2_itv[1]+h2_itv[0])
    c2=-1*h2_itv[1]*h2_itv[0]/(h2_itv[1]-h2_itv[0])

    print(h1_itv[0],h1_itv[1])
    w31 = 0.1
    w32 = 0.2


    relu1 = m.addVar(vtype=GRB.CONTINUOUS, name="relu1")
    relu2 = m.addVar(vtype=GRB.CONTINUOUS, name="relu2")
    h1 = m.addVar(lb=h1_itv[0],ub=h1_itv[1],vtype=GRB.CONTINUOUS, name="h1")
    h2 = m.addVar(lb=h2_itv[0],ub=h2_itv[1],vtype=GRB.CONTINUOUS, name="h2")

    
    # Set objective
    m.setObjective(w31*relu1 + w32*relu2, GRB.MAXIMIZE)
    
    m.addConstr(relu1>=0, "c1")
    m.addConstr(relu1>= h1, "c2")
    m.addConstr(relu1<= h1*m1+c1,"c3")
    m.addConstr(relu2>=0, "c4")
    m.addConstr(relu2>= h2, "c5")
    m.addConstr(relu2<= h2*m2+c2,"c6")
    
    m.optimize()
    m.setObjective(w31*relu1 + w32*relu2, GRB.MINIMIZE)
    m.optimize()
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

