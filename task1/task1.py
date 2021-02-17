import pennylane as qml
from pennylane import numpy as np

t_0 = 0.125*np.pi # ~ 0.3927
p_0 = 0.25*np.pi  # ~ 0.7854

num_shots = 1000
dev = qml.device('default.qubit', wires=3, shots=num_shots)

@qml.qnode(dev)
def var_swap(params):
  # random state preparation
  qml.QubitStateVector([
      np.cos(t_0/2.),
      np.sqrt(0.5)*(1+1j)*np.sin(t_0/2.)
  ], wires=1)
  # actual circuit follows:
  qml.RY(params[0], wires=2)
  qml.RZ(params[1], wires=2)
  qml.Hadamard(wires=0)
  qml.CSWAP(wires=[0,1,2])
  qml.Hadamard(wires=0)
  return qml.sample(qml.PauliZ(0))

def cost(params):
    return 1-(1./num_shots)*np.sum(var_swap(params))

params = np.array([0.,0.])

opt = qml.RotosolveOptimizer()

steps = 100
best_cost = [cost(params)]
best_params = params
for i in range(steps):
  params = opt.step(cost, params)
  new_cost = cost(params)
  if(new_cost < best_cost[-1]):
    best_params = params
  best_cost.append(new_cost)

print(best_params)

n_qubits = 4 * 3

num_shots = 1000
dev4 = qml.device('default.qubit', wires=n_qubits, shots=num_shots)

def init_random_state(state):
  qml.BasisState(state, wires=[1,4,7,10])

def swap(init_wire, theta):
  qml.RY(theta, wires=init_wire+2)
  qml.Hadamard(wires=init_wire)
  qml.CSWAP(wires=[init_wire, init_wire+1, init_wire+2])
  qml.Hadamard(wires=init_wire)

@qml.qnode(dev4)
def n_swap(params):
  init_random_state(np.array([0,0,1,1]))
  swap(0, params[0])
  swap(3, params[1])
  swap(6, params[2])
  swap(9, params[3])
  return [qml.sample(qml.PauliZ(i)) for i in range(0,11,3)]

def cost(params):
  results = n_swap(params)
  return [1-(1./num_shots)*np.sum(results[i]) for i in range(4)]

param_set = [np.array([np.pi*i,np.pi*j,np.pi*k,np.pi*l])
             for i in range(0,2)
             for j in range(0,2)
             for k in range(0,2)
             for l in range(0,2)]
cost_set = []
for param in param_set:
  cost_set.append(cost(param))

best_cost = 1
best_params = []
for r in range(len(cost_set)):
  run_cost = np.sum(cost_set[r])
  if run_cost < best_cost:
    best_cost = run_cost
    best_params = param_set[r]

print(best_params, best_cost)
