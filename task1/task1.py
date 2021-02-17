import pennylane as qml
from pennylane import numpy as np

t_0 = 0.125*np.pi
p_0 = 0.25*np.pi

num_shots = 100
dev = qml.device('default.qubit', wires=3, shots=num_shots)

@qml.qnode(dev)
def var_swap(params):
  # random state preparation
  qml.RY(t_0, wires=1)
  qml.RZ(p_0, wires=1)
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

steps = 1000

best_cost = [cost(params)]
best_params = params
for i in range(steps):
  params = opt.step(cost, params)
  new_cost = cost(params)
  if(new_cost < best_cost[-1]):
    best_params = params
  best_cost.append(new_cost)

print(best_params)
