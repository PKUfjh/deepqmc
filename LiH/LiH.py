from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet

mol = Molecule.from_name('LiH')
net = PauliNet.from_hf(mol)
train(net,batch_size=1024,n_steps=200)
evaluate(net)
