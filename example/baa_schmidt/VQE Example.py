import datetime

import qiskit
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.aqua.components.initial_states import Custom
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel

from qclib.state_preparation import baa_schmidt
from qclib.state_preparation.util import baa
from qclib.state_preparation.util.baa import Node


def get_qubit_op(dist):
    driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM,
                         charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    freeze_list = [0]
    remove_list = [-3, -2]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift


if __name__ == "__main__":
    backend = BasicAer.get_backend("statevector_simulator")
    distances = np.arange(3.0, 6.0, 0.5)
    exact_energies = []
    vqe_energies = []
    optimizer = SLSQP(maxiter=3)
    for dist in distances:
        qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op(dist)
        result = NumPyEigensolver(qubitOp).run()
        exact_energies.append(np.real(result.eigenvalues) + shift)
        initial_state = HartreeFock(
            num_spin_orbitals,
            num_particles,
            qubit_mapping='parity'
        )
        eigenstate = result['eigenstates'][0].primitive.data
        qc: qiskit.QuantumCircuit
        node: Node
        qc, node = baa_schmidt.initialize(eigenstate, max_fidelity_loss=1.0, return_node=True)
        initial_state_baa = Custom(qc.num_qubits, circuit=qc)
        gm = baa.geometric_entanglement(eigenstate)
        start = datetime.datetime.now()
        var_form = UCCSD(
            num_orbitals=num_spin_orbitals,
            num_particles=num_particles,
            initial_state=initial_state_baa,
            qubit_mapping='parity'
        )
        vqe = VQE(qubitOp, var_form, optimizer)
        end = datetime.datetime.now()
        result = vqe.run(backend)
        qcs = vqe.construct_circuit(list(result.optimal_parameters.values()))
        qcs_t = qiskit.transpile(qcs, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)
        vqe_result = np.real(result['eigenvalue'] + shift)
        vqe_energies.append(vqe_result)
        print("Interatomic Distance:", np.round(dist, 2), "VQE Result:", vqe_result, "Exact Energy:",
              exact_energies[-1], "Time taken: ", end - start, "Entanglement: ", gm, "Approx. Error", node.total_fidelity_loss)

    print("All energies have been calculated")