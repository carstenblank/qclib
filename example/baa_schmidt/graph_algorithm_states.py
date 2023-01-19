import datetime
import itertools
import logging
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qiskit
import qiskit.providers.aer as aer
import qutip
from networkx.drawing.nx_agraph import graphviz_layout
from numpy import ndarray
from qiskit.circuit.random import random_circuit
from qutip.sparse import sp_eigs

logging.basicConfig(format='%(asctime)s::' + logging.BASIC_FORMAT, level='ERROR')
LOG = logging.getLogger(__name__)
LOG.setLevel('INFO')


def calculate_entropy_meyer_wallach(vector: np.ndarray):
    num_qb = int(np.ceil(np.log2(vector.shape[0])))
    meyer_wallach_entry = np.zeros(shape=(num_qb, 1))
    for j in range(num_qb):
        psi_0 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=np.complex)  # np.zeros(shape=())
        psi_1 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=np.complex)  # np.zeros(shape=())
        for basis_state, entry in enumerate(vector):
            delta_0, new_basis_state_0 = get_iota(j, num_qb, 0, basis_state)
            delta_1, new_basis_state_1 = get_iota(j, num_qb, 1, basis_state)

            if delta_0:
                psi_0[new_basis_state_0] = entry
            if delta_1:
                psi_1[new_basis_state_1] = entry

        entry = generalized_cross_product(psi_0, psi_1)
        meyer_wallach_entry[j] = entry

    return np.sum(meyer_wallach_entry) * (4/num_qb)


def get_bipartite_systems(vector: np.ndarray) -> List[Tuple[int, ...]]:
    # Bundle the state vector into a qutip Qobj
    num_qb = int(np.ceil(np.log2(vector.shape[0])))
    qobj = qutip.Qobj(inpt=vector, dims=[num_qb * [2], [1]])

    # The density matric rho to use for qutip processing
    rho = qutip.ket2dm(qobj)
    # To make the below code more clear: I use qutip's entropy_vn (von Neuman) function, and before that I do a
    # partial trace of one of the systems. The other subsystem is the complement,
    # and by the HS decomposition it must have the same EVs, so I don't compute. All combinations are done,
    # the maximum value gives the maximum entropy. So far, that is what I think is what needed to be done.
    size_biggest_coef = []
    # This loop looks at the principal subsystem size: starting from 1 to at most the half of the system
    # (+1 to have in included). The reason is that HS decomposition says that the EVs are the same for the 'other'
    # system.
    for size in range(1, num_qb // 2 + 1):
        sub_systems = itertools.combinations(range(num_qb), size)
        size_biggest_coef += sub_systems
    return size_biggest_coef


def calculate_biggest_system_divide_error(vector: np.ndarray) -> Dict[int, float]:
    # Bundle the state vector into a qutip Qobj
    num_qb = int(np.ceil(np.log2(vector.shape[0])))
    qobj = qutip.Qobj(inpt=vector, dims=[num_qb * [2], [1]])

    # The density matric rho to use for qutip processing
    rho = qutip.ket2dm(qobj)
    # To make the below code more clear: I use qutip's entropy_vn (von Neuman) function, and before that I do a
    # partial trace of one of the systems. The other subsystem is the complement,
    # and by the HS decomposition it must have the same EVs, so I don't compute. All combinations are done,
    # the maximum value gives the maximum entropy. So far, that is what I think is what needed to be done.
    size_biggest_coef = {}
    # This loop looks at the principal subsystem size: starting from 1 to at most the half of the system
    # (+1 to have in included). The reason is that HS decomposition says that the EVs are the same for the 'other'
    # system.
    for size in range(1, num_qb//2 + 1):
        sub_systems = itertools.combinations(range(num_qb), size)
        for system in sub_systems:
            rho_traced = qutip.ptrace(rho, list(system))
            vals = sp_eigs(rho_traced.data, rho_traced.isherm, vecs=False, sparse=False)
            nzvals = vals[abs(vals) > 1e-3]
            # biggest_value = size_biggest_coef.get(size, 0)
            # size_biggest_coef[size] = biggest_value if biggest_value > max(nzvals) else max(nzvals)
            if size not in size_biggest_coef:
                size_biggest_coef[system] = []
            size_biggest_coef[system].append(max(nzvals))
            LOG.debug(f'The SubSystem {list(system)} has the biggest Schmidt coefficient {size_biggest_coef[system]}')
    fidelity_error = dict([(k, sum([(1-v)**2 for v in list_of_max_coef])) for k, list_of_max_coef in size_biggest_coef.items()])
    return fidelity_error  # [max(fidelity_error.keys())]


def generate_tree_simple(g: nx.DiGraph, parent_node: any):
    vector: np.ndarray = g.nodes[parent_node]['vector']
    qubits = int(np.ceil(np.log2(vector.shape[0])))

    nodes_weights = calculate_biggest_system_divide_error(vector)

    n: Tuple[int, ...]
    for n, v in nodes_weights.items():
        # Compute the Schmidt States
        M = get_separation_matrix(vector, n)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        vector_1 = U[:, 0]
        vector_2 = Vh.T[:, 0]

        # Add to the DiGraph
        node_key = len(g.nodes) - 1  # (num_qubits, parent_node, n)

        g.add_node(node_key, split=[get_complementary_subsystem(n, qubits), n])

        g.add_node(node_key + 1, vector=vector_1, subsystem=get_complementary_subsystem(n, qubits))
        g.add_node(node_key + 2, vector=vector_2, subsystem=n)
        g.add_edges_from([
            (parent_node, node_key),
            (node_key, node_key + 1, {'weight': v}),
            (node_key, node_key + 2, {'weight': v})
        ])

        # Propagate Tree
        generate_tree_simple(g, node_key + 1)
        generate_tree_simple(g, node_key + 2)


class Split:

    fidelity_loss: float
    subsystem: Optional[Tuple[int, ...]]

    def __init__(self, subsystem: Optional[Tuple[int, ...]], fidelity_loss: float):
        self.subsystem = subsystem
        self.fidelity_loss = fidelity_loss

    def __str__(self):
        return f'{type(self).__name__}|{self.subsystem}|'

    def __repr__(self):
        return str(self)


def unitary_cnots(k):
    return 23 / 48 * 2 ** (2 * k)  # - 3 / 2 * 2 ** k + 4 / 3


def sp_cnots(n) -> int:
    if n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 4
    elif n % 2 == 0:
        k = n/2
        return int(2 ** k - k - 1 + k + 23/24*2**(2*k) - 3/2 * 2**(k+1) + 8/3)
    else:
        k = (n-1)/2
        return int(2 ** k - k - 1 + k + 23/48*2**(2*k) - 3/2 * 2**(k) + 4/3 + 23/48*2**(2*k + 2) - 3/2 * 2**(k + 1) + 4/3)


def to_qubits(d):
    return int(np.ceil(np.log2(d)))


class Node:
    split_program: Tuple[Split, ...]
    vectors: List[ndarray]
    cnot_saving: int
    fidelity_loss: float

    def __init__(self, split_program: Tuple[Split, ...], vectors: List[np.ndarray], fidelity_loss: float, cnot_saving: int):
        self.fidelity_loss = fidelity_loss
        self.cnot_saving = cnot_saving
        self.vectors = vectors
        self.split_program = split_program

    def __getitem__(self, item):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return data[item]

    def __iter__(self):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return iter(data)

    def __str__(self):
        return f'Node{(self.split_program, self.fidelity_loss, self.cnot_saving, self.vectors)}'

    def __repr__(self):
        return str(self)


def get_nodes_from_activations(vectors: List[np.ndarray], subsystem_list: List[List[Tuple[int, ...]]],
                               activations: np.ndarray) -> List[Node]:
    result = []
    # This step is the main step, I need to explain this a bit better!
    all_paths = [[Split(dd, -1.0) for dd in d] if a == 1 else [Split(None, 0.0)] for d, a in
                 zip(subsystem_list, activations)]

    # The product makes all possible cross-product combinations as given above. Need to explain this better!
    for split_program in itertools.product(*all_paths):
        # apply the split program to the vectors and generate the new children
        new_vectors = []
        new_fidelity = 1.0
        saved_cnots = 0
        split: Split
        for vector, split in zip(vectors, split_program):
            if split.subsystem is None:
                new_vectors.append(vector)
            else:
                # Compute the Schmidt States
                M = get_separation_matrix(vector, split.subsystem)
                U, S, Vh = np.linalg.svd(M, full_matrices=False)
                vector_1 = U[:, 0]
                vector_2 = Vh.T[:, 0]
                new_vectors.append(vector_1)
                new_vectors.append(vector_2)
                # k = to_qubits(min(vector_1.shape[0], vector_2.shape[0]))
                cnots_phase_3 = sp_cnots(to_qubits(vector_1.shape[0]))
                cnots_phase_4 = sp_cnots(to_qubits(vector_2.shape[0]))
                cnots_originally = sp_cnots(to_qubits(vector.shape[0]))
                saved_cnots += cnots_originally - cnots_phase_3 - cnots_phase_4
                split.fidelity_loss = 1 - (S ** 2)[0]
                new_fidelity *= 1 - split.fidelity_loss
        result.append(Node(split_program, new_vectors, 1 - np.around(new_fidelity, 6), saved_cnots))
    return result


def generate_subsystem_partitions(vectors: List[np.ndarray], no_branching=False):
    subsystem_list: List[List[Tuple[int], ...]] = [get_bipartite_systems(vector) for vector in vectors]

    result = []
    v = np.zeros(len(vectors))
    v[0] = 1
    if no_branching:
        result += get_nodes_from_activations(vectors, [[d[0]] for d in subsystem_list if len(d) > 0], v)
    else:
        for activations in [np.roll(v, i) for i in range(len(vectors))]:
            if set(activations) == {0}:
                continue
            result += get_nodes_from_activations(vectors, subsystem_list, activations)

    return result


def generate_tree(g: nx.DiGraph, parent_node: any, no_branching=False):
    vectors: List[np.ndarray] = g.nodes[parent_node]['vectors']

    qubits = set([int(np.ceil(np.log2(vector.shape[0]))) for vector in vectors])

    if set(qubits) == {1}:
        g.add_edges_from([(parent_node, -1, {'weight': 0})])
        return

    result = generate_subsystem_partitions(vectors, no_branching)
    for node_split_program, node_vectors, edge_fidelity_loss, saved_cnots in result:
        # Add to the DiGraph
        node_key = len(g.nodes)  # (num_qubits, parent_node, n)
        g.add_node(node_key, vectors=node_vectors, split_program=node_split_program)
        g.add_edges_from([(parent_node, node_key, {'weight': edge_fidelity_loss, 'cnots': saved_cnots})])
        # Propagate Tree
        generate_tree(g, node_key, no_branching)


def get_complementary_subsystem(subsystem: Tuple[int, ...], num_qubits: int):
    subsystem_c = tuple(set(range(num_qubits)).difference(set(subsystem)))
    return subsystem_c


def get_separation_matrix(vector: np.ndarray, subsystem_2: Tuple[int, ...]):
    num_qubits =  int(np.ceil(np.log2(vector.shape[0])))
    subsystem_1 = get_complementary_subsystem(subsystem_2, num_qubits)

    new_shape = (2 ** len(subsystem_1), 2 ** len(subsystem_2))
    M = np.zeros(shape=new_shape, dtype=complex)

    for n, v in enumerate(vector):
        current = f'{n:b}'.zfill(num_qubits)
        number_2 = ''.join([c for i, c in enumerate(current) if i in subsystem_2])
        number_1 = ''.join([c for i, c in enumerate(current) if i in subsystem_1])
        M[int(number_1, 2), int(number_2, 2)] = v

    return M


def plot_graph(g: nx.Graph):
    nx.nx_agraph.write_dot(g, 'test.dot')
    plt.figure(figsize=(60, 60))
    pos = graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(g, pos)
    plt.show()


def calculate_state(vectors: List[np.ndarray]):
    state = np.ones(1)
    for p in vectors:
        state = np.kron(p, state)
    return state


def get_fidelity_loss(vector: np.ndarray, return_product_state=False):
    LOG.debug('Creating Graph')
    g = nx.DiGraph()
    g.add_node(0, vectors=[vector])  # root node
    g.add_node(-1)  # end node
    generate_tree(g, 0, no_branching=True)

    LOG.debug('Calculating Shortest Path.')
    shortest_path = nx.shortest_path(g, source=0, target=-1)
    expected_fidelity = np.prod([1 - g.edges[edge]['weight'] for edge in nx.path_graph(shortest_path).edges()])
    expected_fidelity_loss = 1 - expected_fidelity
    product_state = g.nodes[shortest_path[-2]]['vectors']

    if return_product_state:
        return expected_fidelity_loss, product_state
    else:
        return expected_fidelity_loss


def search_best_node(vectors: List[ndarray], running_cnot_saving: int, running_fidelity_loss: float, max_fidelity_loss: float) -> Optional[Node]:
    data: List[Node] = generate_subsystem_partitions(vectors)
    possible_data: List[Node] = [d for d in data if 1 - (1 - d.fidelity_loss) * (1 - running_fidelity_loss) <= max_fidelity_loss]
    better = [search_best_node(p.vectors, running_cnot_saving + p.cnot_saving, running_fidelity_loss + p.fidelity_loss, max_fidelity_loss) for p in possible_data]
    if len(better) == 0 or (set(better) == {None}):
        if len(possible_data) == 0:
            return None
        else:
            best_possible = max(possible_data, key=lambda n: n.cnot_saving)
            total_fidelity_loss = 1 - (1 - best_possible.fidelity_loss) * (1 - running_fidelity_loss)
            total_cnot_savings = running_cnot_saving + best_possible.cnot_saving
            return Node((), best_possible.vectors, total_fidelity_loss, total_cnot_savings)
    else:
        return max([b for b in better if b is not None], key=lambda n: n.cnot_saving)


def adaptive_approximation(vector: np.ndarray, max_fidelity_loss: float) -> Optional[Node]:
    LOG.debug('Creating Graph')
    # fidelity_loss, product_state = get_fidelity_loss(vector, return_product_state=True)

    # if fidelity_loss <= max_fidelity_loss:
    #     LOG.debug(f'Product State is within limits {fidelity_loss} <= {max_fidelity_loss}.')
    #     return product_state

    best_node: Optional[Node] = search_best_node([vector], 0, 0.0, max_fidelity_loss)
    LOG.debug(f'Best Node: {best_node}')
    return best_node


if __name__ == "__main__":
    exp_time_start = pd.Timestamp.now()
    LOG.setLevel('INFO')
    num_qubits = 10
    mw_limit_lower = 0.5
    mw_limit_upper = 0.6
    for _ in range(10):
        mw = -1.0
        while mw < mw_limit_lower or mw > mw_limit_upper:
            qc: qiskit.QuantumCircuit = random_circuit(num_qubits, 2*num_qubits)
            job: aer.AerJob = qiskit.execute(qc, backend=aer.StatevectorSimulator())
            vector = job.result().get_statevector()
            mw = calculate_entropy_meyer_wallach(vector)

        LOG.debug(f"The Circuit\n{qc.draw(fold=-1)}")
        LOG.debug(f"Vector: {np.linalg.norm(vector)}\n {vector}")
        LOG.debug(f"Meyer-Wallach: {mw}.")

        start = datetime.datetime.now()
        max_fidelity_loss = 0.1
        node = adaptive_approximation(vector, max_fidelity_loss)
        end = datetime.datetime.now()

        if node is None:
            LOG.info(f'[{max_fidelity_loss}] No approximation could be found (MW: {mw}). ({end - start})')
        else:
            moettoenen_sp = sum([2**n for n in range(1, num_qubits)])
            LOG.info(f'[{max_fidelity_loss}] With fidelity loss {node.fidelity_loss} (MW: {mw}) we can '
                     f'save {node.cnot_saving} of {sp_cnots(num_qubits)}/{moettoenen_sp} CNOT-gates. ({end - start})')
