# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bounded Approximation Algorithm.
https://arxiv.org/abs/2111.03132
"""
import sys
from dataclasses import dataclass
from itertools import combinations, chain
from typing import List, Optional, Tuple
from math import log2, sqrt
import numpy as np
import tensorly as ty

from qclib.entanglement import geometric_entanglement
from qclib.state_preparation.schmidt import cnot_count as schmidt_cnots, \
                                            schmidt_decomposition, \
                                            schmidt_composition, \
                                            low_rank_approximation

# pylint: disable=missing-class-docstring

def adaptive_approximation(state_vector, max_figure_of_merit, strategy='greedy',
                           max_combination_size=0, use_low_rank=False, figure_of_merit = 'fidelity_loss'):
    """
    It reduces the entanglement of the given state, producing an approximation
    to reduce the complexity of the quantum circuit needed to prepare it.
    `https://arxiv.org/abs/2111.03132`_.
    Args:
        state_vector (list):
            A state vector to be approximated by a less complex state.
        max_figure_of_merit (float):
            Maximum fidelity loss allowed to the approximated state.
        strategy (string):
            Method to search for the best approximation ('brute_force' or 'greedy').
            For states larger than 2**8, the greedy strategy should preferably be used.
            Default is strategy='greedy'.
        max_combination_size (int):
            Maximum size of the combination ``C(n_qubits, max_combination_size)``
            between the qubits of an entangled subsystem of length ``n_qubits`` to
            produce the possible bipartitions
            (1 <= ``max_combination_size`` <= ``n_qubits``//2).
            For example, if ``max_combination_size``==1, there will be ``n_qubits``
            bipartitions between 1 and ``n_qubits``-1 qubits.
            The default value is 0 (the size will be maximum for each level).
        use_low_rank (bool):
            If set to True, ``rank``>1 approximations are also considered. This is fine
            tuning for high-entanglement states and is slower.
            The default value is False.
    Returns:
        Node: a node with the data required to build the quantum circuit.
    """

    n_qubits = _to_qubits(len(state_vector))

    # Completely separates the state to estimate the maximum possible fidelity loss.
    # If max_fidelity_loss input is lower than the estimated loss, it runs the full
    # routine with potentially exponential cost.
    entanglement, product_state = geometric_entanglement(state_vector, return_product_state=True)
    if max_figure_of_merit >= entanglement and figure_of_merit == 'fidelity_loss':
        qubits = [(n,) for n in range(n_qubits)]
        ranks = [1] * n_qubits
        partitions = [None] * n_qubits
        return Node(
            0, 0, entanglement, entanglement, product_state, qubits, ranks, partitions, []
        )

    vectors = [state_vector]
    qubits = [tuple(range(n_qubits))]
    ranks = [0]
    partitions = [None]

    root_node = Node(0, 0, 0.0, 0.0, vectors, qubits, ranks, partitions, [])
    _build_approximation_tree(root_node, max_figure_of_merit, strategy,
                              max_combination_size, use_low_rank, figure_of_merit)

    leaves = []
    _search_leaves(root_node, leaves)

    best_node = _search_best(leaves, max_figure_of_merit, state_vector, figure_of_merit)

    return best_node

@dataclass
class Entanglement:
    """
    Entanglement reduction information.

    This class contains the information about the entanglement reduction
    of a bipartition. It can be used to assemble an approximate state
    (rank>1) or two completely separate states (rank=1).
    """
    rank: int
    svd_u: np.ndarray
    svd_v: np.ndarray
    svd_s: np.ndarray

    register: Tuple[int]
    partition: Tuple[int]
    local_partition: Tuple[int]

    def fidelity_loss(self) -> float:
        return 1.0 - sum(self.svd_s ** 2)

    def approximate_state(self):
        normed_svd_s = self.svd_s/sqrt( 1.0 - self.fidelity_loss() )
        return schmidt_composition(
            self.svd_u, self.svd_v, normed_svd_s, self.local_partition
        )

@dataclass
class Node:
    """
    Tree node used in _approximation_tree function.
    """
    node_saved_cnots: int
    total_saved_cnots: int

    node_fidelity_loss: float
    total_fidelity_loss: float

    vectors: List[List[complex]]
    qubits: List[Tuple[int]]
    ranks: List[int]
    partitions: List[Optional[Tuple[int]]]

    nodes: List['Node']

    node_figure_of_merit: float = 0.0
    total_figure_of_merit: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """
        True if the all vectors have reached an approximation assessment. There
        is no more decomposition/approximation possible. Therefore, the node is
        a leaf.
        """
        return all(np.asarray(self.ranks) >= 1)

    def num_qubits(self) -> int:
        """ Complete state number of qubits. """
        return len([e for qb_list in self.qubits for e in qb_list])

    def state_vector(self) -> np.ndarray:
        """ Complete state vector. """
        state = ty.tenalg.kronecker(self.vectors) # pylint: disable=no-member
        return state

    def __str__(self):
        str_vectors = '\n'.join([str(np.around(i,2)) for i in self.vectors])
        str_qubits = ' '.join([str(i) for i in self.qubits])
        str_ranks = ' '.join([str(i) for i in self.ranks])
        return f'saved cnots node={self.node_saved_cnots} ' + \
               f'total={self.total_saved_cnots}\n' + \
               f'fidelity loss node={round(self.node_fidelity_loss,6)} ' + \
               f'total={round(self.total_fidelity_loss,6)}\n' + \
               f'states\n{str_vectors}\n' + \
               f'qubits\n{str_qubits}\n' + \
               f'ranks\n{str_ranks}'

def _build_approximation_tree(node, max_figure_of_merit, strategy='brute_force', max_k=0,
                              use_low_rank=False, figure_of_merit='fidelity_loss'):
    # Ignore states that are already completely or partially disentangled.
    node_data = [(q, v) for q, v, k in zip(node.qubits, node.vectors, node.ranks) if k == 0]

    for entangled_qubits, entangled_vector in node_data:

        if not 1 <= max_k <= len(entangled_qubits)//2:
            max_k = len(entangled_qubits)//2

        if strategy == 'greedy':
            combs = _greedy_combinations(entangled_vector, entangled_qubits, max_k)
        else:
            combs = _all_combinations(entangled_qubits, max_k)

        # Disentangles or reduces the entanglement of each bipartion of
        # entangled_qubits.
        for partition in combs:
            # Computes the two state vectors after disentangling "partition".
            # If the bipartition cannot be fully disentangled, an approximate
            # state is returned.
            entanglement_info = _reduce_entanglement(
                entangled_vector, entangled_qubits, partition, use_low_rank
            )

            for e_info in entanglement_info:
                # Recursion should not continue for this branch if
                # "total_fidelity_loss" has reached "max_fidelity_loss".
                # The leaf corresponds to the node of the best approximation of
                # "max_fidelity_loss" on the branch.
                new_node = _create_node(node, e_info, figure_of_merit)
                if new_node.total_figure_of_merit <= max_figure_of_merit:
                    if new_node.total_saved_cnots > 0:
                        node.nodes.append(new_node)

    if len(node.nodes) > 0:  # If it is not the end of the recursion,
        node.vectors.clear() # clear vectors and qubits to save memory.
        node.qubits.clear()  # This information is no longer needed from this point
                             # on (but may be needed in the future).
    if strategy == 'greedy' and len(node.nodes) > 0:
        # Locally optimal choice at each stage.
        node.nodes = [_search_best(node.nodes, 1.0)]

    for new_node in node.nodes:
        # call _build_approximation_tree recurrently for each new node.
        if not new_node.is_leaf: # Saves one call for each leaf node.
            _build_approximation_tree(
                new_node, max_figure_of_merit, strategy, max_k, use_low_rank
            )

def _all_combinations(entangled_qubits, max_k):
    combs = tuple(combinations(entangled_qubits, max_k))
    if len(entangled_qubits)%2 == 0 and len(entangled_qubits)//2 == max_k:
        # Ignore redundant complements. Only when max_k is exactly
        # half the length of entangled_qubits. Reduces the number of branches.
        # (0,1,2,3) -> (0,1), (0,2), (0,3); ignore (2,3), (1,3), (1,2) .
        combs = combs[:len(combs)//2]

    return chain(*(combinations(entangled_qubits, k)
                                            for k in range(1, max_k)), combs)

def _greedy_combinations(entangled_vector, entangled_qubits, max_k):
    """
    Combinations with a qubit-by-qubit analysis.
    Returns only one representative of the partitions of size k (1<=k<=max_k).
    The increment in the partition size is done by choosing the qubit that has
    the lowest fidelity-loss when removed from the remaining entangled subsystem.
    """
    node = Node( 0, 0, 0.0, 0.0, [entangled_vector], [entangled_qubits], [0], [None], [] )
    for _ in range(max_k):
        current_vector = node.vectors[-1] # Last item is the current entangled state.
        current_qubits = node.qubits[-1]

        nodes = []
        # Disentangles one qubit at a time.
        for qubit_to_disentangle in current_qubits:
            entanglement_info = \
                _reduce_entanglement(current_vector, current_qubits, (qubit_to_disentangle,))

            new_node = _create_node(node, entanglement_info[0])

            nodes.append(new_node)
        # Search for the node with lowest fidelity-loss.
        node = _search_best(nodes, 1.0)

    # Build the partitions by incrementing the number of selected qubits.
    # Returns only one partition for each length k.
    # All disentangled qubits are in the slice "node.qubits[0:max_k]", in the order in which
    # they were selected. Each partition needs to be sorted to ensure that the correct
    # construction of the circuit.
    return ( tuple(sorted( chain(*node.qubits[:k]) )) for k in range(1, max_k+1) )

def _reduce_entanglement(state_vector, register, partition, use_low_rank=False):
    local_partition = []
    # Maintains the relative position between the qubits of the two subsystems.
    for qubit_to_disentangle in partition:
        local_partition.append(sum(i < qubit_to_disentangle for i in register))

    local_partition = tuple(local_partition)

    svd_u, svd_s, svd_v = schmidt_decomposition(state_vector, local_partition)

    entanglement_info = []

    max_ebits = 0
    if use_low_rank:
        # Limit the maximum low_rank to "2**(total_ebits-1)" to not repeat the original state.
        max_ebits = _to_qubits(svd_s.shape[0]) - 1

    for ebits in range(0, max_ebits+1):
        low_rank = 2**ebits

        rank, low_rank_u, low_rank_v, low_rank_s = \
            low_rank_approximation(low_rank, svd_u, svd_v, svd_s)

        if rank < low_rank:
            break # No need to go any further, as the maximum effective rank has been reached.

        entanglement_info.append(Entanglement(
            rank, low_rank_u, low_rank_v, low_rank_s,
            register, partition, local_partition
        ))
    return entanglement_info

def calculate_figure_of_merit(parent_node: Node, entanglement_info: Entanglement, figure_of_merit: str):

    if figure_of_merit == 'fidelity_loss':
        child_fidelity_loss = entanglement_info.fidelity_loss()
        total_fidelity_loss = 1.0 - (1.0 - entanglement_info.fidelity_loss()) * \
                              (1.0 - parent_node.total_figure_of_merit)
        return child_fidelity_loss, total_fidelity_loss
    elif figure_of_merit == 'mse':
        # The MSE is harder to recursively calculate. But we can use the sub-additivity of the Euclidean norm
        # The current (upper bound) of the MSE is given from the last node
        d_x_1_y = parent_node.total_figure_of_merit
        # The current node's state vector must be calculated to a probability distribution
        x_1 = np.abs(parent_node.state_vector()) ** 2
        # As well as the (possible) sub-nodes of it
        x_2 = np.abs(entanglement_info.approximate_state()) ** 2
        # The MSE between both current node and all its (possible) children
        d_x_2_x_1 = np.average((x_2 - x_1 / np.max(x_2 - x_1))**2)
        # The upper bound of the total figure of merit (MSE) is given here
        d_x_2_y_ub = d_x_2_x_1 + d_x_1_y
        # This is what we will save. It will be an overestimate.
        # This results in too many children needing evaluation!
        # In the lower comment, there is a way to calculate it more precisely,
        # if we save more data along the way.
        # This is not correct, as the sign is lost in the "inner products"
        # new_total_figure_of_merit = d_x_2_x_1 + d_x_1_y + 2 * (
        #     np.sqrt(1 - node_fidelity_loss)
        #     - np.sqrt(1 - total_fidelity_loss)
        #     - np.linalg.norm(x_1)
        #     + np.sqrt(1 - node.total_fidelity_loss)
        # )
        return d_x_2_x_1, d_x_2_y_ub
    else:
        raise ValueError(f"The figure of merit '{figure_of_merit}' is not recognized")

def _create_node(parent_node: Node, e_info: Entanglement, figure_of_merit: str = 'fidelity_loss'):

    vectors = parent_node.vectors.copy()
    qubits = parent_node.qubits.copy()
    ranks = parent_node.ranks.copy()
    partitions = parent_node.partitions.copy()

    index = parent_node.qubits.index(e_info.register)
    original_vector = vectors.pop(index)
    original_qubits = qubits.pop(index)
    original_rank = ranks.pop(index)
    original_partition = partitions.pop(index)

    if e_info.rank == 1:
        # The partition qubits have been completely disentangled from the
        # rest of the register. Therefore, the original entangled state is
        # removed from the list and two new separate states are included.
        partition1 = tuple( sorted(set(original_qubits).difference(set(e_info.partition))) )
        partition2 = e_info.partition

        vectors.append(e_info.svd_v.T[:, 0])
        qubits.append(partition2)
        ranks.append(1 if len(partition2) == 1 else 0) # Single qubit states can
        partitions.append(None)                        # no longer be disentangled.

        vectors.append(e_info.svd_u[:, 0])
        qubits.append(partition1)
        ranks.append(1 if len(partition1) == 1 else 0)
        partitions.append(None)

        node_saved_cnots = _count_saved_cnots(
                                original_vector, vectors[-1], vectors[-2],
                                original_partition, None,
                                original_rank
                            )
    else:
        # The entanglement between partition qubits and the rest of the
        # register has been reduced, but not eliminated. Therefore, the
        # original state is replaced by an approximate state.
        vectors.append(e_info.approximate_state())
        qubits.append(original_qubits)
        ranks.append(e_info.rank)
        partitions.append(e_info.local_partition)

        node_saved_cnots = _count_saved_cnots(
                                original_vector, vectors[-1], None,
                                original_partition, e_info.local_partition,
                                original_rank, e_info.rank
                            )

    total_saved_cnots = parent_node.total_saved_cnots + node_saved_cnots
    total_fidelity_loss = 1.0 - (1.0 - e_info.fidelity_loss()) * \
                                (1.0 - parent_node.total_fidelity_loss)
    node_figure_of_merit, total_figure_of_merit = calculate_figure_of_merit(parent_node, e_info, figure_of_merit)

    child_node = Node(node_saved_cnots, total_saved_cnots, e_info.fidelity_loss(),
                      total_fidelity_loss, vectors, qubits, ranks, partitions, [])
    child_node.node_figure_of_merit = node_figure_of_merit
    child_node.total_figure_of_merit = total_figure_of_merit

    return child_node

def _search_leaves(node, leaves):
    # It returns the leaves of the tree. These nodes are the ones with
    # total_fidelity_loss closest to max_fidelity_loss for each branch.
    if len(node.nodes) == 0:
        leaves.append(node)
    else:
        for child in node.nodes:
            _search_leaves(child, leaves)

def evaluate_fidelity_loss(max_figure_of_merit):
    def _inner(node):
        return node.total_fidelity_loss if node.total_fidelity_loss <= max_figure_of_merit else sys.maxsize
    return _inner

def evaluate_mse(max_figure_of_merit: float, original_state: np.ndarray):
    def _inner(node: Node):
        mse = np.linalg.norm(np.abs(original_state) ** 2 - np.abs(node.state_vector()) ** 2)
        return mse if mse <= max_figure_of_merit else sys.maxsize
    return _inner

def _search_best(nodes, max_figure_of_merit, original_state: Optional[list] = None, figure_of_merit = 'fidelity_loss'):
    # Nodes with the greatest reduction in the number of CNOTs.
    # There may be several with the same number.
    max_total_saved_cnots = max(nodes, key=lambda n: n.total_saved_cnots).total_saved_cnots
    max_saved_cnots_nodes = [node for node in nodes
                                if node.total_saved_cnots == max_total_saved_cnots]
    # Nodes with the minimum depth (which depends on the size of the node's largest subsystem).
    # Shallower circuits with the same number of CNOTs means more parallelism.
    min_depth = _max_subsystem_size(min(max_saved_cnots_nodes, key=_max_subsystem_size))
    min_depth_nodes = [node for node in max_saved_cnots_nodes
                                if _max_subsystem_size(node) == min_depth]

    # Node with the lowest fidelity loss among the nodes with
    # the highest reduction in the number of CNOTs.
    if figure_of_merit == 'fidelity_loss':
        key = evaluate_fidelity_loss(max_figure_of_merit)
    elif figure_of_merit == 'mse' and original_state is not None:
        original_state = np.asarray(original_state)
        key = evaluate_mse(max_figure_of_merit, original_state)
    else:
        raise ValueError(f"The figure of merit '{figure_of_merit}' is not or not correctly configured.")
    return min(min_depth_nodes, key=key)

def _max_subsystem_size(node):
    return len(max(node.qubits, key=len))

def _to_qubits(n_state_vector):
    return int(log2(n_state_vector))

def _count_saved_cnots(original_vector, subsystem1_vector, subsystem2_vector,
                            original_partition=None,subsystem_local_partition=None,
                            original_rank=0, subsystem_rank=0):
    method = 'estimate'

    cnots_originally = schmidt_cnots(
                        original_vector, method=method, partition=original_partition,
                        low_rank=original_rank
                    )
    cnots_phase_3 = schmidt_cnots(
                        subsystem1_vector, method=method, partition=subsystem_local_partition,
                        low_rank=subsystem_rank
                    )

    cnots_phase_4 = 0
    if subsystem2_vector is not None:
        cnots_phase_4 = schmidt_cnots(subsystem2_vector, method=method)

    return cnots_originally - cnots_phase_3 - cnots_phase_4
