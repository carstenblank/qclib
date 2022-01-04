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

from dataclasses import dataclass
from itertools import combinations, chain
from typing import List
from math import log2, sqrt
import numpy as np
import tensorly as ty

from qclib.entanglement import geometric_entanglement
from qclib.state_preparation.schmidt import cnot_count as schmidt_cnots, \
                                            schmidt_decomposition, \
                                            schmidt_composition, \
                                            low_rank_approximation

# pylint: disable=missing-class-docstring

def adaptive_approximation(state_vector, max_fidelity_loss, strategy='greedy',
                                        max_combination_size=0, use_low_rank=False):
    """
    It reduces the entanglement of the given state, producing an approximation
    to reduce the complexity of the quantum circuit needed to prepare it.
    `https://arxiv.org/abs/2111.03132`_.
    Args:
        state_vector (list):
            A state vector to be approximated by a less complex state.
        max_fidelity_loss (float):
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
    qubits = [list(range(n_qubits))]
    vectors = [state_vector]

    entanglement, product_state = geometric_entanglement(state_vector, return_product_state=True)
    if max_fidelity_loss >= entanglement:
        full_cnots = schmidt_cnots(state_vector)
        qubits = [[n] for n in range(len(product_state))]
        ranks = [1 for _ in range(len(product_state))]
        return Node(
            full_cnots, full_cnots, entanglement, entanglement, product_state, qubits, ranks, []
        )

    root_node = Node(0, 0, 0.0, 0.0, vectors, qubits, [0], [])
    _build_approximation_tree(root_node, max_fidelity_loss, strategy,
                                    max_combination_size, use_low_rank)

    leafs = []
    _search_leafs(root_node, leafs)

    best_node = _search_best(leafs)

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

    state: List[complex]
    register: List[int]
    partition: List[int]
    local_partition: List[int]

    fidelity_loss: float

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
    qubits: List[List[int]]
    ranks: List[int]

    nodes: List['Node']

    @property
    def is_leaf(self) -> bool:
        """
        True if the all vectors have reached an approximation assessment. There is no more
        decomposition/approximation possible. Therefore, the node is a leaf.
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


def _entanglement_evaluations(entangled_vector, entangled_qubits, disentanglement_list, use_low_rank):
    # Disentangles or reduces the entanglement of each bipartition of
    # entangled_qubits.
    entanglement_info_list = []
    for partition in disentanglement_list:
        # Computes the two state vectors after disentangling "partition".
        # If the bipartition cannot be fully disentangled, an approximate
        # state is returned.
        e_info_list = _reduce_entanglement(
            entangled_vector, entangled_qubits, partition, use_low_rank
        )
        entanglement_info_list += e_info_list
    return entanglement_info_list


def _create_all_entanglement_informations(node, strategy, max_k, use_low_rank):
    # Ignore the completely disentangled qubits.
    data = [(q, v) for q, v, k in zip(node.qubits, node.vectors, node.ranks) if k == 0]

    entanglement_info_list = []
    for entangled_qubits, entangled_vector in data:

        if not 1 <= max_k <= len(entangled_qubits)//2:
            max_k = len(entangled_qubits)//2

        if strategy == 'greedy':
            combs = _greedy_combinations(entangled_vector, entangled_qubits, max_k)
        else:
            combs = _all_combinations(entangled_qubits, max_k)

        entanglement_info_list += _entanglement_evaluations(
            entangled_vector, entangled_qubits, list(combs), use_low_rank
        )
    return entanglement_info_list


def _search_level(node, max_fidelity_loss, strategy, max_k, use_low_rank=False) -> Node:
    entanglement_info_list = _create_all_entanglement_informations(
        node, strategy, max_k, use_low_rank
    )

    node_fidelity_loss = np.array(
        [info.fidelity_loss for info in entanglement_info_list]
    )
    total_fidelity_loss = 1.0 - (1.0 - node_fidelity_loss) * \
                                (1.0 - node.total_fidelity_loss)
    # Removing all those nodes, whose total fidelity loss exceed beyond the required threshold
    filtered_nodes = [
        _create_node(node, e_info)
        for e_info, loss in zip(entanglement_info_list, total_fidelity_loss)
        if loss <= max_fidelity_loss
    ]
    # Also, we remove all those partitions, that don't give us any advantage! This saves a lot
    # of recursions!
    node.nodes = [n for n in filtered_nodes if n.node_saved_cnots > 0]

    if strategy == 'greedy' and len(node.nodes) > 0:
        # Locally optimal choice at each stage.
        node.nodes = [_search_best(node.nodes)]

    return node


def _next_level(nodes, max_fidelity_loss, strategy, max_k, use_low_rank=False) -> None:
    for new_node in nodes:
        # call _build_approximation_tree recurrently for each new node.
        # except that the vectors are matrices. In this case we are done.
        if not new_node.is_leaf:
            _build_approximation_tree(
                new_node, max_fidelity_loss, strategy, max_k, use_low_rank
            )


def _build_approximation_tree(node, max_fidelity_loss, strategy='brute_force', max_k=0,
                              use_low_rank=False, clear_memory=True):
    searched_node = _search_level(node, max_fidelity_loss, strategy, max_k, use_low_rank)

    # If it is not the end of the recursion,
    if clear_memory and len(searched_node.nodes) > 0:
        # clear vectors and qubits to save memory.
        searched_node.vectors.clear()
        # This information is no longer needed from this point
        # on (but may be needed in the future).
        searched_node.qubits.clear()

    _next_level(searched_node.nodes, max_fidelity_loss, strategy, max_k, use_low_rank)


def _all_combinations(entangled_qubits, max_k):
    return chain.from_iterable(combinations(entangled_qubits, k)
                                            for k in range(1, max_k+1))

def _greedy_combinations(entangled_vector, entangled_qubits, max_k):
    """
    Combinations with a qubit-by-qubit analysis.
    Returns only one representative of the bipartitions of size k (1<=k<=max_k).
    The increment in the partition size is done by choosing the qubit that has
    the lowest fidelity-loss when removed from the remaining entangled subsystem.
    """
    node = Node( 0, 0, 0.0, 0.0, [entangled_vector], [entangled_qubits], [0], [] )
    for _ in range(max_k):
        current_vector = node.vectors[-1] # Last item is the current entangled state.
        current_qubits = node.qubits[-1]

        nodes = []
        # Disentangles one qubit at a time.
        for qubit_to_disentangle in current_qubits:
            entanglement_info = \
                _reduce_entanglement(current_vector, current_qubits, [qubit_to_disentangle])

            new_node = _create_node(node, entanglement_info[0])

            nodes.append(new_node)
        # Search for the node with lowest fidelity-loss.
        node = _search_best(nodes)

    # Build the partitions by incrementing the number of selected qubits.
    # Returns only one partition for each length k.
    # All disentangled qubits are in the slice "node.qubits[0:max_k]", in the order in which
    # they were selected. Each partition needs to be sorted to ensure that the correct
    # construction of the circuit.
    return tuple( sorted( chain(*node.qubits[:k]) ) for k in range(1, max_k+1) )

def _reduce_entanglement(state_vector, register, partition, use_low_rank=False):
    local_partition = []
    # Maintains the relative position between the qubits of the two subsystems.
    for qubit_to_disentangle in partition:
        local_partition.append(
                        sum(i < qubit_to_disentangle for i in register))

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

        fidelity_loss = 1.0 - sum(low_rank_s**2)

        entanglement_info.append(Entanglement(rank, low_rank_u, low_rank_v, low_rank_s,
                                              state_vector,
                                              register,
                                              partition,
                                              local_partition,
                                              fidelity_loss))
    return entanglement_info

def _create_node(parent_node: Node, e_info: Entanglement):

    vectors = parent_node.vectors.copy()
    qubits  = parent_node.qubits.copy()
    ranks = parent_node.ranks.copy()

    index = parent_node.qubits.index(e_info.register)
    vectors.pop(index)
    qubits.pop(index)
    ranks.pop(index)

    if e_info.rank == 1:
        # The partition qubits have been completely disentangled from the
        # rest of the register. Therefore, the original entangled state is
        # removed from the list and two new separate states are included.
        partition1 = tuple(set(e_info.register).difference(set(e_info.partition)))
        partition2 = e_info.partition

        vectors.append( e_info.svd_v.T[:, 0] )
        qubits.append( partition2 )
        partition2_rank = 1 if e_info.svd_v.T[:, 0].shape[0] == 2 else 0
        ranks.append(partition2_rank)

        vectors.append( e_info.svd_u[:, 0] )
        qubits.append( partition1 )
        partition1_rank = 1 if e_info.svd_u.T[:, 0].shape[0] == 2 else 0
        ranks.append(partition1_rank)

        node_saved_cnots = _count_saved_cnots(e_info.state, vectors[-1], vectors[-2])
    else:
        # The entanglement between partition qubits and the rest of the
        # register has been reduced, but not eliminated. Therefore, the
        # original state is replaced by an approximate state.
        normed_svd_s = e_info.svd_s/sqrt( 1.0 - e_info.fidelity_loss )
        approximate_state = schmidt_composition(e_info.svd_u, e_info.svd_v,
                                                normed_svd_s, e_info.local_partition)
        vectors.append( approximate_state )
        qubits.append( e_info.register )
        ranks.append(e_info.rank)

        node_saved_cnots = _count_saved_cnots(e_info.state, vectors[-1], None)

    total_saved_cnots = parent_node.total_saved_cnots + node_saved_cnots
    total_fidelity_loss = 1.0 - (1.0 - e_info.fidelity_loss) * \
                                (1.0 - parent_node.total_fidelity_loss)

    return Node(
        node_saved_cnots, total_saved_cnots, e_info.fidelity_loss,
        total_fidelity_loss, vectors, qubits, ranks, []
    )

def _search_leafs(node, leafs):
    # It returns the leaves of the tree. These nodes are the ones with
    # total_fidelity_loss closest to max_fidelity_loss for each branch.
    if len(node.nodes) == 0:
        leafs.append(node)
    else:
        for child in node.nodes:
            _search_leafs(child, leafs)

def _search_best(nodes):
    # Nodes with the greatest reduction in the number of CNOTs.
    # There may be several with the same number.
    max_total_saved_cnots = max(nodes, key=lambda n: n.total_saved_cnots).total_saved_cnots
    max_saved_cnots_nodes = [node for node in nodes
                                if node.total_saved_cnots == max_total_saved_cnots]
    # Nodes with the minimum depth (wich depends on the size of the node's largest subsystem).
    # Shallower circuits with the same number of CNOTs means more parallelism.
    min_depth = _max_subsystem_size(min(max_saved_cnots_nodes, key=_max_subsystem_size))
    min_depth_nodes = [node for node in max_saved_cnots_nodes
                                if _max_subsystem_size(node) == min_depth]
    # Node with the lowest fidelity loss among the nodes with
    # the highest reduction in the number of CNOTs.
    return min(min_depth_nodes, key=lambda n: n.total_fidelity_loss)

def _max_subsystem_size(node):
    return len(max(node.qubits, key=len))

def _to_qubits(n_state_vector):
    return int(log2(n_state_vector))

def _count_saved_cnots(entangled_vector, subsystem1_vector, subsystem2_vector):
    method = 'estimate'

    cnots_originally = schmidt_cnots(entangled_vector, method=method)
    cnots_phase_3 = schmidt_cnots(subsystem1_vector, method=method)

    cnots_phase_4 = 0
    if subsystem2_vector is not None:
        cnots_phase_4 = schmidt_cnots(subsystem2_vector, method=method)

    return cnots_originally - cnots_phase_3 - cnots_phase_4
