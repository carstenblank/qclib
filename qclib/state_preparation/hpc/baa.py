from typing import List

import numpy as np
from dask.distributed import get_client
from dask.distributed import Client
from distributed import Future, as_completed

import qclib.state_preparation.util.baa as baa


class ApproximationTreeBuilder:
    max_fidelity_loss: float
    strategy: str
    max_k: int
    use_low_rank: bool
    clear_memory: bool

    def __init__(self, max_fidelity_loss, strategy='brute_force', max_k=0, use_low_rank=False,
                 clear_memory=True) -> None:
        self.max_fidelity_loss = max_fidelity_loss
        self.strategy = strategy
        self.max_k = max_k
        self.use_low_rank = use_low_rank
        self.clear_memory = clear_memory

    def build(self, node):
        client = get_client()


def _start_reduce_entanglement(d):
    return baa._reduce_entanglement(*d)


class HPCBAA:

    client: Client

    def __init__(self, client: Client):
        self.client = client

    def _entanglement_evaluations(self, entangled_vector, entangled_qubits, disentanglement_list,
                                  use_low_rank):
        # Disentangles or reduces the entanglement of each bipartition of
        # entangled_qubits.
        # Computes the two state vectors after disentangling "partition".
        # If the bipartition cannot be fully disentangled, an approximate
        # state is returned.
        data = [(entangled_vector, entangled_qubits, partition, use_low_rank)
                for partition in disentanglement_list]
        entanglement_info_list_futures: List[Future] = self.client.map(_start_reduce_entanglement, data)
        return entanglement_info_list_futures

    def _create_all_entanglement_informations(self, node, strategy, max_k, use_low_rank):
        # Ignore the completely disentangled qubits.
        data = [(q, v) for q, v, k in zip(node.qubits, node.vectors, node.ranks) if k == 0]

        entanglement_info_list: List[Future] = []
        for entangled_qubits, entangled_vector in data:

            if not 1 <= max_k <= len(entangled_qubits) // 2:
                max_k = len(entangled_qubits) // 2

            if strategy == 'greedy':
                combs = baa._greedy_combinations(entangled_vector, entangled_qubits, max_k)
            else:
                combs = baa._all_combinations(entangled_qubits, max_k)

            entanglement_info_list += self._entanglement_evaluations(
                entangled_vector, entangled_qubits, list(combs), use_low_rank
            )
        return entanglement_info_list

    def override_search_level(self):
        def _search_level(node, max_fidelity_loss, strategy, max_k, use_low_rank=False) -> baa.Node:
            entanglement_info_futures: List[Future] = self._create_all_entanglement_informations(
                node, strategy, max_k, use_low_rank
            )

            filtered_nodes = []
            for future in as_completed(entanglement_info_futures):
                info_list: List[baa.Entanglement] = future.result()
                node_fidelity_loss = np.array(
                    [info.fidelity_loss for info in info_list]
                )
                total_fidelity_loss = 1.0 - (1.0 - node_fidelity_loss) * (1.0 - node.total_fidelity_loss)
                for info, loss in zip(info_list, total_fidelity_loss):
                    # Removing all those nodes, whose total fidelity loss exceed beyond the required threshold
                    if loss <= max_fidelity_loss:
                        new_node = baa._create_node(node, info)
                        # Also, we remove all those partitions, that don't give us any advantage! This saves a lot
                        # of recursions!
                        if new_node.node_saved_cnots > 0:
                            print(f"Found good node: {new_node.total_fidelity_loss} / {new_node.total_saved_cnots} / {new_node.qubits} / {new_node.ranks}")
                            # Send to queue
                            # Then execute new round:
                            # if not new_node.is_leaf:
                            #     baa._build_approximation_tree(
                            #         new_node, max_fidelity_loss, strategy, max_k, use_low_rank
                            #     )
                            filtered_nodes.append(new_node)
            # Update the nodes now
            node.nodes = filtered_nodes

            if strategy == 'greedy' and len(node.nodes) > 0:
                # Locally optimal choice at each stage.
                node.nodes = [baa._search_best(node.nodes)]

            return node
        return _search_level

    def adaptive_approximation(self, state_vector, max_fidelity_loss, strategy='greedy',
                               max_combination_size=0, use_low_rank=False):
        # Monkey patching:
        original = baa._search_level
        baa._search_level = self.override_search_level()
        node = baa.adaptive_approximation(
            state_vector, max_fidelity_loss, strategy, max_combination_size, use_low_rank
        )
        baa._search_level = original
        return node
