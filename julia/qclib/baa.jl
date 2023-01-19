#=
baa:
- Julia version: 
- Author: Carsten Blank <blank@data-cybernetics.com>
- Date: 2022-04-13
=#

 mutable struct Node

    node_saved_cnots::Int32
    total_saved_cnots::Int32

    node_fidelity_loss::Real
    total_fidelity_loss::Real

    vectors::Array{Array{Complex}}

    qubits::Array{Array{Int64}}
    ranks::Array{Int64}
#     partitions: List[Optional[Tuple[int]]]
    nodes::Array{Node}

    Node() = new()
end

function main()
    # do stuff
    node = Node()
    node.node_saved_cnots = 10
    node.node_fidelity_loss = 0.0
    node.total_fidelity_loss = 0.0
    node.vectors = [[0.0, 0.1], [0.5], [0.7]]
    print(node)
end

main()
