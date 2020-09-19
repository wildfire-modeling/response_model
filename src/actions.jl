# import POMDPs
# Action space
const MAX_ACT = 1

function POMDPs.actions(pomdp::FireWorld)
    total_size = pomdp.map_size[1] * pomdp.map_size[2]
    x = Array[1:total_size]
    actions = []
    for i in 1:MAX_ACT # restricting resources, MAX_ACT number of airplanes or less
        push!(actions, collect(combinations(x[1],i)))
    end
    return collect(Iterators.flatten(actions))
#     return collect(combinations(x[1]))
end

# action_space = actions(mdp);
# POMDPs.n_actions(pomdp::FireWorld) = length(actions(pomdp))
POMDPs.actionindex(pomdp::FireWorld, a::Array{Int64,1}) = findall(x->x==a, actions(pomdp))[1]

