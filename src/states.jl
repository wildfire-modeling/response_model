import POMDPs
# State space; not complete and no need to initiate

# Initial state distribution
# function POMDPs.initialstate_distribution(pomdp::FireWorld)
function POMDPs.initialstate(pomdp::FireWorld)
    total_size = pomdp.grid_size * grid_size
    s = FireState[]
    probs = Array{Float64,1}(undef,0)
    
    init_state = pomdp.state
    burning = init_state.burning
    burn_probs = init_state.burn_probs
    fuels = init_state.fuels

    burn_probs_for_obs = deepcopy(burn_probs)
    not_burning = findall(x->x==0, burning)
    all_cells = [1:total_size]
    cells_burning = all_cells[1][minus(not_burning, all_cells[1])]
    for b_i in cells_burning
        burn_probs_for_obs[b_i] = 1.0
    end

    thresholds = collect(1:19)/20
    individual_prob = 1/length(thresholds)
    burns = []
    for threshold_k in thresholds
        burning_k = burn_probs_for_obs .> threshold_k
        # need to check if already in
        if burning_k in burns
            b_idx = findall(x->x==burning_k, burns)
            prob = probs[b_idx[1]]
            probs[b_idx[1]] = prob + individual_prob
        else
            push!(burns, burning_k)
            push!(probs, individual_prob)
            push!(s, FireState(burning_k, burn_probs, fuels))
        end
    end
    return SparseCat(s, normalize!(probs,1))
end
