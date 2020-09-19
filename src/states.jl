import POMDPs
# State space; not complete and no need to initiate

# Initial state distribution
# function POMDPs.initialstate_distribution(pomdp::FireWorld)
function POMDPs.initialstate(pomdp::FireWorld)
    s = FireState[]
    probs = Array{Float64,1}(undef,0)
    
    init_state = pomdp.state
    burning = init_state.burning
#     @show burning
    burn_probs = init_state.burn_probs
#     @show burn_probs
    fuels = init_state.fuels

    burn_probs_for_obs = deepcopy(burn_probs)
    not_burning = findall(x->x==0, burning)
    all_cells = [1:total_size]
    cells_burning = all_cells[1][minus(not_burning, all_cells[1])]
#     @show not_burning
    for b_i in cells_burning
        burn_probs_for_obs[b_i] = 1.0
    end
#     @show burn_probs_for_obs
    # now pick thresholds 
    thresholds = collect(1:19)/20
    individual_prob = 1/length(thresholds)
    burns = []
    for threshold_k in thresholds
        burning_k = burn_probs_for_obs .> threshold_k
        # @show burning_k
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
#     println("initial belief, ", s)
#     println("initial belief probs, ", probs)
    return SparseCat(s, normalize!(probs,1))
end

# init_belief = initialstate_distribution(pomdp)
# init_belief = initialstate(pomdp)


# function make_burn_size(total_size::Int64, burn_perc::Float64)
#     burn_size = floor(Int, total_size * burn_perc)
#     if burn_size in [0, 1]
#         burn_size = 2
#     end
#     not_burn_size = total_size - burn_size
#     return [burn_size, not_burn_size]
# end
    
# function burn_map(burn_threshold::Float64, burns_size::Array{Int64,1}, rng::AbstractRNG)
#     burn_prob_array = rand(rng, Uniform(burn_threshold,1), burns_size[1])
#     # println("burn_prob_array ", burn_prob_array)
#     not_burn_prob_array = rand(rng, Uniform(0, burn_threshold), burns_size[2])
#     burn_map = vcat(burn_prob_array, not_burn_prob_array)
#     return shuffle!(rng, burn_map)
# end


# function make_cost_size(total_size::Int64, costs_perc::Array{Float64,1})
#     high_cost_size = floor(Int, total_size * costs_perc[1])
#     mid_cost_size = floor(Int, total_size * costs_perc[2])
#     low_cost_size = total_size - high_cost_size - mid_cost_size
#     return [high_cost_size, mid_cost_size, low_cost_size]
# end
    
# function cost_map(costs_array::Array{Int64,1}, costs_size::Array{Int64,1}, rng::AbstractRNG)
#     high_cost_array = repeat([costs_array[1]], costs_size[1])
#     mid_cost_array = repeat([costs_array[2]], costs_size[2])
#     low_cost_array = repeat([costs_array[3]], costs_size[3])
#     costs = vcat(high_cost_array, mid_cost_array, low_cost_array)
#     return shuffle!(rng, costs)
# end