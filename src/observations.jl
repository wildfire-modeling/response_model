# Observations
# observation(pomdp, a, sp)
function POMDPs.observation(pomdp::FireWorld, a::Array{Int64,1}, sp::FireState)
    neighbors = FireObs[]
    probabilities = Array{Float64,1}(undef,0)
    
    burning = sp.burning
    fuels = sp.fuels
    
    P_xy = fire_spread(pomdp.wind, sp)
    s_new = update_burn_probs(sp, P_xy)
    burn_probs_new = s_new.burn_probs
    
    # idea is
    # update burning in observations
    # only change the cells that were not acted upon
    # because if action was applied, fp and fn rates are 0 (eyes on fire)
    # otherwise, some delay happens and observation may not be the same
#     all_cells = collect(1:total_size)
#     no_action_cells = all_cells[minus(a, all_cells)]
    # update the burn_probs map for actions taken to be 1.0 (so it won't be sensitive to threshold we pick)
    burn_probs_for_obs = deepcopy(burn_probs_new)
    for a_i in a
        burn_probs_for_obs[a_i] = 1.0
    end
    # now pick thresholds 
    thresholds = collect(12:19)/20
    individual_prob = 1/length(thresholds)
    burns = []
    for threshold_k in thresholds
        burning_k = burn_probs_for_obs .> threshold_k
        # need to check if already in
        if burning_k in burns
            b_idx = findall(x->x==burning_k, burns)
            prob = probabilities[b_idx[1]]
            probabilities[b_idx[1]] = prob + individual_prob
        else
            push!(burns, burning_k)
            push!(probabilities, individual_prob)
            push!(neighbors, FireObs(burning_k))
        end
    end
    return SparseCat(neighbors, probabilities)
end