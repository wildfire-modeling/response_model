import POMDPs
using POMDPModelTools

mutable struct HistoryUpdater <: POMDPs.Updater end

# update(up, pomdp, b, a, o)
# function POMDPs.update(up::HistoryUpdater, pomdp::FireWorld, b, a::Array{Int64,1}, o::FireObs)
function POMDPs.update(up::HistoryUpdater, pomdp::FireWorld, b::SparseCat{Array{FireState,1},Array{Float64,1}}, a::Array{Int64,1}, o::FireObs)
    # particle filter without rejection
    states = FireState[]
    probabilities = Array{Float64,1}(undef,0)
    
    next_states = FireState[]
    weights = Array{Float64,1}(undef,0)
    
    belief_particles = ParticleCollection(b.vals) 
    for i in 1:n_particles(belief_particles)
        s_i = rand(rng, belief_particles)
        sp_gen = rand(rng, transition(pomdp, s_i, a))
        obs_dist = observation(pomdp, a, sp_gen)
        w_i = in_dist_obs(obs_dist, o)
        push!(next_states, sp_gen)
        push!(weights, w_i)
    end
    if sum(weights) == 0 # all next states do not have observation o
        println("All zero. No observation o.")
        weights = ones(length(weights)) * (1/length(weights))
    end
    for_sampling = SparseCat(next_states, weights)
    
    # resample
    for i in 1:n_particles(belief_particles)
        sp_sampled = sample(next_states, Weights(weights))
        in_sample, idx = in_dist_states(states, sp_sampled)
        if !in_sample # new state
            push!(states, sp_sampled)
            push!(probabilities, pdf(for_sampling, sp_sampled))
        else # state already represented
            probabilities[idx] += probabilities[idx]
        end
    end
    return SparseCat(states, normalize!(probabilities,1))
end

POMDPs.update(up::HistoryUpdater, b::SparseCat{Array{FireState,1},Array{Float64,1}}, a::Array{Int64,1}, o::FireObs) = update(up, pomdp, b, a, o)
POMDPs.initialize_belief(updater::HistoryUpdater, belief::Any) = belief


# returns probability of observing o
function in_dist_obs(obs_dist::SparseCat{Array{FireObs,1},Array{Float64,1}}, o::FireObs)
    in_dist = false
    prob = 0.0
    for obs in obs_dist.vals
        burning = obs.burning
        if burning == o.burning
            in_dist = true
            prob = pdf(obs_dist, obs)
        end
    end
    return prob
end

# returns boolean whether a state is in distribution and its index
function in_dist_states(s_dist::Array{FireState,1}, s::FireState)
    in_dist = false
    idx = 0
    for i in 1:length(s_dist)
        state = s_dist[i]
        if isequal(state, s)  
            in_dist = true
            idx = i
        end
    end
    return in_dist, idx
end