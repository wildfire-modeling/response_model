module FirePOMDP

using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using MCTS
using D3Trees
using Random
using StaticArrays
using Parameters
using DiscreteValueIteration
using LinearAlgebra
using Combinatorics
using Distances
using ParticleFilters
using BeliefUpdaters
using StatsBase
using BasicPOMCP
using Distributions
import Distributions: Uniform
using POMCPOW
using JLD

export
    FireWorld,
    FireState,
    FireObs


# initial burn map
# criteria: only a % burning - set maximum number
const BURN_PERC = 0.1
const BURN_THRESHOLD = 0.6
function make_burn_size(total_size::Int64, burn_perc::Float64)
    burn_size = floor(Int, total_size * burn_perc)
    if burn_size in [0, 1]
        burn_size = 2
    end
    not_burn_size = total_size - burn_size
    return [burn_size, not_burn_size]
end
    
function burn_map(burn_threshold::Float64, burns_size::Array{Int64,1}, rng::AbstractRNG)
    burn_prob_array = rand(rng, Uniform(burn_threshold,1), burns_size[1])
    # println("burn_prob_array ", burn_prob_array)
    not_burn_prob_array = rand(rng, Uniform(0, burn_threshold), burns_size[2])
    burn_map = vcat(burn_prob_array, not_burn_prob_array)
    return shuffle!(rng, burn_map)
end

# make initial cost map
# high_cost_perc = 0.2 # according to the 2010 Census, 
#                      # 95 percent of Californians live on just 5.3 percent of the land in the state
#                      # Less than 17 percent of the San Francisco Bay Area is developed
# mid_cost_perc = 0.3 # percentage of California forest coverage
# low_cost_perc = 1 - high_cost_perc - mid_cost_perc # 0.5

const COSTS_PERC = [0.2, 0.3, 0.5]
const COSTS_ARRAY = [-10, -5, -1]
# costs = rand(costs_array, total_size);

function make_cost_size(total_size::Int64, costs_perc::Array{Float64,1})
    high_cost_size = floor(Int, total_size * costs_perc[1])
    mid_cost_size = floor(Int, total_size * costs_perc[2])
    low_cost_size = total_size - high_cost_size - mid_cost_size
    return [high_cost_size, mid_cost_size, low_cost_size]
end
    
function cost_map(costs_array::Array{Int64,1}, costs_size::Array{Int64,1}, rng::AbstractRNG)
    high_cost_array = repeat([costs_array[1]], costs_size[1])
    mid_cost_array = repeat([costs_array[2]], costs_size[2])
    low_cost_array = repeat([costs_array[3]], costs_size[3])
    costs = vcat(high_cost_array, mid_cost_array, low_cost_array)
    return shuffle!(rng, costs)
end

function make_cost_map(GRID_SIZE::Int64, COSTS_PERC::Array{Float64,1}, COSTS_ARRAY::Array{Int64,1}, rng::AbstractRNG)
    costs_size = make_cost_size(GRID_SIZE * GRID_SIZE, COSTS_PERC)
    COSTS = cost_map(COSTS_ARRAY, costs_size, rng)
    return COSTS
end


# initial fuel level
const DEFAULT_FUEL = 5

# initial weather condition
const WIND = [1, 1, 5]

# make initial state
const RAND_INIT = 264
rng = MersenneTwister(RAND_INIT);

# initial state
function make_initial_state(GRID_SIZE::Int64)
    burns_size = make_burn_size(GRID_SIZE * GRID_SIZE, BURN_PERC)
    burn_prob_map = burn_map(BURN_THRESHOLD, burns_size, rng)
    init_burn = burn_prob_map .> BURN_THRESHOLD
    init_fuels = ones(Int, GRID_SIZE * GRID_SIZE) * DEFAULT_FUEL
    return FireState(init_burn, burn_prob_map, init_fuels)
end


# POMDP State
struct FireState
    burning::BitArray{1} # an array maintaining what cells are burning
    burn_probs::Array{Float64,1} # an array of probability of cells burning
    fuels::Array{Int64,1} # an array maintaining each cell's fuel level
end

# POMDP Observation: only observes burning or not
struct FireObs
    burning::BitArray{1} # an array what cells are seen to be burning
end

# @show init_state = FireState(init_burn, burn_prob_map, init_fuels)

# POMDP{State, Action, Observation}
@with_kw struct FireWorld{GRID_SIZE} <: POMDP{FireState, Array{Int64, 1}, FireObs} 
    # size of grid
    grid_size::Int64 = GRID_SIZE
    # initial state
    state::FireState = make_initial_state(grid_size)
    # fire total size, for visualization
    map_size::Tuple{Int,Int} = (grid_size, grid_size)
    # cost map
    costs::Array{Int64,1} = make_cost_map(grid_size, COSTS_PERC, COSTS_ARRAY, rng)
    # 1-sensitivity; probability of an observation of no burning when it actually is, e.g. burning but sensor does not detect
    bprob_fn::Float64 = 0.2
    # 1-specificity; probability of an observation of burning when it isn't, false positive; due to lag cells are likely to be burning (more likely if p_small is lower)
    bprob_fp::Float64 = 0.1
    # probability of successfully putting out a fire
    tprob::Float64 = 1.0
    # discount factor
    discount::Float64 = 0.95
    # consists of strength, acceleration, and direction
    wind::Array{Int64,1} = WIND
end

# Discount factor
POMDPs.discount(pomdp::FireWorld) = pomdp.discount;

# Terminal condition
POMDPs.isterminal(pomdp::FireWorld, state::FireState) = sum(state.burning) == 0

# Equal condition
POMDPs.isequal(s1::FireState, s2::FireState) = s1.burning == s2.burning && s1.fuels == s2.fuels && s1.burn_probs == s2.burn_probs


include("updater.jl")
include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")
# include("visualization.jl")

end # module