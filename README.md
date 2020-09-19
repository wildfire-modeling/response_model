# Wildfire Modeling Response Model



## Problem description


- **States**: whether each cell of the grid is burning, the probability of burning at each cell, and the amount of fuel level left in each cell.

- **Actions**: any of the cells to apply suppression effort to, maximum of one action at a time assumed.  

- **Transition model**: when a suppression action is taken, with probability `tprob` the action is successful, in which case the cell that the action is applied to is no longer burning. All previous cells that were burning incur one level of fuel decrement. Then, the fire spreads by one time step and the probability burning of at each cell updates. Any cell with fuel left is subjected to burning, including the cell that a successful action has been applied to.

- **Observation model**: the agent observes whether a cell is on fire or not after taking a suppression action and the fire spreads. The cell that the agent applies an action to has full observability whereas the other cells are approximated through the burning probabilities and the burning thresholds.

- **Reward model**: the agent receives the sum of the negative reward (negative number) of all cells that are burning. Each cell may have a different reward or cost to burn.

### Example

```julia
using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using Random
using LinearAlgebra
using StatsBase
using Combinatorics
using FirePOMDP

GRID_SIZE = 12
COSTS = cost_map(GRID_SIZE)
MAX_ACT = 1

pomdp = FirePOMDP(state=FireState(GRID_SIZE), 
                        map_size = (GRID_SIZE, GRID_SIZE),
                        costs = COSTS,
                        bprob_fn = 0.2,
                        bprob_fp = 0.1,
                        tprob = 0.8,
                        discount=0.95.
                        wind = [1, 1, 5]) 

a_default = sortperm(pomdp.costs)[1:MAX_ACT]
solver = POMCPOWSolver(rng=MersenneTwister(264), default_action = a_default, tree_queries = 1000, max_time = 60);

planner = solve(solver, pomdp);
up = HistoryUpdater();

s0 = rand(MersenneTwister(264), initialstate(pomdp))
b0 = initialize_belief(up, initialstate(pomdp))

for (s,a,r,sp,o) in stepthrough(pomdp, planner, up, b0, s0, "s, a, r, sp, o")
    println("in state $s")
    println("took action $a")
    println("received observation $o and reward $r")
end
```


**`FirePOMDP` Parameters:** 

- constructor: `FireWorld(kwargs...)` 
- keyword arguments: 
  - `state::FireState` , the initial state of the fire world
  - `map_size` the size of the fire grid,  default (3,3)
  - `costs` the negative utility of each cell being on fire
  - `bprob_fn::Float64`, probability of an observation of no burning when it actually is, e.g. burning but sensor does not detect, default 0.2
  - `bprob_fp::Float64`, probability of an observation of burning when it isn't, false positive; due to lag cells are likely to be burning, default 0.1
  - `tprob::Float64` probability of successfully putting out a fires, default 0.8
  - `discount::Float64` default 0.95
  - `wind`, the environment factors characterized by wind; shown as a vector of strength, acceleration, and direction, default [1, 1, 5] indicating lowest wind with south direction

**Internal types:**

`FireState` : represents the state of the fire world with whether each cell is on fire, its probability of being on fire, and fuel level left.
`FireObs`: represents the observation of the fire world with whether each cell is on fire.
