# using POMDPModelTools
const NEIGHBOR_DIST = 1 # how big a neighborhood, in terms of Euclidean distance 

# # not used now
# const WIND_DIR_DICT = Dict(:north=>1, 
#                         :north_east=>2, 
#                         :east=>3, 
#                         :south_east=>4, 
#                         :south=>5, 
#                         :south_west=>6, 
#                         :west=>7, 
#                         :north_west=>8)

function POMDPs.transition(pomdp::FireWorld, state::FireState, action::Array{Int64,1})
    # 1. update transition - it happens when you start from a state and take an action
#     sp_trans = rand(rng, transition_of_states(pomdp, state, action))
#     @show sp_trans
    sp_trans_dist = transition_of_states(pomdp, state, action)
    # println("sp_trans_dist", sp_trans_dist)
    
    neighbors = FireState[] # empty arrays of FireStates
    # probabilities = Array{Float64,1}(undef,0)
    probabilities = Float64[]
    
    for sp_trans in sp_trans_dist.vals
        # 2. update fuels - other cells that were burning but no action was taking to put it out
        prob = pdf(sp_trans_dist, sp_trans)

        sp_burn = sp_trans.burning
        sp_burn_probs = sp_trans.burn_probs
        sp_fuels = sp_trans.fuels

        # get indices of cells where no put out action was applied to
        all_cells = Array[1:total_size]
        no_action_cells = all_cells[1][minus(action, all_cells[1])]
    #     @show no_action_cells

        # update fuel levels of cells that were burning yet no action was applied to
        sp_fuels_new = deepcopy(sp_fuels)
        for cell in no_action_cells
            if sp_burn[cell]
    #             sp_fuels_new[cell] = max(sp_fuels[cell]-1,0)
                new_fuel = sp_fuels[cell]-1
                if new_fuel < 1
                    sp_fuels_new[cell] = 0
                    sp_burn[cell] = false
                else
                    sp_fuels_new[cell] = new_fuel
                end
            end
        end
        # want to keep the burning indicator even if fuel of a cell is 0 for step 3 below
        # as that cell may spread the fire to other cells despite itself being burned to fuel exhaustion
        sp_new = FireState(sp_burn, sp_burn_probs, sp_fuels_new)
    #     @show sp_new

        # 3. update fire spread
        # wind is a dictionary that contains wind_strength, wind_acc (for speed), and wind_dir
#         wind = wind_init(rng)
        P_xy = fire_spread(pomdp.wind, sp_new)
        sp = update_burn_probs(sp_new, P_xy)
#         @show sp
        
        push!(neighbors, sp)
        push!(probabilities, prob)
    end
    return SparseCat(neighbors, normalize!(probabilities, 1))
end

# Transition helper
function transition_of_states(pomdp::FireWorld, state::FireState, action::Array{Int64,1})
    tprob = pomdp.tprob
    burning = state.burning
    burn_probs = state.burn_probs
    fuels = state.fuels
    neighbors = FireState[]
    probabilities = Array{Float64,1}(undef,0)
    
    # action is an array of indices that we will apply fire fighting actions to
    # those cells may or may not be burning
    # 1. find cell indices of the action set where cell is burning now
    act_burn = burning[action] .* action 
    act_on_burning_cells = act_burn[findall(x->x>0, act_burn)]
    num_act_n_burning = length(act_on_burning_cells)
    
    # check if any cell is burning now
    if num_act_n_burning > 0 
        # 1.(a) decrement fuels in those cells
        fuels_new = [i in act_on_burning_cells ? max(0, fuels[i]-1) : fuels[i] for i in 1:length(fuels) ]
        # 1.(b) account for varying fire fighting outcomes
        # some cells may be put out successfully and some may not be
        # for i in space([false, true], num_act_n_burning) # tuple is faster
        for i in space((false, true), num_act_n_burning) # different permutations of fire fighting outcomes
            burn = collect(i)
            burning_new = update_fullburn(burning, burn, act_on_burning_cells)
            prob = (1-tprob)^sum(burn)*(tprob)^(num_act_n_burning - sum(burn)) # basically binomial
            push!(neighbors, FireState(burning_new, burn_probs, fuels_new))
            push!(probabilities, prob)
        end
    else # none of the cells we apply action to is burning to begin with
        push!(neighbors, FireState(burning, burn_probs, fuels))
        push!(probabilities, 1.0)
    end
    return SparseCat(neighbors, probabilities)
end


# update burning or not for cells that were burning
function update_fullburn(burning::BitArray{1}, burn_perm::Array{Bool,1}, burn_indices::Array{Int64,1})
    burns = deepcopy(burning)
    for i in 1:length(burn_perm)
        burn_new = burn_perm[i]
        idx = burn_indices[i]
        burns[idx] = burn_new
    end
    return burns
end

# FIRE SPREAD
# returns new FireState with different probability burn map
function update_burn_probs(s::FireState, P_xy::Array{Float64,2})
    burn = s.burning
    burn_probs = s.burn_probs
    # println("s burn_probs ", burn_probs)
    fuels = s.fuels
    # println("s fuels ", fuels)
    
    burn_probs_update = zeros(total_size)
    fuels_left = findall(x->x>0, fuels)
    # println("fuels_left", fuels_left)
    for i in fuels_left
        burn_probs_update[i] = burn_probs[i]
    end
    # @show burn_probs_update
    #     #normalize non-zeros
#     burn_probs_update = burn_probs_update ./ sum(burn_probs_update)

    burn_probs_result = zeros(total_size)
    for i in 1:total_size
        not_spread_i = 1
        for j in 1:total_size
            # @show i,j
            P_xy[i,j] , burn_probs_update[j]
            not_spread_j = 1 - P_xy[i,j] * burn_probs_update[j]
#             @show not_spread_j
            not_spread_i *= not_spread_j
            burn_probs_result[i] = 1 - not_spread_i 
        end
    end
    # @show burn_probs_result
    all_cells = collect(1:total_size)
    no_fuels = all_cells[minus(fuels_left, all_cells)]
    for i in no_fuels
        burn_probs_result[i] = 0
    end
#     @show burn_probs_result
    # println("burn_probs_result ", burn_probs_result)
    return FireState(burn, burn_probs_result, fuels)
end

function fire_spread(wind::Array{Int64,1}, s::FireState)
#     "Lambda won't be input; 
#     this needs to take input of distance, wind, and constant lambda_b (terrain-specifc)
#     to update lambda
#     Distance: can calculate from grid; the further, the harder to spread
#    # 2020/05/18 update: restrict to neighboring cells only
#     Wind: Rating of how strong wind is - want to get at direction and speed (relative to direction of two cells)
#     i.e. need 8 directions (for cell and wind)
#     Lambda_b: say, fuel level at the cell"
    wind_strength = wind[1]
    wind_acc = wind[2]
    wind_dir = wind[3]
    longest_dist = euclidean([1,1], [GRID_SIZE, GRID_SIZE])
    to_norm = wind_strength * wind_acc * DEFAULT_FUEL
    lambdas = zeros((total_size, total_size))
    P_xy = zeros((total_size, total_size))
    burning = s.burning
    # println("burning ", burning)
    fuels = s.fuels
    for i in 1:total_size
#         println("i is, $i")
        fuel_level = fuels[i]
        cart_i = cartesian[i]
        for j in 1:total_size
#             println("j is, $j")
            cart_j = cartesian[j]
            rel_pos = relative_direction(cart_i, cart_j)
            wind_factor = find_wind_dir_factor(wind_dir, rel_pos)
            distance_ij = euclidean([cart_i[1],cart_i[2]], [cart_j[1], cart_j[2]])
#             println("dist is, $distance_ij")
            if distance_ij > NEIGHBOR_DIST
                distance_ij = 0
            end
            # println("distance_ij ", distance_ij)
            rel_dist = distance_ij/longest_dist
            lambda_ij = wind_strength*(wind_acc*rel_dist)*wind_factor*fuel_level/to_norm
#             println("lambda is, $lambda_ij")
            lambdas[i,j] = lambda_ij
            P_xy[i,j]= 1 - exp(-lambda_ij)
        end
    end
    # @show P_xy
    return P_xy
end

function find_wind_dir_factor(wind_dir::Int, cell_dir::Int)
    alignment = abs(wind_dir - cell_dir)
    if alignment == 0
        return 1
    elseif alignment == 1 || alignment == 7
        return 0.5
    else
        return 0 # the rest are not considered
    end
end

function relative_direction(cart_i::CartesianIndex{2}, cart_j::CartesianIndex{2})
    row_i = cart_i[1]
    col_i = cart_i[2]
    row_j = cart_j[1]
    col_j = cart_j[2]
    if row_i == row_j # same row
        if col_j > col_i # j is east of i
            return 3
        else
            return 7
        end
    elseif row_j > row_i # j is south of i
        if col_i == col_j
            return 4
        else
            return 6
        end
    elseif row_j < row_i # j is west of i
        if col_i == col_j
            return 2
        else
            return 8
        end
    else # same column
        if row_j > row_i # j is south of i
            return 5
        else
            return 1
        end
    end
end



# to get cells no actions applied to
minus(indx, x) = setdiff(1:length(x), indx)

# get permutations
space(x, n) = vec(collect(Iterators.product([x for i = 1:n]...)))

