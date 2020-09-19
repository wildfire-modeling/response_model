import POMDPs
# Rewards
# Option 1: sum of costs of burning cells
function POMDPs.reward(pomdp::FireWorld, s::FireState)
    burning = s.burning
    r = dot(burning, pomdp.costs)
    return r
end

# # Option 2: sum of sum product of (costs, fuels left)
# function POMDPs.reward(pomdp::FireWorld, s::FireState)
#     burning = s.burning
#     fuels = s.fuels
#     cost_w_fuels = pomdp.costs .* fuels
#     r = dot(burning, cost_w_fuels)
#     return r
# end

POMDPs.reward(pommdp::FireWorld, s::FireState, a::Array{Int64,1}) = reward(pomdp, s)