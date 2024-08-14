using HomotopyContinuation
using Symbolics
using Random, Distributions
using TaylorSeries
using LinearAlgebra
using DataFrames, CSV, Dates

include("competition_models.jl")
include("taylorseries_patch.jl")

#----------------------------------------------
# DIMS: 2
original_x0 = [1.1, 1.2]
n = length(original_x0)

original_params = NamedTuple([
    :a_11 => 0.1,
    :a_12 => 0.2,
    :a_21 => 0.3,
    :a_22 => 0.4,
    :r_1 => 0.4,
    :r_2 => 0.5,
])

#----------------------------------------------
# DIMS: 3
original_x0 = [1.1, 1.2, 1.3]
n = length(original_x0)

original_params = NamedTuple([
    :a_11 => 0.1,
    :a_12 => 0.2,
    :a_13 => 0.25,
    :a_21 => 0.3,
    :a_22 => 0.4,
    :a_23 => 0.45,
    :a_31 => 0.5,
    :a_32 => 0.55,
    :a_33 => 0.6,
    :r_1 => 0.4,
    :r_2 => 0.5,
    :r_3 => 0.6
])

#----------------------------------------------
# simulation settings
steps = n + 1 # simulation steps aka prolongs
Ntaylor = 5 # max taylor approx.
Nsims = 10 # sims per parameter set
interval_ranges = [0.05, 0.1, 0.2, 0.25, 0.5]
#----------------------------------------------

# Symbolic parameters
a_names = [Symbol("a_$(j)$(i)") for i in 1:n, j in 1:n]
r_names = [Symbol("r_$(i)") for i in 1:n]
param_names = (a_names..., r_names...) # used for NamedTuple
sym_params = collect(Symbolics.variable.(param_names)) # used for equations

# presample all parameter sets
# parameter t is sampled from range [t ± p%]
function create_intervals(range, interval_centre=values(original_params))
    NamedTuple{param_names}([RealInterval((1 - range) * p, (1 + range) * p) for p in interval_centre])
end

function sample_intervals(intervals)
    NamedTuple{param_names}([rand(Uniform(interval.lb, interval.ub)) for interval in values(intervals)])
end
#----------------------------------------------
# results file headings
df = DataFrame([
    "sim_num" => Int[],
    "interval_range" => Float64[],
    "taylor_n" => Int[],
    "pred_parameters" => Any[],
    "sampled_parameters" => Any[],
    "param_intervals" => Any[]
]);

@time for I_range in interval_ranges
    println()
    println("The interval size = $(I_range*100)%.")
    param_intervals = create_intervals(I_range)

    for i in 1:Nsims
       println("The simulation number = $i.")

       # sample new parameters and ICs for given interval
       sampled_params = sample_intervals(param_intervals)
       sampled_x0 = original_x0

       data = run_simulation(LPA, sampled_x0, sampled_params, steps)

       for q in 1:Ntaylor
            println("Degree of Taylor polynomial = $q.")
            println()
            eqns = prolongate_LPA(data, sym_params, original_params, steps, q)
            hc_eqns = convert_to_HC_expression.(eqns)

            F = Array{Any}(undef, n)
            sys_vars = Array{Any}(undef, n)
            res_pred = Array{Any}(undef, n)
            res_all_real = Array{Any}(undef, n)

            # Breaking down the system of equations into n independent parts
            for s in 1:n
                pivots = [j * n + s for j in 0:n]
                F[s] = System(hc_eqns[pivots])

                # Solve HC system and keep real solutions (or first complex)

                sys_vars[s] = Symbol.(variables(F[s]))
                res_pred[s], res_all_real[s] = solve_and_filter_solutions(F[s], param_intervals[sys_vars[s]])

                println("Predicted solution: ", res_pred[s])
                println("Original solution: ", collect(sampled_params[sys_vars[s]]))
                println("-------------------------")
            end
            println()

            # pasting solutions for partitioned system together

            pred_params = NamedTuple(collect(Iterators.flatten([sys_vars[s] .=> res_pred[s] for s in 1:n])))

            push!(df, (i, I_range, q,
                   NamedTuple{param_names}(pred_params),
                   sampled_params,
                   param_intervals
                   )
             )
        end
    end
end

# write results to CSV file
timestamp() = Dates.format(now(UTC), "yy-mm-ddTHH")
CSV.write("competition-model/tables/competition_model_dim_$(n)_$(timestamp()).csv", df)
