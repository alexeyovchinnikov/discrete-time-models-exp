using HomotopyContinuation
using Symbolics
using Random, Distributions
using TaylorSeries
using LinearAlgebra
using DataFrames, CSV, Dates

include("competition_models.jl")
include("simulation_functions.jl")
include("taylorseries_patch.jl")

#----------------------------------------------
# START
# DIMS: 2
original_x0 = [1.1, 1.2]
Ndims = length(original_x0)

original_params = NamedTuple([
    :a_11 => 0.1,
    :a_12 => 0.2,
    :a_21 => 0.3,
    :a_22 => 0.4,
    :r_1 => 0.4,
    :r_2 => 0.5,
])

ox = [1.1, 1.2]
op = [0.1, 0.2, 0.3, 0.4, 0.4, 0.5]

#----------------------------------------------
# simulation settings

Ntaylor = 2 # max taylor approx.
Nsims = 1 # sims per parameter set
interval_ranges = [0.1] #[0.05, 0.1, 0.2, 0.25, 0.5]
#----------------------------------------------

original_params = create_parameter_tuples(2)

# END
#----------------------------------------------

# results file headings
results_df = DataFrame([
    "sim_num" => Int[],
    "interval_range" => Float64[],
    "taylor_n" => Int[],
    "pred_parameters" => Any[],
    "sampled_parameters" => Any[],
    "param_intervals" => Any[]
]);

df = @time run_simulation(
    original_x0,
    original_params,
    interval_ranges,
    Nsims,
    Ntaylor,
    Ndims
)
df

# write results to CSV file
timestamp() = Dates.format(now(UTC), "yy-mm-ddTHH")
CSV.write("competition-model/tables/competition_model_dim_$(n)_$(timestamp()).csv", result_df)
