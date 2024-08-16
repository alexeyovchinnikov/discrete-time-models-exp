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
# simulation settings
Ntaylor = 5 # max taylor approx.
Nsims = 10 # sims per parameter set
interval_ranges = [0.1, 0.05, 0.1, 0.2, 0.25, 0.5]
max_d = 4
#----------------------------------------------

for d in 2:max_d
    println("Running simulations for dim = $d.")

    # 'original' initial conditions and parameters
    x0_vals = range(1, 2, length=d)
    param_vals = range(0.1, 1.1, length=d^2+d)

    original_x0 = create_x0_tuple(x0_vals)
    original_params = create_parameter_tuple(param_vals, d)

    result_df = @time run_simulation(
        original_x0,
        original_params,
        interval_ranges,
        Nsims,
        Ntaylor,
        d
    )

    # write results to CSV file
    timestamp() = Dates.format(now(UTC), "yy-mm-ddTHH")
    CSV.write("competition-model/tables/competition_model_dim_$(d)_$(timestamp()).csv", result_df)
end
