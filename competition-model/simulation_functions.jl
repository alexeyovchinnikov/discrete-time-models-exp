# ----------------------------------------------
function create_x0_tuple(x0_values)
    n = length(x0_values)
    x0_names = [Symbol("x_$(i)") for i in 1:n]
    return NamedTuple{(x0_names...,)}(x0_values)
end

function create_parameter_tuple(param_values, n)
    a_names = [Symbol("a_$(j)$(i)") for i in 1:n, j in 1:n]
    r_names = [Symbol("r_$(i)") for i in 1:n]
    param_names = (a_names..., r_names...)
    return NamedTuple{param_names}(param_values)
end

# presample all parameter sets
# parameter t is sampled from range [t ± p%]
function create_intervals(param_tuple, range)
    param_names, param_values = keys(param_tuple), values(param_tuple)
    NamedTuple{param_names}([RealInterval((1 - range) * p, (1 + range) * p) for p in param_values])
end
function sample_intervals(interval_tuple)
    param_names, interval_values = keys(interval_tuple), values(interval_tuple)
    NamedTuple{param_names}([rand(Uniform(interval.lb, interval.ub)) for interval in interval_values])
end

symbolic_parameters(param_names) = collect(Symbolics.variable.(param_names))

#----------------------------------------------
function run_simulation(
        original_x0,
        original_params,
        interval_ranges,
        Nsims,
        Ntaylor,
        n
    )

    # results file headings
    df = DataFrame([
        "sim_num" => Int[],
        "interval_range" => Float64[],
        "taylor_n" => Int[],
        "pred_parameters" => Any[],
        "sampled_parameters" => Any[],
        "param_intervals" => Any[]
    ]);

    # symbolic parameters
    sym_params = symbolic_parameters(keys(original_params))

    for I_range in interval_ranges
        println()
        println("The interval size = $(I_range*100)%.")
        param_intervals = create_intervals(original_params, I_range)

        for i in 1:Nsims
        println("The simulation number = $i.")

        # sample new parameters and ICs for given interval
        sampled_params = sample_intervals(param_intervals)
        sampled_x0 = original_x0

        steps = n + 1 # simulation steps aka prolongs
        data = solve_model(LPA, sampled_x0, sampled_params, steps)

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
                    # NamedTuple{keys(original_params)}(pred_params),
                    pred_params,
                    sampled_params,
                    param_intervals
                    )
                )
            end
        end
    end
    return df
end

#----------------------------------------------
function solve_model(model, u0, params, nsteps)
    sol = Array{Any}(undef, nsteps + 1)
    sol[1] = u0
    for i in 1 : (nsteps)
      sol[i + 1] = model(sol[i], params)
    end
    return sol
end

function convert_to_HC_expression(eqn)
    eqn_vars = Symbolics.get_variables(eqn)
    hc_vars = HomotopyContinuation.Variable.(Symbolics.tosymbol.(eqn_vars))
    sub = substitute(eqn, Dict(eqn_vars .=> hc_vars))
end

function solve_and_filter_solutions(sys, intervals)
    result = HomotopyContinuation.solve(sys, show_progress = false)

    # only interested in real solutions
    all_real_sols = real_solutions(result)

    # filter solution(s) within intervals
    real_sol = filter_solutions(all_real_sols, intervals)

    return real_sol, all_real_sols
end

function filter_solutions(real_sols, intervals)
    for rsol in real_sols
        if all(in.(rsol, values(intervals)))
            return rsol
        end
    end
    return fill(0., length(intervals)) # default
end
#----------------------------------------------
