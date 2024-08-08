#----------------------------------------------
function filter_solutions(real_sols, intervals)
    for rsol in real_sols
        if all(in.(rsol, values(intervals)))
            return rsol
        end
    end
    return fill(0., length(intervals)) # default
end

function solve_and_filter_solutions(sys, intervals)
    result = HomotopyContinuation.solve(sys, show_progress = false)

    # only interested in real solutions
    all_real_sols = real_solutions(result)

    # filter solution(s) within intervals
    real_sol = filter_solutions(all_real_sols, intervals)

    return real_sol, all_real_sols
end

function lagrange_error_bound(f, c; a = 0, tol = 1e-3, max_k = 19)
    # for function f(x) x∈[a,c] determine order k such that erro |ε| < tol
    # See: https://en.wikipedia.org/wiki/Taylor%27s_theorem#Estimates_for_the_remainder
    M = ceil(f(c))
    for k in 1 : max_k
        ε = M * (abs(c - a)^(k + 1))/factorial(k + 1)
        if ε < tol
            return k, ε
        end
    end
    error("max k = $max_k reached")
end

c = 4
k, ε = lagrange_error_bound(exp, c; a = 0, tol = 1e-3)

# Original LPA model (exp version)

function LPA(x, p)
    n  = length(x)
    a  = transpose(reshape(collect(p)[1:(n^2)], n, n))
    r  = collect(p)[(n^2+1):length(p)]

    exprs = [r[j] - sum(a[j, s] * x[s] for s in 1:n) for j in 1:n]

    return [x[q] * exp(exprs[q]) for q = 1:n]
end

# Taylor series approximated LPA model with centering at each step
# Centering is determined by user inputed intervals (need not be exact)
function LPA_taylor_centred(dx, x, p, midpoints, order)
    n  = length(x)
    a  = transpose(reshape(p[1:(n^2)], n, n))
    r  = p[(n^2+1):length(p)]
    
    exprs = [r[j] - sum(a[j, s] * x[s] for s in 1:n) for j in 1:n]
    
    # centres are 'initial guesses' for parameters within some interval
    Midpoints = [substitute(exprs[j], p .=> collect(midpoints)) for j in 1:n]
    
    # implicit form (output should be ≈ 0)
    return -dx + [x[q] * exp_shift(exprs[q], Midpoints[q]; order) for q = 1:n]  
end

exp_shift(x, c; order) = taylor_expand(exp, c; order)(x - c)

#----------------------------------------------
function prolongate_LPA(data, params, midpoints, nsteps, order)
    prolongations = Num[]
    for i in 1 : nsteps
        append!(prolongations, LPA_taylor_centred(data[i+1], data[i], params, midpoints, order))
    end
    return prolongations
end

function convert_to_HC_expression(eqn)
    eqn_vars = Symbolics.get_variables(eqn)
    hc_vars = HomotopyContinuation.Variable.(Symbolics.tosymbol.(eqn_vars))
    sub = substitute(eqn, Dict(eqn_vars .=> hc_vars))
end

function run_simulation(model, u0, params, nsteps)
    sol = Array{Any}(undef, nsteps + 1)
    sol[1] = u0
    for i in 1 : (nsteps)
      sol[i + 1] = model(sol[i], params)
    end
    return sol
end

#----------------------------------------------
