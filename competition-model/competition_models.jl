# Competition models

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

# Taylor series expansion of exp(x) around c
exp_shift(x, c; order) = taylor_expand(exp, c; order)(x - c)

# Prolongate (extend) LPA model
function prolongate_LPA(data, params, midpoints, nsteps, order)
    prolongations = Num[]
    for i in 1 : nsteps
        append!(prolongations, LPA_taylor_centred(data[i+1], data[i], params, midpoints, order))
    end
    return prolongations
end
