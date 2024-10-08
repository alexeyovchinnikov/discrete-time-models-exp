# Patch for TaylorSeries.jl to make it compatible with Symbolics.jl
# Source: https://github.com/JuliaDiff/TaylorSeries.jl/blob/cd96717c57d4d17a1c1fb65e7bf0c6de573ffa0b/src/symbolics.jl
# See also: https://discourse.julialang.org/t/taylor-expansion-in-symbolics/74309/5

using Symbolics
import Base: +, -, *, /

# Promotion rules
promote_rule(::Type{Taylor1{T}}, ::Type{Num}) where {T<:Number} =
    Taylor1{promote_type(T,Num)}

function promote(a::Taylor1{T}, b::S) where {T<:Real,S<:Num}
    R = promote_type(T,S)
    return Taylor1(convert(Array{R},a.coeffs)), Taylor1(convert(R,b), a.order)
end
promote(b::S, a::Taylor1{T}) where {T<:Real,S<:Num} = reverse(promote(a,b))

promote(a::Taylor1{Num}, b::Num) = a, Taylor1(b, a.order)
promote(b::Num, a::Taylor1{Num}) = reverse(promote(a,b))



# Arithmetics
for f in (:+, :-)
    @eval begin
        ($f)(a::Taylor1{T}, b::Num) where {T<:Number} = ($f)(promote(a,b)...)
        ($f)(b::Num, a::Taylor1{T}) where {T<:Number} = ($f)(promote(b,a)...)
    end
end

function *(a::Num, b::Taylor1{T}) where {T<:Number}
    @inbounds aux = a * b.coeffs[1]
    v = Array{typeof(aux)}(undef, length(b.coeffs))
    @__dot__ v = a * b.coeffs
    return Taylor1(v, b.order)
end
*(b::Taylor1{T}, a::Num) where {T<:Number} = a*b

function *(a::T, b::Taylor1{T}) where {T<:Num}
    @inbounds aux = a * b.coeffs[1]
    v = Array{typeof(aux)}(undef, length(b.coeffs))
    @__dot__ v = a * b.coeffs
    return Taylor1(v, b.order)
end
*(b::Taylor1{T}, a::T) where {T<:Num} = a*b

function /(a::Taylor1{T}, b::Num) where {T<:Number}
    v = a.coeffs ./ b
    return Taylor1(v, a.order)
end

Symbolics.substitute(a::Taylor1{Num}, d::Pair) = Taylor1(substitute.(a.coeffs, (d,)))
