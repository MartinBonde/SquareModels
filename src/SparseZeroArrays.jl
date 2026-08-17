# SparseZeroArrays - Domain-aware sparse arrays with zero default
# Optional component of SquareModels (enabled by default)

using JuMP.Containers: SparseAxisArray

# ==============================================================================
# Zero sentinel — adding Zero() to any JuMP expression is a no-op
# ==============================================================================
struct Zero end
Base.:+(x, ::Zero) = x
Base.:+(::Zero, x) = x
Base.:+(::Zero, ::Zero) = Zero()
Base.:*(::Zero) = Zero()
Base.:*(::Zero, ::Zero) = Zero()
Base.:*(::Zero, _) = Zero()
Base.:*(_, ::Zero) = Zero()
Base.:-(::Zero) = Zero()
Base.:-(x, ::Zero) = x
Base.:-(::Zero, x) = -x
Base.:/(::Zero, _) = Zero()
Base.zero(::Type{Zero}) = Zero()
Base.iszero(::Zero) = true
Base.show(io::IO, ::Zero) = print(io, "Zero()")

# MutableArithmetics / JuMP integration — Zero() is a no-op in expression building
JuMP.add_to_expression!(expr::AffExpr, ::Zero) = expr
JuMP.add_to_expression!(expr::AffExpr, ::Any, ::Zero) = expr
JuMP.add_to_expression!(expr::AffExpr, ::Zero, ::Any) = expr
JuMP.add_to_expression!(expr::QuadExpr, ::Zero) = expr
JuMP.add_to_expression!(expr::QuadExpr, ::Any, ::Zero) = expr
JuMP.add_to_expression!(expr::QuadExpr, ::Zero, ::Any) = expr

# ==============================================================================
# SparseZeroArray — domain-aware wrapper for SparseAxisArray with zero default
# ==============================================================================
"""
    SparseZeroArray{T, N, KT}

Domain-aware wrapper around JuMP's `SparseAxisArray` providing GAMS-like semantics:
- Missing entries within the domain return `Zero()` (a no-op additive identity)
- Out-of-domain access throws an error

This allows writing `sum(x[i, d, t] for d in D)` without filter clauses, while
still catching typos and invalid indices via domain checking.

# Fields
- `data::SparseAxisArray{T, N, KT}`: The underlying JuMP container
- `domain::NTuple{N, Set}`: Valid values per dimension (for domain checking)

# AbstractArray subtyping
`SparseZeroArray` subtypes `AbstractArray` following JuMP's `SparseAxisArray` precedent, but
intentionally violates the `AbstractArray` contract: `size` is unsupported (it's a dictionary,
not a dense array) and `keys` returns dictionary keys (tuples) instead of `LinearIndices`.
This is necessary because many functions in SquareModels and JuMP dispatch on `AbstractArray`
(e.g. `ModelDictionary` indexing, `fix`, `set_start_value`, `Block` construction).
Use `eachindex` to iterate over indices, which is the correct `AbstractArray` method for this.
"""
struct SparseZeroArray{T, N, KT} <: AbstractArray{T, N}
    data::SparseAxisArray{T, N, KT}
    domain::NTuple{N, Set}
end

function SparseZeroArray(data::AbstractDict)
    saa = SparseAxisArray(data)
    return SparseZeroArray(saa, _domain_from_keys(saa))
end

function Base.getindex(s::SparseZeroArray{T, N, KT}, args...) where {T, N, KT}
    if args isa KT  # scalar access — compile-time specialized
        for (arg, dom) in zip(args, s.domain)
            arg in dom || error("Index $arg is not in the domain $dom")
        end
        return get(s.data.data, args, Zero())
    else  # slice (Colon, Vector, etc.) — delegate
        return s.data[args...]
    end
end

# Mirror SparseAxisArray: size is intentionally unsupported (conceptually a dictionary)
function Base.size(::SparseZeroArray)
    return error(
        "`Base.size` is not implemented for `SparseZeroArray` because " *
        "although it is a subtype of `AbstractArray`, it is conceptually " *
        "closer to a dictionary with `N`-dimensional keys. If you encounter " *
        "this error and you didn't call `size` explicitly, it is because " *
        "you called a method that is unsupported for `SparseZeroArray`s.",
    )
end

# Intentionally breaks AbstractArray contract: returns dictionary keys (tuples) not LinearIndices.
# The default AbstractArray `keys` calls `size` which is unsupported.
Base.keys(s::SparseZeroArray) = keys(s.data.data)
Base.haskey(s::SparseZeroArray, key) = haskey(s.data, key)
Base.length(s::SparseZeroArray) = length(s.data)
Base.IteratorSize(::Type{<:SparseZeroArray}) = Base.HasLength()
Base.iterate(s::SparseZeroArray, args...) = iterate(s.data, args...)
Base.first(s::SparseZeroArray) = first(s.data)
Base.eltype(::Type{SparseZeroArray{T,N,KT}}) where {T,N,KT} = T
Base.eachindex(s::SparseZeroArray) = keys(s.data.data)
Base.hash(s::SparseZeroArray, h::UInt) = hash(s.data, h)
Base.:(==)(s1::SparseZeroArray, s2::SparseZeroArray) = s1.data == s2.data
Base.mapreduce(f, op, s::SparseZeroArray) = mapreduce(f, op, values(s.data.data))

Base.similar(s::SparseZeroArray{S,N,KT}, ::Type{T}, length::Integer=0) where {S,T,N,KT} = similar(s.data, T, length)
Base.BroadcastStyle(::Type{<:SparseZeroArray{T,N,KT}}) where {T,N,KT} = Base.BroadcastStyle(SparseAxisArray{T,N,KT})
Base.Broadcast.broadcast_preserving_zero_d(f, A::SparseZeroArray, As...) = broadcast(f, A, As...)
Base.Broadcast.broadcast_preserving_zero_d(f, x, A::SparseZeroArray, As...) = broadcast(f, x, A, As...)
Base.Broadcast.broadcast_preserving_zero_d(f, A::SparseZeroArray, B::SparseZeroArray, args...) = broadcast(f, A, B, args...)

function Base.summary(io::IO, s::SparseZeroArray)
    num_entries = length(s)
    return print(io, typeof(s), " with ", num_entries, isone(num_entries) ? " entry" : " entries")
end

function Base.show(io::IO, ::MIME"text/plain", s::SparseZeroArray)
    summary(io, s)
    if !isempty(s.data.data)
        println(io, ":")
        show(io, s)
    end
    return
end

Base.show(io::IO, s::SparseZeroArray) = show(convert(IOContext, io), s)

function Base.show(io::IOContext, s::SparseZeroArray)
    if isempty(s)
        return show(io, MIME("text/plain"), s)
    end
    show(io, s.data)
end

"""
    ∑(args...; kwargs...)

Sum with `Zero()` as the initial value, so summing over empty or all-missing
sparse dimensions yields `Zero()` instead of erroring.
"""
∑(args...; kwargs...) = sum(args...; init=Zero(), kwargs...)

# ==============================================================================
# Configuration — toggle SparseZeroArray wrapping on/off
# ==============================================================================
const _use_sparse_zero_array = Ref(true)

"""
    use_sparse_zero_array!(enabled::Bool=true)

Enable or disable automatic `SparseZeroArray` wrapping in `@variables`.

When enabled (default), `SparseAxisArray` variables declared via `@variables` are
automatically wrapped in `SparseZeroArray`, providing GAMS-like zero-default
semantics for missing entries.

When disabled, `@variables` creates standard JuMP `SparseAxisArray` containers.

# Examples
```julia
# Disable SparseZeroArray wrapping (use plain JuMP SparseAxisArrays)
use_sparse_zero_array!(false)

# Re-enable (default)
use_sparse_zero_array!(true)
```
"""
use_sparse_zero_array!(enabled::Bool=true) = (_use_sparse_zero_array[] = enabled; nothing)

# ==============================================================================
# JuMP custom container — create SparseZeroArray directly via @variable
# ==============================================================================
_domain_key(k::Tuple) = only(k)
_domain_key(k) = k

_domain_from_keys(saa::SparseAxisArray{T,1}) where {T} = (Set(_domain_key(k) for k in keys(saa.data)),)
_domain_from_keys(saa::SparseAxisArray{T,N}) where {T,N} =
    ntuple(dim -> Set(k[dim] for k in keys(saa.data)), N)

"""
    _domain_from_nested(ni::Containers.NestedIterator, N)

Extract per-dimension domains from a `NestedIterator`'s base iterators,
ignoring the filter condition. This gives the full "original" domain for
each dimension, so that indices filtered out by the condition still pass
domain checks (and return `Zero()`) rather than throwing.
"""
function _domain_from_nested(ni::Containers.NestedIterator, N::Int)
    domains = ntuple(_ -> Set(), N)
    _collect_nested_domains!(domains, ni.iterators, 1, ())
    return domains
end

function _collect_nested_domains!(domains, iterators, dim, prev_args)
    dim > length(iterators) && return
    for val in iterators[dim](prev_args...)
        push!(domains[dim], val)
        _collect_nested_domains!(domains, iterators, dim + 1, (prev_args..., val))
    end
end

function Containers.container(f::Function, indices::Containers.NestedIterator, ::Type{SparseZeroArray})
    saa = Containers.container(f, indices, SparseAxisArray)
    N = length(indices.iterators)
    return SparseZeroArray(saa, _domain_from_nested(indices, N))
end

function Containers.container(f::Function, indices, ::Type{SparseZeroArray})
    saa = Containers.container(f, indices, SparseAxisArray)
    return SparseZeroArray(saa, _domain_from_keys(saa))
end

Containers.container(f::Function, indices, ::Type{SparseZeroArray}, names) =
    Containers.container(f, indices, SparseZeroArray)
