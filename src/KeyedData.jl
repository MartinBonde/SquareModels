# KeyedData — observed sparse data keyed by index tuples
#
# A stored cell is observed data, including a stored `0.0`. An unstored cell is
# unobserved and reads as `nothing`. This type does not define arithmetic.
# Equation-time `Zero()` for in-domain unstored cells stays in SparseZeroArrays.jl.

"""
    KeyedData(data::Dict{<:Tuple})
    KeyedData(pairs)

Sparse observed data keyed by index tuples.

A stored coordinate is observed data; a stored `0.0` is an observation. An
unstored coordinate is unobserved, and `data[i, j]` reads `nothing` there. Every
key needs the same number of indices, which `ndims` reports.

A `Colon` or a vector in an index position returns a `KeyedData` with the keys
projected onto those positions.

`KeyedData` can also supply stored coordinates to [`select_axes`](@ref),
[`merge_indices`](@ref), and tuple-destructured [`@variables`](@ref)
declarations. Iterating `KeyedData` itself yields `key => value` pairs; use
`keys(data)` when an ordinary coordinate iterator is needed.

An empty input must have a tuple key type with a fixed number of fields, such as
`Dict{Tuple{Symbol,Int},Float64}()`, because the rank cannot otherwise be
inferred.
"""
struct KeyedData{N,K<:Tuple,V}
    data::Dict{K,V}

    function KeyedData{N,K,V}(data::Dict{K,V}) where {N,K<:NTuple{N,Any},V}
        return new{N,K,V}(data)
    end
end

function _keyed_data_rank(data::AbstractDict)
    arities = unique(length(key) for key in keys(data) if key isa Tuple)
    all(key -> key isa Tuple, keys(data)) || error("KeyedData keys must be tuples")
    length(arities) <= 1 ||
        error("KeyedData needs the same number of indices in every key. Found arities $(sort(arities)).")
    !isempty(arities) && return only(arities)

    K = keytype(data)
    K <: Tuple || error("Cannot infer the number of KeyedData index positions from empty input")
    try
        return fieldcount(K)
    catch error
        error isa ArgumentError || rethrow()
        throw(ArgumentError(
            "Cannot infer the number of KeyedData index positions from empty input with key type $K",
        ))
    end
end

function KeyedData(data::AbstractDict)
    N = _keyed_data_rank(data)
    K = keytype(data)
    V = valtype(data)
    normalized = K <: NTuple{N,Any} && data isa Dict ?
        data : Dict{NTuple{N,Any},V}(data)
    return KeyedData{N,keytype(normalized),valtype(normalized)}(normalized)
end

KeyedData(pairs) = KeyedData(Dict(pairs))

Base.ndims(::KeyedData{N}) where {N} = N
Base.eltype(::Type{KeyedData{N,K,V}}) where {N,K,V} = Pair{K,V}
Base.length(kd::KeyedData) = length(kd.data)
Base.keys(kd::KeyedData) = keys(kd.data)
Base.values(kd::KeyedData) = values(kd.data)
Base.pairs(kd::KeyedData) = pairs(kd.data)
Base.eachindex(kd::KeyedData) = keys(kd.data)
Base.haskey(kd::KeyedData, key) = haskey(kd.data, key)
Base.get(kd::KeyedData, key, default) = get(kd.data, key, default)
Base.iterate(kd::KeyedData, state...) = iterate(kd.data, state...)
# `Ref` makes `w .= kd` a zero-dimensional broadcast, which Base unwraps to `kd`.
Base.broadcastable(kd::KeyedData) = Ref(kd)

_is_slice(::Colon) = true
_is_slice(::AbstractVector) = true
_is_slice(_) = false

_in_slice(_, ::Colon) = true
_in_slice(index, arg::AbstractVector) = index in arg
_in_slice(index, arg) = index == arg

function Base.getindex(kd::KeyedData{M,K,V}, args::Vararg{Any,N}) where {M,K,V,N}
    # Without this, too few indices read as `nothing` and a slice projects a prefix.
    N == M || error("KeyedData has $M index positions, got $N")
    any(_is_slice, args) || return get(kd.data, args, nothing)
    kept = [position for position in 1:N if _is_slice(args[position])]
    key_type = Tuple{fieldtype.(K, kept)...}
    return KeyedData(Dict{key_type,V}(
        Tuple(key[position] for position in kept) => value
        for (key, value) in kd.data
        if all(_in_slice(index, arg) for (index, arg) in zip(key, args))
    ))
end

Base.:(==)(a::KeyedData, b::KeyedData) = a.data == b.data
