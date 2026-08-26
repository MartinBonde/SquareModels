# tagged_variables.jl - @variables macro with tags and descriptions
#
# Syntax (using :: for tags, like Julia's Holy trait pattern):
#
#   # Apply tags to individual variables:
#   @variables container begin
#       var_name[indices] :: (tag1, tag2), "Description"
#       var_name[indices] :: tag1
#       var_name[indices], "Description"
#       var_name[indices]
#   end
#
#   # Apply tags to ALL variables in the block:
#   @variables container :: tag begin
#       var_name[indices], "Description"
#       ...
#   end
#
#   # Combine block-level and variable-level tags (they accumulate):
#   @variables container :: BlockTag begin
#       var_name[indices] :: VarTag, "Description"  # Has both BlockTag and VarTag
#   end
#
# Tags are trait-like markers for variable categorization.
# Use JuMP.@variables for JuMP's original macro.

# ==============================================================================
# Tag Definition (Holy Trait Pattern)
# ==============================================================================
"""
    Tag

A trait-like tag for categorizing variables, following Julia's Holy trait pattern.

Tags are used to mark variables with properties like "growth adjusted" or
"inflation adjusted", enabling trait-based dispatch and grouping.

# Example
```julia
# Define tags as trait markers
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)
const FlatForecast = Tag(:flat_forecast)

# Use with :: syntax (like type annotations)
@variables db begin
    vGDP[t] :: (GrowthAdjusted, InflationAdjusted), "Nominal GDP"
    pGDP[t] :: InflationAdjusted, "GDP deflator"
    qGDP[t] :: GrowthAdjusted, "Real GDP"
end
```
"""
struct Tag
    name::Symbol
end

Base.show(io::IO, t::Tag) = print(io, "Tag(:", t.name, ")")

# ==============================================================================
# Variable Metadata
# ==============================================================================
"""
    VariableMetadata

Stores metadata for a variable: its tags and description.
"""
struct VariableMetadata
    tags::Set{Tag}
    description::String
end

VariableMetadata() = VariableMetadata(Set{Tag}(), "")
VariableMetadata(tags) = VariableMetadata(Set{Tag}(tags), "")
VariableMetadata(tags, desc::AbstractString) = VariableMetadata(Set{Tag}(tags), String(desc))

# Global registry: variable name (Symbol) => VariableMetadata
const _variable_metadata = Dict{Symbol, VariableMetadata}()

"""
    description(var) → String

Get the description of a variable. For indexed variables like `X[2020]`,
returns the description of the base variable `X`.
"""
description(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata()).description
description(var::Symbol) = get(_variable_metadata, var, VariableMetadata()).description

"""
    tags(var) → Set{Tag}

Get all tags associated with a variable. For indexed variables like `X[2020]`,
returns the tags of the base variable `X`.
"""
tags(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata()).tags
tags(var::Symbol) = get(_variable_metadata, var, VariableMetadata()).tags

"""
    has_tag(var, tag::Tag) → Bool

Check if a variable has a specific tag.
"""
has_tag(var::AbstractVariableRef, tag::Tag) = tag ∈ tags(var)
has_tag(var::Symbol, tag::Tag) = tag ∈ tags(var)

"""
    tagged(tag::Tag) → Vector{Symbol}

Get all variable base names that have a specific tag.
"""
tagged(tag::Tag) = [k for (k, m) in _variable_metadata if tag ∈ m.tags]

"""
    metadata(var) → VariableMetadata

Get full metadata for a variable. For indexed variables like `X[2020]`,
returns the metadata of the base variable `X`.
"""
metadata(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata())
metadata(var::Symbol) = get(_variable_metadata, var, VariableMetadata())

# ==============================================================================
# Parsing Helpers
# ==============================================================================

"""
Parse a variable declaration line.

Handles these forms:
- `var[idx]` → (var_def, [], "")
- `var[idx] :: tag` → (var_def, [tag], "")
- `var[idx] :: (tag1, tag2)` → (var_def, [tag1, tag2], "")
- `var[idx] :: tag, "desc"` → (var_def, [tag], "desc")
- `var[idx], "desc"` → (var_def, [], "desc")

Returns: (var_definition, tag_exprs, description)
"""
function _parse_var_line(expr)
    # Handle tuple form: (var_stuff, "description")
    if isexpr(expr, :tuple)
        if length(expr.args) >= 2 && expr.args[end] isa String
            desc = expr.args[end]
            inner = expr.args[1]
            var_def, tag_exprs, _ = _parse_var_line(inner)
            return (var_def, tag_exprs, desc)
        elseif length(expr.args) == 1
            return _parse_var_line(expr.args[1])
        end
    end

    # Simple variable: symbol, `:ref`, or `:typed_vcat` (`x[i=I; cond]`)
    if expr isa Symbol || isexpr(expr, :ref) || isexpr(expr, :typed_vcat)
        return (expr, [], "")
    end

    # Type annotation expression: var[t] :: tag or var[t] :: (tag1, tag2)
    if isexpr(expr, :(::))
        var_def = expr.args[1]
        tag_part = expr.args[2]
        # Handle single tag or tuple of tags
        if isexpr(tag_part, :tuple)
            tag_exprs = collect(tag_part.args)
        else
            tag_exprs = [tag_part]
        end
        return (var_def, tag_exprs, "")
    end

    # Fallback: treat whole thing as var definition
    return (expr, [], "")
end

"""
Parse block-level tags from `container :: tag` or `container :: (tag1, tag2)`.

Returns: (container_expr, block_tag_exprs)
"""
function _parse_block_tags(expr)
    if isexpr(expr, :(::))
        container = expr.args[1]
        tag_part = expr.args[2]
        if isexpr(tag_part, :tuple)
            return (container, collect(tag_part.args))
        else
            return (container, [tag_part])
        end
    end
    return (expr, [])
end

# ==============================================================================
# SparseZeroArray container detection
# ==============================================================================

"""True when a variable definition has a semicolon filter. `x[i=I, t=T; cond]` is `:ref` with a `:parameters` node. `x[i=I; cond]` is `:typed_vcat`."""
function _has_filter_condition(var_def::Expr)
    isexpr(var_def, :ref) && return any(a -> isexpr(a, :parameters), var_def.args)
    isexpr(var_def, :typed_vcat) && return length(var_def.args) >= 3
    return false
end
_has_filter_condition(::Symbol) = false

"""Return true when an index declaration has a tuple on the left of `=`."""
function _has_tuple_destructuring(var_def::Expr)
    isexpr(var_def, :ref) || return false
    return any(var_def.args[2:end]) do axis
        isexpr(axis, :kw) && isexpr(axis.args[1], :tuple)
    end
end
_has_tuple_destructuring(::Symbol) = false

_expr_uses_symbol(s::Symbol, names) = s in names
_expr_uses_symbol(e::Expr, names) = any(arg -> _expr_uses_symbol(arg, names), e.args)
_expr_uses_symbol(_, _) = false

_or_clauses(expr) =
    isexpr(expr, :||) ? vcat(_or_clauses(expr.args[1]), _or_clauses(expr.args[2])) : [expr]

function _membership_lhs_rhs(expr)
    isexpr(expr, :call) && length(expr.args) == 3 && expr.args[1] in (:in, :∈) || return nothing
    lhs, rhs = expr.args[2], expr.args[3]
    names = if lhs isa Symbol
        [lhs]
    elseif isexpr(lhs, :tuple) && all(name -> name isa Symbol, lhs.args)
        collect(lhs.args)
    else
        return nothing
    end
    return names, rhs
end

function _parse_membership_filter(condition)
    parsed = map(_membership_lhs_rhs, _or_clauses(condition))
    any(isnothing, parsed) && return nothing
    names = first.(parsed)
    all(==(names[1]), names) || return nothing
    return names[1], last.(parsed)
end

_is_named_axis(axis) = (isexpr(axis, :kw) || isexpr(axis, :(=))) && axis.args[1] isa Symbol

function _named_ref_axes(var_def::Expr)
    isexpr(var_def, :ref) || isexpr(var_def, :typed_vcat) || return nothing
    axes = var_def.args[2:end]
    condition = nothing
    if isexpr(var_def, :typed_vcat)
        length(axes) > 2 && return nothing
        if length(axes) == 2
            condition = axes[2]
            axes = axes[1:1]
        end
    end
    names = Symbol[]
    values = Any[]
    for axis in axes
        if isexpr(axis, :parameters)
            length(axis.args) == 1 || return nothing
            condition = axis.args[1]
            continue
        end
        _is_named_axis(axis) || return nothing
        push!(names, axis.args[1])
        push!(values, axis.args[2])
    end
    return names, values, condition
end

function _parse_membership_filter_declaration(var_def::Expr)
    parsed_axes = _named_ref_axes(var_def)
    parsed_axes === nothing && return nothing
    names, values, condition = parsed_axes
    condition === nothing && return nothing
    isempty(names) && return nothing
    membership = _parse_membership_filter(condition)
    membership === nothing && return nothing
    filter_names, filter_sets = membership
    allunique(names) || return nothing
    allunique(filter_names) && all(in(names), filter_names) || return nothing
    any(
        _expr_uses_symbol(value, setdiff(names, [name]))
        for (name, value) in zip(names, values)
    ) && return nothing
    # Fast path walks membership sets in isolation. Axis names in a set, such as
    # `p in byyear[t]`, need JuMP's nested iterators so later axes exist.
    any(_expr_uses_symbol(set, names) for set in filter_sets) && return nothing
    return names, values, filter_names, filter_sets
end
_parse_membership_filter_declaration(::Symbol) = nothing

_has_fast_membership_filter(var_def) = _parse_membership_filter_declaration(var_def) !== nothing

function _parse_destructured_axes(var_def::Expr)
    any(a -> isexpr(a, :parameters), var_def.args) &&
        error("A tuple-destructured declaration cannot have a semicolon filter")
    sources = Any[]
    arities = Int[]
    unpack = Bool[]
    names = Symbol[]
    for axis in var_def.args[2:end]
        isexpr(axis, :kw) ||
            error("Each axis in a tuple-destructured declaration must use `name = values`")
        unpack_axis = isexpr(axis.args[1], :tuple)
        axis_names = unpack_axis ? axis.args[1].args : [axis.args[1]]
        all(name -> name isa Symbol, axis_names) ||
            error("Each destructured index name must be a symbol")
        isempty(axis_names) && error("A destructured axis must have at least one index name")
        append!(names, axis_names)
        push!(sources, axis.args[2])
        push!(arities, length(axis_names))
        push!(unpack, unpack_axis)
    end
    allunique(names) || error("Index names in a tuple-destructured declaration must be unique")
    return sources, arities, unpack, names
end

function _destructured_value(value, arity)
    coordinate = arity == 1 && !(value isa Tuple) ? (value,) : Tuple(value)
    length(coordinate) == arity ||
        error("Expected a coordinate with $arity items, got $coordinate")
    return coordinate
end

_unpack_parts(source, arity) = [_destructured_value(key, arity) for key in source]

function _destructured_axis(source::SparseZeroArray, arity, unpack)
    unpack || error("Unpack a SparseZeroArray with (i, j, ...) = array")
    length(source.domain) == arity ||
        error("The destructured index has $arity names, but the array has $(length(source.domain)) axes")
    return _unpack_parts(keys(source), arity), map(copy, source.domain)
end

function _destructured_axis(source::SparseAxisArray, arity, unpack)
    unpack || error("Unpack a SparseAxisArray with (i, j, ...) = array")
    domain = _domain_from_keys(source)
    length(domain) == arity ||
        error("The destructured index has $arity names, but the array has $(length(domain)) axes")
    return _unpack_parts(keys(source.data), arity), domain
end

function _destructured_axis(source::KeyedData, arity, unpack)
    unpack || error("Unpack a KeyedData with (i, j, ...) = data")
    domain = _domain_from_keys(source)
    length(domain) == arity ||
        error("The destructured index has $arity names, but the data has $(length(domain)) axes")
    return _unpack_parts(keys(source), arity), domain
end

function _destructured_axis(source::SparseIndexPattern, arity, unpack)
    unpack || error("Unpack selected axes with (i, j, ...) = select_axes(...)")
    length(source.domain) == arity ||
        error("The destructured index has $arity names, but the source has $(length(source.domain)) axes")
    return _unpack_parts(source, arity), map(copy, source.domain)
end

function _destructured_axis(source, arity, unpack)
    parts = [unpack ? _destructured_value(value, arity) : (value,) for value in source]
    domain = ntuple(i -> Set(part[i] for part in parts), arity)
    return parts, domain
end

function _destructured_index_data(sources::Tuple, arities::Tuple, unpack::Tuple)
    axes = map(_destructured_axis, sources, arities, unpack)
    coordinates = [tuple(Iterators.flatten(parts)...) for parts in Iterators.product(first.(axes)...)]
    domain = tuple(Iterators.flatten(last.(axes))...)
    return coordinates, domain
end

function _attach_sparse_variable(model, name, coordinates, domain, index_names)
    haskey(model, name) && error("An object of name $name is already attached to this model")
    # Empty `Any[]` has no tuple eltype. JuMP's `_container_dict` needs `NTuple{N,Any}`.
    index_keys = isempty(coordinates) ? NTuple{length(index_names),Any}[] : coordinates
    sparse_variable = Containers.container(
        (indices...) -> JuMP.@variable(model, base_name = string(name, "[", join(indices, ","), "]")),
        index_keys,
        SparseAxisArray,
        index_names,
    )
    variable = _use_sparse_zero_array[] ? SparseZeroArray(sparse_variable, domain) : sparse_variable
    model[name] = variable
    return variable
end

function _build_destructured_variable(model, name, sources, arities, unpack, index_names)
    coordinates, domain = _destructured_index_data(sources, arities, unpack)
    return _attach_sparse_variable(model, name, coordinates, domain, index_names)
end

function _assemble_index(n, filter_pos, remain_pos, filter_vals, remain_vals)
    ntuple(n) do i
        filter_at = findfirst(==(i), filter_pos)
        filter_at !== nothing && return filter_vals[filter_at]
        remain_vals[findfirst(==(i), remain_pos)]
    end
end

function _membership_key(value, arity)
    arity == 1 && return (value,)
    return _destructured_value(value, arity)
end

function _membership_coordinates(axis_lists, axis_names, filter_names, filter_sets)
    n = length(axis_names)
    name_pos = Dict(name => i for (i, name) in enumerate(axis_names))
    filter_pos = [name_pos[name] for name in filter_names]
    remain_pos = [i for (i, name) in enumerate(axis_names) if name ∉ filter_names]
    domain = ntuple(i -> Set(axis_lists[i]), n)
    filter_keys = unique(
        _membership_key(raw, length(filter_names))
        for set in filter_sets
        for raw in set
    )
    for filter_vals in filter_keys
        for (pos, val) in zip(filter_pos, filter_vals)
            val in domain[pos] || error("Index $val is not in the domain $(domain[pos])")
        end
    end
    remain_lists = [axis_lists[i] for i in remain_pos]
    coordinates = [
        _assemble_index(n, filter_pos, remain_pos, filter_vals, remain_vals)
        for filter_vals in filter_keys
        for remain_vals in Iterators.product(remain_lists...)
    ]
    return coordinates, domain
end

function _build_membership_variable(model, name, axis_values, axis_names, filter_names, filter_sets, index_names)
    axis_lists = map(collect, axis_values)
    coordinates, domain = _membership_coordinates(axis_lists, axis_names, filter_names, filter_sets)
    return _attach_sparse_variable(model, name, coordinates, domain, index_names)
end

# ==============================================================================
# @variables Macro
# ==============================================================================
"""
    @variables container begin ... end
    @variables container :: tag begin ... end

Create JuMP variables with optional tags and descriptions.

This is SquareModels' `@variables` macro. It adds variable metadata, sparse
coordinate unpacking, and `SparseZeroArray` wrapping. Use `JuMP.@variables`
for JuMP's original macro.

# Syntax

Tags can be applied at two levels:
- **Block-level**: `@variables container :: tag begin ... end` applies tag(s) to all variables
- **Variable-level**: `var :: tag` applies tag(s) to that variable only

When both are used, tags accumulate (union).

Each variable line can have:
- Variable definition (required): `var` or `var[indices]`
- Tags (optional): `:: tag` or `:: (tag1, tag2)` after variable
- Description (optional): `, "description"` at end

```julia
# Define tags as trait markers
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)

# Block-level tag: all variables get GrowthAdjusted
@variables db :: GrowthAdjusted begin
    vGDP[t] :: InflationAdjusted, "Nominal GDP"  # Has both tags
    qGDP[t], "Real GDP"                          # Has GrowthAdjusted only
end

# Variable-level tags only
@variables db begin
    pGDP[t] :: InflationAdjusted, "GDP deflator"
    σ, "Substitution elasticity"
end
```

# Access metadata
```julia
description(:vGDP)  # "Nominal GDP"
tags(:vGDP)         # Set([GrowthAdjusted, InflationAdjusted])
has_tag(:vGDP, GrowthAdjusted)  # true
tagged(GrowthAdjusted)  # [:vGDP, :qGDP, ...]
```

A semicolon `in` filter walks the membership set, not the Cartesian product of
the named axis sets. The named sets are the domain used for `Zero()` lookup:

```julia
@variables db begin
    value[p = product, i = industry, t = years; (p, i) in pairs]
    price[p = product, i = industry, t = years; (p, i, t) in keys(value)]
end
```

Use a tuple on the left of `=` to unpack sparse coordinates. Passing a
`SparseZeroArray` copies its stored cells and its domain. `keys(array)` supplies
only those tuples. [`select_axes`](@ref) picks axes. [`merge_indices`](@ref)
unions arrays. A one-axis unpack needs a trailing comma.

```julia
@variables db begin
    use[(product, industry) = product_industry_pairs, year = years]
    price[(product, industry, year) = value]
    from_keys[(product, industry, year) = keys(value)]
    share[(product, year) = select_axes(value, 1, 3)]
    output[(industry,) = select_axes(value, 2), year = years]
    total[(product, use, origin, year) = merge_indices(purchaser, margin)]
end
```

Use a plain name to keep tuple values on one axis:

```julia
@variables db begin
    flow[edge = edges, year = years]
end
```
"""
macro variables(container_expr, block)
    sm = @__MODULE__
    jump_variable = GlobalRef(JuMP, Symbol("@variable"))
    model_dictionary = GlobalRef(sm, :ModelDictionary)
    sparse_zero_array = GlobalRef(sm, :SparseZeroArray)
    sparse_axis_array = GlobalRef(sm, :SparseAxisArray)
    use_sparse_zero_array = GlobalRef(sm, :_use_sparse_zero_array)
    variable_metadata = GlobalRef(sm, :_variable_metadata)
    variable_metadata_type = GlobalRef(sm, :VariableMetadata)
    build_destructured_variable = GlobalRef(sm, :_build_destructured_variable)
    build_membership_variable = GlobalRef(sm, :_build_membership_variable)

    # Parse block-level tags from container expression
    container, block_tag_exprs = _parse_block_tags(container_expr)

    # Validate block structure
    if !isexpr(block, :block)
        error("@variables requires a begin...end block")
    end

    code = Expr(:block)
    model_expr = :(
        ($container) isa $model_dictionary ?
            ($container).model :
            ($container)
    )
    var_names = Symbol[]

    for line in block.args
        # Skip line numbers
        if line isa LineNumberNode
            push!(code.args, line)  # Preserve for error messages
            continue
        end

        # Parse the line
        var_def, var_tag_exprs, desc = _parse_var_line(line)
        var_name = _get_name(var_def)
        push!(var_names, var_name)

        # Build tuple-destructured coordinates from their stated sources.
        if _has_tuple_destructuring(var_def)
            sources, arities, unpack, index_names = _parse_destructured_axes(var_def)
            sources_expr = Expr(:tuple, sources...)
            arities_expr = Expr(:tuple, arities...)
            unpack_expr = Expr(:tuple, unpack...)
            names_expr = Expr(:vect, map(QuoteNode, index_names)...)
            var_name_node = QuoteNode(var_name)
            push!(code.args, :(
                $(var_name) = $build_destructured_variable(
                    $model_expr,
                    $var_name_node,
                    $sources_expr,
                    $arities_expr,
                    $unpack_expr,
                    $names_expr,
                )
            ))
        elseif _has_fast_membership_filter(var_def)
            axis_names, axis_values, filter_names, filter_sets =
                _parse_membership_filter_declaration(var_def)
            var_name_node = QuoteNode(var_name)
            push!(code.args, :(
                $(var_name) = $build_membership_variable(
                    $model_expr,
                    $var_name_node,
                    $(Expr(:tuple, axis_values...)),
                    $(Expr(:tuple, map(QuoteNode, axis_names)...)),
                    $(Expr(:tuple, map(QuoteNode, filter_names)...)),
                    $(Expr(:tuple, filter_sets...)),
                    $(Expr(:vect, map(QuoteNode, axis_names)...)),
                )
            ))
        # Generate @variable call, using SparseZeroArray for filtered variables.
        elseif _has_filter_condition(var_def)
            container_expr = :(
                ($use_sparse_zero_array)[] ?
                    $sparse_zero_array :
                    $sparse_axis_array
            )
            push!(code.args, Expr(:macrocall, jump_variable, __source__, model_expr, var_def, Expr(:(=), :container, container_expr)))
        else
            push!(code.args, Expr(:macrocall, jump_variable, __source__, model_expr, var_def))
        end

        # Combine block-level and variable-level tags
        all_tag_exprs = vcat(block_tag_exprs, var_tag_exprs)

        # Register metadata
        if !isempty(all_tag_exprs) || !isempty(desc)
            tags_tuple = Expr(:tuple, all_tag_exprs...)
            push!(code.args, :(
                ($variable_metadata)[$(QuoteNode(var_name))] =
                    $variable_metadata_type([$tags_tuple...], $desc)
            ))
        else
            push!(code.args, :(
                ($variable_metadata)[$(QuoteNode(var_name))] =
                    $variable_metadata_type()
            ))
        end
    end

    # Return a named tuple of variables, as JuMP.@variables does.
    if length(var_names) == 1
        push!(code.args, var_names[1])
    else
        named_tuple = Expr(:tuple, [Expr(:(=), n, n) for n in var_names]...)
        push!(code.args, named_tuple)
    end

    return esc(code)
end
