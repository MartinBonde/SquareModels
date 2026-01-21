# ModelDictionaries - Variable-to-value mappings for JuMP models
# Integrated into SquareModels

using Dictionaries
using Arrow
using DataFrames

"""
    ModelDictionary

A dictionary mapping JuMP variable names (as Symbols) to numeric values.

`ModelDictionary` provides convenient syntax for storing and retrieving values
associated with JuMP variables. It supports indexing by variable references,
symbols, or dot notation, and integrates with JuMP's `fix` and `set_start_value`
functions.

# Fields
- `model::AbstractModel`: The JuMP model whose variables are tracked
- `dictionary::Dictionary{Symbol, Union{Nothing, Number}}`: The underlying storage

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)

# Set values using different access methods
d[x] = 1.0
d[:y] = [1, 2, 3]     # Symbol access
d.y = [1, 2, 3]       # Dot notation

# Get values
d[x]       # 1.0
d.y[1]     # 1
```

See also: [`fix`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
struct ModelDictionary
	model::AbstractModel
	dictionary::Dictionary{Symbol, Union{Nothing, Number}}
end
@forward ModelDictionary.dictionary (
	Base.keys,
	Base.values,
	Base.isassigned,
	Base.length,
	Base.iterate,
	Base.filter,
	Base.haskey,
	Base.get,
)

function Base.show(io::IO, ::MIME"text/plain", md::ModelDictionary)
	n = length(md)
	print(io, "ModelDictionary with ", n, " entries")
	n == 0 && return
	println(io, ":")
	# Show first few and last few entries, similar to Vector display
	max_show = get(io, :limit, false) ? 10 : n
	half = max_show ÷ 2
	ks = collect(keys(md.dictionary))
	vs = collect(values(md.dictionary))
	key_width = maximum(length ∘ string, ks; init=1)
	for i in eachindex(ks)
		if n > max_show && i == half + 1
			println(io, "  ⋮")
			continue
		elseif n > max_show && half < i < n - half + 1
			continue
		end
		print(io, "  ", lpad(ks[i], key_width), " => ", vs[i])
		i < n && println(io)
	end
end

Base.show(io::IO, md::ModelDictionary) = show(io, MIME"text/plain"(), md)

"""
    ModelDictionary(model::AbstractModel)

Create a dictionary mapping all variables in `model` to values (initially `nothing`).

Supports convenient syntax for getting/setting values using variable references,
symbols, or dot notation.

# Arguments
- `model::AbstractModel`: The JuMP model whose variables to track

# Returns
A `ModelDictionary` with all model variables initialized to `nothing`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]  # Dot notation

fix(d)  # Fix all variables to their values in d
```

See also: [`fix`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
function ModelDictionary(m)
	md = ModelDictionary(m, Dictionary{Symbol, Union{Nothing, Number}}())
	update!(md)
	return md
end

"""
    ModelDictionary(model::AbstractModel, values::Union{Number, AbstractVector})

Create a dictionary with all variables set to the provided values.

# Arguments
- `model::AbstractModel`: The JuMP model whose variables to track
- `values`: A single number (applied to all) or vector of values

# Returns
A `ModelDictionary` with variables initialized to the given values.
"""
function ModelDictionary(m::AbstractModel, values::Union{Number, AbstractVector})
	d = ModelDictionary(m)
	setindex!.(Ref(d), values, all_variables(m))
	return d
end

Base.copy(md::ModelDictionary) = ModelDictionary(md.model, copy(md.dictionary))

"""Add any missing JuMP variables to the ModelDictionary"""
function update!(md::ModelDictionary)
	for v in Symbol.(all_variables(md.model))
		if v ∉ keys(md.dictionary)
			insert!(md.dictionary, v, nothing)
		end
	end
end

function Base.setindex!(d::ModelDictionary, value, index::Symbol)
	index ∈ keys(d.dictionary) || update!(d)
	index ∉ keys(d.dictionary) && index ∈ keys(d.model.obj_dict) && return setindex!(d, value, d.model.obj_dict[index])
	return setindex!(d.dictionary, value, index)
end
Base.setindex!(d::ModelDictionary, value, index) = setindex!(d, value, Symbol(index))
Base.setindex!(d::ModelDictionary, value, index::AbstractArray) = setindex!.(Ref(d), value, index)

function Base.getindex(d::ModelDictionary, index::Symbol)
	# If key not found, update and try again
	index ∈ keys(d.dictionary) || update!(d)
	index ∈ keys(d.dictionary) && return getindex(d.dictionary, index)
	index ∈ keys(d.model.obj_dict) && return getindex(d, d.model.obj_dict[index])
	return d.dictionary[index] # IndexError
end
Base.getindex(d::ModelDictionary, index) = getindex(d, Symbol(index))
function Base.getindex(d::ModelDictionary, container::AbstractArray{Symbol})
	update!(d)
	# Indices of the variables in the dictionary
	idx = indexin([container...], [keys(d.dictionary)...])
	# References to the values in the dictionary
	data_view = @view(d.dictionary.values[idx])
	return create_window(data_view, container)
end
Base.getindex(d::ModelDictionary, container::AbstractArray) = getindex(d, Symbol.(container))

# Filtering with a boolean ModelDictionary (e.g., d[d .> 0])
function Base.getindex(d::ModelDictionary, mask::ModelDictionary)
	ks = collect(keys(d.dictionary))
	vs = collect(values(d.dictionary))
	mask_vs = collect(values(mask.dictionary))
	selected = mask_vs .== true
	ModelDictionary(d.model, Dictionary(ks[selected], vs[selected]))
end


"""
    Window{T, S}

A view into a subset of a `ModelDictionary`, indexed like a JuMP variable container.

`Window` provides array-like access to a slice of a `ModelDictionary` that corresponds
to an indexed JuMP variable (e.g., `y[1:3]`). It allows reading and writing values
using the same indices as the original variable.

This is an internal type typically created automatically when indexing a
`ModelDictionary` with a variable container.

# Fields
- `data_view::T`: View into the underlying dictionary values
- `indices::S`: Index mapping matching the variable container's axes

# Examples
```julia
model = Model()
@variable(model, y[1:3])

d = ModelDictionary(model)
d.y = [10, 20, 30]

w = d[y]     # Returns a Window
w[1]         # 10
w[2] = 25    # Modify through the window
d[y[2]]      # 25
```
"""
struct Window{T, S}
	data_view::T
	indices::S
end
function create_window(data_view, container)
	indices = (_->0).(container)
	for (i, idx) in enumerate(eachindex(indices))
		indices[idx] = i
	end
	Window(data_view, indices)
end

function Base.getproperty(w::Window, name::Symbol)
	name == :shaped_view && return reshape(w.data_view, size(w.indices))
	return getfield(w, name)
end

@forward Window.indices (
	Base.length,
	Base.size,
	Base.axes,
	Base.ndims,
	Base.keys,
	Base.lastindex,
)
@forward Window.shaped_view (
	Base.iterate,
	Base.collect,
)

Base.getindex(w::Window, index::AbstractArray) = length(index) == 1 ? getindex(w, index[]) : getindex.(Ref(w), index)
Base.getindex(w::Window, indices...) = getindex.(Ref(w.data_view), w.indices[indices...])

Base.setindex!(w::Window, value, index::AbstractArray) = setindex!.(Ref(w), value, index)
Base.setindex!(w::Window, value, indices...) = setindex!.(Ref(w.data_view), value, w.indices[indices...])

Base.in(index::Symbol, d::ModelDictionary) = index ∈ keys(d.dictionary)
Base.in(index, d::ModelDictionary) = Symbol(index) ∈ d
Base.in(index::AbstractArray, d::ModelDictionary) = all(Symbol.(index) .∈ Ref(d))

function Base.replace!(d::ModelDictionary, old_new::Pair...)
	for (k, v) in zip(keys(d), replace(collect(d), old_new...))
		d[k] = v
	end
	return d
end

function Base.replace(d::ModelDictionary, old_new::Pair...)
	d2 = copy(d)
	return replace!(d2, old_new...)
end

# ----------------------------------------------------------------------------------------------------------------------
# JuMP extensions for ModelDictionary
# ----------------------------------------------------------------------------------------------------------------------
"""
    fix(var::VariableRef, d::ModelDictionary)

Fix a single variable to its value in the dictionary.

# Arguments
- `var::VariableRef`: The variable to fix
- `d::ModelDictionary`: Dictionary containing the target value

# Examples
```julia
d = ModelDictionary(model)
d[x] = 5.0
fix(x, d)  # x is now fixed to 5.0
```

See also: [`ModelDictionary`](@ref), [`set_start_value`](@ref)
"""
JuMP.fix(var::VariableRef, d::ModelDictionary) = fix(var, d[var], force=true)

"""
    fix(variables::AbstractArray, d::ModelDictionary)

Fix a collection of variables to their values in the dictionary.

# Arguments
- `variables::AbstractArray`: Array of variable references
- `d::ModelDictionary`: Dictionary containing the target values
"""
JuMP.fix(variables::AbstractArray, d::ModelDictionary) = fix.(variables, Ref(d))

"""
    fix(model::AbstractModel, d::ModelDictionary)
    fix(d::ModelDictionary)

Fix all variables in a JuMP model to their corresponding values in a ModelDictionary.

Variables with `nothing` values in the dictionary are skipped.

# Arguments
- `model::AbstractModel`: The model whose variables to fix (optional if using `fix(d)`)
- `d::ModelDictionary`: Dictionary containing the target values

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]

fix(d)  # Fix all variables to their values in d
# Equivalent to: fix(model, d)
```

See also: [`ModelDictionary`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
function JuMP.fix(model::AbstractModel, d::ModelDictionary)
	variables = filter(v -> !isnothing(d[v]), all_variables(model))
	fix(variables, d)
end
function JuMP.fix(d::ModelDictionary)
	vars = all_variables(d.model)
	if length(d) == length(vars)
		# Fast path: full dictionary, iterate over model variables directly
		for (var, v) in zip(vars, d.dictionary.values)
			isnothing(v) || fix(var, v, force=true)
		end
	else
		# Slow path: subset dictionary, must look up by name
		for (k, v) in pairs(d.dictionary)
			isnothing(v) || fix(variable_by_name(d.model, string(k)), v, force=true)
		end
	end
end

"""
    set_start_value(var::VariableRef, d::ModelDictionary)

Set the starting value of a variable from a ModelDictionary.

# Arguments
- `var::VariableRef`: The variable to set the start value for
- `d::ModelDictionary`: Dictionary containing the start value

See also: [`ModelDictionary`](@ref), [`fix`](@ref)
"""
JuMP.set_start_value(var::VariableRef, values::ModelDictionary) = set_start_value(var, values[var]::Number)

"""
    set_start_value(variables::AbstractArray, d::ModelDictionary)

Set the starting values of a collection of variables from a ModelDictionary.

# Arguments
- `variables::AbstractArray`: Array of variable references
- `d::ModelDictionary`: Dictionary containing the start values
"""
JuMP.set_start_value(variables::AbstractArray, values::ModelDictionary) = set_start_value.(variables, Ref(values))

"""
    set_start_value(model::AbstractModel, d::ModelDictionary)
    set_start_value(d::ModelDictionary)

Set starting values for all variables in a model from a ModelDictionary.

Starting values provide hints to the solver about where to begin the optimization.

# Arguments
- `model::AbstractModel`: The model whose variables to set (optional if using `set_start_value(d)`)
- `d::ModelDictionary`: Dictionary containing the start values

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]

set_start_value(d)  # Set start values for all variables
```

See also: [`ModelDictionary`](@ref), [`fix`](@ref), [`value_dict`](@ref)
"""
JuMP.set_start_value(model::AbstractModel, d::ModelDictionary) = set_start_value.(all_variables(model), Ref(d))
function JuMP.set_start_value(d::ModelDictionary)
	vars = all_variables(d.model)
	if length(d) == length(vars)
		# Fast path: full dictionary
		for (var, v) in zip(vars, d.dictionary.values)
			isnothing(v) || set_start_value(var, v)
		end
	else
		# Slow path: subset dictionary
		for (k, v) in pairs(d.dictionary)
			isnothing(v) || set_start_value(variable_by_name(d.model, string(k)), v)
		end
	end
end

"""
    value_dict(model::AbstractModel) → ModelDictionary

Extract the solution values of all variables as a ModelDictionary.

Call this after `optimize!(model)` to capture the solution in a dictionary
that can be used for warm-starting, comparing solutions, or fixing variables.

# Arguments
- `model::AbstractModel`: A solved JuMP model

# Returns
A `ModelDictionary` containing the optimal value of each variable.

# Examples
```julia
model = Model(Ipopt.Optimizer)
@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + y == 10)
@objective(model, Max, x + 2y)

optimize!(model)

d = value_dict(model)
d[x]  # Optimal value of x
d[y]  # Optimal value of y

# Use solution as starting point for another solve
set_start_value(d)
```

See also: [`ModelDictionary`](@ref), [`fix`](@ref), [`set_start_value`](@ref)
"""
value_dict(model::AbstractModel) = ModelDictionary(model, value.(all_variables(model)))

# ----------------------------------------------------------------------------------------------------------------------
# Arrow/Parquet serialization
# ----------------------------------------------------------------------------------------------------------------------
"""
    parse_variable_name(name::String) → (base_name, indices)

Parse a JuMP variable name into its base name and index string.

# Examples
```julia
parse_variable_name("K[2025]")       # ("K", "2025")
parse_variable_name("cᵃ[15,2025]")   # ("cᵃ", "15,2025")
parse_variable_name("σˣ")            # ("σˣ", "")
```
"""
function parse_variable_name(name::String)
	m = match(r"^(.+?)\[(.+)\]$", name)
	isnothing(m) && return (name, "")
	return (m.captures[1], m.captures[2])
end

"""
    save(path::AbstractString, d::ModelDictionary)

Save a ModelDictionary to an Arrow file.

The dictionary is stored as a table with columns:
- `variable`: The base variable name (e.g., "K", "cᵃ")
- `indices`: The index string (e.g., "2025", "15,2025", "" for scalars)
- `value`: The numeric value

# Arguments
- `path`: File path (typically ending in .arrow or .parquet)
- `d`: The ModelDictionary to save

# Examples
```julia
d = value_dict(model)
save("solution.arrow", d)
```

See also: [`load`](@ref), [`ModelDictionary`](@ref)
"""
function save(path::AbstractString, d::ModelDictionary)
	rows = NamedTuple{(:variable, :indices, :value), Tuple{String, String, Float64}}[]
	for (k, v) in pairs(d.dictionary)
		isnothing(v) && continue
		base, indices = parse_variable_name(string(k))
		push!(rows, (; variable=base, indices=indices, value=Float64(v)))
	end
	Arrow.write(path, DataFrame(rows))
end

"""
    load(path::AbstractString, model::AbstractModel) → ModelDictionary

Load a ModelDictionary from an Arrow file.

Iterates over all variables in the model and looks up their values in the data file.
Variables not found in the file will have `nothing` values.

# Arguments
- `path`: Path to the Arrow file
- `model`: The JuMP model to associate with the dictionary

# Returns
A `ModelDictionary` populated with values from the file.
Variables in the model that aren't in the file will have `nothing` values.

# Examples
```julia
d = load("solution.arrow", model)
set_start_value(d)  # Use loaded values as starting point
```

See also: [`save`](@ref), [`ModelDictionary`](@ref)
"""
function load(path::AbstractString, model::AbstractModel)
	df = DataFrame(Arrow.Table(path))

	# Build index for O(1) lookup: (variable, indices) => value
	data_index = Dict{Tuple{String, String}, Float64}()
	for row in eachrow(df)
		data_index[(row.variable, row.indices)] = row.value
	end

	d = ModelDictionary(model)
	for var in all_variables(model)
		key = _var_to_arrow_key(var)
		if haskey(data_index, key)
			d[var] = data_index[key]
		end
	end
	return d
end

"""
Convert a JuMP variable to the (variable, indices) key format used in Arrow files.
E.g., x → ("x", ""), y[1,2] → ("y", "1,2")
"""
function _var_to_arrow_key(var::AbstractVariableRef)
	base, indices = parse_variable_name(name(var))
	return (base, indices)
end

# ----------------------------------------------------------------------------------------------------------------------
# Dot access
# ----------------------------------------------------------------------------------------------------------------------
Base.setproperty!(d::ModelDictionary, name::Symbol, value) = setindex!(d, value, name)
Base.getproperty(d::ModelDictionary, sym::Symbol) = sym in fieldnames(typeof(d)) ? getfield(d, sym) : d[sym]

# ----------------------------------------------------------------------------------------------------------------------
# Broadcasting
# ----------------------------------------------------------------------------------------------------------------------
struct ModelDictionaryStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:ModelDictionary}) = ModelDictionaryStyle()
Base.BroadcastStyle(::ModelDictionaryStyle, ::Broadcast.DefaultArrayStyle{0}) = ModelDictionaryStyle()
Base.BroadcastStyle(s::ModelDictionaryStyle, ::ModelDictionaryStyle) = s

# ModelDictionary participates directly in broadcasting (not converted via collect)
Base.broadcastable(md::ModelDictionary) = md
Base.axes(md::ModelDictionary) = (Base.OneTo(length(md)),)
Base.getindex(md::ModelDictionary, i::Int) = md.dictionary.values[i]

# Find the first ModelDictionary in broadcast arguments (including nested Broadcasted)
_find_model_dict(md::ModelDictionary) = md
_find_model_dict(bc::Broadcast.Broadcasted) = _find_model_dict(bc.args)
_find_model_dict(::Any) = nothing
function _find_model_dict(args::Tuple)
	for arg in args
		result = _find_model_dict(arg)
		isnothing(result) || return result
	end
	nothing
end

_bc_collect(md::ModelDictionary) = collect(md.dictionary.values)
_bc_collect(x) = x

function Base.copy(bc::Broadcast.Broadcasted{ModelDictionaryStyle})
	md = _find_model_dict(bc.args)
	flat = Broadcast.flatten(bc)
	# Unwrap ModelDictionaries to their values, broadcast scalars normally
	unwrapped = map(_bc_collect, flat.args)
	new_values = broadcast(flat.f, unwrapped...)
	ModelDictionary(md.model, Dictionary(keys(md.dictionary), new_values))
end
