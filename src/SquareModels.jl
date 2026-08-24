# Copyright 2022, Martin Kirk Bonde and contributors
# Licensed under the MIT License. See LICENSE.md for details.

"""
SquareModels
A JuMP extension for writing modular models with square systems of equations
"""
module SquareModels

export @block, @test_constraint, Block, TestConstraint, Equation, @endo_exo_swap!, @variables, add_equation!
export endogenous, residuals, residual, variables, exogenous, is_endogenous, overlaps, shared_endogenous
export VariableRef  # Re-exported from JuMP for macro hygiene
export ModelDictionary, fix, unfix, set_start_value, value, value_dict, add_missing_model_variables!
export keys_match, assert_no_diff, assert_residuals_small, assert_test_constraints, test_constraints, test_constraint_variables
export SquareModelError, ResidualError, ToleranceError, TestConstraintError, NonSquareError
export unload, load, read_indices, read_sparse_array, read_variable
export RESIDUAL_SUFFIX
export solve, solve!, diagnose, annotate_lst!, square_model
export Tag, description, tags, has_tag, tagged, metadata
export SparseZeroArray, select_axes, merge_indices, ∑, use_sparse_zero_array!
export ModelExpressions, ModelPlotting, @plot, @evalexpr, @prt, plotvar, plotseries, alternating_dash!, labeled, LabeledSeries, LabeledArray, MultiVarResult, AbstractSeries, set_plot_finalize!, reset_plot_finalize!, plot_finalize
export set_default_source!, set_default_operator!, set_default_periods!, set_column_label_total_width!, reset_print_defaults!

"""
    RESIDUAL_SUFFIX

Suffix appended to endogenous variable names to name their residual variables
(default `"_J"`, J for "junk" or adjustment). See [`residuals`](@ref).
"""
RESIDUAL_SUFFIX = "_J"

# ----------------------------------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------------------------------
using Base.Meta: isexpr
using StatsBase: countmap
using Lazy
using JuMP: JuMP, AbstractModel, AbstractVariableRef, VariableRef, Containers
using JuMP: AffExpr, QuadExpr, NonlinearExpr
using JuMP.Containers: DenseAxisArray, SparseAxisArray
using JuMP: @variable
using JuMP: set_name, name, fix, is_fixed, unfix, all_variables, value, set_start_value
import MathOptInterface as MOI
const _name_lookup_cache = WeakKeyDict{AbstractModel, Dict{String, VariableRef}}()

include("errors.jl")
include("utils.jl")
include("SparseZeroArrays.jl")
include("TableDisplay.jl")

"""
    AbstractSeries

Supertype for labelled, plottable data series. Concrete subtypes — `Window`
(a view onto model data) and `LabeledSeries` (a single eager, computed line) —
implement `ModelPlotting.expand(s) -> Vector{LabeledSeries}`, which splits the data
into one labelled line per leading-index combination (the last dimension is the
x-axis, e.g. `y[region, year]` becomes one line per region over the years).
Dispatch on `AbstractSeries` to write plotting code that accepts either.
"""
abstract type AbstractSeries end

"""
    Equation

Lightweight storage for a model equation: the expression and its constraint set.
Unlike `ConstraintRef`, this does NOT register with JuMP/MOI, avoiding backend bridging overhead.
"""
struct Equation
	func::Any
	set::MOI.AbstractScalarSet
end

"""
    TestConstraint

A constraint that tests a solution but does not determine an endogenous variable.

`@test_constraint` entries in a [`@block`](@ref) create one `TestConstraint` per
index. Each test constraint stores a variable used for its name and relative
tolerance, its JuMP equation, an optional message, and optional tolerance
overrides.
"""
struct TestConstraint
	variable::VariableRef
	equation::Equation
	message::String
	atol::Union{Nothing,Float64}
	rtol::Union{Nothing,Float64}
end

TestConstraint(variable, equation, message) = TestConstraint(variable, equation, message, nothing, nothing)

collect_variables!(vars::Set{VariableRef}, eq::Equation) = collect_variables!(vars, eq.func)

"""
    collect_variables!(vars::Set{VariableRef}, expr) → Set{VariableRef}

Recursively collect all VariableRef objects from a JuMP expression.
Works with AffExpr (linear), QuadExpr (quadratic), and NonlinearExpr (nonlinear).
"""
function collect_variables!(vars::Set{VariableRef}, expr)
    if expr isa VariableRef
        push!(vars, expr)
    elseif expr isa AffExpr
        union!(vars, keys(expr.terms))
    elseif expr isa QuadExpr
        union!(vars, keys(expr.aff.terms))
        for (pair, _) in expr.terms
            push!(vars, pair.a)
            push!(vars, pair.b)
        end
    elseif expr isa NonlinearExpr
        for arg in expr.args
            collect_variables!(vars, arg)
        end
    end
    return vars
end
collect_variables!(vars::Set{VariableRef}, ::Union{Number, Zero}) = vars

"""
    Block

A mapping between equations and their associated endogenous variables in a JuMP model.

Blocks represent "square" systems where each equation is paired with exactly one
variable, enabling modular model construction and endo-exo swaps (changing which
variable is determined by which equation).

Blocks store `Equation` objects (lightweight func + set pairs). When using `solve`,
these equations are transformed (substituting exogenous values from the data) and
added to an intermediate solve model.

# Fields
- `model::AbstractModel`: The JuMP model containing the variables
- `endogenous::Vector{VariableRef}`: Vector of endogenous variable references
- `residuals::Vector{VariableRef}`: Vector of residual variable references
- `variables::Set{VariableRef}`: All variables appearing in the block's equations
- `_endogenous_set::Set{VariableRef}`: Set for O(1) membership checking of endogenous variables
- `equations::Vector{Equation}`: Equation expressions (func + set pairs)
- `test_constraints::Vector{TestConstraint}`: Constraints that test the solution but do not enter the solve system

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

length(b)  # 4 (one scalar + three indexed)
x ∈ b      # true
```

See also: [`@block`](@ref), [`@endo_exo_swap!`](@ref), [`endogenous`](@ref), [`variables`](@ref), [`solve`](@ref)
"""
struct Block
	model::AbstractModel
	endogenous::Vector{VariableRef}
	residuals::Vector{VariableRef}
	variables::Set{VariableRef}
	_endogenous_set::Set{VariableRef}
	equations::Vector{Equation}
	test_constraints::Vector{TestConstraint}

	function Block(
		model::AbstractModel,
		endogenous::Vector{VariableRef},
		residuals::Vector{VariableRef},
		variables::Set{VariableRef},
		equations::Vector{Equation},
		test_constraints::Vector{TestConstraint}=TestConstraint[]
	)
		length(equations) == length(endogenous) ||
			error("Block must be square: got $(length(equations)) equations and $(length(endogenous)) endogenous variables")

		endogenous_set = Set{VariableRef}(endogenous)
		if length(endogenous_set) != length(endogenous)
			throw(NonSquareError(
				"Non-unique mapping between endogenous variables and equations in block definition:",
				non_unqiue_pairs(endogenous, equations),
			))
		end

		new(model, endogenous, residuals, variables, endogenous_set, equations, test_constraints)
	end
end

Block(model) = Block(model, VariableRef[], VariableRef[], Set{VariableRef}(), Equation[], TestConstraint[])

Base.length(b::Block) = length(b.endogenous)
Base.iterate(b::Block) = iterate(b.endogenous)
Base.iterate(b::Block, state) = iterate(b.endogenous, state)
Base.copy(b::Block) = Block(b.model, copy(b.endogenous), copy(b.residuals), copy(b.variables), copy(b.equations), copy(b.test_constraints))

"""
    is_endogenous(var::VariableRef, b::Block) → Bool

Check if a variable is endogenous in the block (i.e., has an associated constraint).
Uses O(1) set lookup.

# Example
```julia
b = @block model begin
    x, x == 1
end
is_endogenous(x, b)  # true
```
"""
is_endogenous(var::VariableRef, b::Block) = var ∈ b._endogenous_set

"""
    var ∈ block → Bool

Check if a variable appears in the block's constraints (either as endogenous or exogenous).
Uses O(1) set lookup.

See also: [`is_endogenous`](@ref) to check specifically for endogenous variables.
"""
Base.in(var::VariableRef, b::Block) = var ∈ b.variables

"""
    endogenous(b::Block) → Vector{VariableRef}

Return the vector of endogenous variable references in the block.

These are the variables being solved for - each paired with a constraint.

# Arguments
- `b::Block`: The block to get endogenous variables from

# Returns
A vector of `VariableRef` objects representing all endogenous variables in the block.
The order corresponds to the order in which constraints were defined.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

for v in endogenous(b)
    println(name(v))
end
```

See also: [`variables`](@ref), [`exogenous`](@ref)
"""
endogenous(b::Block) = b.endogenous

"""
    residuals(b::Block) → Vector{VariableRef}

Return the residual variables corresponding to each endogenous variable in the block.

Residual variables are automatically created when defining blocks and are named
with the suffix defined by `RESIDUAL_SUFFIX` (default "_J"). They are fixed to 0
by default and can be used to:
- Check for data inconsistencies (unfix residual, fix endo, solve, check residual value)
- Temporarily disable equations (exogenize endo, endogenize residual)
- Debug model issues

# Arguments
- `b::Block`: The block to get residuals from

# Returns
A vector of `VariableRef` objects representing the residual variables.
The order corresponds to `endogenous(b)`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

res = residuals(b)
# res[1] is x_J, res[2:4] are y_J[1], y_J[2], y_J[3]
```

See also: [`endogenous`](@ref), [`residuals(::AbstractModel)`](@ref)
"""
residuals(b::Block) = b.residuals

"""
    test_constraints(b::Block) -> Vector{TestConstraint}

Return the test constraints stored in `b`. Test constraints do not enter the
square solve system. Use [`assert_test_constraints`](@ref) to test them against
a [`ModelDictionary`](@ref).
"""
test_constraints(b::Block) = b.test_constraints

"""
    test_constraint_variables(b::Block) -> Vector{VariableRef}

Return all variables needed to test the constraints stored in `b`. These
variables stay separate from [`variables`](@ref) and [`exogenous`](@ref), which
describe the square solve system.
"""
function test_constraint_variables(b::Block)
	vars = Set{VariableRef}(c.variable for c in b.test_constraints)
	foreach(c -> collect_variables!(vars, c.equation), b.test_constraints)
	return collect(vars)
end


"""
    variables(b::Block) → Vector{VariableRef}

Return a vector of all variables that appear in the block's solve constraints.

This includes both endogenous variables (being solved for) and exogenous variables
(parameters to this block). It does not include variables used only by test
constraints. Only variables that are actually used in the solve expressions are
included - unused indices are not present.

# Arguments
- `b::Block`: The block to get variables from

# Returns
A `Vector{VariableRef}` of all variables referenced in the block's constraints.

See also: [`endogenous`](@ref), [`exogenous`](@ref), [`test_constraint_variables`](@ref)
"""
variables(b::Block) = collect(b.variables)

"""
    exogenous(b::Block) → Vector{VariableRef}

Return a vector of exogenous variables that appear in the block's solve constraints.

These are variables that are referenced in the constraint expressions but are not
endogenous (not being solved for) within this block. Only variables that are
actually used are included - unused variable indices are not present.

# Arguments
- `b::Block`: The block to get exogenous variables from

# Returns
A `Vector{VariableRef}` of all exogenous variables referenced in the block.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])
@variable(model, z[1:3])

b = @block model begin
    x, x == sum(y[i] for i in 1:3)
    z[i ∈ 1:3], z[i] == y[i] * 2
end

exo = exogenous(b)  # Contains y[1], y[2], y[3]
```

See also: [`endogenous`](@ref), [`variables`](@ref), [`test_constraint_variables`](@ref)
"""
exogenous(b::Block) = collect(setdiff(b.variables, b._endogenous_set))

"""
    residuals(model::AbstractModel) → Vector{VariableRef}

Return all residual variables in the model.

Residual variables are identified by their name suffix (defined by `RESIDUAL_SUFFIX`,
default "_J"). This function collects all such variables from the model.

# Arguments
- `model::AbstractModel`: The JuMP model to search for residual variables

# Returns
A vector of `VariableRef` objects representing all residual variables in the model.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

res = residuals(model)
# Returns [x_J, y_J[1], y_J[2], y_J[3]]
```

See also: [`residuals(::Block)`](@ref), [`RESIDUAL_SUFFIX`](@ref)
"""
residuals(model::AbstractModel) = filter(v -> endswith(base_name(v), RESIDUAL_SUFFIX), all_variables(model))

"""
    Base.summary(io::IO, b::Block)

Print a one-line summary of a block showing the number of equations and variables.

# Arguments
- `io::IO`: The IO stream to print to
- `b::Block`: The block to summarize

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

summary(stdout, b)  # prints: "Block with 4 equations over 4 variables"
```

See also: [`Block`](@ref)
"""
function Base.summary(io::IO, b::Block)
	n = length(b)
	print(io, "Block with $n equations over $n variables")
	isempty(b.test_constraints) || print(io, " and $(length(b.test_constraints)) test constraints")
end

"""
    overlaps(a::Block, b::Block) → Bool

Check if two blocks share any endogenous variables.

Returns `true` if a variable is endogenous in both blocks, which indicates
that the blocks cannot be combined without creating duplicate equations.
Variables that are exogenous in one or both blocks do not count as overlap.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
`true` if the blocks have at least one endogenous variable in common, `false` otherwise

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b1 = @block model begin
    x, x == 1
    y[i ∈ 1:2], y[i] == i
end

b2 = @block model begin
    y[i ∈ 2:3], y[i] == i  # y[2] appears in both blocks
end

overlaps(b1, b2)  # true
```

See also: [`shared_endogenous`](@ref), [`Block`](@ref)
"""
overlaps(a::Block, b::Block) = !isempty(intersect(a._endogenous_set, b._endogenous_set))

"""
    shared_endogenous(a::Block, b::Block) → Vector{VariableRef}

Return the endogenous variables that appear in both blocks.

Useful for understanding how blocks are interconnected and for detecting
accidental duplicate equations.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
A vector of `VariableRef` objects that are endogenous in both blocks (may be empty)

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b1 = @block model begin
    x, x == 1
    y[i ∈ 1:2], y[i] == i
end

b2 = @block model begin
    y[i ∈ 2:3], y[i] == i
end

shared = shared_endogenous(b1, b2)  # [y[2]]
y[2] ∈ shared  # true
y[1] ∈ shared  # false
```

See also: [`overlaps`](@ref), [`Block`](@ref)
"""
shared_endogenous(a::Block, b::Block) = collect(intersect(a._endogenous_set, b._endogenous_set))

"""Format variables grouped by base name for readable error messages."""
function format_variables(vars::AbstractVector{VariableRef})
	groups = Dict{String, Vector{VariableRef}}()
	for var in vars
		bn = base_name(var)
		push!(get!(groups, bn, VariableRef[]), var)
	end

	lines = String[]
	for (bn, group) in sort(collect(groups), by=first)
		if length(group) == 1
			push!(lines, "  $bn: $(group[1])")
		else
			examples = string.(group[1:min(3, length(group))])
			examples_str = join(examples, ", ")
			if length(group) > 3
				examples_str *= ", ..."
			end
			push!(lines, "  $bn: $(length(group)) elements (e.g., $examples_str)")
		end
	end
	return join(lines, "\n")
end

function Block(
	model::AbstractModel,
	endogenous::AbstractArray{V},
	residuals::AbstractArray{R},
	variables::Set{VariableRef},
	equations::Vector{Equation},
	test_constraints::Vector{TestConstraint}=TestConstraint[]
) where {V<:VariableRef, R<:VariableRef}
	Block(model, VariableRef[endogenous...], VariableRef[residuals...], variables, equations, test_constraints)
end

function Base.:+(a::Block, b::Block)
	a.model == b.model || error("Cannot add $a and $b. Blocks must belong to the same model.")

	if overlaps(a, b)
		shared = shared_endogenous(a, b)
		formatted = format_variables(shared)
		error("Cannot combine blocks: $(length(shared)) endogenous variable(s) appear in both blocks.\n" *
		      "Overlapping endogenous variables:\n$formatted\n" *
		      "This would create a non-square system with more constraints than unique endogenous variables.")
	end

	combined_vars = union(a.variables, b.variables)
	combined_eqs = vcat(a.equations, b.equations)
	combined_test_constraints = vcat(a.test_constraints, b.test_constraints)
	Block(a.model, vcat(a.endogenous, b.endogenous), vcat(a.residuals, b.residuals), combined_vars, combined_eqs, combined_test_constraints)
end

function Base.:-(a::Block, b::Block)
	a.model == b.model || error("Cannot subtract $b from $a. Blocks must belong to the same model.")
	mask = [v ∉ b._endogenous_set for v in a.endogenous]
	filtered_eqs = a.equations[mask]

	all_vars = Set{VariableRef}()
	for eq in filtered_eqs
		collect_variables!(all_vars, eq.func)
	end

	filtered_test_constraints = copy(a.test_constraints)
	for constraint in b.test_constraints
		index = findfirst(candidate -> candidate === constraint, filtered_test_constraints)
		index === nothing || deleteat!(filtered_test_constraints, index)
	end
	Block(a.model, a.endogenous[mask], a.residuals[mask], all_vars, filtered_eqs, filtered_test_constraints)
end

make_residual_name(var) = string(var) * SquareModels.RESIDUAL_SUFFIX

"""Cached version of JuMP.variable_by_name — O(1) after first call per model."""
function variable_by_name(model::AbstractModel, var_name::AbstractString)
	lookup = get!(_name_lookup_cache, model) do
		Dict{String, VariableRef}(name(v) => v for v in all_variables(model))
	end
	key = String(var_name)
	v = get(lookup, key, nothing)
	v !== nothing && return v
	for v in all_variables(model)
		n = name(v)
		haskey(lookup, n) || (lookup[n] = v)
	end
	return get(lookup, key, nothing)
end

"""
    add_equation!(block::Block, endo::VariableRef, lhs, rhs=0) → Block

Add an endogenous variable and its equation to `block` in place.

This creates and fixes a residual variable. If `endo` occurs in the equation,
the residual adjusts its first stored occurrence. If `endo` does not occur, the
equation becomes `lhs == rhs + residual`. An error is thrown if `endo` is already
endogenous in the block.

# Example
```julia
block = Block(model)
add_equation!(block, x, x, 1)
```
"""
function add_equation!(block::Block, endo::VariableRef, lhs, rhs=0)
	endo ∉ block._endogenous_set || error("Cannot add equation: $(name(endo)) is already endogenous in this block.")
	resid = _residual_for(endo)
	eq = _equation_with_residual(endo, resid, lhs, rhs)
	push!(block.endogenous, endo)
	push!(block.residuals, resid)
	push!(block.equations, eq)
	collect_variables!(block.variables, eq)
	push!(block._endogenous_set, endo)
	return block
end

"""Helper function to extract base name from variable reference"""
_get_name(s::Symbol) = s
_get_name(e::Expr) = e.args[1]

"""Return `expr` with `resid` added at its first stored occurrence of `endo`."""
_with_residual(expr, endo::VariableRef, resid::VariableRef) = (expr, false)

_with_residual(expr::VariableRef, endo::VariableRef, resid::VariableRef) =
	expr == endo ? (expr + resid, true) : (expr, false)

function _with_residual(expr::AffExpr, endo::VariableRef, resid::VariableRef)
	coef = get(expr.terms, endo, 0.0)
	iszero(coef) && return expr, false
	return expr + coef * resid, true
end

function _with_residual(expr::QuadExpr, endo::VariableRef, resid::VariableRef)
	for (pair, coef) in expr.terms
		iszero(coef) && continue
		pair.a == endo && pair.b == endo &&
			return expr + coef * (2 * endo * resid + resid^2), true
		pair.a == endo && return expr + coef * resid * pair.b, true
		pair.b == endo && return expr + coef * pair.a * resid, true
	end
	new_aff, found = _with_residual(expr.aff, endo, resid)
	found || return expr, false
	return QuadExpr(new_aff, copy(expr.terms)), true
end

function _with_residual(expr::NonlinearExpr, endo::VariableRef, resid::VariableRef)
	new_args = copy(expr.args)
	for i in eachindex(new_args)
		new_args[i], found = _with_residual(new_args[i], endo, resid)
		found && return NonlinearExpr(expr.head, new_args), true
	end
	return expr, false
end

"""Build an equation with one residual insertion for both `@block` and `add_equation!`."""
function _equation_with_residual(endo::VariableRef, resid::VariableRef, lhs, rhs)
	new_lhs, found = _with_residual(lhs, endo, resid)
	found && return Equation(new_lhs - rhs, MOI.EqualTo(0.0))
	new_rhs, found = _with_residual(rhs, endo, resid)
	return Equation(found ? lhs - new_rhs : lhs - rhs - resid, MOI.EqualTo(0.0))
end

"""Create the residual container for `endo` when needed and return its matching item."""
function _residual_for(endo::VariableRef)
	m = endo.model
	base, indices = split_name(endo)
	residual_base = make_residual_name(base)
	haskey(m, Symbol(residual_base)) || copy_variable(residual_base, m[Symbol(base)])
	resid = variable_by_name(m, residual_base * indices)
	resid === nothing && error("Cannot find residual for $(name(endo))")
	return resid
end

"""Collect index tuples from a JuMP container in iteration order."""
_all_keys(c::AbstractArray) = vec(collect(Iterators.product(axes(c)...)))
_all_keys(c::SparseAxisArray) = keys(c.data)
_all_keys(c::SparseZeroArray) = _all_keys(c.data)

"""Flatten nested tuples in a key.
E.g. `((:a, :b), 1)` becomes `(:a, :b, 1)`."""
_flatten_key(k::Tuple) = tuple(Iterators.flatten(map(x -> x isa Tuple ? x : (x,), k))...)

"""Index a variable with a constraint key, flattening only when the variable has more dimensions.
Handles the difference between `x[a,b,t]` (3D) and `y[(a,b),t]` (2D with tuple index)."""
_ndims(v::SparseAxisArray{T,N}) where {T,N} = N
_ndims(v::SparseZeroArray{T,N}) where {T,N} = N
_ndims(v::AbstractArray) = ndims(v)
function _index_var(var, k::Tuple)
	fk = _flatten_key(k)
	fk === k && return var[k...]
	_ndims(var) == length(fk) ? var[fk...] : var[k...]
end

"""Return stored keys when a sparse mapped variable has the stated number of axes."""
_mapped_sparse_constraint_keys(_, ::Val) = nothing
_mapped_sparse_constraint_keys(var::SparseAxisArray{T,N}, ::Val{N}) where {T,N} = keys(var.data)
_mapped_sparse_constraint_keys(var::SparseZeroArray{T,N}, ::Val{N}) where {T,N} = keys(var)

"""Parse one scalar named index such as `i = I` or `i in I`."""
function _named_constraint_axis(axis)
	if (isexpr(axis, :kw) || isexpr(axis, :(=))) && axis.args[1] isa Symbol
		return axis.args[1], axis.args[2]
	elseif isexpr(axis, :call) && length(axis.args) == 3 &&
	       axis.args[1] in (:in, :∈) && axis.args[2] isa Symbol
		return axis.args[2], axis.args[3]
	end
	return nothing
end

_constraint_expr_uses_symbol(symbol::Symbol, names) = symbol in names
_constraint_expr_uses_symbol(expr::Expr, names) =
	any(arg -> _constraint_expr_uses_symbol(arg, names), expr.args)
_constraint_expr_uses_symbol(_, _) = false

"""Return named axes and a semicolon filter, or `nothing` for other JuMP index forms."""
function _named_constraint_indices(ref_vars)
	isexpr(ref_vars, :ref) || isexpr(ref_vars, :typed_vcat) || return nothing
	axes = Any[ref_vars.args[2:end]...]
	condition = nothing

	if isexpr(ref_vars, :typed_vcat)
		length(axes) <= 2 || return nothing
		length(axes) == 2 && (condition = pop!(axes))
	else
		parameters = findall(axis -> isexpr(axis, :parameters), axes)
		length(parameters) <= 1 || return nothing
		if !isempty(parameters)
			parameter = axes[only(parameters)]
			length(parameter.args) == 1 || return nothing
			condition = only(parameter.args)
			deleteat!(axes, only(parameters))
		end
	end

	isempty(axes) && return nothing
	parsed = map(_named_constraint_axis, axes)
	any(isnothing, parsed) && return nothing
	names = first.(parsed)
	allunique(names) || return nothing
	return names, last.(parsed), condition
end

"""Wrap a named-axis value so `in` treats a scalar as a one-element set."""
_axis_collection(axis::AbstractString) = (axis,)
_axis_collection(axis::Symbol) = (axis,)
_axis_collection(axis) = applicable(iterate, axis) ? axis : (axis,)

"""Build the normal JuMP indices and an optional stored-key form for a mapped variable."""
function _constraint_index_plan(ref_vars, stored_keys)
	indices = Expr(isexpr(ref_vars, :ref) ? :vect : :vcat, ref_vars.args[2:end]...)
	parsed = _named_constraint_indices(ref_vars)
	parsed === nothing && return indices, nothing, nothing

	names, values, condition = parsed
	bound_values = Any[]
	bindings = Pair{Symbol,Any}[]
	prior_names = Symbol[]
	axis_collection = GlobalRef(@__MODULE__, :_axis_collection)
	for (name, value) in zip(names, values)
		if _constraint_expr_uses_symbol(value, prior_names)
			push!(bound_values, value)
		else
			bound_value = gensym(name)
			push!(bound_values, bound_value)
			push!(bindings, bound_value => value)
		end
		push!(prior_names, name)
	end
	checks = Any[
		Expr(:call, :in, name, Expr(:call, axis_collection, value))
		for (name, value) in zip(names, bound_values)
	]
	condition === nothing || push!(checks, condition)
	filter_condition = reduce((left, right) -> Expr(:&&, left, right), checks)
	key_names = Expr(:tuple, names...)
	sparse_indices = Expr(:vcat, Expr(:call, :in, key_names, stored_keys), filter_condition)
	return indices, sparse_indices, (length(names), bindings)
end

_expression_macrocall(jump_expression, source, indices, expr) =
	Expr(:macrocall, jump_expression, source, :_m, indices, expr)

"""Assign `@expression(_m, indices, expr)` for each name => expr pair."""
function _named_index_expression_code(ref_vars, jump_expression, source, assigns::Pair...)
	stored_keys = gensym(:stored_keys)
	indices, sparse_indices, sparse_plan = _constraint_index_plan(ref_vars, stored_keys)
	dense_block = Expr(:block, (
		:($(lhs) = $(_expression_macrocall(jump_expression, source, indices, rhs)))
		for (lhs, rhs) in assigns
	)...)
	sparse_plan === nothing && return dense_block

	arity, bindings = sparse_plan
	mapped_sparse_keys_ref = GlobalRef(@__MODULE__, :_mapped_sparse_constraint_keys)
	val_ref = GlobalRef(Base, :Val)
	sparse_setup = Expr(:block, (:(local $name = $value) for (name, value) in bindings)...)
	sparse_block = Expr(:block, (
		:($(lhs) = $(_expression_macrocall(jump_expression, source, sparse_indices, rhs)))
		for (lhs, rhs) in assigns
	)...)
	quote
		$stored_keys = $mapped_sparse_keys_ref($(_get_name(ref_vars)), $val_ref($arity))
		if $stored_keys === nothing
			$dense_block
		else
			$sparse_setup
			$sparse_block
		end
	end
end

"""Extract the JuMP model from a container (ModelDictionary or Model)"""
_get_model(m::AbstractModel) = m
# _get_model for ModelDictionary is defined after ModelDictionaries.jl is included

"""Split `lhs op rhs` into `(lhs - rhs)` at the AST level."""
function _constraint_to_diff(expr::Expr)
	if expr.head == :call && expr.args[1] in (:(==), :(<=), :≤, :(>=), :≥) && length(expr.args) == 3
		lhs, rhs = expr.args[2], expr.args[3]
		return :($lhs - ($rhs))
	end
	Expr(expr.head, [_constraint_to_diff(a) for a in expr.args]...)
end
_constraint_to_diff(x) = x

"""Helper macro for Block macro - returns (endogenous, residuals, equations) where equations are vectors parallel to endogenous"""
macro _block(container, ref_vars, constraint, extra...)
	_error(str...) = JuMP._macro_error(:block, (container, ref_vars, constraint, extra...), __source__, str...)
	sm = @__MODULE__
	jump_expression = GlobalRef(JuMP, Symbol("@expression"))
	get_model = GlobalRef(sm, :_get_model)
	residual_for_ref = GlobalRef(sm, :_residual_for)
	equation_ref = GlobalRef(sm, :Equation)
	equation_with_residual_ref = GlobalRef(sm, :_equation_with_residual)
	all_keys_ref = GlobalRef(sm, :_all_keys)
	index_var_ref = GlobalRef(sm, :_index_var)
	code = Expr(:block)
	base_sym = _get_name(ref_vars)

	model_expr = :($get_model($container))

	lhs, rhs = constraint.args[2], constraint.args[3]

	if isa(ref_vars, Symbol)
		lhs_call = Expr(:macrocall, jump_expression, __source__, :_m, lhs)
		rhs_call = Expr(:macrocall, jump_expression, __source__, :_m, rhs)
		macrocall = quote
			let _m = $model_expr
				endo = $ref_vars
				resid = $residual_for_ref(endo)
				eqs = $equation_ref[$equation_with_residual_ref(endo, resid, $lhs_call, $rhs_call)]
				([endo], [resid], eqs)
			end
		end
	elseif isexpr(ref_vars, :ref) || isexpr(ref_vars, :typed_vcat)
		expr_code = _named_index_expression_code(
			ref_vars, jump_expression, __source__, :_lhs => lhs, :_rhs => rhs,
		)
		macrocall = quote
			let _m = $model_expr
				$expr_code
				_ks = $all_keys_ref(_lhs)
				endos = [$index_var_ref($base_sym, k) for k in _ks]
				resids = [$residual_for_ref(endo) for endo in endos]
				eqs = $equation_ref[$equation_with_residual_ref(
					endo,
					resid,
					_lhs[k...],
					_rhs[k...],
				) for (k, endo, resid) in zip(_ks, endos, resids)]
				(endos, resids, eqs)
			end
		end
	else
		_error("Reference must be a variable")
	end
	push!(code.args, macrocall)
	return esc(code)
end

"""Build indexed `TestConstraint` objects without residual variables or solve constraints."""
macro _test_constraint(container, ref_vars, constraint, message, atol, rtol)
	_error(str...) = JuMP._macro_error(:test_constraint, (container, ref_vars, constraint, message, atol, rtol), __source__, str...)
	sm = @__MODULE__
	jump_expression = GlobalRef(JuMP, Symbol("@expression"))
	get_model = GlobalRef(sm, :_get_model)
	test_constraint_ref = GlobalRef(sm, :TestConstraint)
	equation_ref = GlobalRef(sm, :Equation)
	all_keys_ref = GlobalRef(sm, :_all_keys)
	index_var_ref = GlobalRef(sm, :_index_var)
	set_ref = GlobalRef(
		MOI,
		constraint.args[1] in (:(<=), :≤) ? :LessThan :
		constraint.args[1] in (:(>=), :≥) ? :GreaterThan : :EqualTo,
	)
	base_sym = _get_name(ref_vars)
	model_expr = :($get_model($container))
	diff_expr = _constraint_to_diff(constraint)

	if isa(ref_vars, Symbol)
		expression_call = Expr(:macrocall, jump_expression, __source__, :_m, diff_expr)
		macrocall = quote
			let _m = $model_expr
				_func = $expression_call
				_test_constraint = $test_constraint_ref(
					$ref_vars,
					$equation_ref(_func, $set_ref(0.0)),
					String($message),
					$atol,
					$rtol,
				)
				[_test_constraint]
			end
		end
	elseif isexpr(ref_vars, :ref) || isexpr(ref_vars, :typed_vcat)
		expr_code = _named_index_expression_code(
			ref_vars, jump_expression, __source__, :_exprs => diff_expr,
		)
		macrocall = quote
			let _m = $model_expr
				$expr_code
				_ks = $all_keys_ref(_exprs)
				[$test_constraint_ref(
					$index_var_ref($base_sym, k),
					$equation_ref(_exprs[k...], $set_ref(0.0)),
					String($message),
					$atol,
					$rtol,
				) for k in _ks]
			end
		end
	else
		_error("Reference must be a variable")
	end
	return esc(macrocall)
end

"""
    @test_constraint(; atol=nothing, rtol=nothing)
    variable, constraint

    @test_constraint(message; atol=nothing, rtol=nothing)
    variable, constraint

Mark the next `variable, constraint` entry as a test constraint. The macro call
must be a separate statement directly before the entry. The test constraint does
not add an endogenous variable, a residual variable, or a solve constraint. The
constraint can use `==`, `<=`, or `>=`; Unicode `≤` and `≥` also work. The
optional string appears in [`TestConstraintError`](@ref) output. The optional
`atol` and `rtol` keywords override the matching tolerance passed to
[`solve`](@ref), [`solve!`](@ref), or [`assert_test_constraints`](@ref) for this
test constraint.

`@test_constraint` is valid only in an `@block` body. `solve` and `solve!` run
all test constraints after a successful solve. Use
[`assert_test_constraints`](@ref) to test a loaded or edited
[`ModelDictionary`](@ref) without a solve.

# Example
```julia
block = @block model begin
    a, a == b + c
    @test_constraint("a aggregation"; atol=1e-8)
    a, a == sum(a_i)
end
```
"""
macro test_constraint(args...)
	error("@test_constraint is valid only in an @block body")
end

"""
    @block model begin ... end

Create a `Block` of equations mapped to their endogenous variables.

Each standard line in the block body specifies a variable (or indexed variable)
followed by its defining equation. Equations are stored as lightweight `Equation`
objects (expression + set) without registering JuMP constraints. A standalone
`@test_constraint` call marks the next entry as a test constraint that does not
enter the square solve system.

SquareModels adds the residual at the first stored occurrence of the mapped
variable. It adds the residual only once. If the variable is absent, it adds the
unscaled residual to the right-hand side.

# Arguments
- `model`: The JuMP model (or ModelDictionary) containing the variables
- `begin ... end`: A block where each entry is `variable, equation_expr`. Put
  `@test_constraint([message]; atol, rtol)` on the prior line to make the entry a
  test constraint.

# Returns
A `Block` containing the equation-to-variable mappings.

# Examples
```julia
model = Model()
@variable(model, p)
@variable(model, w[1:3])
@variable(model, L[1:3])
@variable(model, ρ[1:3])
@variable(model, N[1:3])

# Define a block with scalar and indexed constraints
my_block = @block model begin
    p, p == 1
    w[j ∈ 1:3], L[j] == ρ[j] * N[j]
end

# Check block properties
length(my_block)  # 4
p ∈ my_block      # true
w[1] ∈ my_block   # true
```

```julia
# Multi-dimensional indexing
@variable(model, z[1:2, [:a, :b]])

b = @block model begin
    z[i ∈ 1:2, j ∈ [:a, :b]], z[i,j] == i
end
```

```julia
# Use a variable's sparse keys and add the standard JuMP filter after `;`.
active = Set([(1, :a), (2, :a), (2, :b)])
@variables model begin
    z_sparse[i = 1:2, j = [:a, :b]; (i, j) in active]
end
b = @block model begin
    z_sparse[(i, j) in keys(z_sparse); i == 2], z_sparse[i, j] == i
end
```

See also: [`Block`](@ref), [`@endo_exo_swap!`](@ref), [`endogenous`](@ref), [`variables`](@ref)
"""
macro block(model, expr)
	_error(line_number, it, msg) = error(
		"Invalid @block expression at $(line_number.file):$(line_number.line): $msg. Got $(sprint(show, it)).",
	)
	_is_equality(it) = isexpr(it, :call) && length(it.args) == 3 && it.args[1] == :(==)
	_is_test_relation(it) = isexpr(it, :call) && length(it.args) == 3 && it.args[1] in (:(==), :(<=), :≤, :(>=), :≥)
	_is_continuation(it) = isexpr(it, :call) && length(it.args) == 2 && it.args[1] in (:+, :-)
	_macro_name(name::Symbol) = name
	_macro_name(name::Expr) = name.head == :. && last(name.args) isa QuoteNode ? last(name.args).value : nothing
	_is_test_constraint(it) = isexpr(it, :macrocall) && _macro_name(it.args[1]) == Symbol("@test_constraint")
	sm = @__MODULE__
	block_macro_ref = GlobalRef(sm, Symbol("@_block"))
	test_constraint_macro_ref = GlobalRef(sm, Symbol("@_test_constraint"))
	get_model_ref = GlobalRef(sm, :_get_model)
	equation_ref = GlobalRef(sm, :Equation)
	test_constraint_ref = GlobalRef(sm, :TestConstraint)
	collect_variables_ref = GlobalRef(sm, :collect_variables!)
	residuals_ref = GlobalRef(sm, :residuals)
	line_number = expr.args[1]
	@assert isa(line_number, LineNumberNode)
	block_items = Tuple{LineNumberNode,Expr}[]
	test_constraint_items = Tuple{LineNumberNode,Expr,Any,Any,Any}[]
	last_tuple = nothing
	pending_test_constraint = nothing
	for it in expr.args
	    if isa(it, LineNumberNode)
	        line_number = it
	    elseif isexpr(it, :tuple) # line with commas
	        length(it.args) == 2 || _error(line_number, it, "Each line must be `variable, equation`")
	        if pending_test_constraint === nothing
	            _is_equality(it.args[2]) || _error(line_number, it, "The equation must use `==`")
	            push!(block_items, (line_number, it))
	        else
	            macro_line, _, message, atol, rtol = pending_test_constraint
	            _is_test_relation(it.args[2]) || _error(line_number, it, "The test constraint must use `==`, `<=`, or `>=`")
	            push!(test_constraint_items, (macro_line, it, message, atol, rtol))
	            pending_test_constraint = nothing
	        end
	        last_tuple = it
	    elseif _is_test_constraint(it)
	        pending_test_constraint === nothing ||
	            _error(line_number, it, "Each `@test_constraint` call must be followed by one `variable, equation` entry")
	        args = Any[it.args[3:end]...]
	        parameters = !isempty(args) && isexpr(first(args), :parameters) ? popfirst!(args) : Expr(:parameters)
	        if length(args) in (1, 2) && isexpr(last(args), :tuple)
	            isempty(parameters.args) ||
	                _error(line_number, it, "Put `@test_constraint(message; atol, rtol)` on its own line to use keywords")
	            message = length(args) == 2 ? first(args) : ""
	            tuple = last(args)
	            _is_test_relation(tuple.args[2]) || _error(line_number, it, "The test constraint must use `==`, `<=`, or `>=`")
	            push!(test_constraint_items, (line_number, tuple, message, nothing, nothing))
	            last_tuple = tuple
	        else
	            length(args) in (0, 1) ||
	                _error(line_number, it, "Put `@test_constraint([message]; atol, rtol)` on its own line before `variable, equation`")
	            message = isempty(args) ? "" : only(args)
	            options = Dict{Symbol,Any}(:atol => nothing, :rtol => nothing)
	            seen_options = Set{Symbol}()
	            for option in parameters.args
	                isexpr(option, :kw) && option.args[1] in keys(options) ||
	                    _error(line_number, it, "Test constraint keywords must be `atol` or `rtol`")
	                option.args[1] in seen_options &&
	                    _error(line_number, it, "Test constraint keyword `$(option.args[1])` occurs more than once")
	                push!(seen_options, option.args[1])
	                options[option.args[1]] = option.args[2]
	            end
	            pending_test_constraint = (line_number, it, message, options[:atol], options[:rtol])
	            last_tuple = nothing
	        end
	    elseif _is_continuation(it) && last_tuple !== nothing
	        eq = last_tuple.args[2]
	        eq.args[3] = Expr(:call, it.args[1], eq.args[3], it.args[2])
	    else
	        pending_test_constraint === nothing ||
	            _error(line_number, it, "Each `@test_constraint` call must be followed by one `variable, equation` entry")
	        _error(line_number, it, "Unexpected code in block body")
	    end
	end
	pending_test_constraint === nothing ||
	    _error(pending_test_constraint[1], pending_test_constraint[2], "Each `@test_constraint` call must be followed by one `variable, equation` entry")
	code = Expr(:tuple)
	for (line_number, it) in block_items
	    macro_call = Expr(
	        :macrocall,
	        block_macro_ref,
	        line_number,
	        model,
	        it.args...,
	    )
	    push!(code.args, esc(macro_call))
	end
	test_constraint_code = Expr(:tuple)
	for (line_number, it, message, atol, rtol) in test_constraint_items
	    macro_call = Expr(
	        :macrocall,
	        test_constraint_macro_ref,
	        line_number,
	        model,
	        it.args...,
	        message,
	        atol,
	        rtol,
	    )
	    push!(test_constraint_code.args, esc(macro_call))
	end
	quote
	    _container = $(esc(model))
	    _model = $get_model_ref(_container)
	    results = [$code...]
	    endogenous = Iterators.flatten([r[1] for r in results])
	    residuals = Iterators.flatten([r[2] for r in results])
	    eqs = Iterators.flatten([r[3] for r in results])
	    endo_vec = VariableRef[endogenous...]
	    res_vec = VariableRef[residuals...]
	    eqs_vec = $equation_ref[eqs...]
	    test_constraint_results = [$test_constraint_code...]
	    test_constraints_vec = $test_constraint_ref[Iterators.flatten(test_constraint_results)...]
	    all_vars = Set{VariableRef}()
	    for eq in eqs_vec
	        $collect_variables_ref(all_vars, eq)
	    end
	    _block = Block(_model, endo_vec, res_vec, all_vars, eqs_vec, test_constraints_vec)
	    if _container isa ModelDictionary
	        _container[$residuals_ref(_block)] .= 0.0
	    end
	    _block
	end
end

"""Split full name JuMP variable into base name and indices"""
function split_name(var::AbstractVariableRef)
	parts = split(string(var), "["; limit=2)
	if length(parts) == 1
		return parts[1], ""  # Scalar variable, no indices
	end
	return parts[1], "[" * parts[2]
end

"""Return base name of JuMP variable"""
base_name(var::AbstractVariableRef) = split_name(var)[1]
base_name(var::AbstractArray{T}) where {T<:AbstractVariableRef} = base_name(first(var))

"""
    residual(var)

Return the residual variable or residual container corresponding to an endogenous
variable or variable container.
"""
residual(var) = first(var).model[Symbol(make_residual_name(base_name(var)))]
residual(var::AbstractVariableRef) = var.model[Symbol(make_residual_name(base_name(var)))]

"""
If a variable Symbol(new_name) does not exist, define a new variable with the same indices as an existing variable.
"""
function copy_variable(new_name::String, original::SparseAxisArray)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    d = Dict(k => VariableRef(m) for k in keys(original.data))
	    new = SparseAxisArray(d)
	    for k in keys(original.data)
	        set_name(new[k...], new_name * split_name(original[k...])[2])
	    end
	    m[sym] = new
	    fix.(new, 0)
	end
	return m[sym]
end
function copy_variable(new_name::String, original::AbstractArray)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    data = reshape([VariableRef(m) for _ in _all_keys(original)], length.(axes(original))...)
	    new = DenseAxisArray(data, axes(original)...)
	    for (x, y) in zip(new, original)
	        set_name(x, new_name * split_name(y)[2])
	    end
	    m[sym] = new
	    fix.(new, 0)
	end
	return m[sym]
end
function copy_variable(new_name::String, original::AbstractVariableRef)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    new = VariableRef(m)
	    set_name(new, new_name)
	    m[sym] = new
	    fix(new, 0)
	end
	return m[sym]
end
copy_variable(new_name::String, original::SparseZeroArray) = copy_variable(new_name, original.data)

base_name(var::SparseZeroArray) = base_name(first(var))

"""
    unfix(b::Block)

Unfix all endogenous variables in a block.

Iterates through all variables in the block and unfixes any that are currently fixed.
Variables that are already unfixed are skipped.

# Arguments
- `b::Block`: The block whose variables should be unfixed

# Returns
`nothing`

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

fix.(b, 1.0)      # Fix all variables in block to 1.0
is_fixed(x)       # true
unfix(b)          # Unfix all variables
is_fixed(x)       # false
```

See also: [`Block`](@ref), [`@endo_exo_swap!`](@ref)
"""
function JuMP.unfix(b::Block)
	for var in b
	    if is_fixed(var)
	        unfix(var)
	    end
	end
	return nothing
end

include("endo_exo_swap.jl")
include("tagged_variables.jl")
include("ModelDictionaries.jl")
include("solve.jl")
include("ModelExpressions.jl")
include("ModelPlotting.jl")
using .ModelExpressions: @evalexpr, @prt, LabeledArray, MultiVarResult, set_default_source!, set_default_operator!, set_default_periods!, set_column_label_total_width!, reset_print_defaults!
using .ModelPlotting: @plot, plotvar, plotseries, alternating_dash!, labeled, LabeledSeries, set_plot_finalize!, reset_plot_finalize!, plot_finalize

# Define _get_model for ModelDictionary (after ModelDictionaries.jl is included)
_get_model(md::ModelDictionary) = md.model

end # Module
