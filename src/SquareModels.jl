# Copyright 2022, Martin Kirk Bonde and contributors
# Licensed under the MIT License. See LICENSE.md for details.

"""
SquareModels
A JuMP extension for writing modular models with square systems of equations
"""
module SquareModels

export @block, Block, @endo_exo!, delete!
export constraints, variables, residuals, overlaps, shared_variables
export ModelDictionary, fix, unfix, set_start_value, value, value_dict
export save, load
export RESIDUAL_SUFFIX

# Constraints are named after the associated endogenous variable
# A prefix is added as a constraint and variable cannot share the same name
CONSTRAINT_PREFIX = "E_"
RESIDUAL_SUFFIX = "_J"  # Suffix for residual variables (J for "junk" or adjustment)

# ----------------------------------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------------------------------
using Base.Meta: isexpr
using StatsBase: countmap
using Lazy
using JuMP: JuMP, AbstractModel, AbstractVariableRef, VariableRef, ConstraintRef, Containers
using JuMP.Containers: DenseAxisArray
using JuMP: @variable, @constraint
using JuMP: set_name, name, variable_by_name, fix, is_fixed, unfix, all_variables, value, set_start_value

include("utils.jl")

"""
    Block

A mapping between constraints and their associated endogenous variables in a JuMP model.

Blocks represent "square" systems where each constraint is paired with exactly one
variable, enabling modular model construction and endo-exo swaps (changing which
variable is determined by which equation).

Internally, constraints and variables are stored in parallel vectors where the same
index corresponds to a constraint-variable pair. A Set is maintained for O(1) membership
checks.

# Fields
- `model::AbstractModel`: The JuMP model containing the constraints and variables
- `constraints::Vector{ConstraintRef}`: Vector of constraint references
- `variables::Vector{VariableRef}`: Vector of endogenous variable references (parallel to constraints)
- `_variable_set::Set{VariableRef}`: Set for O(1) membership checking

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

See also: [`@block`](@ref), [`@endo_exo!`](@ref), [`constraints`](@ref), [`variables`](@ref)
"""
struct Block
	model::AbstractModel
	constraints::Vector{ConstraintRef}
	variables::Vector{VariableRef}
	residuals::Vector{VariableRef}
	_variable_set::Set{VariableRef}

	function Block(model::AbstractModel, constraints::Vector{ConstraintRef}, variables::Vector{VariableRef}, residuals::Vector{VariableRef})
		# Validate square
		length(constraints) == length(variables) ||
			error("Block must be square: got $(length(constraints)) constraints and $(length(variables)) variables")

		# Validate unique variables
		variable_set = Set{VariableRef}(variables)
		if length(variable_set) != length(variables)
			display(non_unqiue_pairs(variables, constraints))
			error("Non-unique mapping between variables and constraints in block definition.\n" *
			      "See non-unique mappings above.")
		end

		new(model, constraints, variables, residuals, variable_set)
	end
end

Block(model) = Block(model, ConstraintRef[], VariableRef[], VariableRef[])

Base.length(b::Block) = length(b.variables)
Base.iterate(b::Block) = iterate(b.variables)
Base.iterate(b::Block, state) = iterate(b.variables, state)
Base.in(var::VariableRef, b::Block) = var ∈ b._variable_set
Base.copy(b::Block) = Block(b.model, copy(b.constraints), copy(b.variables), copy(b.residuals))

function Base.getindex(b::Block, var::VariableRef)
	idx = findfirst(==(var), b.variables)
	isnothing(idx) && throw(KeyError(var))
	return b.constraints[idx]
end

function Base.getindex(b::Block, c::ConstraintRef)
	idx = findfirst(==(c), b.constraints)
	isnothing(idx) && throw(KeyError(c))
	return b.variables[idx]
end

"""
    constraints(b::Block) → Vector{ConstraintRef}

Return the vector of constraint references in the block.

# Arguments
- `b::Block`: The block to get constraints from

# Returns
A vector of `ConstraintRef` objects representing all constraints in the block.
The order corresponds to the order of `variables(b)`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

for c in constraints(b)
    println(name(c))
end
```

See also: [`variables`](@ref)
"""
constraints(b::Block) = b.constraints

"""
    variables(b::Block) → Vector{VariableRef}

Return the vector of endogenous variable references in the block.

# Arguments
- `b::Block`: The block to get variables from

# Returns
A vector of `VariableRef` objects representing all endogenous variables in the block.
The order corresponds to the order of `constraints(b)`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

for v in variables(b)
    println(name(v))
end
```

See also: [`constraints`](@ref), [`residuals`](@ref)
"""
variables(b::Block) = b.variables

"""
    residuals(b::Block) → Vector{VariableRef}

Return the residual variables corresponding to each endogenous variables in the block.

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
The order corresponds to `variables(b)`.

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

See also: [`variables`](@ref), [`constraints`](@ref)
"""
residuals(b::Block) = b.residuals

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
end

"""
    overlaps(a::Block, b::Block) → Bool

Check if two blocks share any common variables.

Returns `true` if any variable appears in both blocks, which may indicate
duplicate equations or intentional variable sharing across model components.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
`true` if the blocks have at least one variable in common, `false` otherwise

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

See also: [`shared_variables`](@ref), [`Block`](@ref)
"""
overlaps(a::Block, b::Block) = !isempty(intersect(a._variable_set, b._variable_set))

"""
    shared_variables(a::Block, b::Block) → Vector{VariableRef}

Return the variables that appear in both blocks.

Useful for understanding how blocks are interconnected and for detecting
accidental duplicate equations.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
A vector of `VariableRef` objects that appear in both blocks (may be empty)

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

shared = shared_variables(b1, b2)  # [y[2]]
y[2] ∈ shared  # true
y[1] ∈ shared  # false
```

See also: [`overlaps`](@ref), [`Block`](@ref)
"""
shared_variables(a::Block, b::Block) = collect(intersect(a._variable_set, b._variable_set))

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
	constraints::AbstractArray{C},
	variables::AbstractArray{V},
	residuals::AbstractArray{R}
) where {C<:ConstraintRef, V<:VariableRef, R<:VariableRef}
	Block(model, ConstraintRef[constraints...], VariableRef[variables...], VariableRef[residuals...])
end

function Base.:+(a::Block, b::Block)
	a.model == b.model || error("Cannot add $a and $b. Blocks must belong to the same model.")

	# Check for overlap before combining
	if overlaps(a, b)
		shared = shared_variables(a, b)
		formatted = format_variables(shared)
		error("Cannot combine blocks: $(length(shared)) variable(s) appear in both blocks.\n" *
		      "Overlapping variables:\n$formatted\n" *
		      "This would create a non-square system with more constraints than unique variables.")
	end

	Block(a.model, vcat(a.constraints, b.constraints), vcat(a.variables, b.variables), vcat(a.residuals, b.residuals))
end

function Base.:-(a::Block, b::Block)
	a.model == b.model || error("Cannot subtract $b from $a. Blocks must belong to the same model.")
	# Keep only pairs where variable is NOT in b
	mask = [v ∉ b._variable_set for v in a.variables]
	if !any(mask)
		return Block(a.model)
	end
	Block(a.model, a.constraints[mask], a.variables[mask], a.residuals[mask])
end

"""
    delete!(model::AbstractModel, block::Block)

Delete all constraints in a block from the model.

# Arguments
- `model::AbstractModel`: The JuMP model containing the constraints
- `block::Block`: The block whose constraints should be deleted

# Examples
```julia
model = Model()
@variable(model, x)

b = @block model begin
    x, x == 1
end

delete!(model, b)  # Removes the constraint from the model
```
"""
function Base.delete!(model::AbstractModel, block::Block)
  for c in constraints(block)
	JuMP.delete(model, c)
  end
end

"""Helper function for Block macro - returns (constraints, variables, residuals) aligned by constraint indices"""
pairconstraints2vars(constraints, var, residual) = (constraints, var[axes(constraints)...], residual[axes(constraints)...])

make_constraint_name(var) = SquareModels.CONSTRAINT_PREFIX * string(var)
make_residual_name(var) = string(var) * SquareModels.RESIDUAL_SUFFIX

"""Helper function to extract base name from variable reference"""
_get_name(s::Symbol) = s
_get_name(e::Expr) = e.args[1]

"""
Replace occurrences of `target` symbol (with optional indexing) with `(target + model[residual_sym][indices])`.
Handles both scalar references like `x` and indexed references like `x[i,j]`.
The `model_sym` is the symbol referring to the model, and `residual_sym` is the Symbol for the residual.
"""
function _substitute_with_residual(expr, target::Symbol, model_sym, residual_sym::Symbol)
	if expr isa Symbol
		if expr == target
			# Scalar variable: x -> (x + model[:x_J])
			return :($expr + $model_sym[$(QuoteNode(residual_sym))])
		end
		return expr
	elseif expr isa Expr
		if expr.head == :ref && expr.args[1] == target
			# Indexed variable: x[i,j] -> (x[i,j] + model[:x_J][i,j])
			indices = expr.args[2:end]
			residual_access = Expr(:ref, :($model_sym[$(QuoteNode(residual_sym))]), indices...)
			return :($expr + $residual_access)
		else
			# Recurse into sub-expressions
			new_args = [_substitute_with_residual(arg, target, model_sym, residual_sym) for arg in expr.args]
			return Expr(expr.head, new_args...)
		end
	else
		return expr
	end
end

"""Helper macro for Block macro - returns (constraints, variables, residuals)"""
macro _block(model, ref_vars, constraint, extra...)
	_error(str...) = JuMP._macro_error(:block, (model, ref_vars, constraint, extra...), __source__, str...)
	code = Expr(:block)
	base_sym = _get_name(ref_vars)
	constraint_name = make_constraint_name(base_sym)
	constraint_symbol = Symbol(constraint_name)
	residual_name = make_residual_name(base_sym)
	residual_symbol = Symbol(residual_name)

	push!(code.args, :(unregister($model, Symbol($constraint_name))))
	# Create residual variable with same shape as original variable (using copy_variable)
	push!(code.args, :(SquareModels.copy_variable($residual_name, $base_sym)))

	if isa(ref_vars, Symbol)
		# Scalar variable case
		# Transform constraint: replace endo with (endo + model[:endo_J])
		transformed_constraint = _substitute_with_residual(constraint, base_sym, model, residual_symbol)
		macrocall = :([@constraint($model, $constraint_symbol, $transformed_constraint, $(extra...))], [$ref_vars], [$model[$(QuoteNode(residual_symbol))]])
	elseif isexpr(ref_vars, :ref)
		index_vars, _ = Containers.build_ref_sets(error, ref_vars)
		indices = ref_vars.args[2:end]
		# Transform constraint: replace endo[i,j] with (endo[i,j] + model[:endo_J][i,j])
		transformed_constraint = _substitute_with_residual(constraint, base_sym, model, residual_symbol)
		macrocall = quote
			SquareModels.pairconstraints2vars(
				@constraint($model, $constraint_symbol[$(indices...)], $transformed_constraint, $(extra...)),
				$base_sym,
				$model[$(QuoteNode(residual_symbol))]
			)
		end
	else
		_error("Reference must be a variable")
	end
	push!(code.args, macrocall)
	return esc(code)
end

"""
    @block model begin ... end

Create a `Block` of constraints mapped to their endogenous variables.

Each line in the block body specifies a variable (or indexed variable) followed by
its defining equation. Constraints are automatically named with the prefix "E_"
followed by the variable name.

# Arguments
- `model`: The JuMP model to add constraints to
- `begin ... end`: A block where each line is `variable, constraint_expr`

# Returns
A `Block` containing the constraint-to-variable mappings.

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

See also: [`Block`](@ref), [`@endo_exo!`](@ref), [`constraints`](@ref), [`variables`](@ref)
"""
macro block(model, expr)
	line_number = expr.args[1]
	@assert isa(line_number, LineNumberNode)
	code = Expr(:tuple)
	for it in expr.args
	    if isa(it, LineNumberNode)
	        line_number = it
	    elseif isexpr(it, :tuple) # line with commas
	        macro_call = Expr(
	            :macrocall,
	            :(SquareModels.var"@_block"),
	            line_number,
	            model,
	            it.args...,
	        )
	        push!(code.args, esc(macro_call))
	    end
	end
	quote
	    constraints, variables, residuals = Iterators.flatten.([zip($code...)...])
	    Block(
	        $(esc(model)),
	        ConstraintRef[constraints...],
	        VariableRef[variables...],
	        VariableRef[residuals...],
	    )
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
If a variable Symbol(new_name) does not exist, define a new variable with the same indices as an existing variable.
"""
function copy_variable(new_name::String, original)
	m = first(original).model
	sym = Symbol(new_name)
	if sym ∉ keys(m.obj_dict)
	    new = DenseAxisArray([VariableRef(m) for _ in keys(original)], axes(original)...)
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
	if sym ∉ keys(m.obj_dict)
	    new = VariableRef(m)
	    set_name(new, new_name)
	    m[sym] = new
	    fix(new, 0)
	end
	return m[sym]
end

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

See also: [`Block`](@ref), [`@endo_exo!`](@ref)
"""
function JuMP.unfix(b::Block)
	for var in b
	    if is_fixed(var)
	        unfix(var)
	    end
	end
	return nothing
end

include("endo_exo.jl")
include("ModelDictionaries.jl")

end # Module
