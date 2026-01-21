# ----------------------------------------------------------------------------------------------------------------------
#  Copyright 2022, Martin Kirk Bonde and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ----------------------------------------------------------------------------------------------------------------------

"""
SquareModels
A JuMP extension for writing modular models with square systems of equations
"""
module SquareModels

export @block, Block, @endo_exo!, delete!
export constraints, variables, overlaps, shared_variables
export ModelDictionary, fix, unfix, set_start_value, value, value_dict
export save, load

# Constraints are named after the associated endogenous variable
# A prefix is added as a constraint and variable cannot share the same name
CONSTRAINT_PREFIX = "E_"

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
	_variable_set::Set{VariableRef}

	function Block(model::AbstractModel, constraints::Vector{ConstraintRef}, variables::Vector{VariableRef})
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

		new(model, constraints, variables, variable_set)
	end
end

Block(model) = Block(model, ConstraintRef[], VariableRef[])

Base.length(b::Block) = length(b.variables)
Base.iterate(b::Block) = iterate(b.variables)
Base.iterate(b::Block, state) = iterate(b.variables, state)
Base.in(var::VariableRef, b::Block) = var ∈ b._variable_set
Base.copy(b::Block) = Block(b.model, copy(b.constraints), copy(b.variables))

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

See also: [`constraints`](@ref)
"""
variables(b::Block) = b.variables

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
	variables::AbstractArray{V}
) where {C<:ConstraintRef, V<:VariableRef}
	Block(model, ConstraintRef[constraints...], VariableRef[variables...])
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

	Block(a.model, vcat(a.constraints, b.constraints), vcat(a.variables, b.variables))
end

function Base.:-(a::Block, b::Block)
	a.model == b.model || error("Cannot subtract $b from $a. Blocks must belong to the same model.")
	# Keep only pairs where variable is NOT in b
	mask = [v ∉ b._variable_set for v in a.variables]
	if !any(mask)
		return Block(a.model)
	end
	Block(a.model, a.constraints[mask], a.variables[mask])
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

"""Helper function for Block macro"""
pairconstraints2vars(constraints, var) = (constraints, var[axes(constraints)...])

make_constraint_name(var) = SquareModels.CONSTRAINT_PREFIX * string(var)

"""Helper function to extract base name from variable reference"""
_get_name(s::Symbol) = s
_get_name(e::Expr) = e.args[1]

"""Helper macro for Block macro"""
macro _block(model, ref_vars, constraint, extra...)
	_error(str...) = JuMP._macro_error(:block, (model, ref_vars, constraint, extra...), __source__, str...)
	code = Expr(:block)
	base_sym = _get_name(ref_vars)
	constraint_name = make_constraint_name(base_sym)
	constraint_symbol = Symbol(constraint_name)
	push!(code.args, :(unregister($model, Symbol($constraint_name))))
	if isa(ref_vars, Symbol)
	    macrocall = :([@constraint($model, $constraint_symbol, $constraint, $(extra...))], [$ref_vars])
	elseif isexpr(ref_vars, :ref)
	    index_vars, _ = Containers.build_ref_sets(error, ref_vars)
	    macrocall = quote
	        SquareModels.pairconstraints2vars(
	            @constraint($model, $constraint_symbol[$(ref_vars.args[2:end]...)], $constraint, $(extra...)),
	            $base_sym
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
	    constraints, variables = Iterators.flatten.([zip($code...)...])
	    Block(
	        $(esc(model)),
	        ConstraintRef[constraints...],
	        VariableRef[variables...],
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
