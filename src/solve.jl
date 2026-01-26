# solve.jl - Functions for solving blocks

using JuMP: Model, VariableRef, ConstraintRef, AffExpr, QuadExpr, NonlinearExpr
using JuMP: @variable, @constraint, constraint_object, set_name, name
using JuMP: set_start_value, fix, has_lower_bound, has_upper_bound
using JuMP: lower_bound, upper_bound, set_lower_bound, set_upper_bound
using JuMP: all_variables, is_fixed, value, add_to_expression!
using JuMP: optimize!, set_silent, unsafe_backend

# ============================================================================
# Expression Transformation
# ============================================================================
# Transform JuMP expressions by:
# - Replacing endogenous VariableRefs with their solve model equivalents
# - Replacing exogenous VariableRefs with their data values (constants)

"""
    transform_expr(expr, var_map, data, endo_set) -> transformed_expr

Transform a JuMP expression by substituting variables.
- Endogenous variables are mapped to solve model variables via `var_map`
- Exogenous variables are replaced with their values from `data`
"""
function transform_expr end

# Passthrough for numbers
transform_expr(x::Number, var_map, data, endo_set) = x

# Transform a single VariableRef
function transform_expr(var::VariableRef, var_map, data, endo_set)
    # Check if it's an endogenous variable or residual (both are in var_map)
    if haskey(var_map, var)
        return var_map[var]
    else
        # Exogenous variable: substitute with data value
        val = data[var]
        val === nothing && error("No data value for exogenous variable $(name(var))")
        return val
    end
end

# Transform AffExpr (linear): constant + Σ(coef * var)
function transform_expr(expr::AffExpr, var_map, data, endo_set)
    new_expr = AffExpr(expr.constant)
    for (var, coef) in expr.terms
        if haskey(var_map, var)
            # Endogenous variable or residual: map to solve model variable
            add_to_expression!(new_expr, coef, var_map[var])
        else
            # Exogenous: substitute with data value
            val = data[var]
            val === nothing && error("No data value for exogenous variable $(name(var))")
            add_to_expression!(new_expr, coef * val)
        end
    end
    return new_expr
end

# Transform QuadExpr (quadratic): aff + Σ(coef * var_a * var_b)
function transform_expr(expr::QuadExpr, var_map, data, endo_set)
    # Transform the affine part
    new_aff = transform_expr(expr.aff, var_map, data, endo_set)
    new_expr = QuadExpr(new_aff)

    for (pair, coef) in expr.terms
        var_a, var_b = pair.a, pair.b

        # Get value or mapped variable for each
        val_a = if haskey(var_map, var_a)
            var_map[var_a]
        else
            v = data[var_a]
            v === nothing && error("No data value for exogenous variable $(name(var_a))")
            v
        end

        val_b = if haskey(var_map, var_b)
            var_map[var_b]
        else
            v = data[var_b]
            v === nothing && error("No data value for exogenous variable $(name(var_b))")
            v
        end

        if val_a isa Number && val_b isa Number
            # Both exogenous: becomes constant
            add_to_expression!(new_expr.aff, coef * val_a * val_b)
        elseif val_a isa Number
            # var_a exogenous: becomes linear term
            add_to_expression!(new_expr.aff, coef * val_a, val_b)
        elseif val_b isa Number
            # var_b exogenous: becomes linear term
            add_to_expression!(new_expr.aff, coef * val_b, val_a)
        else
            # Both endogenous: stays quadratic
            add_to_expression!(new_expr, coef, val_a, val_b)
        end
    end
    return new_expr
end

# Transform NonlinearExpr (tree structure with head and args)
function transform_expr(expr::NonlinearExpr, var_map, data, endo_set)
    new_args = Any[]
    for arg in expr.args
        if arg isa VariableRef
            if haskey(var_map, arg)
                push!(new_args, var_map[arg])
            else
                val = data[arg]
                val === nothing && error("No data value for exogenous variable $(name(arg))")
                push!(new_args, val)
            end
        elseif arg isa Union{AffExpr, QuadExpr, NonlinearExpr}
            push!(new_args, transform_expr(arg, var_map, data, endo_set))
        else
            push!(new_args, arg)  # Numbers, symbols, etc.
        end
    end
    # Use splatting to pass args correctly to NonlinearExpr constructor
    return NonlinearExpr(expr.head, Any[new_args...])
end

# ============================================================================
# _build_model (internal)
# ============================================================================

# Internal function to build a solve model from a block
function _build_model(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing
)
    # Get optimizer from the block's model
    optimizer = typeof(unsafe_backend(block.model))
    solve_model = Model(optimizer)
    endo_set = block._endogenous_set

    # Create endogenous variables in solve model
    var_map = sizehint!(Dict{VariableRef, VariableRef}(), length(block.endogenous))
    for endo_var in block.endogenous
        new_var = @variable(solve_model)
        set_name(new_var, name(endo_var))
        var_map[endo_var] = new_var

        # Transfer bounds from original model
        if has_lower_bound(endo_var)
            set_lower_bound(new_var, lower_bound(endo_var))
        end
        if has_upper_bound(endo_var)
            set_upper_bound(new_var, upper_bound(endo_var))
        end

        # Set start value (start_values takes precedence over data)
        start_val = nothing
        if start_values !== nothing
            try
                start_val = start_values[endo_var]
            catch
            end
        end
        if start_val === nothing
            try
                start_val = data[endo_var]
            catch
            end
        end
        if start_val !== nothing && !isnan(start_val)
            set_start_value(new_var, start_val)
        end
    end

    # Transform and add constraints (residuals not in endogenous set get substituted from data)
    for con_ref in block.constraints
        con_obj = constraint_object(con_ref)
        new_func = transform_expr(con_obj.func, var_map, data, endo_set)
        @constraint(solve_model, new_func in con_obj.set)
    end

    return solve_model, var_map
end

# ============================================================================
# solve
# ============================================================================

"""
    solve(block::Block, data::ModelDictionary; start_values=nothing, silent=true)

Build, optimize, and extract solution in one step.

Uses the optimizer from the block's model. Creates an intermediate solve model with only
endogenous variables, optimizes it, and returns a new ModelDictionary with the solution.

# Arguments
- `block::Block`: Block defined on a model with an optimizer set
- `data::ModelDictionary`: Data dictionary with values for all variables
- `start_values::Union{Nothing, ModelDictionary}`: Optional starting values (overrides `data`)
- `silent::Bool`: Whether to suppress solver output (default: true)

# Returns
A new `ModelDictionary` containing the solution values for endogenous variables,
with exogenous values copied from `data`.

# Example
```julia
using Ipopt
model = Model(Ipopt.Optimizer)
@variables model begin
    x
    y
end
data = ModelDictionary(model)
data[x] = 1.0
data[y] = 2.0

block = @block model begin
    x, x == 10
    y, y == 20
end
data[residuals(block)] .= 0.0

solution = solve(block, data)
solution[x]  # 10.0
solution[y]  # 20.0
```
"""
function solve(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing,
    silent::Bool = true
)
    result = copy(data)
    solve!(block, result; start_values, silent)
    return result
end

"""
    solve!(block::Block, data::ModelDictionary; start_values=nothing, silent=true)

Build, optimize, and update data in-place.

Like `solve`, but mutates `data` instead of returning a new ModelDictionary.

# Arguments
- `block::Block`: Block defined on a model with an optimizer set
- `data::ModelDictionary`: Data dictionary to update with solution values
- `start_values::Union{Nothing, ModelDictionary}`: Optional starting values (overrides `data`)
- `silent::Bool`: Whether to suppress solver output (default: true)

# Returns
The mutated `data` ModelDictionary.

# Example
```julia
solve!(block, data)  # data is updated in-place
```
"""
function solve!(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing,
    silent::Bool = true
)
    model, var_map = _build_model(block, data; start_values)
    silent && set_silent(model)
    optimize!(model)

    for (original_var, solve_var) in var_map
        data[original_var] = value(solve_var)
    end
    return data
end
