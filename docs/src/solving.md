# Solving

SquareModels solves a [`Block`](@ref) by building an intermediate JuMP model with
the block's endogenous variables. Exogenous variables are substituted from a
[`ModelDictionary`](@ref), the intermediate model is optimized, and the solution
is written back to a dictionary.

Use [`solve`](@ref) when you want a new dictionary:

```@example solving
import JuMP
using JuMP: set_silent
using Ipopt
using SquareModels

model = square_model(Ipopt.Optimizer)
set_silent(model)

@variables model begin
    x, "First endogenous variable"
    y, "Second endogenous variable"
    a, "Exogenous shift"
end

data = ModelDictionary(model)
data[x] = 1.0
data[y] = 1.0
data[a] = 3.0

block = @block data begin
    x, x == 2
    y, y == a + x
end

solution = solve(block, data)
round(solution[y], digits=4)
```

Use [`solve!`](@ref) when you want to update the existing dictionary in place:

```@example solving
data[a] = 8.0
solve!(block, data)
round(data[y], digits=4)
```

## Calibration

Calibration is the same solve pipeline with a different endogenous set. Start
from a copy of the behavioral block and use [`@endo_exo_swap!`](@ref):

```@example solving
calibration = copy(block)
@endo_exo_swap! calibration begin
    a, y
end

data[y] = 20.0
calibrated = solve(calibration, data)
round(calibrated[a], digits=4)
```

## Starting Values

`solve` and `solve!` take optional `start_values` and `replace_nothing` keywords:

```julia
solution = solve(block, data; start_values=baseline, replace_nothing=1.0)
```

`replace_nothing` is useful during early calibration when a variable exists in
the model but has no data value yet.

## Test Constraints

After a successful solve, `solve` and `solve!` run each `@test_constraint` entry
in the block. Test constraints can use `==`, `<=`, or `>=` and do not enter the
solve model. Equalities default to `atol=1e-6` and `rtol=1e-8`. Inequalities
default to zero for both tolerances. Set defaults for all constraint types with
`test_constraint_atol` and `test_constraint_rtol`:

```julia
solution = solve(block, data; test_constraint_atol=1e-8, test_constraint_rtol=1e-6)
```

Set a tolerance for one test in its macro call:

```julia
block = @block data begin
    @test_constraint("a aggregation"; atol=1e-10, rtol=1e-7)
    a[t ∈ periods], a[t] == ∑(a_i[i, t] for i ∈ industries)
end
```

The macro `atol` and `rtol` values replace the matching solve default for this
test. You can set one or both. The test passes when its gap is at most
`max(atol, rtol * abs(data[a[t]]))`. The mapped variable sets the relative
tolerance scale.

Set `run_test_constraints=false` for a partial solve that does not yet meet all
test constraints. If one fails after `solve!`, the dictionary keeps the solved
values. If one fails after `solve`, the `TestConstraintError.data` field holds
the solved copy. Use [`assert_test_constraints`](@ref) to test a loaded or edited
dictionary without another solve.

## Choosing a Solver

[`square_model`](@ref) constructs a JuMP model configured as a square nonlinear
system (feasibility sense, no objective). Pass any JuMP optimizer, or use the
`gamsdir` keyword to solve through GAMS as a constrained nonlinear system (CNS):

```julia
model = square_model(Ipopt.Optimizer; tol=1e-10)

using GAMS  # loads the optional GAMS extension
model = square_model(; gamsdir="C:/GAMS/53", solver="CONOPT4")
```

With the GAMS backend, `solve!` annotates the generated `moi.lst` listing with
the model's variable and equation names (see [`annotate_lst!`](@ref)), which
makes GAMS solver output readable for debugging.

## Diagnostics

Before solving, SquareModels checks whether substituting exogenous data makes the
system effectively non-square. For example, equations that collapse to constants
or endogenous variables that disappear from every equation are reported before
the optimizer runs.

Call [`diagnose`](@ref) directly when you want to inspect those structural issues:

```julia
trivial, orphans = diagnose(block, data)
```

Set `presolve_diagnostics=false` only when you have already validated the block
and need to avoid the extra pre-solve pass.
