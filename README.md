# SquareModels

[![CI](https://github.com/MartinBonde/SquareModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MartinBonde/SquareModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MartinBonde/SquareModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MartinBonde/SquareModels.jl)

A JuMP extension for writing modular models with **square systems of equations** — systems where the number of constraints equals the number of endogenous variables.

**Documentation: [martinbonde.github.io/SquareModels.jl](https://martinbonde.github.io/SquareModels.jl)**

## Installation

Install SquareModels together with a solver such as Ipopt:

```julia
using Pkg
Pkg.add(["SquareModels", "Ipopt"])
```

Solvers are not included with SquareModels; use any solver supported by the
workflow you choose.

## Motivation

Large-scale macroeconomic models are typically "square" — each equation determines one endogenous variable. This package provides tools to:

- **Map constraints to endogenous variables** — Each constraint is explicitly paired with the variable it determines
- **Build models modularly** — Define separate `Block`s of equations that can be combined
- **Swap endogenous/exogenous variables** — Use `@endo_exo_swap!` to change which variables are endogenous for calibration or counterfactual scenarios
- **Test redundant constraints** — Add `@test_constraint` entries without adding solve constraints
- **Inspect results** — Print, evaluate, and plot model data with `@prt`, `@evalexpr`, and `@plot`

## Quick Example

> 📄 Full runnable version: [`examples/quick_example.jl`](examples/quick_example.jl)

```julia
import JuMP
using JuMP: Model, set_silent
using Ipopt, SquareModels

data = ModelDictionary(Model(Ipopt.Optimizer))
set_silent(data.model)

j = 1:2  # Types of labor

@variables data.model begin
    L[j], "Labor demand"
    w[j], "Wage"
    Y, "Output"
    Cj[j], "Consumption by labor type"
    C, "Consumption"
    p, "Price"

    N[j], "Labor force (exogenous)"
    ρ[j], "Productivity (calibrated)"
    μ[j], "Scale parameter (calibrated)"
    σ, "Substitution elasticity (exogenous)"
end

# Define a Block: each line pairs an endogenous variable with its equation
model_block = @block data begin
    L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y   # Labor demand
    w[j ∈ j], L[j] == ρ[j] * N[j]                 # Labor market clearing
    Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)   # Zero profit
    Cj[j ∈ j], Cj[j] == w[j] * ρ[j] * N[j] / p    # Consumption by labor type
    C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p  # Budget constraint
    @test_constraint "Consumption aggregation" C, C == ∑(Cj[j] for j ∈ j)
    p,        p == 1                               # Numeraire
end

# Set data values
data[σ] = 2.0
data[w] .= 1
data[N] = [3200, 500]
data[L] = [800, 200]

# Calibration: swap observed values with parameters to be calibrated
calibration = copy(model_block)
@endo_exo_swap! calibration begin
    μ, L
    ρ, w
end

baseline = solve(calibration, data; replace_nothing=1.0)

# Counterfactual: shock exogenous variables
scenario = copy(baseline)
scenario[N] .= [2700, 1000]  # Population shock
solve!(model_block, scenario)

println("Multipliers: ", scenario ./ baseline .- 1)
```

## Documentation

The [documentation pages](https://martinbonde.github.io/SquareModels.jl) cover:

- **Getting Started** — a compact calibration and scenario workflow
- **Core Concepts** — blocks, endo-exo swapping, model dictionaries, data I/O, variable metadata, sparse arrays
- **Solving** — `solve`, `solve!`, calibration, solver choice, and diagnostics
- **Plotting and Printing** — `plotvar`, `@plot`, `@prt`, and expression evaluation
- **Modular Models** — organizing larger models into Julia modules
- **Optimization** — embedding square systems in JuMP optimization problems
- **Examples** — the runnable scripts in [`examples/`](examples/)
- **API Reference**

## Optional Extensions

Loaded automatically via Julia's package extension system when the corresponding package is present:

- **Makie** — plotting support for model data (`plotvar`, `@plot`, ...); add any Makie backend (e.g. `CairoMakie`)
- **GAMS** — solve square systems with GAMS CNS solvers (e.g. CONOPT) via `square_model(gamsdir=...)`
- **GDXInterface** — load data from GAMS GDX files into a `ModelDictionary`

## License

This project is licensed under an MIT license — see [LICENSE](LICENSE.md) for details.

## Acknowledgments

This work is part of the [DREAM](https://dreamgruppen.dk/) group's effort to modernize economic modeling tools in Denmark and the rest of the world.
