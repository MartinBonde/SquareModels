# SquareModels

A JuMP extension for writing modular models with **square systems of equations** — systems where the number of constraints equals the number of endogenous variables.

## Motivation
Large-scale macroeconomic models are typically "square" — each equation determines one endogenous variable. This package provides tools to:

- **Map constraints to endogenous variables** — Each constraint is explicitly paired with the variable it determines
- **Build models modularly** — Define separate `Block`s of equations that can be combined
- **Swap endogenous/exogenous variables** — Use `@endo_exo!` to change which variables are endogenous for calibration or counterfactual scenarios

## Quick Example

> 📄 Full runnable version: [`examples/quick_example.jl`](examples/quick_example.jl)

```julia
using JuMP, Ipopt, SquareModels

model = Model(Ipopt.Optimizer)

j = 1:2  # Types of labor

@variables model begin
    L[j]  # Labor demand
    w[j]  # Wage
    Y     # Output
    C     # Consumption
    p     # Price

    N[j]  # Labor force (exogenous)
    ρ[j]  # Productivity (exogenous)
    μ[j]  # Scale parameter (calibrated)
    σ     # Substitution elasticity (exogenous)
end

# Define a Block: each line pairs an endogenous variable with its equation
model_block = @block model begin
    L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y   # Labor demand
    w[j ∈ j], L[j] == ρ[j] * N[j]                 # Labor market clearing
    Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)   # Zero profit
    C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p  # Budget constraint
    p,        p == 1                               # Numeraire
end

# Calibration: fix data, solve for unknowns
fix(σ, 2)
fix(Y, 1000)
fix.(N, [3200, 500])
fix.(L, [800, 200])
set_start_value.(model_block, 1.0)
optimize!(model)
baseline = value_dict(model)

# Counterfactual: fix calibrated parameters, shock exogenous variables
fix(baseline)
unfix(model_block)
fix.(N, [2700, 1000])  # Population shock
set_start_value(baseline)
optimize!(model)
println("Multipliers: ", (value_dict(model) .- baseline) ./ baseline .- 1)

```

## Key Concepts

### Blocks

A `Block` is a collection of constraints paired with their endogenous variables:

```julia
block = @block model begin
  x,           x == a + b
  y[i ∈ 1:3],  y[i] == i * z
end
```

Blocks can be combined with `+`:
```julia
full_model = consumers + production + government
```

### Endo-Exo Swapping

During calibration, you often want to treat normally-endogenous variables as exogenous (data) and solve for parameters instead. The `@endo_exo!` macro swaps variable roles within a block:

```julia
calibration_block = copy(base_block)
@endo_exo! calibration_block begin
  μ,  Y      # Solve for μ given Y (instead of Y given μ)
  δ,  K[t₁]  # Solve for δ given initial capital
end
```

## Solver Notes

### Current Status
The primary solver for large-scale nonlinear square systems is **CONOPT**. Currently, CONOPT is accessed via the [GAMS.jl](https://github.com/GAMS-dev/gams.jl) package, which requires a GAMS installation and license.

### Roadmap

The CONOPT developers are working on a direct JuMP interface for CONOPT. Once available, this will remove the GAMS dependency and simplify the setup significantly.

For smaller models or testing, **Ipopt** works well and is freely available:

```julia
using Ipopt
model = Model(Ipopt.Optimizer)
```

## Project Structure

```
SquareModels/
├── src/
│   ├── SquareModels.jl   # Main module: Block, @block, @endo_exo!
│   ├── endo_exo.jl       # Endo-exo swap implementation
│   └── utils.jl          # Helper functions
├── examples/
│   └── quick_example.jl  # Labor market model (tested)
├── ModelDictionaries/    # Helper package for variable-value mappings
├── GamsGDX/              # Utility for reading GAMS GDX files (optional)
└── test/
    └── runtests.jl
```

## Related Packages

- **ModelDictionaries** (included): Provides `ModelDictionary` for mapping JuMP variables to values, with convenient `fix()`, `set_start_value()`, and `value()` extensions
- **GamsGDX** (included, optional): Reads GAMS GDX files into Julia DataFrames for data transfer

## Requirements

- Julia 1.9+
- JuMP 1.15+ (uses the unified nonlinear interface)
- A nonlinear solver (Ipopt for testing, GAMS+CONOPT for production)

## License
This project is licensed under an MIT license — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work is part of the [DREAM](https://dreamgruppen.dk/) group's effort to modernize economic modeling tools in Denmark and the rest of the world.
