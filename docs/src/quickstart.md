# Getting Started

This example calibrates a small labor-market model, then solves a counterfactual
scenario. The full version lives in `examples/quick_example.jl`.

```@example quickstart
import JuMP
using JuMP: Model, set_silent
using Ipopt
using SquareModels

data = ModelDictionary(Model(Ipopt.Optimizer))
set_silent(data.model)

j = 1:2

@variables data.model begin
    L[j], "Labor demand"
    w[j], "Wage"
    Y, "Output"
    Cj[j], "Consumption by labor type"
    C, "Consumption"
    p, "Price"

    N[j], "Labor force"
    σ, "Substitution elasticity"
    ρ[j], "Productivity"
    μ[j], "Scale parameter"
end

data[σ] = 2.0
data[w] .= 1
data[N] = [3200, 500]
data[L] = [800, 200]
nothing
```

`@block` pairs every endogenous variable with the equation that determines it.
Indexed declarations expand to one equation per index.

```@example quickstart
model_block = @block data begin
    L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y
    w[j ∈ j], L[j] == ρ[j] * N[j]
    Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)
    Cj[j ∈ j], Cj[j] == w[j] * ρ[j] * N[j] / p
    C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p
    @test_constraint("Consumption aggregation")
    C,        C == ∑(Cj[j] for j ∈ j)
    p,        p == 1
end

length(model_block)
```

Calibration is an endo-exo swap: observed variables become fixed data, while
normally exogenous parameters become endogenous calibration targets.

```@example quickstart
calibration = copy(model_block)
@endo_exo_swap! calibration begin
    μ, L
    ρ, w
end

baseline = solve(calibration, data; replace_nothing=1.0)
round.(baseline[μ], digits=4)
```

A scenario starts from the calibrated baseline, changes exogenous assumptions,
and solves the original model block.

```@example quickstart
scenario = copy(baseline)
scenario[N] .= [2700.0, 1000.0]
solve!(model_block, scenario)

round.(scenario[L] ./ baseline[L] .- 1, digits=4)
```
