# Quick Example - A simple labor market model
#
# This example demonstrates the core features of SquareModels:
# - Defining blocks of equations with paired endogenous variables
# - Calibrating parameters from data
# - Running counterfactual scenarios

using JuMP
using SquareModels
using Ipopt

# ------------------------------------------------------------------------------
# Model and solver
# ------------------------------------------------------------------------------
model = Model(Ipopt.Optimizer)

# ------------------------------------------------------------------------------
# Sets
# ------------------------------------------------------------------------------
j = 1:2  # Types of labor

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
@variables model begin
    L[j]  # Labor demand
    w[j]  # Wage
    Y     # Output
    C     # Consumption
    p     # Price

    N[j]  # Labor force (exogenous)
    σ     # Substitution elasticity (exogenous)

    ρ[j]  # Productivity (calibrated)
    μ[j]  # Scale parameter (calibrated)
end

# ------------------------------------------------------------------------------
# Equations
# ------------------------------------------------------------------------------
# Define a Block: each line pairs an endogenous variable with its equation
model_block = @block model begin
    L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y   # Labor demand
    w[j ∈ j], L[j] == ρ[j] * N[j]                 # Labor market clearing
    Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)   # Zero profit
    C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p  # Budget constraint
    p,        p == 1                               # Numeraire
end

# ------------------------------------------------------------------------------
# Exogenous values
# ------------------------------------------------------------------------------
data = ModelDictionary(model)

data[σ] = 2.0
data[w] .= 1
data[N] = [3200, 500]
data[L] = [800, 200]

# Set residual values
data[residuals(model_block)] .= 0.0

# ------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------
# For calibration, swap observed values with parameters to be calibrated
calibration = copy(model_block)
@endo_exo! calibration begin
    μ, L
    ρ, w
end

start_values = copy(data)
start_values[endogenous(calibration)] .= 1.0

baseline = solve(calibration, data; start_values)

# ------------------------------------------------------------------------------
# Counterfactual scenario
# ------------------------------------------------------------------------------
# Start from baseline and apply shock
shock = copy(baseline)
shock[N] .= [2700.0, 1000.0]
solve!(model_block, shock)

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
differences = shock .- baseline
multipliers = shock ./ baseline .- 1

println("Baseline: ", baseline)
println("Shocked:  ", shock)
println("Multipliers: ", multipliers[multipliers .≠ 0])
