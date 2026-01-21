# Quick Example - A simple labor market model
#
# This example demonstrates the core features of SquareModels:
# - Defining blocks of equations with paired endogenous variables
# - Calibrating parameters from data
# - Running counterfactual scenarios

using JuMP
using SquareModels

# ------------------------------------------------------------------------------
# Initialize a new model and choose solver
# ------------------------------------------------------------------------------
using Ipopt
model = Model(Ipopt.Optimizer)

# For large-scale models, use GAMS with CONOPT:
# using GAMS
# model = Model(GAMS.Optimizer)

set_silent(model)

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
    ρ[j]  # Productivity (exogenous)
    μ[j]  # Scale parameter (calibrated)
    σ     # Substitution elasticity (exogenous)
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
# Calibration
# ------------------------------------------------------------------------------
# Fix variables that we have data for
fix(σ, 2)
fix(Y, 1000)
fix.(N, [3200, 500])
fix.(L, [800, 200])

# Provide starting guesses to help the solver
set_start_value.(model_block, 1.0)

# Solve calibration model
optimize!(model)

# Save all results as a Dictionary
baseline = value_dict(model)

# ------------------------------------------------------------------------------
# Counterfactual scenario
# ------------------------------------------------------------------------------
# Fix all variables to their baseline values
fix(baseline)

# Unfix endogenous variables (variables attached to our block of equations)
unfix(model_block)

# Shock the population
fix.(N, [2700, 1000])

# Solve model using baseline as starting guesses
set_start_value(baseline)
optimize!(model)
shock = value_dict(model)

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
differences = shock .- baseline
multipliers = shock ./ baseline .- 1

println("Baseline: ", baseline)
println("Shocked:  ", shock)
println("Multipliers: ", multipliers[multipliers .≠ 0])
