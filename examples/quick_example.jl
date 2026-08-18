# Quick Example - A simple labor market model
#
# This example demonstrates the core features of SquareModels:
# - Defining blocks of equations with paired endogenous variables
# - Testing a redundant aggregate constraint outside the solve system
# - Calibrating parameters from data
# - Running counterfactual scenarios

import JuMP
using JuMP: Model, set_silent
using Ipopt
using SquareModels

# ------------------------------------------------------------------------------
# Model and data container
# ------------------------------------------------------------------------------
data = ModelDictionary(Model(Ipopt.Optimizer))

# ------------------------------------------------------------------------------
# Sets
# ------------------------------------------------------------------------------
j = 1:2  # Types of labor

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
@variables data.model begin
	L[j], "Labor demand"
	w[j], "Wage"
	Y, "Output"
	Cj[j], "Consumption by labor type"
	C, "Consumption"
	p, "Price"

	N[j], "Labor force (exogenous)"
	σ, "Substitution elasticity (exogenous)"

	ρ[j], "Productivity (calibrated)"
	μ[j], "Scale parameter (calibrated)"
end

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
data[σ] = 2.0
data[w] .= 1
data[N] = [3200, 500]
data[L] = [800, 200]

# ------------------------------------------------------------------------------
# Equations
# ------------------------------------------------------------------------------
# Define a Block: each line pairs an endogenous variable with its equation
model_block = @block data begin
	L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y   # Labor demand
	w[j ∈ j], L[j] == ρ[j] * N[j]                 # Labor market clearing
	Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)   # Zero profit
	Cj[j ∈ j], Cj[j] == w[j] * ρ[j] * N[j] / p    # Consumption by labor type
	C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p  # Budget constraint
	@test_constraint("Consumption aggregation")
	C,        C == ∑(Cj[j] for j ∈ j)
	p,        p == 1                               # Numeraire
end

# ------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------
# For calibration, swap observed values with parameters to be calibrated
calibration = copy(model_block)
@endo_exo_swap! calibration begin
	μ, L
	ρ, w
end

baseline = solve(calibration, data; replace_nothing=1.0)

# ------------------------------------------------------------------------------
# Counterfactual scenario
# ------------------------------------------------------------------------------
# Start from baseline and apply shock
scenario = copy(baseline)
scenario[N] .= [2700.0, 1000.0]
solve!(model_block, scenario)

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
differences = scenario .- baseline
multipliers = scenario ./ baseline .- 1

println("Baseline: ", baseline)
println("Scenario: ", scenario)
println("Multipliers: ", multipliers[multipliers .≠ 0])
