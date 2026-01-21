using JuMP
# using Ipopt
using GAMS
using Dictionaries
using SquareModels

# ------------------------------------------------------------------------------
# Initalize a new model and choose solver
# ------------------------------------------------------------------------------
# model = Model(Ipopt.Optimizer)
model = Model(GAMS.Optimizer)

# ------------------------------------------------------------------------------
# Sets
# ------------------------------------------------------------------------------
j = 1:2 # Typer arbejdskraft

# ------------------------------------------------------------------------------
# Define variables
# ------------------------------------------------------------------------------
@variables model begin
	L[j]  # Arbejdskraft
	w[j]  # Løn
	Y  # Output
	C  # Forbrug
	p  # Pris

	N[j]  # Arbejdsstyrke
	ρ[j]  # Produktivitet
	μ[j]  # Skala-parameter
	σ  # Substitutionselasticitiet
end

# ------------------------------------------------------------------------------
# Equations
# ------------------------------------------------------------------------------
# We define each equation with an explicitly attached endogenous variable
model_block = @block model begin
	L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y
	w[j ∈ j], L[j] == ρ[j] * N[j]
	Y, p * Y == ∑(w[j] * L[j] for j ∈ j)
	C, C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p
	p, p == 1
end

# ------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------
# Fix variable that we have data for
fix(σ, 2)
fix(Y, 1000)
fix.(N, [3200, 500])
fix.(L, [800, 200])

# Provide better starting guesses to help the solver
set_start_value.(model_block, 1.0)

# Solve calibration model
optimize!(model)

# Save all results as a Dictionary
baseline = value_dict(model)

# ------------------------------------------------------------------------------
# Endo exo
# ------------------------------------------------------------------------------
# Fix all variables to their baseline values
fix(baseline)

# Unfix endogenous variables (variables attached to our block of equations)
unfix(model_block)

# ------------------------------------------------------------------------------
# Shock
# ------------------------------------------------------------------------------
# Shock population
fix.(N, [2700, 1000])

# Solve model using baseline as starting guesses
set_start_value(baseline)
optimize!(model)
shock = value_dict(model)

# Look at output
difference = shock .- baseline
multiplier = (shock .- baseline) ./ baseline
