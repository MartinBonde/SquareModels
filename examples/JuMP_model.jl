using JuMP
using JuMP.Containers: DenseAxisArray
using GAMS
using Dictionaries
using DataFrames
using SQLite
using BSON, OrderedCollections
using SquareModels

# ------------------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------------------
function save(file_path, dictionary)
	dict = Dictionary(string.(keys(dictionary)), dictionary) |> pairs |> OrderedDict
	bson(file_path, dict)
end
function load(file_path, model)
	dict = BSON.load(file_path)
	vars = variable_by_name.(Ref(model), keys(dict))
	return Dictionary(vars, values(dict))
end

# ------------------------------------------------------------------------------
# Initalize a new model and choose solver
# ------------------------------------------------------------------------------
gams_dir = splitdir(Sys.which("gams"))[1]
ws = GAMS.GAMSWorkspace(abspath("examples/GAMS"), gams_dir)
# model = Model(() -> GAMS.Optimizer(ws))
model = direct_model(GAMS.Optimizer(ws));
set_optimizer_attribute(model, GAMS.ModelType(), "CNS")
set_optimizer_attribute(model, "solver", "CONOPT4")
set_optimizer_attribute(model, "HoldFixed", 1)
set_optimizer_attribute(model, "lmmxsf", 1)
set_optimizer_attribute(model, "RTREDG", 1.e-9)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
# For time and age sets it is more convenient to only define the start and end conditions
# and use these to create ranges as needed
tData₀ = 2024 # First year
t₁ = 2025 # First endogenous year
T = 2099 # Terminal year

a₁ = 15 # First age
A = 100 # Terminal age

fq = 1.01
fp = 1.02

# ------------------------------------------------------------------------------
# Define variables
# ------------------------------------------------------------------------------
t = tData₀:T
a = a₁:A
a⁺ = [:tot, a₁:A...]

variables = @variables model begin
	Y[t]  # Output
	L[t]  # Arbejdskraft
	K[t]  # Kapital
	I[t]  # Investeringer

	X[t]  # Eksport
	σˣ  # Eksportpriselasticitet
	μˣ  # Skalaparameter for X

	cᵃ[a, t]  # Forbrug fordelt på alder
	C[t]  # Samlet forbrug

	p[t]  # Pris
	w[t]  # Løn
	pᵏ[t]  # Usercost på Kapital
	pᶠ[t]  # Eksportkonkurrerende pris

	N[a⁺, t]  # Arbejdsstyrke

	μᴷ  # Skalaparameter for K
	μᴸ  # Skalaparameter for L
	σᴷᴸ  # Substitutionselasticitiet

	r[t]  # Renten
	δ  # Afskrivningsrate

	MPC # Marginal forbrugstilbøjelighed
end

# ------------------------------------------------------------------------------
# Equation  blocks
# ------------------------------------------------------------------------------
# We define each equation with an explicitly attached endogenous variable
consumers = @block model begin
	cᵃ[a ∈ a₁:A, t ∈ t₁:T],
	cᵃ[a, t] == MPC * w[t] / p[t]


	C[t = t₁:T],
	C[t] == ∑(N[a, t] * cᵃ[a, t] for a in a₁:A)
end

exports = @block model begin
	X[t = t₁:T],
	X[t] == μˣ * (p[t] / pᶠ[t])^-σˣ
end

io = @block model begin
	Y[t = t₁:T],
	Y[t] == C[t] + X[t] + I[t]
end

labor_market = @block model begin
	L[t = t₁:T],
	L[t] == ∑(N[a, t] for a = a₁:A)
end

production = @block model begin
	p[t = t₁:T],
	p[t] * Y[t] == pᵏ[t] * K[t-1] + w[t] * L[t]


	K[t = t₁:T-1],
	K[t] == μᴷ * (pᵏ[t+1] / p[t+1])^-σᴷᴸ * Y[t+1] * fq

	K[t = [T]],
	K[t] == K[t-1]


	w[t = t₁:T],
	L[t] == μᴸ * (w[t] / p[t])^-σᴷᴸ * Y[t]


	pᵏ[t = t₁+1:T],
	pᵏ[t] == δ * r[t] * p[t]

	pᵏ[t = [t₁]],
	K[t-1] == μᴷ * (pᵏ[t] / p[t])^-σᴷᴸ * Y[t]


	I[t = t₁:T],
	K[t] == (1 - δ) * K[t-1] / fq + I[t]
end

base_bocks = [
	consumers
	exports
	labor_market
	io
	production
]
base = sum(base_bocks)  # Our baseline model is the sum of all the blocks

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
# Create Dictionary matching all variables with a numerical value - initialy zero
data = zeros(Indices(all_variables(model)))

# Read data from an SQLite database (created by converting a gdx file)
db = SQLite.DB("examples/MAKRO_stylized_baseline.db")

# Retrieve DataFrame of symbol from database
get_dataframe(db::SQLite.DB, table::AbstractString) =
	DBInterface.execute(db, "SELECT * FROM $table") |> DataFrame

# Extract level values from DataFrame for each index in a DenseAxisArray
function get_data(df::AbstractDataFrame, var::DenseAxisArray{VariableRef})
	df_indices = [(row...,) for row in eachrow(Matrix(df[:, 1:end-4]))]
	d = Dictionary(df_indices, df.level)
	var_indices = [string.(rec) for rec in Base.Iterators.product(var.axes...)]
	return getindex.(Ref(d), var_indices)
end

# Extract level values from database for each index in a DenseAxisArray
function get_data(db::SQLite.DB, table::AbstractString, var::DenseAxisArray{VariableRef}, fixed...)
	df = get_dataframe(db, table)
	get_data(df, var, fixed...)
end

function get_data(df::AbstractDataFrame, var::DenseAxisArray{VariableRef}, fixed...)
	n = length(fixed)
	get_data(
	    df[all.([df[:, i] .== fixed[i] for i in 1:n]...), n+1:end],
	    var
	)
end

data[N] = get_data(db, "nLHh", N)
data[N[:tot, :]] = data[N[:tot, :]]
data[L] = data[N[:tot, :]]

# data[Y] = get_data(db, "qY", Y, "tot")

data[I] = get_data(db, "qI", I, "iTot")

data[cᵃ] = get_data(db, "qC_a", cᵃ)
data[C] = get_data(db, "qC", C, "cTot")

data[X] = get_data(db, "qX", C, "xTot")

data[Y] = data[C] + data[I] + data[X]

data[w] = get_data(db, "vhW", w)

data[p] = get_data(db, "pY", p, "tot")

# data[r] = get_data(db, "rRente", r, "RealKred")
data[r] .= 0.04


# ------------------------------------------------------------------------------
# Exogenous parameters
# ------------------------------------------------------------------------------
data[σᴷᴸ] = 0.9
data[δ] = 0.1

data[K] .= data[I[t₁]] / data[δ]

data[σˣ] = 2
data[pᶠ] .= 1

# ------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------
calibrate_consumers = copy(consumers)
@endo_exo! calibrate_consumers begin
	MPC, C[t₁]
end

calibrate_exports = copy(exports)
@endo_exo! calibrate_exports begin
	μˣ, X[t₁]
end

calibrate_io = copy(io)

calibrate_labor_market = copy(labor_market)

calibrate_production = copy(production)
@endo_exo! calibrate_production begin
	μᴸ, w[t₁]
	μᴷ, K[t₁]
end

calibration_blocks = [
	calibrate_consumers
	calibrate_exports
	calibrate_labor_market
	calibrate_io
	calibrate_production
]
calibration = sum(calibration_blocks)

for block in [
	# calibration_blocks...,
	calibration
]
	fix(data)
	unfix.(residuals(base))
	fix.(residuals(block), 0.0)
	unfix.(block)
	set_lower_bound.(block, 0.00001)

	set_start_value([values(block)...], data)
	previous_solution = load("examples/multi_module_example.bson", model)
	set_start_value([i for i in block if i in keys(previous_solution)], previous_solution)
	optimize!(model)
end
output = value_dict(model)
# save("multi_module_example.bson", output)

using GLMakie

function plot(data, t, vars...)
	fig = Figure()
	axis = Axis(fig[1, 1])
	for var in vars
	    lines!(axis, collect(t), data[var[t]], label = base_name(var))
	end
	Legend(fig[2, 1], axis, tellwidth = false, orientation = :horizontal, framevisible = false)
	return fig
end

plot(output, 2025:2080, C, I, X)
plot(output, 2025:2080, w, p)
plot(output, 2025:2080, K, N[:tot, :])
plot(output, 2025:2080, pᵏ, r)
