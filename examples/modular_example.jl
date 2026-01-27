# Modular Dynamic CGE Model Example
#
# A minimal two-sector dynamic CGE model demonstrating:
# - Module-based organization with explicit cross-module imports
# - Sector indices defined in one module, used by others
# - Time index defined in main file with non-constant T (for T-loops)
# - Self-contained calibration blocks
#
# Modules:
# - Production: Defines sectors, output, and prices
# - HouseHolds: Consumption demand using production variables

using JuMP
using Ipopt
using SquareModels

# ==============================================================================
# Global model container and time configuration
# ==============================================================================
db = ModelDictionary(Model(Ipopt.Optimizer))
set_silent(db.model)

# Time configuration (T is non-const for T-loop calibration)
const t₀ = 2020    # First year for variable definitions
const max_T = 2200 # Maximum possible terminal year
const t = t₀:max_T  # Full time range for variable definitions

t₁::Int = t₀+1    # First endogenous year (can be modified later)
T::Int = max_T    # Terminal year (can be modified later)

# ==============================================================================
# Submodel: Production
# ==============================================================================
module Production
	using JuMP, SquareModels
	import ..db, ..t, ..t₁, ..T

	# Sectors defined in this module
	const s = [:agri, :manuf]

	@variables db.model begin
		Y[s,t]     # Output by sector and time
		p[s,t]     # Price by sector and time
		K[s,t]     # Capital input (exogenous)
		A[s]       # Productivity (calibrated, time-invariant)
		g          # Growth rate (calibrated)
	end

	function define_equations()
		return @block db begin
			# Production function
			Y[s = s, t = t₁:T],
			Y[s,t] == A[s] * K[s,t]

			# Numeraire: agriculture price = 1
			p[s = [:agri], t = t₁:T],
			p[s,t] == 1

			# Manufacturing price grows at rate g
			p[s = [:manuf], t = [t₁]],
			p[s,t] == 1

			p[s = [:manuf], t = t₁+1:T],
			p[s,t] == p[s,t-1] * (1 + g)
		end
	end

	function set_data!(db)
		db[Y] .= 100.0
		db[p] .= 1.0
		db[K] .= 50.0
		return nothing
	end

	function define_calibration()
		block = define_equations()
		@endo_exo! block begin
			A, Y[s, t₁]  # Calibrate productivity to match initial output
			g, p[:manuf, t₁+1]  # Calibrate growth to match second period price
		end
		return block
	end
end

# ==============================================================================
# Submodel: HouseHolds
# ==============================================================================
module HouseHolds
	using JuMP, SquareModels
	import ..db, ..t, ..t₁, ..T

	# Import sectors from Production
	s = Main.Production.s

	# Import variables from Production
	Y = Main.Production.Y
	p = Main.Production.p

	@variables db.model begin
		C[s,t]     # Consumption by sector and time
		U[t]       # Utility per period
		α[s]       # Consumption shares (calibrated, time-invariant)
		I[t]       # Income (exogenous)
	end

	function define_equations()
		return @block db begin
			# Consumption demand (Cobb-Douglas)
			C[s = s, t = t₁:T],
			p[s,t] * C[s,t] == α[s] * I[t]

			# Utility (log utility)
			U[t = t₁:T],
			U[t] == sum(α[s] * log(C[s,t]) for s in s)
		end
	end

	function set_data!(db)
		db[C] .= 80.0
		db[U] .= 5.0
		db[I] .= 160.0
		return nothing
	end

	function define_calibration()
		block = define_equations()
		@endo_exo! block begin
			α, C[s, t₁]  # Calibrate shares to match initial consumption
		end
		return block
	end
end

# ==============================================================================
# Model Assembly
# ==============================================================================
submodels = [
	Production,
	HouseHolds
]

for m in submodels
	m.set_data!(db)
end

base_model() = sum(m.define_equations() for m in submodels)
calibration() = sum(m.define_calibration() for m in submodels)

# ==============================================================================
# Solve calibration
# ==============================================================================
baseline = solve(calibration(), db; replace_nothing=1.0)

println("Calibrated A: ", [baseline[Production.A[s]] for s in Production.s])
println("Calibrated α: ", [baseline[HouseHolds.α[s]] for s in Production.s])
println("Calibrated g: ", baseline[Production.g])

# ==============================================================================
# Scenario: Productivity shock in manufacturing
# ==============================================================================
T = 2060 # Terminal year can be changed
scenario = copy(baseline)
scenario[Production.K[:manuf, t]] = 60.0  # 20% increase in manufacturing capital

solve!(base_model(), scenario)

println("\nScenario results (T=$T):")
println("Y[manuf,:]: ", scenario[Production.Y[:manuf, t₁:T]])
println("p[manuf,:]: ", scenario[Production.p[:manuf, t₁:T]])
println("U: ", scenario[HouseHolds.U[t₁:T]])
