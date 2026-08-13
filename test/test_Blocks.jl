module TestSquareModels

using Test
using JuMP
using SquareModels
using JuMP.Containers: DenseAxisArray, SparseAxisArray
using Ipopt

module NoJuMPImportBlockTest
using Test
using SquareModels: @variables, @block, @test_constraint, test_constraints, test_constraint_variables, is_endogenous

function run(m)
	@variables m begin
		x
		y[1:2]
	end

	b = @block m begin
		x, x == 1
		@test_constraint "x aggregation" x, x == 1
		y[i ∈ 1:2], y[i] == i
	end

	@test length(b) == 3
	@test length(test_constraints(b)) == 1
	@test test_constraint_variables(b) == [x]
	@test is_endogenous(x, b)
	@test all(is_endogenous(y[i], b) for i ∈ 1:2)
end
end

@testset "copy_variable" begin
	m = Model()
	JuMP.@variables m begin
		x
		y[1:5]
	end

	J_x = SquareModels.copy_variable("J_x", x)
	@test J_x == m[:J_x]

	J_y = SquareModels.copy_variable("J_y", y)
	@test J_y == m[:J_y]
	@test length(J_y) == length(y)

	@testset "SparseAxisArray" begin
		@variable(m, s[i=1:3, j=1:3; i != j])
		@test s isa SparseAxisArray

		J_s = SquareModels.copy_variable("J_s", s)
		@test J_s == m[:J_s]
		@test J_s isa SparseAxisArray
		@test length(J_s) == length(s)
	end
end

@testset "@block works without importing JuMP" begin
	NoJuMPImportBlockTest.run(Model())
end

@testset "@_block" begin
	m = Model()
	JuMP.@variables m begin
		x
		y[1:5]
		z[1:3, [:a, :b]]
		q
	end

	@testset "x" begin
		v1, r1, eqs1 = SquareModels.@_block(m, x, x == 1)
		b1 = SquareModels.Block(m, v1, r1, Set{VariableRef}(), eqs1)
		@test typeof(v1) <: AbstractVector{VariableRef}
		@test typeof(eqs1) <: AbstractVector{Equation}
		@test length(v1) == length(eqs1) == length(b1) == 1
		@test is_endogenous(x, b1)
	end

	@testset "y[1:4]" begin
		v2, r2, eqs2 = SquareModels.@_block(m, y[i ∈ 1:4], y[i] == 1)
		b2 = SquareModels.Block(m, v2, r2, Set{VariableRef}(), eqs2)
		@test typeof(v2) <: AbstractVector{VariableRef}
		@test length(v2) == length(eqs2) == length(b2) == 4
		@test all(is_endogenous(y[i], b2) for i ∈ 1:4)
	end

	@testset "y[5]" begin
		v3, r3, eqs3 = SquareModels.@_block(m, y[i ∈ [5]], y[i] == 1)
		b3 = SquareModels.Block(m, v3, r3, Set{VariableRef}(), eqs3)
		@test typeof(v3) <: AbstractVector{VariableRef}
		@test length(v3) == length(eqs3) == length(b3) == 1
		@test is_endogenous(y[5], b3)
	end

	@testset "z" begin
		t₁ = 1
		T = 3
		v4, r4, eqs4 = SquareModels.@_block(m, z[i ∈ t₁:T, j ∈ [:a, :b]], z[i, j] == 1)
		b4 = SquareModels.Block(m, v4, r4, Set{VariableRef}(), eqs4)
		@test typeof(v4) <: AbstractVector{VariableRef}
		@test length(v4) == length(eqs4) == length(b4) == 6
		@test all(is_endogenous(z[i, j], b4) for i ∈ t₁:T, j ∈ [:a, :b])
	end
end

@testset "@block accepts multiline equation continuations" begin
	m = Model()
	JuMP.@variables m begin
		x
		a
		b
		c
		q
	end

	block = @block m begin
		x, x == a
			+ b
			- c
	end

	@test length(block) == 1
	@test all(v ∈ block.variables for v in [a, b, c])
	@test q ∉ block.variables
end

@testset "@block rejects stray block expressions" begin
	@test_throws LoadError @eval let m = Model()
		@variable(m, q)
		@block m begin
			+ q
		end
	end

	@test_throws LoadError @eval let m = Model()
		JuMP.@variables m begin
			x
			q
		end
		@block m begin
			x, x == 1
			q
		end
	end

	@test_throws LoadError @eval let m = Model()
		@variable(m, x[1:2])
		@block m begin
			x[t], x[t]
		end
	end

	@test_throws LoadError @eval let m = Model()
		JuMP.@variables m begin
			x
			q
		end
		@block m begin
			x, x == 1, q
		end
	end
end

@testset "@block malformed syntax is rejected by Julia parser" begin
	@test_throws Meta.ParseError Meta.parse("""
		@block m begin
			x[t], x[t] ==
		end
	""")

	@test_throws Meta.ParseError Meta.parse("""
		@block m begin
			x[t],
		end
	""")

	@test_throws Meta.ParseError Meta.parse("""
		@block m begin
			// wrong comment
		end
	""")
end

@testset "@block" begin
	m = Model(Ipopt.Optimizer)
	set_silent(m)
	JuMP.@variables m begin
		x
		y[1:5]
		z[1:3, [:a, :b]]
		q
	end

	b = @block m begin
		x, x == 1
		y[i ∈ 1:4], y[i] == 1
		y[i ∈ [5]], y[5] == 1
		z[i ∈ 1:3, j ∈ [:a, :b]], z[i, j] == 1
	end
	@test length(b) == sum(length([x, y..., z...]))
	@test x ∈ b
	@test all(y[i] ∈ b for i ∈ 1:5)
	@test all(z[i, j] ∈ b for i ∈ 1:3, j ∈ [:a, :b])
	@test q ∉ b

	db = ModelDictionary(m, 0.0)
	result = solve(b, db)
	for i in [x, y..., z...]
		@test result[i] ≈ 1 atol=1e-6
	end
end

@testset "solve block" begin
	m = Model(Ipopt.Optimizer)
	set_silent(m)
	JuMP.@variables m begin
		x
		y[1:5]
		z[1:3, [:a, :b]]
	end

	b = @block m begin
		x, x == 1
		y[i ∈ 1:4], y[i] == 1
		y[i ∈ [5]], y[5] == 1
		z[i ∈ 1:3, j ∈ [:a, :b]], z[i, j] == 1
	end

	db = ModelDictionary(m, 0.0)
	result = solve(b, db)
	@test all(isapprox(result[i], 1; atol=1e-6) for i in b)
end

@testset "_endo_exo_swap!" begin
	m = Model(Ipopt.Optimizer)
	JuMP.@variables m begin
		x
		y[1:5]
		x_exo
		y_exo[1:5]
	end

	# Constraints include both x/y and x_exo/y_exo so swaps are valid
	b = @block m begin
		x, x + x_exo == 1
		y[i ∈ 1:5], y[i] + y_exo[i] == 1
	end

	@testset "x" begin
		@test !is_fixed(x)
		fix.(b, 1)
		@test is_fixed(x)
		unfix(b)
		@test !is_fixed(x)
		SquareModels._endo_exo_swap!(b, x_exo, x, "")
		fix.(b, 1)
		@test !is_fixed(x)
		@test is_fixed(x_exo)
		unfix(b)
	end

	@testset "y" begin
		@test !any(is_fixed.(y))
		fix.(b, 1)
		@test all(is_fixed.(y))
		unfix(b)
		@test !any(is_fixed.(y))
		SquareModels._endo_exo_swap!(b, y_exo, y, "")
		fix.(b, 1)
		@test !any(is_fixed.(y))
		@test all(is_fixed.(y_exo))
		unfix(b)
	end

	@testset "SparseAxisArray" begin
		m2 = Model(Ipopt.Optimizer)
		@variable(m2, s[i=1:3, j=1:3; i != j])
		@variable(m2, s_exo[i=1:3, j=1:3; i != j])

		b2 = @block m2 begin
			s[i ∈ 1:3, j ∈ 1:3; i != j], s[i, j] + s_exo[i, j] == 1
		end

		SquareModels._endo_exo_swap!(b2, s_exo, s, "")
		fix.(b2, 1)
		@test !any(is_fixed(s[i, j]) for (i, j) in keys(s.data))
		@test all(is_fixed(s_exo[i, j]) for (i, j) in keys(s_exo.data))
		# Verify correct pairing: each s_exo[i,j] replaced the matching s[i,j]
		for (i, j) in keys(s.data)
			@test is_endogenous(s_exo[i, j], b2)
			@test !is_endogenous(s[i, j], b2)
		end
		unfix(b2)
	end
end

@testset "@endo_exo_swap!" begin
	m = Model(Ipopt.Optimizer)
	JuMP.@variables m begin
		x
		y[1:5]
		x_exo
		y_exo[1:5]
	end

	# Constraints include both x/y and x_exo/y_exo so swaps are valid
	b = @block m begin
		x, x + x_exo == 1
		y[i ∈ 1:5], y[i] + y_exo[i] == 1
	end

	@testset "x" begin
		@endo_exo_swap!(b, x_exo, x)
		fix.(b, 1)
		@test !is_fixed(x)
		@test is_fixed(x_exo)
		unfix(b)
	end

	@testset "y" begin
		@endo_exo_swap!(b, y_exo, y)
		fix.(b, 1)
		@test !any(is_fixed.(y))
		@test all(is_fixed.(y_exo))
		unfix(b)
	end

	@testset "x and y" begin
		# Fresh block with both vars in constraints
		b = @block m begin
		x, x + x_exo == 1
		y[i ∈ 1:5], y[i] + y_exo[i] == 1
		end

		@endo_exo_swap! b begin
		x_exo, x
		y_exo, y
		end
		fix.(b, 1)
		@test !is_fixed(x)
		@test is_fixed(x_exo)
		@test !any(is_fixed.(y))
		@test all(is_fixed.(y_exo))
	end
end

@testset "@endo_exo_swap! error messages" begin
	m = Model()
	JuMP.@variables m begin
		x
		y
		z
	end

	b = @block m begin
		x, x == 1
	end

	@testset "variable not in block" begin
		err = try
			@endo_exo_swap!(b, z, y)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("y is not endogenous", err.msg)
		@test occursin("Endogenous variables in block", err.msg)
	end

	@testset "endo not in constraints" begin
		# w doesn't appear in any constraint in b
		@variable(m, w)
		err = try
			@endo_exo_swap!(b, w, x)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("w does not appear in the block's constraints", err.msg)
	end

	@testset "swapped order suggestion" begin
		# Create a block where swap detection makes sense: y is exogenous, x is endogenous
		b_swap = @block m begin
			x, x + y == 1
		end
		# Swapped args: trying to make x endogenous (but it already is) and y exogenous (but it's not endo)
		err = try
			@endo_exo_swap!(b_swap, x, y)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("y is not endogenous", err.msg)
		# Should suggest swap since x is endogenous and y appears in block
		@test occursin("Did you swap the arguments?", err.msg)
		@test occursin("@endo_exo_swap!(block, y, x)", err.msg)
	end

	@testset "no swap suggestion when unhelpful" begin
		# z is not in any constraint, so swapping wouldn't help
		err = try
			@endo_exo_swap!(b, x, z)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("z is not endogenous", err.msg)
		# Should NOT suggest swap since z is not in block.variables
		@test !occursin("swap", err.msg)
	end

	@testset "length mismatch" begin
		JuMP.@variables m begin
			a[1:3]
			b[1:5]
		end
		b2 = @block m begin
			a[i ∈ 1:3], a[i] == 1
		end
		err = try
			@endo_exo_swap!(b2, b, a)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("Number of variables do not match", err.msg)
		@test occursin("endo variables (5)", err.msg)
		@test occursin("exo variables (3)", err.msg)
	end
end

@testset "Block subtraction" begin
	m = Model()
	JuMP.@variables m begin
		x
		y
		z
	end

	b1 = @block m begin
		x, x == 1
		y, y == 2
	end

	b2 = @block m begin
		y, y == 2
	end

	b3 = b1 - b2
	@test length(b3) == 1
	@test x ∈ b3
	@test y ∉ b3

	b4 = @block m begin
		z, z == 3
	end
	@test length(b1 - b4) == length(b1)
end

@testset "Block addition with overlapping variables" begin
	m = Model()
	JuMP.@variables m begin
		x
		y
		z[1:3]
	end

	b1 = @block m begin
		x, x == 1
		y, y == 2
	end

	b2 = @block m begin
		z[i ∈ 1:3], z[i] == i
	end

	# Non-overlapping blocks can be added
	combined = b1 + b2
	@test length(combined) == 5

	# Overlapping blocks cannot be added (would create non-square system)
	b3 = @block m begin
		x, x == 10  # x already in b1
	end

	err = try
		b1 + b3
		nothing
	catch e
		e
	end
	@test err isa ErrorException
	@test occursin("Cannot combine blocks", err.msg)
	@test occursin("Overlapping endogenous variables:", err.msg)
	@test occursin("x", err.msg)
	@test occursin("non-square", err.msg)

	# Multiple overlapping variables
	b4 = @block m begin
		x, x == 10
		y, y == 20
	end

	err2 = try
		b1 + b4
		nothing
	catch e
		e
	end
	@test err2 isa ErrorException
	@test occursin("2 endogenous variable(s)", err2.msg)
	@test occursin("Overlapping endogenous variables:", err2.msg)

	# Indexed variable overlap
	b5 = @block m begin
		z[i ∈ 2:3], z[i] == i * 10  # z[2], z[3] overlap with b2
	end

	err3 = try
		b2 + b5
		nothing
	catch e
		e
	end
	@test err3 isa ErrorException
	@test occursin("2 endogenous variable(s)", err3.msg)
	@test occursin("elements", err3.msg)  # Groups show count when >1

	# Large indexed variable overlap - verify error is readable
	@variable(m, big_var[1:100, 1:100])
	b6 = @block m begin
		big_var[i ∈ 1:50, j ∈ 1:100], big_var[i,j] == i + j
	end
	b7 = @block m begin
		big_var[i ∈ 25:75, j ∈ 1:100], big_var[i,j] == i * j  # 26*100=2600 overlap with b6
	end

	err4 = try
		b6 + b7
		nothing
	catch e
		e
	end
	@test err4 isa ErrorException
	@test occursin("2600 endogenous variable(s)", err4.msg)
	@test occursin("big_var:", err4.msg)
	@test occursin("elements", err4.msg)
	@test occursin("e.g.,", err4.msg)
	# Error message should NOT be thousands of lines
	@test count('\n', err4.msg) < 20
end

@testset "Trade model definition" begin
  m = Model()
  D = S = 1:2
  JuMP.@variables m begin
		C[D] >= 1e-6 # CES aggregate consumption in country d
		c[D,S] >= 1e-6 # Consumption in country d from country s
		pᶜ[D] >= 1e-6 # CES price index in country d
		w[D] >= 1e-6 # Price of output in country s
		X[S] # Exports of country s
		M[D] # Imports of country d

		σ # Elasticity of substitution
		μ[D,S] # Preference parameter, country d's preference for country s
		y[D] # GDP in country s
		τ[D,S] # Trade cost from country s to country d
  end

  variable, residual, eqs = SquareModels.@_block(m, C[d ∈ D], w[d] * y[d] == pᶜ[d] * C[d])
  @test isa(variable, AbstractVector{VariableRef})
  @test isa(eqs, AbstractVector{Equation})
  variable, residual, eqs = SquareModels.@_block(m, c[d ∈ D, s ∈ S], c[d,s] == μ[d,s] * C[d] * (w[s] / pᶜ[d])^(-σ))
  @test isa(variable, AbstractVector{VariableRef})
  @test isa(eqs, AbstractVector{Equation})

  ert_tuples = [
		SquareModels.@_block(m, C[d ∈ D], w[d] * y[d] == pᶜ[d] * C[d]),
		SquareModels.@_block(m, c[d ∈ D, s ∈ S], c[d,s] == μ[d,s] * C[d] * (w[s] / pᶜ[d])^(-σ)),
		SquareModels.@_block(m, pᶜ[d ∈ D], pᶜ[d] * C[d] == ∑(w[s] * c[d,s] for s ∈ S)),
		SquareModels.@_block(m, w[s ∈ D[2:end]], y[s] == ∑(c[d,s] for d ∈ D)),
		SquareModels.@_block(m, X[s ∈ S], X[s] == ∑(c[d,s] for d ∈ D if d ≠ s)),
		SquareModels.@_block(m, M[d ∈ D], M[d] == ∑(c[d,s] for s ∈ S if d ≠ s))
  ]
  variables = VariableRef[Iterators.flatten([t[1] for t in ert_tuples])...]
  residuals = VariableRef[Iterators.flatten([t[2] for t in ert_tuples])...]
  equations = Equation[Iterators.flatten([t[3] for t in ert_tuples])...]
  @test all(isa.(variables, VariableRef))
  @test all(isa.(equations, Equation))

  Block(m, variables, residuals, Set{VariableRef}(), equations)

  base_model = @block m begin
		C[d ∈ D],
			w[d] * y[d] == pᶜ[d] * C[d]

		c[d ∈ D, s ∈ S],
			c[d,s] == μ[d,s] * C[d] * (w[s] / pᶜ[d])^(-σ)

		pᶜ[d ∈ D],
			pᶜ[d] * C[d] == ∑(w[s] * c[d,s] for s ∈ S)

		w[s ∈ D[2:end]], # We leave out the condition for the first country and set its price to 1
			y[s] == ∑(c[d,s] for d ∈ D)

		X[s ∈ S],
			X[s] == ∑(c[d,s] for d ∈ D if d ≠ s)

		M[d ∈ D],
			M[d] == ∑(c[d,s] for s ∈ S if d ≠ s)
  end
end

@testset "Block diagnostics" begin
	m = Model()
	JuMP.@variables m begin
		x
		y[1:3]
		z
	end

	b1 = @block m begin
		x, x == 1
		y[i ∈ 1:2], y[i] == i
	end

	b2 = @block m begin
		y[i ∈ 2:3], y[i] == i
	end

	@test overlaps(b1, b2)  # y[2] is in both
	@test y[2] ∈ shared_endogenous(b1, b2)
	@test y[1] ∉ shared_endogenous(b1, b2)

	b3 = @block m begin
		z, z == y[1]
	end
	@test !overlaps(b1, b3)  # y[1] is shared, but exogenous in b3
	@test isempty(shared_endogenous(b1, b3))

	# Test summary
	io = IOBuffer()
	summary(io, b1)
	@test occursin("3", String(take!(io)))  # Should mention 3 equations
end

@testset "Edge cases" begin
	m = Model()
	JuMP.@variables m begin
		x
		y[1:5]
		αβγδ_long_name_with_unicode_σ
	end

	@testset "Empty block" begin
		empty = Block(m)
		@test length(empty) == 0
		@test isempty(endogenous(empty))
	end

	@testset "Single equation" begin
		single = @block m begin
			x, x == 1
		end
		@test length(single) == 1
	end

	@testset "Unicode variable names" begin
		b = @block m begin
			αβγδ_long_name_with_unicode_σ, αβγδ_long_name_with_unicode_σ == 1
		end
		@test length(b) == 1
	end

	@testset "Block + empty = Block" begin
		b = @block m begin
			x, x == 1
		end
		empty = Block(m)
		@test length(b + empty) == length(b)
	end
end

@testset "Residual variables" begin
	m = Model()
	JuMP.@variables m begin
		x
		y[1:3]
		z[1:2, [:a, :b]]
	end

	@testset "Scalar residual" begin
		b = @block m begin
			x, x == 5
		end
		# Residual variable should be created
		@test haskey(m, :x_J)
		@test is_fixed(m[:x_J])
		# residuals function should return the residual
		res = residuals(b)
		@test length(res) == 1
		@test name(res[1]) == "x_J"
	end

	@testset "Vector residual" begin
		b = @block m begin
			y[i ∈ 1:3], y[i] == i
		end
		# Residual variable should be created with same shape as original
		@test haskey(m, :y_J)
		@test length(m[:y_J]) == 3
		@test all(is_fixed.(m[:y_J]))
		# residuals function should return all residuals
		res = residuals(b)
		@test length(res) == 3
		@test name(res[1]) == "y_J[1]"
	end

	@testset "Matrix residual" begin
		b = @block m begin
			z[i ∈ 1:2, j ∈ [:a, :b]], z[i, j] == i
		end
		@test haskey(m, :z_J)
		@test size(m[:z_J]) == (2, 2)
		@test all(is_fixed.(m[:z_J]))
		res = residuals(b)
		@test length(res) == 4
	end

	@testset "SparseAxisArray residual" begin
		m_sparse = Model()
		@variable(m_sparse, s[i=1:3, j=1:3; i != j])
		@test s isa SparseAxisArray

		b = @block m_sparse begin
			s[i ∈ 1:3, j ∈ 1:3; i != j], s[i, j] == i + j
		end
		@test haskey(m_sparse, :s_J)
		@test m_sparse[:s_J] isa SparseAxisArray
		@test length(residuals(b)) == 6
		@test all(is_fixed.(m_sparse[:s_J]))
	end

	@testset "Partial index range uses full residual" begin
		# Create a fresh model
		m2 = Model()
		@variable(m2, w[1:5])

		# Define block with subset of indices
		b1 = @block m2 begin
			w[i ∈ 1:3], w[i] == i
		end
		# Residual should have full shape of original variable
		@test haskey(m2, :w_J)
		@test length(m2[:w_J]) == 5

		# Second block with different indices should reuse residual
		b2 = @block m2 begin
			w[i ∈ 4:5], w[i] == i
		end
		@test length(m2[:w_J]) == 5  # Still same size
	end

	@testset "Residual substitution in different equation positions" begin
		@testset "Endo on LHS (simple)" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			JuMP.@variables m begin
				GDP
				C
				I
				G
			end
			b1 = @block m begin
				GDP, GDP == C + I + G
			end
			@test haskey(m, :GDP_J)
			db = ModelDictionary(m)
			db[GDP] = 100.0; db[C] = 100.0; db[I] = 50.0; db[G] = 30.0
			db[m[:GDP_J]] = 0.0
			@endo_exo_swap!(b1, m[:GDP_J], GDP)
			result = solve(b1, db)
			@test result[m[:GDP_J]] ≈ 80 atol=1e-6
		end

		@testset "Endo with coefficient (not first term)" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			@variable(m, a)
			b2 = @block m begin
				a, 2 * a == 10
			end
			@test haskey(m, :a_J)
			db = ModelDictionary(m)
			db[a] = 4.0; db[m[:a_J]] = 0.0
			@endo_exo_swap!(b2, m[:a_J], a)
			result = solve(b2, db)
			@test result[m[:a_J]] ≈ 1 atol=1e-6
		end

		@testset "Endo appears multiple times" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			@variable(m, b[1:2])
			b3 = @block m begin
				b[i ∈ 1:2], b[i] + b[i] == 4
			end
			@test haskey(m, :b_J)
			db = ModelDictionary(m)
			db[b] .= 0.0; db[m[:b_J]] .= 0.0
			@endo_exo_swap!(b3, m[:b_J], b)
			result = solve(b3, db)
			@test all(result[m[:b_J][i]] ≈ 2 for i in 1:2)
		end

		@testset "Endo in complex expression (power)" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			@variable(m, c[1:2])
			b4 = @block m begin
				c[i ∈ 1:2], c[i]^2 + c[i] == 6
			end
			@test haskey(m, :c_J)
			db = ModelDictionary(m)
			db[c] .= 0.0; db[m[:c_J]] .= 2.0
			@endo_exo_swap!(b4, m[:c_J], c)
			result = solve(b4, db)
			@test all(result[m[:c_J][i]] ≈ 2 for i in 1:2)
		end

		@testset "Endo on RHS" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			JuMP.@variables m begin
				GDP
				C
				I
			end
			b5 = @block m begin
				GDP, C + I == GDP
			end
			@test haskey(m, :GDP_J)
			db = ModelDictionary(m)
			db[GDP] = 100.0; db[C] = 100.0; db[I] = 50.0; db[m[:GDP_J]] = 0.0
			@endo_exo_swap!(b5, m[:GDP_J], GDP)
			result = solve(b5, db)
			@test result[m[:GDP_J]] ≈ 50 atol=1e-6
		end

		@testset "Lagged self-reference" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			@variable(m, x[1:3])
			b6 = @block m begin
				x[t ∈ 2:3], x[t] == x[t-1] + 1
			end
			@test haskey(m, :x_J)
			db = ModelDictionary(m)
			db[x] .= [10.0, 20.0, 30.0]; db[m[:x_J]] .= 0.0
			@endo_exo_swap!(b6, m[:x_J][2:3], x[2:3])
			result = solve(b6, db)
			@test result[m[:x_J][2]] ≈ -9 atol=1e-6
			@test result[m[:x_J][3]] ≈ -9 atol=1e-6
		end

		@testset "Lagged self-reference with additional leading index" begin
			m = Model(Ipopt.Optimizer)
			set_silent(m)
			@variable(m, x[1:2, 1:3])
			b7 = @block m begin
				x[s ∈ 1:2, t ∈ 2:3], x[s, t] == x[s, t-1] + 1
			end
			@test haskey(m, :x_J)
			db = ModelDictionary(m)
			db[x] .= [10.0 20.0 30.0; 40.0 50.0 60.0]; db[m[:x_J]] .= 0.0
			@endo_exo_swap!(b7, vec(m[:x_J][:, 2:3]), vec(x[:, 2:3]))
			result = solve(b7, db)
			@test result[m[:x_J][1, 2]] ≈ -9 atol=1e-6
			@test result[m[:x_J][1, 3]] ≈ -9 atol=1e-6
			@test result[m[:x_J][2, 2]] ≈ -9 atol=1e-6
			@test result[m[:x_J][2, 3]] ≈ -9 atol=1e-6
		end
	end

	@testset "residuals(model) collects all residuals" begin
		m = Model()
		@variable(m, x)
		@variable(m, y[1:3])
		@variable(m, z)

		b1 = @block m begin
			x, x == 1
			y[i ∈ 1:3], y[i] == i
		end

		b2 = @block m begin
			z, z == 5
		end

		all_res = residuals(m)
		@test length(all_res) == 5  # x_J, y_J[1:3], z_J
		@test m[:x_J] ∈ all_res
		@test m[:z_J] ∈ all_res
		@test all(m[:y_J][i] ∈ all_res for i in 1:3)
	end

end

@testset "equations are stored" begin
	m = Model()
	@variable(m, x)
	@variable(m, y[1:3])

	b = @block m begin
		x, x == 1
		y[i ∈ 1:3], y[i] == i
	end

	# Verify equations are stored - one per endogenous variable (4 total: 1 for x + 3 for y)
	@test length(b.equations) == 4
	@test all(isa.(b.equations, Equation))
end

@testset "@test_constraint stores and tests constraints outside the solve system" begin
	m = Model(Ipopt.Optimizer)
	set_silent(m)
	industries = 1:2
	periods = 1:2
	JuMP.@variables m begin
		a_i[industries, periods]
		b_i[industries, periods]
		c_i[industries, periods]
		a[periods]
		b[periods]
		c[periods]
		observed_a[periods]
	end

	block = @block m begin
		a_i[i = industries, t = periods], a_i[i, t] == b_i[i, t] + c_i[i, t]
		a[t = periods], a[t] == b[t] + c[t]
		@test_constraint "a aggregation" a[t = periods], a[t] == sum(a_i[i, t] for i in industries)
		b[t = periods], b[t] == sum(b_i[i, t] for i in industries)
		c[t = periods], c[t] == sum(c_i[i, t] for i in industries)
	end

	@test length(block) == 10
	@test length(test_constraints(block)) == 2
	@test length(block.equations) == length(block.endogenous) == length(block.residuals)
	@test all(c -> c isa TestConstraint && c.message == "a aggregation", test_constraints(block))
	@test [c.variable for c in test_constraints(block)] == collect(a)
	@test occursin("2 test constraints", sprint(summary, block))

	data = ModelDictionary(m, 0.0)
	data[b_i] .= [1.0 3.0; 2.0 4.0]
	data[c_i] .= [5.0 7.0; 6.0 8.0]
	solution = solve(block, data)
	@test assert_test_constraints(block, solution)

	solution[a_i[1, 1]] += 1.0
	err = try
		assert_test_constraints(block, solution; msg="Block test constraints failed")
		nothing
	catch e
		e
	end
	@test err isa TestConstraintError
	@test err.data === solution
	@test only(err.violations)[1] == "a[1]"
	@test occursin("Block test constraints failed", sprint(showerror, err))
	@test occursin("a aggregation", sprint(showerror, err))
	solution[a_i[1, 1]] -= 1.0

	m_bad = Model(Ipopt.Optimizer)
	set_silent(m_bad)
	@variable(m_bad, x)
	failing_block = @block m_bad begin
		x, x == 1
		@test_constraint "automatic test constraint" x, x == 1.01
	end
	failing_data = ModelDictionary(m_bad, 0.0)
	solve_error = try
		solve(failing_block, failing_data)
		nothing
	catch e
		e
	end
	@test solve_error isa TestConstraintError
	@test solve_error.data !== failing_data
	@test solve_error.data[x] ≈ 1 atol=1e-6
	@test solve(failing_block, failing_data; run_test_constraints=false)[x] ≈ 1 atol=1e-6
	@test solve(failing_block, failing_data; presolve_diagnostics=false, run_test_constraints=false)[x] ≈ 1 atol=1e-6
	@test solve(failing_block, failing_data; test_constraint_atol=0.02)[x] ≈ 1 atol=1e-6
	@test solve(failing_block, failing_data; test_constraint_rtol=0.02)[x] ≈ 1 atol=1e-6
	@test_throws TestConstraintError solve!(failing_block, failing_data)
	@test failing_data[x] ≈ 1 atol=1e-6

	test_message = "observed aggregation"
	test_constraint_only = @block m begin
		SquareModels.@test_constraint test_message observed_a[t = periods], observed_a[t] == sum(a_i[i, t] for i in industries)
	end
	@test isempty(endogenous(test_constraint_only))
	@test isempty(test_constraint_only.equations)
	@test isempty(test_constraint_only.variables)
	@test length(test_constraints(test_constraint_only)) == 2
	@test all(c -> c.message == test_message, test_constraints(test_constraint_only))
	@test Set(test_constraint_variables(test_constraint_only)) == Set(vcat(collect(observed_a), vec(collect(a_i))))
	@test !haskey(m, :observed_a_J)

	solution[observed_a] .= solution[a]
	@test assert_test_constraints(test_constraint_only, solution)
	@test length(test_constraints(copy(block))) == 2
	@test length(test_constraints(block + test_constraint_only)) == 4
	@test length(test_constraints((block + test_constraint_only) - test_constraint_only)) == 2

	rebuilt_test_constraints = @block m begin
		@test_constraint "a aggregation" a[t = periods], a[t] == sum(a_i[i, t] for i in industries)
	end
	@test length(test_constraints(block + rebuilt_test_constraints)) == 4
	@test length(test_constraints(block - rebuilt_test_constraints)) == 2
	@test test_constraints((block + rebuilt_test_constraints) - rebuilt_test_constraints) == test_constraints(block)

	different_message = @block m begin
		@test_constraint "other aggregation" a[t = periods], a[t] == sum(a_i[i, t] for i in industries)
	end
	@test length(test_constraints(block - different_message)) == 2

	scaled_solution = copy(solution)
	scaled_solution[observed_a[1]] = 1e8
	scaled_solution[a_i[1, 1]] = 1e8
	scaled_solution[a_i[2, 1]] = 0.5
	@test assert_test_constraints(test_constraint_only, scaled_solution)
	@test_throws TestConstraintError assert_test_constraints(test_constraint_only, scaled_solution; rtol=0.0)

	nonfinite = @block m_bad begin
		@test_constraint "finite failure" x, x == 2
		@test_constraint "NaN failure" x, x == NaN
	end
	nonfinite_error = try
		assert_test_constraints(nonfinite, failing_data)
		nothing
	catch e
		e
	end
	@test nonfinite_error isa TestConstraintError
	@test isnan(first(nonfinite_error.violations)[2])

	inequalities = @block m_bad begin
		@test_constraint "lower bound" x, x >= 0.9
		@test_constraint "upper bound" x, x <= 1.1
		@test_constraint "Unicode lower bound" x, x ≥ 0.9
		@test_constraint "Unicode upper bound" x, x ≤ 1.1
	end
	@test assert_test_constraints(inequalities, failing_data)

	failing_inequalities = @block m_bad begin
		@test_constraint "lower bound failure" x, x >= 1.1
		@test_constraint "upper bound failure" x, x <= 0.9
	end
	inequality_error = try
		assert_test_constraints(failing_inequalities, failing_data)
		nothing
	catch e
		e
	end
	@test inequality_error isa TestConstraintError
	@test length(inequality_error.violations) == 2

	manual_residual = @block m begin
		@test_constraint "a aggregation with residual" a[t = periods],
			a[t] + residual(a)[t] == sum(a_i[i, t] for i in industries)
	end
	@test assert_test_constraints(manual_residual, solution)
	@test all(residual(a)[t] in test_constraint_variables(manual_residual) for t in periods)
end

@testset "@block performance" begin
	# Test that @block scales linearly with problem size, not quadratically
	# A 100x100 indexed variable creates 10,000 constraints
	# With the O(n²) bug, this would iterate 100M times; with the fix, only 10K
	m = Model()
	N = 100
	@variable(m, large[1:N, 1:N])
	@variable(m, param[1:N, 1:N])

	# Warm-up compilation run with smaller size
	m_warmup = Model()
	@variable(m_warmup, w[1:5, 1:5])
	@variable(m_warmup, wp[1:5, 1:5])
	@block m_warmup begin
		w[i ∈ 1:5, j ∈ 1:5], w[i,j] == wp[i,j]
	end

	# Time the actual test - should complete in under 5 seconds with the fix
	# (would take minutes with O(n²) behavior)
	t = @elapsed begin
		b = @block m begin
			large[i ∈ 1:N, j ∈ 1:N], large[i,j] == param[i,j] * 2
		end
	end

	@test length(b) == N * N
	@test t < 5.0  # Should be well under 1 second, but allow margin for CI
end

@testset "SparseAxisArray with tuple destructuring" begin
	pairs = [(:a, :b), (:c, :d)]
	pairs_set = Set(pairs)

	@testset "@_block" begin
		m = Model()
		@variable(m, s[i=[:a, :c], d=[:b, :d], t=1:2; (i, d) in pairs_set])
		@test s isa SparseAxisArray

		v, r, cons = SquareModels.@_block(m, s[(i_e, d_e) = pairs, t ∈ 1:2], s[i_e, d_e, t] == 1)
		@test length(v) == 4
		@test all(isa.(v, VariableRef))
	end

	@testset "@block" begin
		m = Model(Ipopt.Optimizer)
		@variable(m, s[i=[:a, :c], d=[:b, :d], t=1:2; (i, d) in pairs_set])

		b = @block m begin
			s[(i_e, d_e) = pairs, t ∈ 1:2], s[i_e, d_e, t] == 1
		end
		@test length(b) == 4
		@test all(is_endogenous(s[i, d, t], b) for (i, d) in pairs for t in 1:2)
	end

	@testset "residual substitution" begin
		m = Model(Ipopt.Optimizer)
		set_silent(m)
		@variable(m, s[i=[:a, :c], d=[:b, :d], t=1:2; (i, d) in pairs_set])

		b = @block m begin
			s[(i_e, d_e) = pairs, t ∈ 1:2], s[i_e, d_e, t] == 5
		end
		@test haskey(m, :s_J)
		@test m[:s_J] isa SparseAxisArray

		db = ModelDictionary(m)
		for (i, d) in pairs, t in 1:2
			db[s[i, d, t]] = 3.0
		end
		db[m[:s_J]] .= 0.0
		@endo_exo_swap!(b, m[:s_J], s)
		result = solve(b, db)
		@test all(result[m[:s_J][i, d, t]] ≈ 2 for (i, d) in pairs for t in 1:2)
	end
end

@testset "DenseAxisArray with tuple indices" begin
	pairs = [(:a, :b), (:c, :d)]

	@testset "@_block" begin
		m = Model()
		@variable(m, y[pairs, 1:2])
		@test y isa DenseAxisArray

		v, r, eqs = SquareModels.@_block(m, y[(i_e, d_e) = pairs, t ∈ 1:2], y[(i_e, d_e), t] == 1)
		@test length(v) == 4
		@test all(isa.(v, VariableRef))
	end

	@testset "@block" begin
		m = Model(Ipopt.Optimizer)
		@variable(m, y[pairs, 1:2])

		b = @block m begin
			y[(i_e, d_e) = pairs, t ∈ 1:2], y[(i_e, d_e), t] == 1
		end
		@test length(b) == 4
		@test all(is_endogenous(y[(i, d), t], b) for (i, d) in pairs for t in 1:2)
	end

	@testset "residual substitution" begin
		m = Model(Ipopt.Optimizer)
		set_silent(m)
		@variable(m, y[pairs, 1:2])

		b = @block m begin
			y[(i_e, d_e) = pairs, t ∈ 1:2], y[(i_e, d_e), t] == 5
		end
		@test haskey(m, :y_J)
		@test m[:y_J] isa DenseAxisArray

		db = ModelDictionary(m)
		for (i, d) in pairs, t in 1:2
			db[y[(i, d), t]] = 3.0
		end
		db[m[:y_J]] .= 0.0
		@endo_exo_swap!(b, m[:y_J], y)
		result = solve(b, db)
		@test all(result[m[:y_J][(i, d), t]] ≈ 2 for (i, d) in pairs for t in 1:2)
	end
end

@testset "add_equation!" begin
	m = Model(Ipopt.Optimizer)
	JuMP.@variables m begin
		x
		y[1:3]
		z
	end

	base = @block m begin
		x, x == 1
	end

	@testset "adds equation to existing block" begin
		add_equation!(base, z, z, 5)
		@test length(base) == 2
		@test is_endogenous(z, base)
		@test z ∈ base
		@test length(base.residuals) == 2
		@test length(base.equations) == 2
	end

	@testset "variables set is updated" begin
		@test z ∈ base.variables
	end

	@testset "rejects duplicate endogenous" begin
		err = try
			add_equation!(base, x, x, 99)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("already endogenous", err.msg)
	end

	@testset "solves correctly" begin
		m2 = Model(Ipopt.Optimizer)
		set_silent(m2)
		JuMP.@variables m2 begin
			a
			b
			c
		end

		blk = @block m2 begin
			a, a == 10
		end
		add_equation!(blk, b, b, 20)
		add_equation!(blk, c, a + b + c, 60)
		@test length(blk) == 3

		db = ModelDictionary(m2, 0.0)
		result = solve(blk, db)
		@test result[a] ≈ 10 atol=1e-6
		@test result[b] ≈ 20 atol=1e-6
		@test result[c] ≈ 30 atol=1e-6
	end

	@testset "residual and equation alignment" begin
		m3 = Model()
		@variable(m3, p)
		@variable(m3, q)

		blk = Block(m3)
		add_equation!(blk, p, p, 1)
		add_equation!(blk, q, q, 2)

		@test length(blk.endogenous) == length(blk.residuals) == length(blk.equations) == 2
		@test name(blk.residuals[1]) == "p_J"
		@test name(blk.residuals[2]) == "q_J"
	end

	@testset "returns block for chaining" begin
		m4 = Model()
		@variable(m4, r)
		@variable(m4, s)

		blk = Block(m4)
		result = add_equation!(blk, r, r, 1)
		@test result === blk
		add_equation!(result, s, s, 2)
		@test length(blk) == 2
	end
end

end # Module
