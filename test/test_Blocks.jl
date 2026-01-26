module TestSquareModels

using Test
using JuMP
using SquareModels
using JuMP.Containers: DenseAxisArray
using Ipopt

@testset "copy_variable" begin
	m = Model()
	@variables m begin
		x
		y[1:5]
	end

	J_x = SquareModels.copy_variable("J_x", x)
	@test J_x == m[:J_x]

	J_y = SquareModels.copy_variable("J_y", y)
	@test J_y == m[:J_y]
	@test length(J_y) == length(y)
end

@testset "@_block" begin
	m = Model()
	@variables m begin
		x
		y[1:5]
		z[1:3, [:a, :b]]
		q
	end

	@testset "x" begin
		v1, r1, cons1 = SquareModels.@_block(m, x, x == 1)
		b1 = SquareModels.Block(m, v1, r1, Set{VariableRef}(), cons1)
		@test typeof(v1) <: AbstractVector{VariableRef}
		@test typeof(cons1) <: AbstractVector{ConstraintRef}
		@test length(v1) == length(cons1) == length(b1) == 1
		@test x ∈ b1
	end

	@testset "y[1:4]" begin
		v2, r2, cons2 = SquareModels.@_block(m, y[i ∈ 1:4], y[i] == 1)
		b2 = SquareModels.Block(m, v2, r2, Set{VariableRef}(), cons2)
		@test typeof(v2) <: AbstractVector{VariableRef}
		@test length(v2) == length(cons2) == length(b2) == 4
		@test all(y[i] ∈ b2 for i ∈ 1:4)
	end

	@testset "y[5]" begin
		v3, r3, cons3 = SquareModels.@_block(m, y[i ∈ [5]], y[i] == 1)
		b3 = SquareModels.Block(m, v3, r3, Set{VariableRef}(), cons3)
		@test typeof(v3) <: AbstractVector{VariableRef}
		@test length(v3) == length(cons3) == length(b3) == 1
		@test y[5] ∈ b3
	end

	@testset "z" begin
		t₁ = 1
		T = 3
		v4, r4, cons4 = SquareModels.@_block(m, z[i ∈ t₁:T, j ∈ [:a, :b]], z[i, j] == 1)
		b4 = SquareModels.Block(m, v4, r4, Set{VariableRef}(), cons4)
		@test typeof(v4) <: AbstractVector{VariableRef}
		@test length(v4) == length(cons4) == length(b4) == 6
		@test all(z[i, j] ∈ b4 for i ∈ t₁:T, j ∈ [:a, :b])
	end
end

@testset "@block" begin
	m = Model(Ipopt.Optimizer)
	@variables m begin
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

	optimize!(m)
	for i in [x, y..., z...]
		@test value(i) == 1
	end
end

@testset "solve block" begin
	m = Model(Ipopt.Optimizer)
	@variables m begin
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

	optimize!(m)
	output = Dict(var => value(var) for var in all_variables(m))
	@test all(output[i] == 1 for i in b)
end

@testset "_endo_exo!" begin
	m = Model(Ipopt.Optimizer)
	@variables m begin
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
		SquareModels._endo_exo!(b, x_exo, x, "")
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
		SquareModels._endo_exo!(b, y_exo, y, "")
		fix.(b, 1)
		@test !any(is_fixed.(y))
		@test all(is_fixed.(y_exo))
		unfix(b)
	end
end

@testset "@endo_exo!" begin
	m = Model(Ipopt.Optimizer)
	@variables m begin
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
		@endo_exo!(b, x_exo, x)
		fix.(b, 1)
		@test !is_fixed(x)
		@test is_fixed(x_exo)
		unfix(b)
	end

	@testset "y" begin
		@endo_exo!(b, y_exo, y)
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

		@endo_exo! b begin
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

@testset "@endo_exo! error messages" begin
	m = Model()
	@variables m begin
		x
		y
		z
	end

	b = @block m begin
		x, x == 1
	end

	@testset "variable not in block" begin
		err = try
			@endo_exo!(b, z, y)
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
			@endo_exo!(b, w, x)
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
			@endo_exo!(b_swap, x, y)
			nothing
		catch e
			e
		end
		@test err isa ErrorException
		@test occursin("y is not endogenous", err.msg)
		# Should suggest swap since x ∈ block and y ∈ block.variables
		@test occursin("Did you swap the arguments?", err.msg)
		@test occursin("@endo_exo!(block, y, x)", err.msg)
	end

	@testset "no swap suggestion when unhelpful" begin
		# z is not in any constraint, so swapping wouldn't help
		err = try
			@endo_exo!(b, x, z)
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
		@variables m begin
			a[1:3]
			b[1:5]
		end
		b2 = @block m begin
			a[i ∈ 1:3], a[i] == 1
		end
		err = try
			@endo_exo!(b2, b, a)
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
	@variables m begin
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
	@variables m begin
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
  @variables m begin
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

  variable, residual, cons = SquareModels.@_block(m, C[d ∈ D], w[d] * y[d] == pᶜ[d] * C[d])
  @test isa(variable, AbstractVector{VariableRef})
  @test isa(cons, AbstractVector{ConstraintRef})
  variable, residual, cons = SquareModels.@_block(m, c[d ∈ D, s ∈ S], c[d,s] == μ[d,s] * C[d] * (w[s] / pᶜ[d])^(-σ))
  @test isa(variable, AbstractVector{VariableRef})
  @test isa(cons, AbstractVector{ConstraintRef})

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
  constraints = ConstraintRef[Iterators.flatten([t[3] for t in ert_tuples])...]
  @test all(isa.(variables, VariableRef))
  @test all(isa.(constraints, ConstraintRef))

  Block(m, variables, residuals, Set{VariableRef}(), constraints)

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
	@variables m begin
		x
		y[1:3]
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

	# Test summary
	io = IOBuffer()
	summary(io, b1)
	@test occursin("3", String(take!(io)))  # Should mention 3 equations
end

@testset "Edge cases" begin
	m = Model()
	@variables m begin
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
		# Verify constraint was created with unicode name
		@test haskey(m, :E_αβγδ_long_name_with_unicode_σ)
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
	@variables m begin
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
		# Test residual substitution by fixing endo to wrong value and solving for residual
		# This verifies the substitution (endo + residual) is happening correctly

		@testset "Endo on LHS (simple)" begin
			# GDP == C + I + G  =>  (GDP + GDP_J) == C + I + G
			# Fix GDP=100, C+I+G=180, so GDP_J should be 80
			m = Model(Ipopt.Optimizer)
			@variables m begin
				GDP
				C
				I
				G
			end
			b1 = @block m begin
				GDP, GDP == C + I + G
			end
			@test haskey(m, :GDP_J)
			fix(GDP, 100, force=true)
			fix(C, 100, force=true)
			fix(I, 50, force=true)
			fix(G, 30, force=true)
			unfix(m[:GDP_J])
			optimize!(m)
			# (100 + GDP_J) == 180 => GDP_J == 80
			@test value(m[:GDP_J]) ≈ 80 atol=1e-6
		end

		@testset "Endo with coefficient (not first term)" begin
			# 2 * a == 10  =>  2 * (a + a_J) == 10
			# Fix a=4: 2*(4 + a_J) == 10 => a_J = 1
			m = Model(Ipopt.Optimizer)
			@variable(m, a)
			b2 = @block m begin
				a, 2 * a == 10
			end
			@test haskey(m, :a_J)
			fix(a, 4, force=true)
			unfix(m[:a_J])
			optimize!(m)
			@test value(m[:a_J]) ≈ 1 atol=1e-6
		end

		@testset "Endo appears multiple times" begin
			# b[i] + b[i] == 4  =>  (b[i] + b_J[i]) + (b[i] + b_J[i]) == 4
			# = 2*(b[i] + b_J[i]) == 4
			# Fix b[i]=0: 2*(0 + b_J[i]) == 4 => b_J[i] = 2
			m = Model(Ipopt.Optimizer)
			@variable(m, b[1:2])
			b3 = @block m begin
				b[i ∈ 1:2], b[i] + b[i] == 4
			end
			@test haskey(m, :b_J)
			fix.(b, 0, force=true)
			unfix.(m[:b_J])
			optimize!(m)
			@test all(value.(m[:b_J]) .≈ 2)
		end

		@testset "Endo in complex expression (power)" begin
			# c[i]^2 + c[i] == 6  =>  (c[i] + c_J[i])^2 + (c[i] + c_J[i]) == 6
			# Fix c=0: (0 + c_J)^2 + (0 + c_J) == 6 => c_J^2 + c_J - 6 = 0
			# => (c_J+3)(c_J-2) = 0 => c_J = 2 (positive root)
			m = Model(Ipopt.Optimizer)
			@variable(m, c[1:2])
			b4 = @block m begin
				c[i ∈ 1:2], c[i]^2 + c[i] == 6
			end
			@test haskey(m, :c_J)
			fix.(c, 0, force=true)
			unfix.(m[:c_J])
			set_start_value.(m[:c_J], 2)  # Start near positive root
			optimize!(m)
			@test all(value.(m[:c_J]) .≈ 2)
		end

		@testset "Endo on RHS" begin
			# Equation: C + I == GDP, endo is GDP
			# Transformed: C + I == (GDP + GDP_J)
			# Fix GDP=100, C+I=150: 150 == (100 + GDP_J) => GDP_J = 50
			m = Model(Ipopt.Optimizer)
			@variables m begin
				GDP
				C
				I
			end
			b5 = @block m begin
				GDP, C + I == GDP
			end
			@test haskey(m, :GDP_J)
			fix(GDP, 100, force=true)
			fix(C, 100, force=true)
			fix(I, 50, force=true)
			unfix(m[:GDP_J])
			optimize!(m)
			@test value(m[:GDP_J]) ≈ 50 atol=1e-6
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

@testset "constraint refs are stored" begin
	m = Model()
	@variable(m, x)
	@variable(m, y[1:3])

	b = @block m begin
		x, x == 1
		y[i ∈ 1:3], y[i] == i
	end

	# Verify constraints are stored - one per endogenous variable (4 total: 1 for x + 3 for y)
	@test length(b.constraints) == 4
	@test all(isa.(b.constraints, ConstraintRef))
	# Names can be retrieved directly from constraints
	@test name(b.constraints[1]) == "E_x"
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

end # Module
