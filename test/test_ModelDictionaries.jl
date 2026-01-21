module TestModelDictionaries

using Test
using JuMP
using SquareModels
using Dictionaries
using Parquet2
using DataFrames

model = Model()
vars = @variables model begin
	x
	y[1:5]
	z[1:5, [:a, :b, :c]]
end

all_nothing(x) = all(isnothing.(x))

@testset "Test missing" begin
	b = ModelDictionary(model)
	@test all_nothing(b[x])
	@test all_nothing(b[y])
	@test all_nothing(b[z])
	@test all_nothing(b[y[1]])
	@test all_nothing(b[z[1, :a]])
	@test length(b) == length(all_variables(model))
end

@testset "Test getting and setting single variables refs" begin
	b = ModelDictionary(model)

	b[x] = 1
	@test b[x] == 1
	@test b[x] == b[:x]

	b[y[1]] = 2.0
	@test b[y[1]] == 2.0
	@test b[y[1]] == b[y][1] == b[:y][1]
	b[y][2] = 2.0
	@test b[y[2]] == 2.0

	b[z[1,:a]] = 1 // 3
	@test b[z[1,:a]] == 1 // 3

	@test length(b) == length(all_variables(model))
end

@testset "Test getting and setting variable containers to scalars" begin
	b = ModelDictionary(model)

	b[y] = 2.0
	@test b[y[1]] == 2.0
	@test all(b[y] .== 2.0)
	@test all(b[y] .== b[:y])

	b[z] = 1 // 3
	@test b[z[1,:a]] == 1 // 3
	@test all(b[z] .== 1 // 3)
	@test all(b[z] .== b[:z])

	@test length(b) == length(all_variables(model))
end

@testset "Test setting single variable refs, but getting container" begin
	b = ModelDictionary(model)
	b[y[1]] = 1
	@test !isnothing(b[y[1]])
	@test isnothing(b[y[2]])
	@test all(b[y] .=== [1, [nothing for _ in 2:5]...])
	@test all(b[y] .== b[:y])

	b[z[1, :a]] = 1
	@test !isnothing(b[z[1, :a]])
	@test isnothing(b[z[1, :b]])
	@test isnothing(b[z[2, :a]])
	@test sum(isnothing.(b[z])) == length(z) - 1
	@test all(b[z] .== b[:z])

	@test length(b) == length(all_variables(model))
end

@testset "Test getting and setting variable containers to arrays" begin
	b = ModelDictionary(model)

	b[y] = [1, 2, 3, 4, 5]
	@test all(b[y] .== [1, 2, 3, 4, 5])
	@test all(b[y[1:3]] .== [1, 2, 3])

	v = [i*j for j=1:5, i=1:3]
	b[z] = v
	@test size(b[z]) == size(v)
	@test all(b[z] .== v)
	b[z[:,:c]] = 5 .* [1, 2, 3, 4, 5]
	@test all(b[z[:,:c]] .== 5 .* [1, 2, 3, 4, 5])

	@test length(b) == length(all_variables(model))
end

@testset "Test getting and setting variable containers with array indices" begin
	b = ModelDictionary(model)

	b[y[1:2]] = 1
	@test all(b[y[1:2]] .== 1)

	b[z[1:2, [:a, :b]]] = 1
	@test all(b[z[1:2, [:a, :b]]] .== 1)

	@test length(b) == length(all_variables(model))
end

@testset "Test getting and setting variable containers with Colon indices" begin
	b = ModelDictionary(model)

	b[y[:]] = 1
	@test all(b[y[:]] .== 1)

	b[z[:, :]] = 1
	@test all(b[z[1:end, :]] .== 1)

	@test length(b) == length(all_variables(model))
end

@testset "Test fixing variables" begin
	# Create new model, as we are changing model state
	model = Model()
	vars = @variables model begin
		x
		y[1:5]
		z[1:5, [:a, :b, :c]]
	end

	b = ModelDictionary(model)
	b[x], b[y], b[z] = 1, 1, 1

	# Fix variables
	@test !is_fixed(x)
	fix(x, b)
	@test fix_value(x) == 1.0

	@test !any(is_fixed.(y))
	fix(y[2], b)
	@test fix_value(y[2]) == 1.0
	fix(y, b)
	@test all(fix_value.(y) .== 1.0)

	@test !any(is_fixed.(z))
	fix(model, b)
	@test all(fix_value.(z) .== 1.0)

	@test length(b) == length(all_variables(model))
end

@testset "Test setting start values" begin
	# Create new model, as we are changing model state
	model = Model()
	vars = @variables model begin
		x
		y[1:5]
		z[1:5, [:a, :b, :c]]
	end

	b = ModelDictionary(model)
	b[x], b[y], b[z] = 1, 1, 1

	@test start_value(x) |> isnothing
	set_start_value(x, b)
	@test start_value(x) == 1.0

	@test all(isnothing.(start_value.(y)))
	set_start_value(y[2], b)
	@test start_value(y[2]) == 1.0
	set_start_value(y, b)
	@test all(start_value.(y) .== 1.0)

	@test all(isnothing.(start_value.(z)))
	set_start_value(model, b)
	@test all(start_value.(z) .== 1.0)
end

@testset "Test ∈" begin
	b = ModelDictionary(model)
	@test :x ∈ b

	@test "x" ∈ b
	@test x ∈ b

	@test y[1] ∈ b
	@test y ∈ b

	@test z[1, :a] ∈ b
	@test z ∈ b
end

@testset "Test dot access syntax" begin
	b = ModelDictionary(model)

	b.x = 1
	@test b.x == b[x] == 1

	b[y[1]] = 1
	@test all(b.y .=== b[y])
	@test b.y[1] == b[y[1]] == 1
	b.y[2] = 2
	@test b.y[2] == 2

	b.z[1,:a] = 1 // 3
	@test b.z[1,:a] == 1 // 3

	b = ModelDictionary(model)
	b.y[1:2] = 1
	@test all(b.y[1:2] .== 1)

	b.z[1:2, [:a, :b]] = 1
	@test all(b.z[1:2, [:a, :b]] .== 1)

	b = ModelDictionary(model)
	@test all_nothing(b.y)
	b.y = 1
	@test all(b.y .== 1)

	@test all_nothing(b.z)
	b.z = 1
	@test all(b.z .== 1)

	b = ModelDictionary(model)
	b.y[:] = 1
	@test all(b.y[:] .== 1)

	b.z[1:end, :] = 1
	@test all(b.z[1:end, :] .== 1)

	@test length(b) == length(all_variables(model))
end

@testset "Test adding variables to model" begin
	b = ModelDictionary(model)
	@variable(model, q)
	@test q ∉ b
	@test isnothing(b[q])
	@test q ∈ b
end

@testset "Test broadcasting" begin
	model = Model()
	@variable(model, x)
	@variable(model, y[1:3])

	b = ModelDictionary(model)
	b[x] = 1.0
	b[y] = [2.0, 3.0, 4.0]

	# Scalar operations
	b2 = b .+ 1
	@test b2 isa ModelDictionary
	@test b2[x] == 2.0
	@test all(b2[y] .== [3.0, 4.0, 5.0])

	b3 = 2 .* b
	@test b3 isa ModelDictionary
	@test b3[x] == 2.0
	@test all(b3[y] .== [4.0, 6.0, 8.0])

	# Standard library functions
	b4 = log.(b)
	@test b4 isa ModelDictionary
	@test b4[x] ≈ log(1.0)
	@test all(b4[y] .≈ log.([2.0, 3.0, 4.0]))

	# User-defined functions
	myfunc = x -> x^2 + 1
	b5 = myfunc.(b)
	@test b5 isa ModelDictionary
	@test b5[x] == 2.0
	@test all(b5[y] .== [5.0, 10.0, 17.0])

	# Two dictionaries
	b6 = ModelDictionary(model)
	b6[x] = 10.0
	b6[y] = [20.0, 30.0, 40.0]

	diff = b6 .- b
	@test diff isa ModelDictionary
	@test diff[x] == 9.0
	@test all(diff[y] .== [18.0, 27.0, 36.0])

	# Chained operations
	b7 = (b .+ 1) .* 2
	@test b7 isa ModelDictionary
	@test b7[x] == 4.0

	# Boolean broadcasting and filtering
	b8 = b .> 2
	@test b8 isa ModelDictionary
	@test b8[x] == false
	@test all(b8[y] .== [false, true, true])

	filtered = b[b .> 2]
	@test filtered isa ModelDictionary
	@test length(filtered) == 2
end

@testset "Test subset ModelDictionary" begin
	model = Model()
	@variable(model, x)
	@variable(model, y[1:3])

	b = ModelDictionary(model)
	b[x] = 1.0
	b[y] = [2.0, 3.0, 4.0]

	# Create a subset via filtering
	subset = b[b .> 2]
	@test length(subset) == 2

	# fix(subset) should only fix the variables in the subset
	fix(subset)
	@test !is_fixed(x)
	@test !is_fixed(y[1])
	@test is_fixed(y[2])
	@test is_fixed(y[3])
	@test fix_value(y[2]) == 3.0
	@test fix_value(y[3]) == 4.0

	# Subset should not have been expanded
	@test length(subset) == 2

	# Unfix for next test
	unfix.(y[2:3])

	# set_start_value(subset) should only set start values for subset variables
	set_start_value(subset)
	@test isnothing(start_value(x))
	@test isnothing(start_value(y[1]))
	@test start_value(y[2]) == 3.0
	@test start_value(y[3]) == 4.0

	# Subset should still not have been expanded
	@test length(subset) == 2
end

@testset "Test save and load" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, x)
		@variable(model, y[1:3])
		@variable(model, z[1:2, [:a, :b]])
		@variable(model, σ)  # Unicode variable name

		d = ModelDictionary(model)
		d[x] = 1.5
		d[y] = [2.0, 3.0, 4.0]
		d[z] = [10.0 20.0; 30.0 40.0]
		d[σ] = 0.5

		# Save and load
		path = joinpath(tmpdir, "test.parquet")
		save(path, d)
		@test isfile(path)

		d2 = load(path, model)
		@test d2 isa ModelDictionary
		@test d2[x] == 1.5
		@test all(d2[y] .== [2.0, 3.0, 4.0])
		@test d2[z[1, :a]] == 10.0
		@test d2[z[2, :b]] == 40.0
		@test d2[σ] == 0.5

		# Variables not in the file should be nothing
		@variable(model, new_var)
		d3 = load(path, model)
		@test isnothing(d3[new_var])
	end
end

@testset "Test parse_variable_name" begin
	using SquareModels: parse_variable_name

	# Scalar variable
	@test parse_variable_name("x") == ("x", "")
	@test parse_variable_name("σ") == ("σ", "")

	# Single index
	@test parse_variable_name("y[1]") == ("y", "1")
	@test parse_variable_name("K[2025]") == ("K", "2025")

	# Multiple indices
	@test parse_variable_name("z[1,a]") == ("z", "1,a")
	@test parse_variable_name("cᵃ[15,2025]") == ("cᵃ", "15,2025")

	# Complex indices
	@test parse_variable_name("N[tot,2025]") == ("N", "tot,2025")
	@test parse_variable_name("emissions[energy,dk,2025,coal]") == ("emissions", "energy,dk,2025,coal")
end

@testset "Test save skips nothing values" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, x)
		@variable(model, y[1:3])

		d = ModelDictionary(model)
		d[x] = 1.0
		# y values are left as nothing

		path = joinpath(tmpdir, "test.parquet")
		save(path, d)

		d2 = load(path, model)
		@test d2[x] == 1.0
		@test isnothing(d2[y[1]])
		@test isnothing(d2[y[2]])
		@test isnothing(d2[y[3]])
	end
end

@testset "Test load with indices outside model range" begin
	# Create a model with limited index range
	model = Model()
	@variable(model, a[2025:2030])
	@variable(model, b[1:2, 2025:2030])

	# Create Parquet data with indices outside the model's range
	data = DataFrame(
		variable = ["a", "a", "a", "b", "b", "b"],
		indices = ["2024", "2025", "2100", "1,2025", "1,2024", "3,2025"],
		value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	)

	mktempdir() do tmpdir
		path = joinpath(tmpdir, "test.parquet")
		Parquet2.writefile(path, data)

		# Load should skip indices that don't exist in model
		d = load(path, model)

		# Only a[2025] and b[1,2025] should be loaded
		@test d[a[2025]] == 2.0
		@test d[b[1, 2025]] == 4.0

		# Other valid model indices should be nothing (not in data or outside data range)
		@test isnothing(d[a[2026]])
		@test isnothing(d[b[2, 2025]])
	end
end

@testset "Test load with partial data" begin
	model = Model()
	@variable(model, x)
	@variable(model, y[1:2])

	mktempdir() do tmpdir
		# Data only has x, not y
		data = DataFrame(
			variable = ["x"],
			indices = [""],
			value = [1.0]
		)
		path = joinpath(tmpdir, "partial.parquet")
		Parquet2.writefile(path, data)

		# Missing variables are nothing
		d = load(path, model)
		@test d[x] == 1.0
		@test isnothing(d[y[1]])
		@test isnothing(d[y[2]])
	end
end

end # module
