module TestModelDictionaries

using Test
using JuMP
using SquareModels
using Dictionaries
using Parquet2
using DataFrames

# GDX tests require the GAMS runtime - skip in CI
const IN_CI = get(ENV, "CI", "false") == "true"
using GAMS: write_gdx

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

@testset "Test add_missing_model_variables!" begin
	# Create a new model and dictionary
	model2 = Model()
	@variable(model2, a)
	@variable(model2, b[1:3])

	d = ModelDictionary(model2)
	@test length(d) == 4  # a + b[1:3]

	# Add new variables to the model
	@variable(model2, c)
	@variable(model2, d_var[1:2])

	# New variables are not yet in dictionary
	@test :c ∉ keys(d.dictionary)
	@test Symbol("d_var[1]") ∉ keys(d.dictionary)

	# Explicitly sync the dictionary
	add_missing_model_variables!(d)

	# Now they should be present (with nothing values)
	@test :c ∈ keys(d.dictionary)
	@test Symbol("d_var[1]") ∈ keys(d.dictionary)
	@test Symbol("d_var[2]") ∈ keys(d.dictionary)
	@test isnothing(d[c])
	@test isnothing(d[d_var[1]])
	@test length(d) == 7  # a + b[1:3] + c + d_var[1:2]
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

@testset "Test unload and load" begin
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
		unload(path, d)
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

@testset "Test unload skips nothing values" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, x)
		@variable(model, y[1:3])

		d = ModelDictionary(model)
		d[x] = 1.0
		# y values are left as nothing

		path = joinpath(tmpdir, "test.parquet")
		unload(path, d)

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

@testset "Test load with renames" begin
	model = Model()
	@variable(model, Y[1:3])
	@variable(model, X)
	@variable(model, Z)

	mktempdir() do tmpdir
		# Data has different names than model
		data = DataFrame(
			variable = ["OtherY", "OtherY", "OtherY", "DataX"],
			indices = ["1", "2", "3", ""],
			value = [10.0, 20.0, 30.0, 100.0]
		)
		path = joinpath(tmpdir, "renamed.parquet")
		Parquet2.writefile(path, data)

		# Test Pair syntax
		d = load(path, model, Y => "OtherY", X => "DataX")
		@test d[Y[1]] == 10.0
		@test d[Y[2]] == 20.0
		@test d[Y[3]] == 30.0
		@test d[X] == 100.0
		@test isnothing(d[Z])

		# Test keyword syntax
		d2 = load(path, model; Y="OtherY", X="DataX")
		@test d2[Y[1]] == 10.0
		@test d2[X] == 100.0

		# Test Symbol key in Pair
		d3 = load(path, model, :Y => "OtherY")
		@test d3[Y[1]] == 10.0
	end
end

@testset "Test load with renames - Gekko format" begin
	# Test with the Gekko format test file
	baseline_path = joinpath(@__DIR__, "test_gekko_format.parquet")
	if isfile(baseline_path)
		model = Model()
		@variable(model, N_a[0:5, 2029:2031])

		d = load(baseline_path, model; N_a="nPop")

		# nPop data exists for ages 0+ and years 2029+
		@test d[N_a[0, 2029]] ≈ 63.861139166386145
		@test d[N_a[1, 2029]] ≈ 63.326547140764674
		@test !isnothing(d[N_a[5, 2031]])
	end
end

@testset "Test load with renames - mismatched indices" begin
	model = Model()
	# Model has indices 5:7, data will have indices 1:10
	@variable(model, Y[5:7])

	mktempdir() do tmpdir
		# Data has more indices than model (1:10 vs model's 5:7)
		data = DataFrame(
			variable = ["DataY", "DataY", "DataY", "DataY", "DataY", "DataY", "DataY", "DataY", "DataY", "DataY"],
			indices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
			value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
		)
		path = joinpath(tmpdir, "mismatch.parquet")
		Parquet2.writefile(path, data)

		d = load(path, model, Y => "DataY")

		# Only model indices 5, 6, 7 should be loaded
		@test d[Y[5]] == 5.0
		@test d[Y[6]] == 6.0
		@test d[Y[7]] == 7.0

		# Data indices outside model range (1-4, 8-10) are ignored - no error
		@test length(d) == 3
	end
end

@testset "Test load with renames - missing data indices" begin
	model = Model()
	# Model has indices 1:5, data only has 2:3
	@variable(model, Y[1:5])

	mktempdir() do tmpdir
		data = DataFrame(
			variable = ["DataY", "DataY"],
			indices = ["2", "3"],
			value = [20.0, 30.0]
		)
		path = joinpath(tmpdir, "partial.parquet")
		Parquet2.writefile(path, data)

		d = load(path, model, Y => "DataY")

		# Indices with data are loaded
		@test d[Y[2]] == 20.0
		@test d[Y[3]] == 30.0

		# Model indices without data are nothing
		@test isnothing(d[Y[1]])
		@test isnothing(d[Y[4]])
		@test isnothing(d[Y[5]])
	end
end

# =============================================================================
# GDX file tests (only run when GAMS is available)
# =============================================================================

if !IN_CI

@testset "Test load from GDX - basic" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, x)
		@variable(model, y[1:3])

		# Create a GDX file with test data
		scalar_df = DataFrame(value = [1.5])
		indexed_df = DataFrame(dim1 = [1, 2, 3], value = [2.0, 3.0, 4.0])

		path = joinpath(tmpdir, "test.gdx")
		write_gdx(path, "x" => scalar_df, "y" => indexed_df)

		d = load(path, model)
		@test d isa ModelDictionary
		@test d[x] == 1.5
		@test d[y[1]] == 2.0
		@test d[y[2]] == 3.0
		@test d[y[3]] == 4.0
	end
end

@testset "Test load from GDX - multi-dimensional" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, z[1:2, [:a, :b]])

		# Create a GDX file with 2D data
		df = DataFrame(
			dim1 = [1, 1, 2, 2],
			dim2 = ["a", "b", "a", "b"],
			value = [10.0, 20.0, 30.0, 40.0]
		)

		path = joinpath(tmpdir, "test2d.gdx")
		write_gdx(path, "z" => df)

		d = load(path, model)
		@test d[z[1, :a]] == 10.0
		@test d[z[1, :b]] == 20.0
		@test d[z[2, :a]] == 30.0
		@test d[z[2, :b]] == 40.0
	end
end

@testset "Test load from GDX - with renames" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, Y[1:3])
		@variable(model, X)
		@variable(model, Z)

		# Create GDX with different names than model
		scalar_df = DataFrame(value = [100.0])
		indexed_df = DataFrame(dim1 = [1, 2, 3], value = [10.0, 20.0, 30.0])

		path = joinpath(tmpdir, "renamed.gdx")
		write_gdx(path, "DataX" => scalar_df, "OtherY" => indexed_df)

		# Test Pair syntax
		d = load(path, model, Y => "OtherY", X => "DataX")
		@test d[Y[1]] == 10.0
		@test d[Y[2]] == 20.0
		@test d[Y[3]] == 30.0
		@test d[X] == 100.0
		@test isnothing(d[Z])

		# Test keyword syntax
		d2 = load(path, model; Y="OtherY", X="DataX")
		@test d2[Y[1]] == 10.0
		@test d2[X] == 100.0
	end
end

@testset "Test load from GDX - partial data" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, x)
		@variable(model, y[1:3])

		# GDX only has x, not y
		df = DataFrame(value = [1.0])

		path = joinpath(tmpdir, "partial.gdx")
		write_gdx(path, "x" => df)

		d = load(path, model)
		@test d[x] == 1.0
		@test isnothing(d[y[1]])
		@test isnothing(d[y[2]])
		@test isnothing(d[y[3]])
	end
end

@testset "Test load from GDX - indices outside model range" begin
	mktempdir() do tmpdir
		# Model has limited index range
		model = Model()
		@variable(model, a[2025:2027])

		# GDX has indices outside model range
		df = DataFrame(
			dim1 = [2024, 2025, 2026, 2027, 2028],
			value = [1.0, 2.0, 3.0, 4.0, 5.0]
		)

		path = joinpath(tmpdir, "range.gdx")
		write_gdx(path, "a" => df)

		d = load(path, model)

		# Only model indices should be loaded
		@test d[a[2025]] == 2.0
		@test d[a[2026]] == 3.0
		@test d[a[2027]] == 4.0
	end
end

end # if !IN_CI

# Note: GDX files don't support Unicode symbol names (GAMS limitation).
# Use ASCII names in GDX and rename when loading into JuMP models with Unicode names.

# =============================================================================
# Slice specification tests
# =============================================================================

@testset "Test _parse_slice_spec" begin
	using SquareModels: _parse_slice_spec

	# Simple rename (no brackets)
	@test _parse_slice_spec("nPop") == ("nPop", String[], Int[])

	# Single fixed index, single wildcard
	gdx_sym, fixed, wildcards = _parse_slice_spec("vC[:cTot,:]")
	@test gdx_sym == "vC"
	@test fixed == ["cTot"]
	@test wildcards == [2]

	# Multiple fixed indices, single wildcard
	gdx_sym, fixed, wildcards = _parse_slice_spec("vK[:iTot,:tot,:]")
	@test gdx_sym == "vK"
	@test fixed == ["iTot", "tot"]
	@test wildcards == [3]

	# Wildcard first
	gdx_sym, fixed, wildcards = _parse_slice_spec("data[:,:fixed]")
	@test gdx_sym == "data"
	@test fixed == ["fixed"]
	@test wildcards == [1]

	# Multiple wildcards
	gdx_sym, fixed, wildcards = _parse_slice_spec("matrix[:,:,:fixed,:]")
	@test gdx_sym == "matrix"
	@test fixed == ["fixed"]
	@test wildcards == [1, 2, 4]

	# No colon prefix on fixed index
	gdx_sym, fixed, wildcards = _parse_slice_spec("vX[xTot,:]")
	@test gdx_sym == "vX"
	@test fixed == ["xTot"]
	@test wildcards == [2]
end

@testset "Test _build_slice_key" begin
	using SquareModels: _build_slice_key

	# Single wildcard at end: C[2025] -> vC[:cTot,:] -> "cTot,2025"
	@test _build_slice_key("2025", ["cTot"], [2]) == "cTot,2025"

	# Multiple fixed indices: K[2025] -> vK[:iTot,:tot,:] -> "iTot,tot,2025"
	@test _build_slice_key("2025", ["iTot", "tot"], [3]) == "iTot,tot,2025"

	# Wildcard first: X[2025] -> data[:,:fixed] -> "2025,fixed"
	@test _build_slice_key("2025", ["fixed"], [1]) == "2025,fixed"

	# Multiple wildcards: Z[1,2] -> matrix[:,:,:fixed,:] -> "1,2,fixed,?"
	# Target has 2 indices, wildcards at positions 1, 2, 4
	@test _build_slice_key("1,2", ["fixed"], [1, 2, 4]) == "1,2,fixed,"

	# Multi-dimensional target: N_a[15,2025] -> pop[:,:] -> "15,2025"
	@test _build_slice_key("15,2025", String[], [1, 2]) == "15,2025"

	# No wildcards (all fixed)
	@test _build_slice_key("", ["a", "b", "c"], Int[]) == "a,b,c"
end

@testset "Test load with slices - Parquet" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, C[2025:2027])
		@variable(model, K[2025:2027])

		# Data has higher-dimensional structure
		data = DataFrame(
			variable = [
				"vC", "vC", "vC",  # vC[:cTot, t]
				"vC", "vC", "vC",  # vC[:cHh, t] (different slice, should be ignored)
				"vK", "vK", "vK",  # vK[:iTot, :tot, t]
			],
			indices = [
				"cTot,2025", "cTot,2026", "cTot,2027",
				"cHh,2025", "cHh,2026", "cHh,2027",
				"iTot,tot,2025", "iTot,tot,2026", "iTot,tot,2027",
			],
			value = [
				100.0, 110.0, 120.0,
				50.0, 55.0, 60.0,
				1000.0, 1100.0, 1200.0,
			]
		)
		path = joinpath(tmpdir, "sliced.parquet")
		Parquet2.writefile(path, data)

		d = load(path, model;
			C = "vC[:cTot,:]",
			K = "vK[:iTot,:tot,:]",
		)

		# C should get vC[:cTot, t] values
		@test d[C[2025]] == 100.0
		@test d[C[2026]] == 110.0
		@test d[C[2027]] == 120.0

		# K should get vK[:iTot, :tot, t] values
		@test d[K[2025]] == 1000.0
		@test d[K[2026]] == 1100.0
		@test d[K[2027]] == 1200.0
	end
end

if !IN_CI
@testset "Test load with slices - GDX" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, X[2025:2027])

		# Create GDX with higher-dimensional data: vX[commodity, year]
		df = DataFrame(
			dim1 = ["xTot", "xTot", "xTot", "xOther", "xOther", "xOther"],
			dim2 = [2025, 2026, 2027, 2025, 2026, 2027],
			value = [200.0, 220.0, 240.0, 10.0, 11.0, 12.0]
		)

		path = joinpath(tmpdir, "sliced.gdx")
		write_gdx(path, "vX" => df)

		d = load(path, model; X = "vX[:xTot,:]")

		# X should get vX[:xTot, t] values
		@test d[X[2025]] == 200.0
		@test d[X[2026]] == 220.0
		@test d[X[2027]] == 240.0
	end
end
end # if !IN_CI

@testset "Test load with mixed renames and slices" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, N_a[1:3, 2025:2027])  # Simple rename
		@variable(model, C[2025:2027])          # Slice

		# Create Parquet with both types
		rows = NamedTuple{(:variable, :indices, :value), Tuple{String, String, Float64}}[]

		# nPop data (simple rename target)
		for a in 1:3, t in 2025:2027
			push!(rows, (variable="nPop", indices="$a,$t", value=Float64(a * 1000 + t)))
		end

		# vC data (slice target)
		for t in 2025:2027
			push!(rows, (variable="vC", indices="cTot,$t", value=Float64(t * 10)))
		end

		path = joinpath(tmpdir, "mixed.parquet")
		Parquet2.writefile(path, DataFrame(rows))

		d = load(path, model;
			N_a = "nPop",           # Simple rename
			C = "vC[:cTot,:]",      # Slice
		)

		# N_a uses simple rename
		@test d[N_a[1, 2025]] == 1000.0 + 2025
		@test d[N_a[2, 2026]] == 2000.0 + 2026
		@test d[N_a[3, 2027]] == 3000.0 + 2027

		# C uses slice
		@test d[C[2025]] == 20250.0
		@test d[C[2026]] == 20260.0
		@test d[C[2027]] == 20270.0
	end
end

@testset "Test load with slices - partial data" begin
	mktempdir() do tmpdir
		model = Model()
		@variable(model, C[2025:2030])

		# Data only has some years
		data = DataFrame(
			variable = ["vC", "vC", "vC"],
			indices = ["cTot,2025", "cTot,2027", "cTot,2029"],
			value = [100.0, 120.0, 140.0]
		)
		path = joinpath(tmpdir, "partial_slice.parquet")
		Parquet2.writefile(path, data)

		d = load(path, model; C = "vC[:cTot,:]")

		# Only matching years should be loaded
		@test d[C[2025]] == 100.0
		@test isnothing(d[C[2026]])
		@test d[C[2027]] == 120.0
		@test isnothing(d[C[2028]])
		@test d[C[2029]] == 140.0
		@test isnothing(d[C[2030]])
	end
end

end # module
