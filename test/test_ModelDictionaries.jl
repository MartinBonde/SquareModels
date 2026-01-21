module TestModelDictionaries

using Test
using JuMP
using SquareModels
using Dictionaries

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

end # module
