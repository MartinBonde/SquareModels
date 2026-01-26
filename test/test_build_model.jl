module TestBuildModel

using Test
using JuMP
using SquareModels
using Ipopt

# Access internal function for testing
const _build_model = SquareModels._build_model

@testset "solve basic" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        Y[1:10]
        K[1:10]
        δ
    end
    
    data = ModelDictionary(model)
    data[δ] = 0.1
    for t in 1:10
        data[Y[t]] = 100.0 + t
        data[K[t]] = 50.0 + t
    end
    
    block = @block model begin
        Y[t = 3:5], Y[t] == K[t] * (1 - δ)
    end
    
    data[residuals(block)] .= 0.0
    
    @test length(block) == 3
    @test length(endogenous(block)) == 3
    
    solution = solve(block, data)
    
    # K[t] * (1 - δ) for t=3,4,5 should be (53, 54, 55) * 0.9
    @test solution[Y[3]] ≈ 53 * 0.9 atol=1e-6
    @test solution[Y[4]] ≈ 54 * 0.9 atol=1e-6
    @test solution[Y[5]] ≈ 55 * 0.9 atol=1e-6
end

@testset "solve with nonlinear constraints" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        x
        y
        a
    end
    
    data = ModelDictionary(model)
    data[a] = 2.0
    data[y] = 3.0
    
    block = @block model begin
        x, x == a * y^2
    end
    
    data[residuals(block)] .= 0.0
    
    solution = solve(block, data)
    
    # x should equal 2 * 3^2 = 18
    @test solution[x] ≈ 18.0 atol=1e-6
end

@testset "solve variable reduction" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        Y[2000:2100]  # 101 time periods
        K[2000:2100]
        C[2000:2100]
        I[2000:2100]
    end
    
    data = ModelDictionary(model)
    for t in 2000:2100
        data[Y[t]] = 100.0
        data[K[t]] = 50.0
        data[C[t]] = 70.0
        data[I[t]] = 30.0
    end
    
    # Count variables before block definition (no residuals yet)
    model_vars_before = length(all_variables(model))
    @test model_vars_before == 404  # 4 * 101 = 404
    
    # Only solve for 11 periods (2030:2040)
    block = @block model begin
        Y[t = 2030:2040], Y[t] == C[t] + I[t]
    end
    
    data[residuals(block)] .= 0.0
    
    # Test internal: intermediate model has only 11 variables
    solve_model, var_map = _build_model(block, data)
    @test length(all_variables(solve_model)) == 11
    
    # Calculate reduction compared to original variables
    reduction = 1 - length(all_variables(solve_model)) / model_vars_before
    @test reduction > 0.9  # More than 90% reduction
end

@testset "solve updates data correctly" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        x
        y
    end
    
    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    
    block = @block model begin
        x, x == 100
        y, y == 200
    end
    
    data[residuals(block)] .= 0.0
    
    solution = solve(block, data)
    
    @test solution[x] ≈ 100.0 atol=1e-6
    @test solution[y] ≈ 200.0 atol=1e-6
    
    # Original data unchanged
    @test data[x] ≈ 1.0 atol=1e-6
    @test data[y] ≈ 2.0 atol=1e-6
end

@testset "solve with start_values" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        x >= 0
        y >= 0
    end
    
    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    
    block = @block model begin
        x, x == 10
        y, y == 20
    end
    
    data[residuals(block)] .= 0.0
    
    # Test internal: start values come from data
    solve_model1, var_map1 = _build_model(block, data)
    @test JuMP.start_value(var_map1[x]) ≈ 1.0 atol=1e-6
    @test JuMP.start_value(var_map1[y]) ≈ 2.0 atol=1e-6
    
    # Create a "previous solution" with different values
    previous_solution = ModelDictionary(model)
    previous_solution[x] = 5.0
    previous_solution[y] = 15.0
    
    # Test internal: start_values overrides data
    solve_model2, var_map2 = _build_model(block, data; start_values=previous_solution)
    @test JuMP.start_value(var_map2[x]) ≈ 5.0 atol=1e-6
    @test JuMP.start_value(var_map2[y]) ≈ 15.0 atol=1e-6
end

@testset "solve transfers bounds" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        x >= 0
        y <= 100
        z >= -10
    end
    set_upper_bound(z, 10)
    
    data = ModelDictionary(model)
    data[x] = 5.0
    data[y] = 50.0
    data[z] = 0.0
    
    block = @block model begin
        x, x == 5
        y, y == 50
        z, z == 0
    end
    
    data[residuals(block)] .= 0.0
    
    # Test internal: bounds are transferred
    solve_model, var_map = _build_model(block, data)
    
    @test has_lower_bound(var_map[x])
    @test lower_bound(var_map[x]) == 0.0
    
    @test has_upper_bound(var_map[y])
    @test upper_bound(var_map[y]) == 100.0
    
    @test has_lower_bound(var_map[z])
    @test has_upper_bound(var_map[z])
    @test lower_bound(var_map[z]) == -10.0
    @test upper_bound(var_map[z]) == 10.0
end

@testset "solve with endo_exo swap" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        Y[1:3]
        μ
    end
    
    data = ModelDictionary(model)
    data[μ] = 1.5
    for t in 1:3
        data[Y[t]] = 10.0 * t
    end
    
    # Base block: Y is endogenous, μ is exogenous
    base_block = @block model begin
        Y[t = 1:3], Y[t] == μ * t
    end
    
    data[residuals(base_block)] .= 0.0
    
    # Calibration: swap μ with Y[1]
    cal_block = copy(base_block)
    @endo_exo! cal_block begin
        μ, Y[1]
    end
    
    # Now μ is endogenous, Y[1] is exogenous
    # Y[1] = μ * 1, so μ = Y[1] = 10
    solution = solve(cal_block, data)
    
    @test solution[μ] ≈ 10.0 atol=1e-6
    @test solution[Y[2]] ≈ 10.0 * 2 atol=1e-6
    @test solution[Y[3]] ≈ 10.0 * 3 atol=1e-6
end

@testset "solve returns new dictionary" begin
    model = Model(Ipopt.Optimizer)
    @variables model begin
        x
        y
        z
    end
    
    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0  # exogenous
    
    block = @block model begin
        x, x == z * 2
        y, y == z * 3
    end
    
    data[residuals(block)] .= 0.0
    
    solution = solve(block, data)
    
    @test solution[x] ≈ 10.0 atol=1e-6
    @test solution[y] ≈ 15.0 atol=1e-6
    @test solution[z] ≈ 5.0 atol=1e-6  # exogenous unchanged
    
    # original data unchanged
    @test data[x] ≈ 1.0 atol=1e-6
    @test data[y] ≈ 2.0 atol=1e-6
end

end # module
