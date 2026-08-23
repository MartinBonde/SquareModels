module TestBuildModel

using Test
using JuMP
using SquareModels
using Ipopt
const MOI = JuMP.MOI

# Access internal function for testing
const _build_model = SquareModels._build_model

@testset "solve basic" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
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
    JuMP.@variables model begin
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
    JuMP.@variables model begin
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

@testset "_build_model names solve constraints after endogenous variables" begin
    model = Model(() -> MOI.Utilities.Model{Float64}())
    JuMP.@variables model begin
        x
        y
        z
    end

    data = ModelDictionary(model)
    data[z] = 2.0

    block = @block model begin
        x, x == z
        y, y == z^2
    end
    data[residuals(block)] .= 0.0

    solve_model, _ = _build_model(block, data)
    @test SquareModels._solve_equation_names(solve_model) == name.(endogenous(block))

    swapped = copy(block)
    @endo_exo_swap! swapped begin
        residuals(swapped)[1], x
    end

    data[x] = 1.0
    solve_model, _ = _build_model(swapped, data)
    equation_names = SquareModels._solve_equation_names(solve_model)
    @test equation_names == name.(endogenous(swapped))
    @test name(residuals(swapped)[1]) in equation_names
end

@testset "solve updates data correctly" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
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
    JuMP.@variables model begin
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
    JuMP.@variables model begin
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
    JuMP.@variables model begin
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
    @endo_exo_swap! cal_block begin
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
    JuMP.@variables model begin
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

@testset "diagnose: no false positive after endo-exo swap" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        z
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 3.0

    block = @block model begin
        x, x == 5
        y, y == z * 2
    end
    data[residuals(block)] .= 0.0

    # Exogenize x (make residual x_J endogenous), so the equation for x becomes:
    # x_value + x_J == 5 → after substitution, x_J == 5 - 1 = 4, still has a variable (x_J)
    # Instead, we need to create a truly trivial equation. Swap y out, z in:
    cal_block = copy(block)
    @endo_exo_swap! cal_block begin
        z, y
    end
    # Now: eq1 is "x + x_J == 5" (x endo) — fine
    #      eq2 is "y_data + y_J == z * 2" (z endo) — fine
    trivial, orphans = diagnose(cal_block, data)
    @test isempty(trivial)
    @test isempty(orphans)
end

@testset "diagnose: no false positive after calibration swap" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        a
    end

    data = ModelDictionary(model)
    data[x] = 5.0
    data[y] = 10.0
    data[a] = 0.0  # multiplier is zero

    # x, x == a * y  → after substitution: x + x_J == 0*10 = 0
    # But x is endogenous, so it stays as a variable. That's not trivial.
    # To get a truly trivial equation, we need all variables in the equation to be exogenous.
    # Exogenize x, endogenize x_J:
    block = @block model begin
        x, x == a * y
        y, y == 10
    end
    data[residuals(block)] .= 0.0

    cal_block = copy(block)
    @endo_exo_swap! cal_block begin
        residuals(cal_block)[1], x
    end
    # Now eq1: "x_data + x_J == a * y" → x_J is endogenous, a and y are exogenous
    # After sub: x_J_solve == 0*10 - 5 = -5. x_J is still a variable, not trivial.

    # For a truly trivial equation: all vars in the equation become exogenous after swap.
    # This happens when the endo var doesn't actually appear in its own equation.
    # Build manually: endo=y, equation="a == a" (where a is exogenous)
    model2 = Model(Ipopt.Optimizer)
    JuMP.@variables model2 begin
        p
        q
        c
    end
    data2 = ModelDictionary(model2)
    data2[p] = 1.0
    data2[q] = 2.0
    data2[c] = 3.0

    block2 = @block model2 begin
        p, p == c
        q, q == c
    end
    data2[residuals(block2)] .= 0.0

    # Exogenize both p and q — their residuals become endo
    cal2 = copy(block2)
    @endo_exo_swap! cal2 begin
        residuals(cal2)[1], p
        residuals(cal2)[2], q
    end
    # eq1: "p_data + p_J == c" → p_J_solve == 3 - 1 = 2 (has variable p_J, not trivial)
    trivial2, orphans2 = diagnose(cal2, data2)
    @test isempty(trivial2)
end

@testset "diagnose detects orphan variable" begin
    # z is endogenous but doesn't appear in its own equation (or any equation)
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        z
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 3.0

    b1 = @block model begin
        x, x == 10
    end
    data[residuals(b1)] .= 0.0

    # Pair z with an equation that only references x and y (z never appears)
    add_equation!(b1, z, y - x, 0)
    data[residuals(b1)] .= 0.0
    block = b1

    # eq2 is "y - x - z_J == 0". After substitution, z_J is exogenous (fixed=0),
    # y and x are mixed: x is endogenous (in eq1), y is exogenous.
    # Result: eq2 still has variable x, so it's NOT trivial.
    # But z is endogenous and appears in NO equation — it's an orphan.
    trivial, orphans = diagnose(block, data)
    @test isempty(trivial)
    orphan_names = [name(o.endogenous) for o in orphans]
    @test "z" in orphan_names
end

@testset "diagnose detects truly trivial equation" begin
    # Build a case where an equation reduces to a constant.
    # This happens when the endogenous variable doesn't appear in the equation,
    # AND all variables in the equation are exogenous.
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        a
        b
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[a] = 3.0
    data[b] = 4.0

    b1 = @block model begin
        x, x == 10
    end
    data[residuals(b1)] .= 0.0

    # Pair y with an equation involving only exogenous a and b
    add_equation!(b1, y, a - b, 0)  # a - b - y_J == 0, y is endo but not in eq
    data[residuals(b1)] .= 0.0
    block = b1

    # eq2: "a - b - y_J == 0". a, b, y_J are all exogenous.
    # After substitution: 3 - 4 + 0 = -1 (constant). Truly trivial.
    # y is endogenous but appears in no equation → orphan.
    trivial, orphans = diagnose(block, data)
    @test length(trivial) == 1
    @test name(trivial[1].endogenous) == "y"
    @test trivial[1].constant_value ≈ -1.0

    orphan_names = [name(o.endogenous) for o in orphans]
    @test "y" in orphan_names
end

@testset "solve errors on orphan variable" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        z
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 3.0

    b1 = @block model begin
        x, x == 10
    end
    data[residuals(b1)] .= 0.0

    add_equation!(b1, z, y - x, 0)
    data[residuals(b1)] .= 0.0
    block = b1

    err = try
        solve(block, data)
        nothing
    catch e
        e
    end
    @test err isa NonSquareError
    compact = sprint(show, err)
    @test count('\n', compact) == 0
    @test length(compact) < 200
    @test occursin("not effectively square", compact)
    @test !occursin("\\n", compact)
    full = sprint(showerror, err)
    @test occursin('\n', full)
    @test occursin('┌', full)
    @test occursin("z", full)

    # Disabling pre-solve diagnostics allows it through (solver may still fail)
    solve_model, var_map = _build_model(block, data; presolve_diagnostics=false)
    @test length(var_map) == 2
end

@testset "diagnose detects effectively trivial equation from zero coefficient" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        a
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[a] = 0.0

    block = @block model begin
        x, a * x == 5
    end
    data[residuals(block)] .= 0.0

    # a=0 zeroes out x: equation becomes 0 == 5 (infeasible trivial)
    trivial, orphans = diagnose(block, data)
    @test length(trivial) == 1
    @test name(trivial[1].endogenous) == "x"
    @test trivial[1].constant_value ≈ -5.0

    orphan_names = [name(o.endogenous) for o in orphans]
    @test "x" in orphan_names
end

@testset "diagnose detects effectively orphaned variable from zero coefficient" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        z
        a
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[z] = 3.0
    data[a] = 0.0

    block = @block model begin
        x, x == 10
        z, x + a * z == 5
    end
    data[residuals(block)] .= 0.0

    # eq2: a=0 zeroes out z, but x keeps the equation non-trivial → z is orphaned
    trivial, orphans = diagnose(block, data)
    @test isempty(trivial)
    @test length(orphans) == 1
    @test name(orphans[1].endogenous) == "z"
end

@testset "diagnose detects orphan from zero multiplication in nonlinear expression" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        z
        a
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[z] = 3.0
    data[a] = 0.0

    b1 = @block model begin
        x, x == 10
    end
    data[residuals(b1)] .= 0.0

    # z's equation involves a * sin(z), but a=0 → sin(z) branch is zeroed out
    add_equation!(b1, z, x + a * sin(z), 5)
    data[residuals(b1)] .= 0.0
    block = b1

    trivial, orphans = diagnose(block, data)
    @test isempty(trivial)
    @test length(orphans) == 1
    @test name(orphans[1].endogenous) == "z"
end

@testset "diagnose: no false positive with nonzero coefficient" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        z
        a
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[z] = 3.0
    data[a] = 2.0

    block = @block model begin
        x, x == 10
        z, x + a * z == 5
    end
    data[residuals(block)] .= 0.0

    trivial, orphans = diagnose(block, data)
    @test isempty(trivial)
    @test isempty(orphans)
end

@testset "solve errors on effectively orphaned variable" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        z
        a
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[z] = 3.0
    data[a] = 0.0

    block = @block model begin
        x, x == 10
        z, x + a * z == 5
    end
    data[residuals(block)] .= 0.0

    @test_throws NonSquareError solve(block, data)

    # Disabling pre-solve diagnostics lets it through
    solve_model, var_map = _build_model(block, data; presolve_diagnostics=false)
    @test length(var_map) == 2
end

@testset "solve succeeds on healthy model (diagnostics enabled)" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        z
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0

    block = @block model begin
        x, x == z * 2
        y, y == z * 3
    end
    data[residuals(block)] .= 0.0

    solution = solve(block, data)
    @test solution[x] ≈ 10.0 atol=1e-6
    @test solution[y] ≈ 15.0 atol=1e-6
end

@testset "solve errors on failed solver status" begin
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, x >= 1)

    data = ModelDictionary(model)
    data[x] = 1.0

    block = @block model begin
        x, x == 0
    end
    data[residuals(block)] .= 0.0

    @test_throws ErrorException solve(block, data)
end

@testset "solve auto-initializes missing residuals" begin
    model = Model(Ipopt.Optimizer)
    JuMP.@variables model begin
        x
        y
        z
    end

    # Block 1: calibration step producing a ModelDictionary
    block1 = @block model begin
        x, x == 10
    end

    data = ModelDictionary(model)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0
    data[residuals(block1)] .= 0.0
    calibrated = solve(block1, data)

    # Block 2: uses variables from block1's result, but its residuals are missing from calibrated
    block2 = @block model begin
        y, y == z * 3
    end

    # This should work without manually setting residuals
    solution = solve(block2, calibrated)
    @test solution[y] ≈ 15.0 atol=1e-6
end

end # module
