module TestIntegration

using Test
using JuMP
using SquareModels
using Ipopt

@testset "Market Clearing" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variables m begin
        p >= 0.01  # Price
        q >= 0     # Quantity
        α          # Demand intercept
        β          # Demand slope
        γ          # Supply intercept
        δ          # Supply slope
    end

    # Supply = Demand equilibrium
    block = @block m begin
        p, q == α - β * p      # Demand
        q, q == γ + δ * p      # Supply
    end

    # Set parameters
    fix(α, 100)
    fix(β, 2)
    fix(γ, 10)
    fix(δ, 3)

    set_start_value.(block, 10.0)
    optimize!(m)

    # Verify equilibrium: α - β*p = γ + δ*p → p = (α-γ)/(β+δ)
    expected_p = (100 - 10) / (2 + 3)
    @test value(p) ≈ expected_p atol=1e-6
    @test value(q) ≈ (100 - 2 * expected_p) atol=1e-6
end

@testset "Calibration round-trip" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variables m begin
        Y >= 0.01  # Output
        L >= 0.01  # Labor
        A          # Productivity (to calibrate)
    end

    block = @block m begin
        Y, Y == A * L
        L, L == 100  # Labor supply fixed
    end

    # Calibration phase: given Y and L (fixed), solve for A (free)
    # Y and L are exogenous in calibration, A is endogenous
    fix(Y, 150; force=true)
    fix(L, 100; force=true)
    # A is NOT in the block and not fixed, so it will be solved for

    set_start_value(A, 1.0)
    optimize!(m)

    baseline_A = value(A)
    @test baseline_A ≈ 1.5 atol=1e-6

    # Simulation phase: fix A and L, solve for Y
    # Now Y is endogenous (unfixed) and A, L are exogenous (fixed)
    unfix(Y)
    fix(A, baseline_A)
    # L stays fixed at 100

    set_start_value(Y, 100.0)
    optimize!(m)
    @test value(Y) ≈ 150 atol=1e-6
end

@testset "Comparative statics" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variables m begin
        Y >= 0.01  # Output
        L >= 0.01  # Labor
        K >= 0.01  # Capital
        A          # Productivity
        α          # Capital share
    end

    # Cobb-Douglas production: Y = A * K^α * L^(1-α)
    block = @block m begin
        Y, Y == A * K^α * L^(1 - α)
        L, L == 100
        K, K == 50
    end

    # Set parameters
    fix(A, 1.0)
    fix(α, 0.3)

    set_start_value.(block, 10.0)
    optimize!(m)

    baseline_Y = value(Y)

    # Apply positive productivity shock: 10% increase in A
    fix(A, 1.1)
    optimize!(m)

    shocked_Y = value(Y)

    # Verify: output should increase when productivity increases
    @test shocked_Y > baseline_Y
    @test shocked_Y ≈ 1.1 * baseline_Y atol=1e-6
end

@testset "Two-good exchange economy" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variables m begin
        p1 >= 0.01  # Price of good 1 (numeraire)
        p2 >= 0.01  # Price of good 2
        x1[1:2] >= 0  # Consumption of good 1 by agent i
        x2[1:2] >= 0  # Consumption of good 2 by agent i
        e1[1:2]  # Endowment of good 1 by agent i
        e2[1:2]  # Endowment of good 2 by agent i
        α[1:2]   # Preference parameter for good 1 by agent i
    end

    # Cobb-Douglas utility: agents maximize x1^α * x2^(1-α) subject to budget
    # Optimal demands: x1_i = α_i * I_i / p1, x2_i = (1-α_i) * I_i / p2
    # where I_i = p1 * e1_i + p2 * e2_i (income)

    block = @block m begin
        p1, p1 == 1  # Numeraire
        # Market clearing for good 2
        p2, sum(x2) == sum(e2)
        # Optimal demands
        x1[i ∈ 1:2], x1[i] == α[i] * (p1 * e1[i] + p2 * e2[i]) / p1
        x2[i ∈ 1:2], x2[i] == (1 - α[i]) * (p1 * e1[i] + p2 * e2[i]) / p2
    end

    # Set endowments and preferences
    fix(e1[1], 10); fix(e1[2], 5)
    fix(e2[1], 5); fix(e2[2], 10)
    fix(α[1], 0.6); fix(α[2], 0.4)

    set_start_value.(block, 1.0)
    optimize!(m)

    # Verify market clearing for good 1 (Walras' law implies this holds)
    @test value(x1[1]) + value(x1[2]) ≈ value(e1[1]) + value(e1[2]) atol=1e-5

    # Verify budget constraints hold
    for i in 1:2
        income = value(p1) * value(e1[i]) + value(p2) * value(e2[i])
        expenditure = value(p1) * value(x1[i]) + value(p2) * value(x2[i])
        @test income ≈ expenditure atol=1e-5
    end
end

@testset "Quick Example (examples/quick_example.jl)" begin
    # Run the example file - this tests that the README example works
    include(joinpath(@__DIR__, "..", "examples", "quick_example.jl"))

    # After include, variables are in scope. Verify results.
    @test baseline isa ModelDictionary
    @test shock isa ModelDictionary
    @test shock[Y] > baseline[Y]  # Output should increase with population shock
end

end # module
