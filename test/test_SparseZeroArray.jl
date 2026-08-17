using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, fix, unfix, is_fixed, value, set_start_value, set_silent, AffExpr, name
using JuMP.Containers: SparseAxisArray
using Ipopt

@testset "Zero sentinel" begin
    z = SquareModels.Zero()
    @test z + 1 === 1
    @test 1 + z === 1
    @test z + z === z
    @test z * z === z
    @test z * 5 === z
    @test 5 * z === z
    @test -z === z
    @test 3 - z === 3
    @test iszero(z)

    # Unary * (used by MutableArithmetics in JuMP constraint rewriting)
    @test *(z) === z

    # Works with JuMP expressions
    m = Model()
    @variable(m, x)
    @test x + z === x
    @test z + x === x
    expr = AffExpr(1.0, x => 2.0)
    @test expr + z === expr
    @test z + expr === expr
end

@testset "Zero in NonlinearExpr constraint (MutableArithmetics add_mul)" begin
    m = Model(Ipopt.Optimizer)
    @variables m begin
        sparse[i=1:3, j=1:3; i <= j], "Sparse"
        rate[1:3], "Rate parameter"
        result[1:3], "Result"
    end
    @test sparse isa SparseZeroArray

    # JuMP's MutableArithmetics @rewrite decomposes constraint expressions into
    # operate!!(add_mul, accumulator, terms...) calls. When a SparseZeroArray
    # lookup returns Zero() as a standalone additive term alongside a NonlinearExpr
    # (e.g. from division), the call becomes operate!!(add_mul, NonlinearExpr, Zero())
    # with a single vararg. JuMP's NonlinearExpr dispatch does *(args...) = *(Zero()),
    # requiring unary * to be defined on Zero.
    # Note: x*y produces QuadExpr, not NonlinearExpr, and QuadExpr is handled by
    # add_to_expression! dispatches that already accept Zero().
    b = @block m begin
        result[i ∈ 1:3], result[i] / rate[i] + sparse[i, 1] == 0
    end
    @test length(b) == 3
end

@testset "SparseZeroArray construction" begin
    m = Model()
    @variable(m, x[i=1:5, j=[:a, :b]; i <= 3])
    @test x isa SparseAxisArray

    domain = (Set(1:5), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)
    @test sz isa SparseZeroArray
    @test length(sz) == length(x)
end

@testset "SparseZeroArray scalar indexing" begin
    m = Model()
    @variable(m, x[i=1:5, j=[:a, :b]; i <= 3])
    domain = (Set(1:5), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)

    # Existing key returns the variable
    @test sz[1, :a] isa VariableRef
    @test sz[1, :a] === x[1, :a]

    # Missing key within domain returns Zero()
    @test sz[4, :a] isa SquareModels.Zero
    @test sz[5, :b] isa SquareModels.Zero

    # Out-of-domain key throws
    @test_throws ErrorException sz[6, :a]
    @test_throws ErrorException sz[1, :c]
end

@testset "SparseZeroArray slicing" begin
    m = Model()
    @variable(m, x[i=1:3, j=[:a, :b]; j == :a || i <= 2])
    domain = (Set(1:3), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)

    # Colon slicing delegates to underlying SparseAxisArray
    slice = sz[:, :a]
    @test length(slice) == 3
end

@testset "SparseZeroArray forwarded methods" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    @test length(sz) == 6
    @test haskey(sz, (1, 2))
    @test !haskey(sz, (1, 1))
    @test first(sz) isa VariableRef
    @test eltype(typeof(sz)) == VariableRef
    vals = VariableRef[]
    for v in sz
        push!(vals, v)
    end
    @test vals isa Vector{VariableRef}
    @test length(vals) == length(sz)
end

@testset "SparseZeroArray sum with ∑" begin
    m = Model()
    @variable(m, x[i=1:5, j=1:5; i <= 2])
    domain = (Set(1:5), Set(1:5))
    sz = SparseZeroArray(x, domain)

    # Sum over all i for a given j — most i's are missing, should get Zero()
    result = ∑(sz[i, 3] for i in 1:5)
    # Only i=1 and i=2 exist, the rest are Zero()
    @test result isa AffExpr
end

@testset "_all_keys for SparseZeroArray" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    ks = SquareModels._all_keys(sz)
    @test length(ks) == 6
    @test (1, 2) in ks
    @test (1, 1) ∉ ks
end

@testset "copy_variable for SparseZeroArray" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    copied = SquareModels.copy_variable("x_copy", sz)
    @test copied isa SparseAxisArray
    @test length(copied) == length(sz)
    @test haskey(m, :x_copy)
end

@testset "base_name for SparseZeroArray" begin
    m = Model()
    @variable(m, my_var[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(my_var, domain)

    @test SquareModels.base_name(sz) == "my_var"
end

@testset "@variables auto-wrapping" begin
    m = Model()
    t = 1:3

    @variables m begin
        sparse_var[i=1:3, j=1:3; i != j], "A sparse variable"
        dense_var[t], "A dense variable"
        scalar_var, "A scalar"
    end

    @test sparse_var isa SparseZeroArray
    @test dense_var isa JuMP.Containers.DenseAxisArray
    @test scalar_var isa VariableRef

    @test sparse_var[1, 2] isa VariableRef
    @test sparse_var[1, 1] isa SquareModels.Zero
    @test sparse_var[2, 2] isa SquareModels.Zero
    @test_throws ErrorException sparse_var[4, 1]
    @test_throws ErrorException sparse_var[1, 4]
end

@testset "@variables with tuple destructuring" begin
    m = Model()
    pairs = [(:a, :b), (:c, :d)]
    pairs_set = Set(pairs)
    t = 1:3

    @variables m begin
        x[i=[:a, :c], d=[:b, :d], tt=t; (i, d) in pairs_set], "Sparse with condition"
    end

    @test x isa SparseZeroArray
    @test x[:a, :b, 1] isa VariableRef
    # Valid domain but missing combination
    @test x[:a, :d, 1] isa SquareModels.Zero
    # Out of domain
    @test_throws ErrorException x[:z, :b, 1]
end

@testset "SparseZeroArray domain covers full index sets, not just filtered keys" begin
    m = Model()
    I = [:a, :b, :c]
    D = [:x, :y, :z]
    # Only (:a,:x), (:a,:y), (:b,:x) survive the filter
    # :c is absent from ALL keys in dim 1, :z is absent from ALL keys in dim 2
    valid = Set([(:a, :x), (:a, :y), (:b, :x)])

    @variables m begin
        v[i=I, d=D; (i, d) in valid], "Filtered sparse var"
    end

    @test v isa SparseZeroArray
    @test v[:a, :x] isa VariableRef

    # :c never appears in any key but IS in the original index set I → Zero()
    @test v[:c, :x] isa SquareModels.Zero
    @test v[:c, :z] isa SquareModels.Zero
    # :z never appears in any key but IS in the original index set D → Zero()
    @test v[:b, :z] isa SquareModels.Zero

    # Out of original domain — should still error
    @test_throws ErrorException v[:d, :x]
    @test_throws ErrorException v[:a, :w]

    # ∑ over full index set should work even when some values are absent from keys
    result = ∑(v[i, :x] for i in I)
    @test result isa AffExpr
end

@testset "ModelDictionary with SparseZeroArray" begin
    m = Model()
    @variables m begin
        x[i=1:3, j=1:3; i != j], "Sparse var"
    end
    @test x isa SparseZeroArray

    d = ModelDictionary(m)
    # getindex delegates to underlying SparseAxisArray
    w = d[x]
    @test w isa SquareModels.Window

    # setindex! delegates
    d[x] .= 1.0
    @test all(d[v] == 1.0 for v in x)
end

@testset "@block with SparseZeroArray" begin
    m = Model(Ipopt.Optimizer)

    @variables m begin
        x[i=1:3, j=1:3; i != j], "Sparse var"
        param[i=1:3, j=1:3; i != j], "Parameter"
    end

    @test x isa SparseZeroArray
    @test param isa SparseZeroArray

    b = @block m begin
        x[i ∈ 1:3, j ∈ 1:3; i != j], x[i, j] == param[i, j] + 1
    end

    @test length(b) == 6
    @test all(is_endogenous(x.data[i, j], b) for i in 1:3, j in 1:3 if i != j)
end

@testset "use_sparse_zero_array! flag" begin
    # Enabled (default): filtered variables produce SparseZeroArray
    m1 = Model()
    @variables m1 begin
        x1[i=1:3, j=1:3; i != j], "Sparse var"
        d1[1:3], "Dense var"
    end
    @test x1 isa SparseZeroArray
    @test !(d1 isa SparseZeroArray)
    @test m1[:x1] isa SparseZeroArray  # model dictionary stores wrapped type

    # Disabled: filtered variables produce plain SparseAxisArray
    m2 = Model()
    use_sparse_zero_array!(false)
    @variables m2 begin
        x2[i=1:3, j=1:3; i != j], "Sparse var"
        d2[1:3], "Dense var"
    end
    use_sparse_zero_array!(true)  # restore default
    @test x2 isa SparseAxisArray
    @test !(x2 isa SparseZeroArray)
    @test !(d2 isa SparseZeroArray)
    @test m2[:x2] isa SparseAxisArray
    @test !(m2[:x2] isa SparseZeroArray)
end

@testset "block-level tags with tuple membership filter" begin
    m = Model()
    T = Tag(:test_tag)
    I = [:a, :b, :c]
    D = [:x, :y]
    valid = Set([(:a, :x), (:b, :y), (:c, :x)])

    @variables m :: T begin
        v[i=I, d=D; (i, d) in valid], "Test var"
        w[I], "Dense var"
    end

    @test v isa SparseZeroArray
    @test m[:v] isa SparseZeroArray
    @test v[:a, :x] isa JuMP.VariableRef
    @test v[:a, :y] isa SquareModels.Zero  # missing but in domain
    @test_throws ErrorException v[:z, :x]  # out of domain
end

@testset "@block with SparseZeroArray and ∑" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variables m begin
        x[i=1:3, j=1:3; i <= j], "Sparse"
        y[1:3], "Dense"
    end

    @test x isa SparseZeroArray

    b = @block m begin
        y[i ∈ 1:3], y[i] == ∑(x[i, j] for j in 1:3)
    end

    @test length(b) == 3

    db = ModelDictionary(m)
    for i in 1:3, j in 1:3
        if i <= j
            db[x.data[i, j]] = Float64(i + j)
        end
    end
    db[y] .= 1.0
    db[residuals(b)] .= 0.0

    result = solve(b, db)

    # y[1] = x[1,1] + x[1,2] + x[1,3] = 2 + 3 + 4 = 9
    @test result[y[1]] ≈ 9.0 atol=1e-6
    # y[2] = Zero() + x[2,2] + x[2,3] = 0 + 4 + 5 = 9
    @test result[y[2]] ≈ 9.0 atol=1e-6
    # y[3] = Zero() + Zero() + x[3,3] = 0 + 0 + 6 = 6
    @test result[y[3]] ≈ 6.0 atol=1e-6
end
