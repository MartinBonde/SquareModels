using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, fix, unfix, is_fixed, value, set_start_value, set_silent, AffExpr, name
using JuMP.Containers: SparseAxisArray
using Ipopt

struct VisitVector{T} <: AbstractVector{T}
    data::Vector{T}
    visits::Ref{Int}
end
Base.size(v::VisitVector) = size(v.data)
Base.getindex(v::VisitVector, i::Integer) = (v.visits[] += 1; v.data[i])

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

    slice_a = sz[:, :a]
    @test slice_a isa SparseZeroArray
    @test length(slice_a) == 3
    @test Set(keys(slice_a)) == Set([(1,), (2,), (3,)])

    slice_b = sz[:, :b]
    @test slice_b isa SparseZeroArray
    @test length(slice_b) == 2
    @test slice_b[3] isa SquareModels.Zero
    @test_throws ErrorException sz[:, [:a, :c]]

    for k in eachindex(sz)
        @test sz[k] === sz[k...]
    end
    for k in eachindex(slice_b)
        @test slice_b[k] === slice_b[k...]
    end
    @test sz[i=1, j=:a] === sz[1, :a]
    @test sz[i=3, j=:b] isa SquareModels.Zero

    block = @block m begin
        slice_a[i = 1:3], slice_a[i] == i
    end
    @test all(is_endogenous(x[i, :a], block) for i in 1:3)

    data = ModelDictionary(m)
    data[slice_b] .= 2.0
    @test all(data[x[i, :b]] == 2.0 for i in 1:2)

    multi = SparseZeroArray(
        SparseAxisArray(Dict((1, 1, 1, 1) => 10.0, (2, 1, 2, 2) => 20.0)),
        ntuple(_ -> Set(1:2), 4),
    )
    multi_slice = multi[:, :, 1, :]
    @test multi_slice isa SparseZeroArray
    @test multi_slice.domain == ntuple(_ -> Set(1:2), 3)
    @test multi_slice[1, 1, 1] == 10.0
    @test multi_slice[2, 2, 2] isa SquareModels.Zero
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

@testset "SparseZeroArray matches SparseAxisArray except Zero()" begin
    data = Dict((1, :a) => 10.0, (2, :a) => 20.0, (1, :b) => 30.0)
    saa = SparseAxisArray(data)
    sz = SparseZeroArray(saa, (Set(1:3), Set([:a, :b])))

    @test sz[1, :a] == saa[1, :a]
    @test sz[(2, :a)] == saa[(2, :a)]
    @test_throws KeyError saa[3, :a]
    @test sz[3, :a] isa SquareModels.Zero

    slice_sz = sz[:, :a]
    slice_saa = saa[:, :a]
    @test Set(eachindex(slice_sz)) == Set(eachindex(slice_saa))
    for k in eachindex(slice_sz)
        @test slice_sz[k] == slice_saa[k]
    end
    @test slice_sz[3] isa SquareModels.Zero

    r_sz = slice_sz .* 2
    r_saa = slice_saa .* 2
    @test r_sz isa SparseZeroArray
    @test r_sz.data == r_saa
    @test r_sz[1] == 20.0
    @test r_sz[3] isa SquareModels.Zero
    @test (sz[:, :a] .+ sz[:, :a]) isa SparseZeroArray
    @test_throws ArgumentError slice_sz .+ 1
    @test_throws ArgumentError iszero.(slice_sz)
    @test_throws ArgumentError slice_sz .^ 2
    @test_throws ArgumentError sqrt.(slice_sz)
    @test_throws ArgumentError 1 ./ slice_sz

    mixed = slice_sz .+ slice_saa
    @test mixed isa SparseZeroArray
    @test mixed.data == slice_saa .+ slice_saa
    @test mixed[3] isa SquareModels.Zero
    @test_throws ArgumentError slice_sz .+ SparseAxisArray(Dict((1,) => 1.0))

    full_saa = SparseAxisArray(Dict((1,) => 10.0, (2,) => 20.0))
    full_sz = SparseZeroArray(full_saa, (Set(1:2),))
    shifted = full_sz .+ 1
    @test shifted isa SparseZeroArray
    @test shifted[1] == 11.0
    @test shifted[2] == 21.0

    short_domain = SparseZeroArray(
        SparseAxisArray(Dict((1,) => 10.0, (2,) => 20.0)),
        (Set(1:3),),
    )
    long_domain = SparseZeroArray(
        SparseAxisArray(Dict((1,) => 10.0, (2,) => 20.0)),
        (Set(1:4),),
    )
    @test_throws ArgumentError short_domain .+ long_domain
    @test_throws ArgumentError long_domain .+ short_domain

    sim = similar(sz, Float64)
    @test sim isa SparseZeroArray
    @test isempty(sim)
    @test sim[1, :a] isa SquareModels.Zero

    sz[3, :a] = 40.0
    @test sz[3, :a] == 40.0
    @test sz[(3, :a)] == 40.0
    @test_throws ErrorException sz[4, :a] = 1.0
    @test_throws ArgumentError sz[:, :a] = 1.0
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

@testset "@variables semicolon filter" begin
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

@testset "@variables tuple destructuring" begin
    m = Model()
    pairs = [(:a, :x), (:b, :y), (:c, :x)]
    years = 2020:2021
    sparse_tag = Tag(:sparse_tag)

    @variables m begin
        dense[i = [:a, :b], year = years], "Dense variable"
        sparse[(product, industry) = pairs, year = years] :: sparse_tag, "Sparse variable"
        flow[edge = pairs, year = years], "Tuple-valued axis"
    end

    @test dense isa JuMP.Containers.DenseAxisArray
    @test flow isa JuMP.Containers.DenseAxisArray
    @test ndims(flow) == 2
    @test flow[(:a, :x), 2020] isa VariableRef

    @test sparse isa SparseZeroArray
    @test ndims(sparse) == 3
    @test length(sparse) == length(pairs) * length(years)
    @test sparse.data.names == (:product, :industry, :year)
    @test name(sparse[:a, :x, 2020]) == "sparse[a,x,2020]"
    @test sparse[:a, :y, 2020] isa SquareModels.Zero
    @test_throws ErrorException sparse[:d, :x, 2020]
    @test m[:sparse] === sparse
    @test description(m, :sparse) == "Sparse variable"
    @test sparse_tag in tags(m, :sparse)

    data = ModelDictionary(m)
    data[sparse] .= 1.0
    @test all(data[variable] == 1.0 for variable in sparse)
end

@testset "@variables copies domain from a SparseZeroArray, not from keys" begin
    m = Model()
    products = [:a, :b, :c]
    industries = [:x, :y, :z]
    years = 2020:2021
    stored = Set([(:a, :x, 2020), (:b, :y, 2021)])
    other_stored = Set([(:a, :x, 2020), (:c, :x, 2021)])

    @variables m begin
        value[p = products, i = industries, t = years; (p, i, t) in stored]
        other[p = products, i = industries, t = years; (p, i, t) in other_stored]
        price[(p, i, t) = value]
        from_keys[(p, i, t) = keys(value)]
        from_filter[p = products, i = industries, t = years; (p, i, t) in keys(value)]
        projected[(p, t) = select_axes(value, 1, 3)]
        projected_keys[(p, t) = select_axes(keys(value), 1, 3)]
        industry[(i,) = select_axes(value, 2)]
        merged[(p, i, t) = merge_indices(value, other)]
    end

    @test price isa SparseZeroArray
    @test ndims(price) == 3
    @test Set(keys(price)) == stored
    @test price[:c, :y, 2020] isa SquareModels.Zero
    @test_throws ErrorException price[:d, :y, 2020]
    @test name(price[:a, :x, 2020]) == "price[a,x,2020]"
    @test m[:price] === price
    @test price[:a, :, 2020] isa SparseZeroArray
    @test Set(keys(price[[:a, :b], :, :])) == stored

    @test Set(keys(from_keys)) == stored
    @test from_keys[:a, :y, 2020] isa SquareModels.Zero
    @test_throws ErrorException from_keys[:c, :x, 2020]
    @test from_filter[:c, :y, 2020] isa SquareModels.Zero
    @test_throws ErrorException from_filter[:d, :y, 2020]

    @test Set(keys(projected)) == Set((p, t) for (p, _, t) in stored)
    @test projected[:c, 2020] isa SquareModels.Zero
    @test_throws ErrorException projected_keys[:c, 2020]
    @test Set(keys(industry)) == Set((i,) for (_, i, _) in stored)
    @test industry[:z] isa SquareModels.Zero
    @test_throws ErrorException industry[:w]
    @test Set(keys(merged)) == union(stored, other_stored)
    @test merged[:c, :y, 2020] isa SquareModels.Zero
    @test_throws ErrorException select_axes(value, 1, 4)
    @test_throws ErrorException merge_indices(value, projected)
end

@testset "axis-dependent membership filter uses nested JuMP evaluation" begin
    m = Model()
    products = [:a, :b]
    years = 2020:2021
    byyear = Dict(2020 => [:a], 2021 => [:a, :b])

    @variables m begin
        x[p = products, t = years; p in byyear[t]]
    end

    @test x isa SparseZeroArray
    @test x[:a, 2020] isa VariableRef
    @test x[:b, 2020] isa SquareModels.Zero
    @test x[:b, 2021] isa VariableRef
    @test_throws ErrorException x[:c, 2020]
end

@testset "empty sparse coordinate sets still create variables" begin
    m = Model()
    products = [:a, :b]
    industries = [:x, :y]
    years = 2020:2021
    empty_pairs = []

    @variables m begin
        empty_filter[p = products, i = industries, t = years; (p, i) in empty_pairs]
        empty_unpack[(p, i) = empty_pairs, t = years]
    end

    @test empty_filter isa SparseZeroArray
    @test isempty(empty_filter)
    @test empty_filter[:a, :x, 2020] isa SquareModels.Zero
    @test_throws ErrorException empty_filter[:z, :x, 2020]

    @test empty_unpack isa SparseZeroArray
    @test isempty(empty_unpack)
    @test_throws ErrorException empty_unpack[:a, :x, 2020]
end

@testset "single-name membership filter keeps tuple values on one axis" begin
    m = Model()
    edges = [(:a, :x), (:b, :y), (:c, :z)]
    chosen = [(:a, :x), (:b, :y)]
    years = 2020:2021

    @variables m begin
        flow[edge = edges, year = years; edge in chosen]
    end

    @test flow isa SparseZeroArray
    @test ndims(flow) == 2
    @test flow[(:a, :x), 2020] isa VariableRef
    @test flow[(:c, :z), 2020] isa SquareModels.Zero
    @test_throws ErrorException flow[(:d, :w), 2020]
end

@testset "single-axis semicolon filter wraps SparseZeroArray" begin
    m = Model()
    I = [:a, :b, :c]
    S = [:a, :b]
    sparse_tag = Tag(:sparse_tag)

    @variables m begin
        x[i = I; i in S] :: sparse_tag
        y[i = 1:3; i <= 2]
    end

    @test x isa SparseZeroArray
    @test x[:a] isa VariableRef
    @test x[:c] isa SquareModels.Zero
    @test_throws ErrorException x[:d]
    @test sparse_tag in tags(m, :x)

    @test y isa SparseZeroArray
    @test y[1] isa VariableRef
    @test y[3] isa SquareModels.Zero
    @test_throws ErrorException y[4]
end

@testset "membership filter scales with keys not the Cartesian product" begin
    products = 1:8
    industries = 1:7
    years = 1:3
    pairs = [(1, 1), (3, 4), (8, 7)]
    visits = Ref(0)
    counted_years = VisitVector(collect(years), visits)

    m = Model()
    @variables m begin
        x[p = products, i = industries, t = counted_years; (p, i) in pairs]
    end

    @test visits[] == length(years)
    @test length(x) == length(pairs) * length(years)
    @test x[2, 1, 1] isa SquareModels.Zero
    @test_throws ErrorException x[9, 1, 1]
end

@testset "tuple destructuring scales with supplied coordinates" begin
    products = 1:8
    industries = 1:7
    years = 1:3
    pairs = [(1, 1), (3, 4), (8, 7)]

    old_checks = Ref(0)
    old_model = Model()
    @variables old_model begin
        old[p = products, i = industries, t = years;
            (old_checks[] += 1; (p, i) in pairs)]
    end

    pair_visits = Ref(0)
    direct_pairs = ((pair_visits[] += 1; pair) for pair in pairs)
    new_model = Model()
    @variables new_model begin
        new[(p, i) = direct_pairs, t = years]
    end

    @test old_checks[] == length(products) * length(industries) * length(years)
    @test pair_visits[] == length(pairs)
    @test length(old) == length(new) == length(pairs) * length(years)
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
    w = d[x]
    @test w isa SquareModels.Window
    @test w.indices isa SparseZeroArray
    @test w.indices.domain == x.domain

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

@testset "@block with tuple-destructured SparseZeroArray" begin
    m = Model(Ipopt.Optimizer)
    pairs = [(:a, :x), (:b, :y)]
    years = 2020:2021

    @variables m begin
        x[(product, industry) = pairs, year = years]
        param[(product, industry) = select_axes(x, 1, 2), year = years]
    end

    b = @block m begin
        x[(product, industry, year) in keys(x)], x[product, industry, year] == param[product, industry, year] + 1
    end

    @test length(b) == length(pairs) * length(years)
    @test all(is_endogenous(x[product, industry, year], b) for (product, industry) in pairs, year in years)
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

    m3 = Model()
    use_sparse_zero_array!(false)
    @variables m3 begin
        x3[(product, industry) = [(:a, :x), (:b, :y)], year = 2020:2021]
    end
    use_sparse_zero_array!(true)
    @test x3 isa SparseAxisArray
    @test !(x3 isa SparseZeroArray)
    @test ndims(x3) == 3
    @test length(x3) == 4
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
