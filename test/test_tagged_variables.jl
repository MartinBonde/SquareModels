using Test
using SquareModels
import JuMP
using JuMP: Model, all_variables, haskey

module NoJuMPImportVariablesTest
using Test
using SquareModels: @variables, use_sparse_zero_array!, _use_sparse_zero_array

function run(model, all_variables)
    @variables model begin
        x
    end
    @test length(all_variables(model)) == 1
    @test haskey(model, :x)

    previous_sparse_setting = _use_sparse_zero_array[]
    try
        use_sparse_zero_array!(false)
        @variables model begin
            y[i=1:3; i <= 2]
        end
        @test length(all_variables(model)) == 3
        @test haskey(model, :y)
    finally
        use_sparse_zero_array!(previous_sparse_setting)
    end
end
end

@testset "Tagged Variables" begin

    @testset "Tag type" begin
        t1 = Tag(:test)
        t2 = Tag(:test)
        t3 = Tag(:other)

        @test t1.name == :test
        @test t1 == t2  # Same name means equal
        @test t1 != t3
        @test sprint(show, t1) == "Tag(:test)"
    end

    @testset "Basic variable creation" begin
        model = Model()
        t = 2020:2022

        @variables model begin
            x
            y[t]
        end

        @test length(all_variables(model)) == 4  # 1 scalar + 3 indexed
        @test haskey(model, :x)
        @test haskey(model, :y)
    end

    @testset "@variables works without importing JuMP" begin
        NoJuMPImportVariablesTest.run(Model(), all_variables)
    end

    @testset "Sparse variables honor the JuMP string-name setting" begin
        model = Model()
        JuMP.set_string_names_on_creation(model, false)
        stored = Set([(1, :a), (2, :b)])
        @variables model begin
            sparse[i = 1:2, j = [:a, :b]; (i, j) in stored]
            copied[(i, j) = sparse]
        end
        @test all(isempty(JuMP.name(variable)) for variable in sparse)
        @test all(isempty(JuMP.name(variable)) for variable in copied)
    end

    @testset "@block residuals use model-dictionary names when string names are off" begin
        model = Model()
        JuMP.set_string_names_on_creation(model, false)
        stored = Set([(1, :a), (2, :b)])
        @variables model begin
            sparse[i = 1:2, j = [:a, :b]; (i, j) in stored]
            other[i = 1:2, j = [:a, :b]; (i, j) in stored]
            y[1:2]
            z
        end
        block = @block model begin
            sparse[(i, j) in stored], sparse[i, j] == i
            other[(i, j) in stored], other[i, j] == i
            y[i = 1:2], y[i] == i
            z, z == 1
        end
        @test length(block) == 7
        @test haskey(model, :sparse_J)
        @test haskey(model, :other_J)
        @test haskey(model, :y_J)
        @test haskey(model, :z_J)
        @test residual(sparse) === model[:sparse_J]
        @test residual(other) === model[:other_J]
        @test residual(z) === model[:z_J]
        @test all(isempty(JuMP.name(variable)) for variable in sparse)
        @test all(isempty(JuMP.name(variable)) for variable in model[:sparse_J])
        @test Set(residuals(model)) == Set(residuals(block))
    end

    @testset "Variables with tags (:: syntax)" begin
        model = Model()
        t = 2020:2022

        tag_a = Tag(:tag_a)
        tag_b = Tag(:tag_b)

        @variables model begin
            v1[t] :: tag_a
            v2[t] :: (tag_a, tag_b)
            v3[t] :: tag_b
            v4[t]
        end
 
        @test tag_a ∈ tags(model, :v1)
        @test tag_b ∉ tags(model, :v1)

        @test tag_a ∈ tags(model, :v2)
        @test tag_b ∈ tags(model, :v2)

        @test tag_a ∉ tags(model, :v3)
        @test tag_b ∈ tags(model, :v3)

        @test isempty(tags(model, :v4))
    end

    @testset "Variables with descriptions" begin
        model = Model()
        t = 2020:2022

        @variables model begin
            vDesc[t], "A variable with description"
            vNoDesc[t]
        end

        @test description(model, :vDesc) == "A variable with description"
        @test description(model, :vNoDesc) == ""
        
        # Test that indexed variable refs also work (lookup by base name)
        @test description(vDesc[2020]) == "A variable with description"
        @test description(vDesc[2022]) == "A variable with description"
    end

    @testset "Variables with tags and descriptions" begin
        model = Model()
        t = 2020:2022

        growth = Tag(:growth)
        inflation = Tag(:inflation)

        @variables model begin
            vGDP[t] :: (growth, inflation), "Gross Domestic Product"
            pGDP[t] :: inflation, "GDP deflator"
            qGDP[t] :: growth, "Real GDP"
        end

        @test description(model, :vGDP) == "Gross Domestic Product"
        @test growth ∈ tags(model, :vGDP)
        @test inflation ∈ tags(model, :vGDP)

        @test description(model, :pGDP) == "GDP deflator"
        @test growth ∉ tags(model, :pGDP)
        @test inflation ∈ tags(model, :pGDP)

        @test description(model, :qGDP) == "Real GDP"
        @test growth ∈ tags(model, :qGDP)
        @test inflation ∉ tags(model, :qGDP)
    end

    @testset "Query functions" begin
        model = Model()
        t = 2020:2022

        tag_x = Tag(:tag_x)
        tag_y = Tag(:tag_y)

        @variables model begin
            a[t] :: tag_x, "Variable A"
            b[t] :: (tag_x, tag_y), "Variable B"
            c[t] :: tag_y, "Variable C"
            d[t], "Variable D"
        end

        # has_tag
        @test has_tag(model, :a, tag_x)
        @test !has_tag(model, :a, tag_y)
        @test has_tag(model, :b, tag_x)
        @test has_tag(model, :b, tag_y)

        # tagged
        with_x = tagged(model, tag_x)
        @test :a ∈ with_x
        @test :b ∈ with_x
        @test :c ∉ with_x

        with_y = tagged(model, tag_y)
        @test :a ∉ with_y
        @test :b ∈ with_y
        @test :c ∈ with_y

        # metadata
        m = metadata(model, :a)
        @test m.description == "Variable A"
        @test tag_x ∈ m.tags
    end

    @testset "Model-owned metadata" begin
        first_model = Model()
        second_model = Model()
        first_tag = Tag(:first_model)
        second_tag = Tag(:second_model)

        @variables first_model begin
            shared_name :: first_tag, "First model"
        end
        @variables second_model begin
            shared_name :: second_tag, "Second model"
        end

        @test metadata(first_model, :shared_name).description == "First model"
        @test metadata(second_model, :shared_name).description == "Second model"
        @test has_tag(first_model[:shared_name], first_tag)
        @test !has_tag(first_model[:shared_name], second_tag)
        @test :shared_name in tagged(first_model, first_tag)
        @test :shared_name ∉ tagged(first_model, second_tag)

        dictionary = ModelDictionary(first_model)
        @test description(dictionary, :shared_name) == "First model"
        @test tags(dictionary, :shared_name) == Set([first_tag])
        @test has_tag(dictionary, :shared_name, first_tag)
        @test tagged(dictionary, first_tag) == [:shared_name]
        @test metadata(dictionary, :shared_name) == metadata(first_model, :shared_name)

        @test !applicable(description, :shared_name)
        @test !applicable(tags, :shared_name)
        @test !applicable(has_tag, :shared_name, first_tag)
        @test !applicable(tagged, first_tag)
        @test !applicable(metadata, :shared_name)
    end

    @testset "Scalar variables" begin
        model = Model()

        scalar_tag = Tag(:scalar_tag)

        @variables model begin
            σ :: scalar_tag, "Substitution elasticity"
            ρ, "Discount rate"
            δ
        end

        @test haskey(model, :σ)
        @test haskey(model, :ρ)
        @test haskey(model, :δ)

        @test scalar_tag ∈ tags(model, :σ)
        @test description(model, :σ) == "Substitution elasticity"
        @test description(model, :ρ) == "Discount rate"
        @test description(model, :δ) == ""
    end

    @testset "ModelDictionary integration" begin
        db = ModelDictionary(Model())
        t = 2020:2022

        md_tag = Tag(:md_tag)

        @variables db begin
            v[t] :: md_tag, "Test variable"
        end

        @test haskey(db.model, :v)
        @test md_tag ∈ tags(db, :v)
        @test description(db, :v) == "Test variable"
    end

    @testset "JuMP.@variables still accessible" begin
        # Users can still use JuMP's original macro if needed
        model = Model()
        t = 2020:2022

        JuMP.@variables model begin
            jump_var[t]
        end

        @test haskey(model, :jump_var)
        # But it won't have our metadata
        @test description(model, :jump_var) == ""
        @test isempty(tags(model, :jump_var))
    end

    @testset "Block-level tags" begin
        model = Model()
        t = 2020:2022

        block_tag = Tag(:block_tag)
        var_tag = Tag(:var_tag)

        # All variables in this block get block_tag
        @variables model :: block_tag begin
            w1[t], "Variable with block tag only"
            w2[t] :: var_tag, "Variable with both tags"
        end

        # w1 should have block_tag only
        @test block_tag ∈ tags(model, :w1)
        @test var_tag ∉ tags(model, :w1)
        @test description(model, :w1) == "Variable with block tag only"

        # w2 should have both tags
        @test block_tag ∈ tags(model, :w2)
        @test var_tag ∈ tags(model, :w2)
        @test description(model, :w2) == "Variable with both tags"
    end

    @testset "Block-level multiple tags" begin
        model = Model()
        t = 2020:2022

        tag1 = Tag(:tag1)
        tag2 = Tag(:tag2)
        tag3 = Tag(:tag3)

        # Multiple block-level tags
        @variables model :: (tag1, tag2) begin
            z1[t]
            z2[t] :: tag3
        end

        # z1 should have both block tags
        @test tag1 ∈ tags(model, :z1)
        @test tag2 ∈ tags(model, :z1)
        @test tag3 ∉ tags(model, :z1)

        # z2 should have all three tags
        @test tag1 ∈ tags(model, :z2)
        @test tag2 ∈ tags(model, :z2)
        @test tag3 ∈ tags(model, :z2)
    end

    @testset "Same name in separate models" begin
        model = Model()
        t = 2020:2022

        old_tag = Tag(:old_tag)
        new_tag = Tag(:new_tag)

        # First definition
        @variables model begin
            redef_var[t] :: old_tag, "Old description"
        end

        @test old_tag ∈ tags(model, :redef_var)
        @test new_tag ∉ tags(model, :redef_var)
        @test description(model, :redef_var) == "Old description"

        # Redefine variable with new tags and description (new model to avoid JuMP error)
        model2 = Model()
        @variables model2 begin
            redef_var[t] :: new_tag, "New description"
        end

        @test old_tag ∈ tags(model, :redef_var)
        @test new_tag ∉ tags(model, :redef_var)
        @test description(model, :redef_var) == "Old description"

        @test old_tag ∉ tags(model2, :redef_var)
        @test new_tag ∈ tags(model2, :redef_var)
        @test description(model2, :redef_var) == "New description"
    end
end
