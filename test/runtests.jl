using Test

@testset "SquareModels Tests" begin
	include("test_Blocks.jl")
	include("test_utils.jl")
	include("test_ModelDictionaries.jl")
	include("test_build_model.jl")
	include("test_integration.jl")
end;
