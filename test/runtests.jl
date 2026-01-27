using Test

@testset "SquareModels Tests" begin
	include("test_Blocks.jl")
	include("test_utils.jl")
	include("test_ModelDictionaries.jl")
	include("test_build_model.jl")
	include("test_integration.jl")

	@testset "Examples" begin
		@testset "quick_example.jl" begin
			include("../examples/quick_example.jl")
		end
		@testset "modular_example.jl" begin
			include("../examples/modular_example.jl")
		end
	end
end;
