using Test

@testset "SquareModels Tests" begin
	include("test_Blocks.jl")
	include("test_utils.jl")
	include("test_ModelDictionaries.jl")
	include("test_build_model.jl")
	include("test_integration.jl")
	include("test_tagged_variables.jl")
	include("test_SparseZeroArray.jl")
	include("test_ModelPlotting.jl")
	include("test_GAMS.jl")

	@testset "Examples" begin
		@testset "modular_example.jl" begin
			include("../examples/modular_example.jl")
		end
		@testset "optimization_example.jl" begin
			include("../examples/optimization_example.jl")
		end
	end
end;
