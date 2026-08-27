module TestUtils

using Test
using SquareModels
using SquareModels: @replace_vars, @q, @pq, @m

@testset "Test @replace_vars" begin
  xy = Dict(:x => 1.1, :y => 1.2)
  z = 0.3

  @test @replace_vars(x + y, xy) == @replace_vars(y + x, xy) == xy[:x] + xy[:y]

  @test @replace_vars(x + z, xy) == @replace_vars(z + x, xy) == xy[:x] + z
end

@testset "Test compare operator macros" begin
  b = Dict(:x => 1, :y => 1)
  s = Dict(:x => 1.1, :y => 1.2)

  @test @q(x*100, s, b) ≈ 0.1
  @test @q(x+y, s, b) ≈ 0.15

  @test @pq(x, s, b) ≈ 10
  @test @pq(x+y, s, b) ≈ 15

  @test @m(x, s, b) ≈ 0.1
  @test @m(x+y, s, b) ≈ 0.3
end

end # Module
