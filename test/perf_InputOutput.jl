using Test
using SquareModels
import JuMP
using Ipopt

const FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "InputOutputPerfFixture.jl")
const SCALE_KEY = "SQUAREMODELS_IO_PERF_SCALE"
const DEFAULT_SCALE = 40

"""Run `f` with the scale value set, then restore the old value."""
function with_scale(f, scale)
    old_value = get(ENV, SCALE_KEY, nothing)
    ENV[SCALE_KEY] = string(scale)
    try
        return f()
    finally
        if old_value === nothing
            delete!(ENV, SCALE_KEY)
        else
            ENV[SCALE_KEY] = old_value
        end
    end
end

"""Include the fixture in a new parent module and return its inner module."""
function load_fixture(scale)
    return with_scale(scale) do
        parent = Module(gensym(:InputOutputPerfRun))
        Base.include(parent, FIXTURE_PATH)
    end
end

"""Read a binding that a new fixture module made in the latest Julia world."""
latest_global(fixture, name) = Base.invokelatest(getproperty, fixture, name)

"""Call the equation method that a new fixture module made."""
function make_block(fixture)
    define_equations = latest_global(fixture, :define_equations)
    return Base.invokelatest(define_equations)
end

function show_time(label, stats)
    seconds = round(stats.time, digits=2)
    allocated_gib = round(stats.bytes / 2.0^30, digits=2)
    gc_seconds = round(stats.gctime, digits=2)
    println("  ", rpad(label, 25), lpad("$seconds s", 10),
        lpad("$allocated_gib GiB", 12), lpad("GC $gc_seconds s", 12))
end

@testset "Perf: InputOutput include and model construction" begin
    scale = parse(Int, get(ENV, SCALE_KEY, string(DEFAULT_SCALE)))
    max_seconds = parse(Float64,
        get(ENV, "SQUAREMODELS_IO_PERF_MAX_SECONDS", "600"))

    include_stats = @timed load_fixture(scale)
    fixture = include_stats.value
    model = latest_global(fixture, :model)
    source_variable_count = length(JuMP.all_variables(model))

    GC.gc()
    block_stats = @timed make_block(fixture)
    block = block_stats.value

    GC.gc()
    data_stats = @timed ModelDictionary(model, 1.0)
    data = data_stats.value

    GC.gc()
    build_stats = @timed SquareModels._build_model(block, data)
    solve_model, variable_map = build_stats.value

    expected_variables = latest_global(fixture, :variable_count)
    expected_equations = latest_global(fixture, :equation_count)
    @test source_variable_count == expected_variables
    @test length(block) == expected_equations
    @test length(variable_map) == expected_equations
    @test length(JuMP.all_variables(solve_model)) == expected_equations

    total_seconds =
        include_stats.time + block_stats.time + data_stats.time + build_stats.time

    println()
    println("  InputOutput scale test")
    product = latest_global(fixture, :product)
    use_axis = latest_global(fixture, :use)
    origin = latest_global(fixture, :origin)
    time_axis = latest_global(fixture, :t)
    println("  Products: $(length(product)), uses: $(length(use_axis)), " *
        "origins: $(length(origin)), periods: $(length(time_axis))")
    println("  Four variable arrays: $(expected_variables) variables")
    println("  Three equation arrays: $(expected_equations) equations")
    println("  Residual variables added by block: " *
        "$(length(JuMP.all_variables(model)) - source_variable_count)")
    println("  Phase                     Time   Allocated     GC time")
    show_time("include and variables", include_stats)
    show_time("block construction", block_stats)
    show_time("data construction", data_stats)
    show_time("solve model construction", build_stats)
    println("  ", rpad("total", 25),
        lpad("$(round(total_seconds, digits=2)) s", 10))

    @test total_seconds < max_seconds
end
