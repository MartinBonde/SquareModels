module TestGAMS

using Test
import JuMP
using JuMP: Model, set_optimizer_attribute, get_optimizer_attribute, @variable, unsafe_backend, MOI
using SquareModels
import GAMS

const GAMS_SYSDIR = raw"C:\GAMS\52"

# Probe whether the GAMS runtime (not just the Julia package) is installed
const GAMS_AVAILABLE = try
    m = Model(GAMS.Optimizer)
    set_optimizer_attribute(m, "sysdir", GAMS_SYSDIR)
    @variable(m, _x)
    JuMP.@constraint(m, _x == 1)
    JuMP.optimize!(m)
    true
catch
    false
end

if GAMS_AVAILABLE

@testset "GAMS CONOPT solve" begin
    m = Model(GAMS.Optimizer)
    set_optimizer_attribute(m, "sysdir", GAMS_SYSDIR)
    set_optimizer_attribute(m, "NLP", "CONOPT")
    set_optimizer_attribute(m, "LogOption", 0)

    JuMP.@variables m begin
        x
        y
        z
    end

    data = ModelDictionary(m)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0

    block = @block m begin
        x, x == z * 2
        y, y == z^2
    end
    data[residuals(block)] .= 0.0

    # Verify that _copy_model_config preserves GAMS-specific attributes
    solve_model, _ = SquareModels._build_model(block, data)
    inner = unsafe_backend(solve_model)
    @test MOI.get(inner, MOI.RawOptimizerAttribute("NLP")) == "conopt"
    @test MOI.get(inner, MOI.RawOptimizerAttribute("LogOption")) == 0

    solution = solve(block, data)

    @test solution[x] ≈ 10.0 atol=1e-6
    @test solution[y] ≈ 25.0 atol=1e-6
end

@testset "square_model with gamsdir" begin
    m = square_model(; gamsdir = GAMS_SYSDIR)
    @test m isa Model

    JuMP.@variables m begin
        x
        y
        z
    end

    data = ModelDictionary(m)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0

    block = @block m begin
        x, x == z * 2
        y, y == z^2
    end
    data[residuals(block)] .= 0.0

    solution = solve(block, data)

    @test solution[x] ≈ 10.0 atol=1e-6
    @test solution[y] ≈ 25.0 atol=1e-6
end

@testset "GAMS failed solve annotates generated files" begin
    dir = mktempdir()
    m = square_model(; gamsdir = GAMS_SYSDIR, working_dir=dir)
    JuMP.@variable(m, foo)

    data = ModelDictionary(m)
    data[foo] = 0.0
    block = @block m begin
        foo, foo == 1 / (foo - foo)
    end
    data[residuals(block)] .= 0.0

    @test_throws ErrorException solve(block, data)

    @test isfile(joinpath(dir, "moi.lst"))
    content = read(joinpath(dir, "moi.gms"), String)
    @test occursin("foo", content)
    @test !occursin(r"\bx1\b", content)
    @test !occursin(r"\beq1\b", content)
end

@testset "annotate_lst! on real GAMS listing" begin
    # GAMS.jl suppresses the symbol listing ($offlisting, limrow/limcol/solprint=0), so a
    # clean solve never names x<i>/eq<i> in the .lst. CONOPT *does* name them when the square
    # system is singular ("ERRORS/WARNINGS IN EQUATION/VARIABLE"), which is exactly the case
    # annotate_lst! exists to make readable. Two linearly dependent equations force that.
    #
    # The working dir is pinned via an explicit workspace so we can read the .lst after the
    # solve (GAMS.jl swaps the args when both "sysdir" and "workdir" attributes are set).
    dir = mktempdir()
    m = Model(() -> GAMS.Optimizer(GAMS.GAMSWorkspace(dir, GAMS_SYSDIR)))
    set_optimizer_attribute(m, GAMS.ModelType(), "CNS")
    set_optimizer_attribute(m, "CNS", "CONOPT")
    set_optimizer_attribute(m, GAMS.Solver(), "CONOPT")

    JuMP.@variables m begin
        foo
        bar
    end

    data = ModelDictionary(m)
    data[foo] = 1.0
    data[bar] = 1.0

    block = @block m begin
        foo, foo + bar == 2
        bar, 2foo + 2bar == 4
    end
    data[residuals(block)] .= 0.0

    # A singular system may or may not be reported as a solver failure; either way solve!
    # annotates the .lst before that decision, which is what we are checking here.
    try
        solve(block, data)
    catch
    end

    lst = joinpath(dir, "moi.lst")
    @test isfile(lst)

    content = read(lst, String)
    # GAMS only ever writes x<i>/eq<i>, so the JuMP names appearing in the singularity
    # diagnostics is definitive proof solve! rewrote (annotated) the listing.
    @test occursin("EQUATION foo", content)
    @test occursin("VARIABLE foo", content)
    @test !occursin(r"\bx1\b", content)
    @test !occursin(r"\beq1\b", content)

    gms = joinpath(dir, "moi.gms")
    @test isfile(gms)
    gms_content = read(gms, String)
    @test occursin("foo", gms_content)
    @test !occursin(r"\beq1\b", gms_content)
end

else
    @warn "Skipping GAMS tests: GAMS runtime not available"
end

# Pure string rewriting: no GAMS runtime required, so run unconditionally.
@testset "annotate_lst!" begin
    # Greek names are valid JuMP identifiers but not valid GAMS symbols, so the
    # annotated output uses names GAMS itself could never have produced.
    m = Model()
    JuMP.@variables m begin
        α
        β
    end

    block = @block m begin
        α, α == 1
        β, β == 2
    end

    content = join([
        "GAMS Rev 24.5 WEX-WEI x86 64bit/MS Windows",  # banner: x86 must survive
        "---- VAR x1",
        "---- VAR x2",
        "---- EQU eq1",
        "---- EQU eq2",
        "out of range x99 eq50",                        # indices > #vars: unchanged
        "boundary xx1 ax1 x1x",                         # not whole tokens: unchanged
    ], "\n")

    dir = mktempdir()
    src = joinpath(dir, "moi.lst")
    write(src, content)

    out = annotate_lst!(block, src)
    @test out == src
    lines = readlines(src)

    @test occursin("WEX-WEI", lines[1]) && occursin("x86", lines[1])  # banner skipped
    @test lines[2] == "---- VAR α"
    @test lines[3] == "---- VAR β"
    @test lines[4] == "---- EQU α"
    @test lines[5] == "---- EQU β"
    @test lines[6] == "out of range x99 eq50"
    @test lines[7] == "boundary xx1 ax1 x1x"

    # out_path leaves the source untouched
    dst = joinpath(dir, "annotated.lst")
    write(src, content)
    annotate_lst!(block, src; out_path=dst)
    @test occursin("x1", readlines(src)[2])
    dst_lines = readlines(dst)
    @test dst_lines[2] == "---- VAR α"
    @test dst_lines[4] == "---- EQU α"

    write(src, content)
    annotate_lst!(block, src; equation_names=["β", "α"])
    lines = readlines(src)
    @test lines[4] == "---- EQU β"
    @test lines[5] == "---- EQU α"
end

end # module
