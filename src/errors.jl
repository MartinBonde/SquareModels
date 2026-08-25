# Copyright 2022, Martin Kirk Bonde and contributors
# Licensed under the MIT License. See LICENSE.md for details.

"""
	SquareModelError <: Exception

Abstract supertype for all errors that represent an *expected model condition
failure* (large residuals, out-of-tolerance differences, a non-square system)
as opposed to a programming bug (`MethodError`, `BoundsError`, ...).

Catch this type to handle/log every SquareModels-specific failure in one place
while letting genuine bugs propagate:

```julia
try
	assert_residuals_small(data)
catch e
	e isa SquareModelError ? log_failure(e) : rethrow()
end
```

Concrete subtypes carry the offending data as fields, so logging code can format
it however it wants instead of parsing a message string.

See also: [`ResidualError`](@ref), [`ToleranceError`](@ref), [`TestConstraintError`](@ref),
[`NonSquareError`](@ref).
"""
abstract type SquareModelError <: Exception end

"""
	ResidualError <: SquareModelError

Thrown by [`assert_residuals_small`](@ref) when one or more residual variables
exceed the tolerance. `violations` holds `(name, |value|, tolerance)` tuples,
where `tolerance` is the effective per-residual threshold (combining `atol`,
`rtol`, and any per-residual overrides), sorted by descending magnitude.
"""
struct ResidualError <: SquareModelError
	violations::Vector{Tuple{String, Float64, Float64}}
	atol::Float64
	rtol::Float64
	msg::String
end
ResidualError(violations, atol::Real, msg::String) = ResidualError(violations, Float64(atol), 0.0, msg)

function Base.showerror(io::IO, e::ResidualError)
	isempty(e.msg) || print(io, e.msg, "\n")
	tol_desc = e.rtol > 0 ? "atol=$(e.atol), rtol=$(e.rtol)" : "atol=$(e.atol)"
	println(io, "$(length(e.violations)) residuals exceed tolerance ($tol_desc):")
	_print_table(io, hcat(getindex.(e.violations, 2), getindex.(e.violations, 3));
		column_labels=["|value|", "tolerance"],
		row_labels=getindex.(e.violations, 1),
		stubhead_label="residual")
end

"""
	ToleranceError <: SquareModelError

Thrown by [`assert_no_diff`](@ref) when one or more values differ by more than
the allowed tolerance. Each `violations` entry is
`(key, abs_diff, rel_diff, value_a, value_b)`; `rel_diff` is `Inf` when the
reference value is itself below `atol`.
"""
struct ToleranceError <: SquareModelError
	violations::Vector{Tuple{String, Float64, Float64, Any, Any}}
	atol::Float64
	rtol::Float64
	msg::String
end

function Base.showerror(io::IO, e::ToleranceError)
	isempty(e.msg) || print(io, e.msg, "\n")
	tol_desc = e.rtol > 0 ? "atol=$(e.atol), rtol=$(e.rtol)" : "atol=$(e.atol)"
	println(io, "$(length(e.violations)) differences exceed tolerance ($tol_desc):")
	relative_differences = [
		isinf(v[3]) ? "" : "$(round(v[3] * 100, digits=2))%" for v in e.violations
	]
	_print_table(io, hcat(
		getindex.(e.violations, 2),
		relative_differences,
		getindex.(e.violations, 4),
		getindex.(e.violations, 5),
	);
		column_labels=["abs diff", "rel diff", "value", "reference"],
		row_labels=getindex.(e.violations, 1),
		stubhead_label="variable")
end

"""
	TestConstraintError <: SquareModelError

Thrown by [`assert_test_constraints`](@ref) when one or more test constraints
exceed their tolerance. Each `violations` entry is
`(name, distance, tolerance, message)`.
`distance` is the distance from the evaluated JuMP expression to its constraint
set. `atol` and `rtol` are the explicit defaults passed to the test, or the
equality defaults when they were omitted. Inequalities use zero by default. A
test constraint can override either tolerance. `data` is the tested dictionary,
including the solved copy when [`solve`](@ref) throws this error.
"""
struct TestConstraintError{D} <: SquareModelError
	violations::Vector{Tuple{String, Float64, Float64, String}}
	atol::Float64
	rtol::Float64
	msg::String
	data::D
end

function Base.showerror(io::IO, e::TestConstraintError)
	isempty(e.msg) || print(io, e.msg, "\n")
	tol_desc = e.rtol > 0 ? "atol=$(e.atol), rtol=$(e.rtol)" : "atol=$(e.atol)"
	println(io, "$(length(e.violations)) test constraints exceed tolerance (configured defaults: $tol_desc; inequalities use zero when not configured):")
	_print_table(io, hcat(
		getindex.(e.violations, 2),
		getindex.(e.violations, 3),
		getindex.(e.violations, 4),
	);
		column_labels=["distance", "tolerance", "message"],
		row_labels=getindex.(e.violations, 1),
		stubhead_label="variable")
end

"""
	NonSquareError <: SquareModelError

Thrown when a block is not square or is not effectively square after data
substitution. `msg` is a one-line summary. Extra rows go in `mappings`,
`trivial`, and `orphans` so `showerror` can print them
as tables. Do not put those lists in `msg`: Julia `show` escapes newlines, and
hosts that call `show` instead of `showerror` then dump one long line.
"""
struct NonSquareError{M} <: SquareModelError
	msg::String
	mappings::M
	trivial::Vector{Tuple{String, Float64}}
	orphans::Vector{String}
end
NonSquareError(msg::String, mappings=nothing; trivial=Tuple{String, Float64}[], orphans=String[]) =
	NonSquareError(msg, mappings, convert(Vector{Tuple{String, Float64}}, trivial), convert(Vector{String}, orphans))

function Base.showerror(io::IO, e::NonSquareError)
	print(io, e.msg)
	if e.mappings !== nothing && !isempty(e.mappings)
		println(io)
		_print_table(io, hcat(
			string.(first.(e.mappings)),
			string.(getproperty.(last.(e.mappings), :func)),
		);
			column_labels=["endogenous variable", "equation expression (= 0)"])
	end
	if !isempty(e.trivial)
		println(io)
		println(io, "$(length(e.trivial)) trivial equation(s) (no endogenous variables effectively present after substituting exogenous data):")
		_print_table(io, hcat(last.(e.trivial), _trivial_status.(last.(e.trivial)));
			column_labels=["constant", "status"],
			row_labels=first.(e.trivial),
			stubhead_label="variable")
	end
	if !isempty(e.orphans)
		println(io)
		println(io, "$(length(e.orphans)) orphan variable(s) (not effectively present in any non-trivial equation):")
		_print_table(io, reshape(e.orphans, :, 1); column_labels=["variable"])
	end
end

_trivial_status(rhs::Float64) =
	isnan(rhs) ? "constant could not be determined" :
	abs(rhs) < 1e-12 ? "redundant" : "infeasible"

# `show` must stay short and on one line. Julia escapes newlines here, and a
# host that prints an exception with `show` then dumps one unreadable line.
# The tables belong to `showerror`.
Base.show(io::IO, e::SquareModelError) =
	print(io, nameof(typeof(e)), "(", repr(first(split(e.msg, '\n'; limit=2))), ")")
