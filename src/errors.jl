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
	print(io, "$(length(e.violations)) residuals exceed tolerance ($tol_desc):")
	for (k, v, tol) in e.violations
		print(io, "\n  $(k): |value|=$(v), tolerance=$(tol)")
	end
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
	print(io, "$(length(e.violations)) differences exceed tolerance ($tol_desc):")
	for (k, d, rd, v1, v2) in e.violations
		line = isinf(rd) ? "  $(k): diff=$(d) ($(v1) vs $(v2))" :
		       "  $(k): diff=$(d) ($(round(rd * 100, digits=2))%) ($(v1) vs $(v2))"
		print(io, "\n", line)
	end
end

"""
	TestConstraintError <: SquareModelError

Thrown by [`assert_test_constraints`](@ref) when one or more test constraints
exceed their tolerance. Each `violations` entry is
`(name, distance, tolerance, message)`.
`distance` is the distance from the evaluated JuMP expression to its constraint
set. `data` is the tested dictionary, including the solved copy when [`solve`](@ref)
throws this error.
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
	print(io, "$(length(e.violations)) test constraints exceed tolerance ($tol_desc):")
	for (name, distance, tolerance, message) in e.violations
		print(io, "\n  $(name): distance=$(distance), tolerance=$(tolerance)")
		isempty(message) || print(io, ": ", message)
	end
end

"""
	NonSquareError <: SquareModelError

Thrown when a block is not effectively square after substituting exogenous data
(trivial equations and/or orphan variables). `msg` holds the formatted
diagnostic describing the offending equations and variables.
"""
struct NonSquareError <: SquareModelError
	msg::String
end

Base.showerror(io::IO, e::NonSquareError) = print(io, e.msg)
