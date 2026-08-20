# Core Concepts

## Blocks

A [`Block`](@ref) stores a square mapping from endogenous variables to equations.
The mapping is explicit: each line in `@block` starts with the variable that the
equation determines.

```julia
block = @block data begin
    x,          x == a + b
    y[i ∈ I],   y[i] == z[i] + 1
end
```

Blocks can be combined with `+` as long as they belong to the same JuMP model and
do not determine the same endogenous variable twice:

```julia
full_model = households + production + government
```

Use [`endogenous`](@ref), [`exogenous`](@ref), [`variables`](@ref), and
[`residuals`](@ref) to inspect the block.

## Test Constraints

Use `@test_constraint` for a constraint that must hold but must not determine
another endogenous variable:

```julia
block = @block data begin
    a_i[i ∈ industries, t ∈ periods], a_i[i, t] == b_i[i, t] + c_i[i, t]
    a[t ∈ periods], a[t] == b[t] + c[t]
    @test_constraint("a aggregation")
    a[t ∈ periods], a[t] == ∑(a_i[i, t] for i ∈ industries)
    b[t ∈ periods], b[t] == ∑(b_i[i, t] for i ∈ industries)
    c[t ∈ periods], c[t] == ∑(c_i[i, t] for i ∈ industries)
end
```

Test constraints create no residual variables and do not enter the solve model.
`solve` and `solve!` evaluate them with JuMP after each successful solve:

```julia
solution = solve(block, data)
```

Test constraints accept `==`, `<=`, and `>=`. SquareModels does not add a
residual to a test constraint. Add one when the test must include the gap that
the solve equation absorbs:

```julia
block = @block data begin
    a[t ∈ periods], a[t] == b[t] + c[t]
    @test_constraint("a aggregation with residual")
    a[t ∈ periods],
        a[t] + residual(a)[t] == ∑(a_i[i, t] for i ∈ industries)
end
```

Here, `residual(a)` returns the residual that the solve equation for `a` creates.
Omit it when the test must compare the reported value of `a` itself.

Use `atol` and `rtol` to set a tolerance for one test constraint:

```julia
block = @block data begin
    @test_constraint("a aggregation"; atol=1e-8, rtol=1e-6)
    a[t ∈ periods], a[t] == ∑(a_i[i, t] for i ∈ industries)
end
```

The macro call must be a separate statement directly before the entry. You can
remove the macro call without changing the variable or equation.

The test passes when the gap is at most
`max(atol, rtol * abs(data[a[t]]))`. The mapped variable, `a[t]` in this
example, sets the scale for the relative tolerance. If you omit a keyword, the
test uses the value from `test_constraint_atol` or `test_constraint_rtol` in
`solve` or `solve!`. These defaults are `1e-6` and `1e-8`.

Set `run_test_constraints=false` to skip all tests. Use
[`assert_test_constraints`](@ref) to test a loaded or edited `ModelDictionary`
without a solve. Its `atol` and `rtol` values act as the defaults in the same
way. [`test_constraint_variables`](@ref) returns the variables that the tests
need. These variables stay out of [`variables`](@ref) and [`exogenous`](@ref)
unless a solve constraint also uses them.

## Endo-Exo Swapping

Calibration often means solving for parameters that are normally exogenous while
holding observed endogenous variables fixed. [`@endo_exo_swap!`](@ref) changes
which variables are endogenous in an existing block:

```julia
calibration = copy(model_block)
@endo_exo_swap! calibration begin
    μ, Y
    δ, K[t₀]
end
```

The left-hand variable becomes endogenous. The right-hand variable must already
be endogenous in the block and becomes exogenous data.

## Model Dictionaries

[`ModelDictionary`](@ref) maps JuMP variables to values and supports scalar,
container, and slice indexing:

```julia
# Single value
data[x[2025]]          # scalar

# Vector of variable references — returns a Window (a view into the dictionary)
data[x[2025:2060]]     # all periods
data[x[[2025, 2030]]]  # selected periods

# Multi-dimensional variables
data[y[:electric, 2025:2060]]  # one fuel type, all periods
data[y[:, 2025]]               # all fuel types, one period

# Assignment works the same way
data[x[2025:2060]] .= 1.0
data[y[:electric, 2025:2060]] .= 0.8
```

Indexing a variable container returns a `Window`, which behaves like a view into
the dictionary and keeps the original model indices. That is what makes slices
usable for printing and plotting. A `Window` supports broadcasting (`.=`, `.*`,
etc.) and iteration, but external libraries may require `collect` or
`Float64.()` to convert to a plain `Vector`. At the REPL, a multi-dimensional
`Window` displays as a table (rows for the leading indices, columns for the last
dimension) via PrettyTables.jl.

## Loading and Saving Data

[`unload`](@ref) saves a `ModelDictionary` to Parquet in a simple tabular format
(`variable`, `indices`, `value`). Entries with `nothing` are omitted. [`load`](@ref)
reads Parquet, CSV, or — with `using GDXInterface` — GDX files and matches rows to
the model's variables by base name and indices.

```julia
unload("solution.parquet", baseline)
baseline = load("solution.parquet", model)
```

**Index matching:** only indices that exist in both the file and the model are loaded.
Extra data indices are ignored; model indices missing from the file stay `nothing`.

**Renames** map model variable base names to differently named data symbols (like GAMS
`$LOAD`). Pass keyword arguments or `Pair`s:

```julia
d = load("data.parquet", model; N_a = "nPop", Y => "OtherY")
```

**Slices** extract a lower-dimensional symbol from higher-dimensional data. Use `:`
for positions filled from the model variable's indices:

```julia
d = load("data.gdx", model;
    N = "nPop",              # simple rename
    C = "vC[:cTot,:]",        # C[t] ← vC[:cTot, t]
    K = "vK[:iTot,:tot,:]",   # K[t] ← vK[:iTot, :tot, t]
)
```

For reading a single variable without building a full dictionary, use
[`read_variable`](@ref), [`read_sparse_array`](@ref), or [`read_indices`](@ref) on
simple-format CSV/Parquet files.

## Variable Metadata

SquareModels `@variables` adds descriptions and tags. Use
`JuMP.@variables` for JuMP's original macro:

```julia
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)

@variables data.model :: GrowthAdjusted begin
    qGDP[t], "Real GDP"
    vGDP[t] :: InflationAdjusted, "Nominal GDP"
end
```

Use [`description`](@ref), [`tags`](@ref), [`has_tag`](@ref), and [`tagged`](@ref)
to query the metadata.

## SparseZeroArray

Sparse models need short expressions, fast declarations, and strict index
checks. [`SparseZeroArray`](@ref) lets an expression such as
`∑(x[a, b, c] * (1 + t[a, b, c]) for a in A)` use missing in-domain cells as
`Zero()` without a lookup guard. An out-of-domain access still fails, so the
model can catch a bad index or index order. A naive implementation of
`x[a = A, b = B, c = C; (a, b, c) in a_b_c_filter]` tests every combination of
`A`, `B`, and `C`. Direct coordinate unpack and the index helpers let one
variable reuse another variable's stored index pattern without that full scan.
They also keep tuple coordinates distinct from tuple-valued axes.

SquareModels wraps sparse `@variables` containers in `SparseZeroArray` by
default:

```julia
@variables data.model begin
    x[i=1:5, j=1:5; i <= j], "Upper triangular variable"
end

x[1, 2]  # VariableRef
x[3, 1]  # Zero()
x[6, 1]  # Error: 6 is outside the declared domain
```

This lets you write sums over sparse domains without filtering every access:

```julia
total = ∑(x[i, j] for i in 1:5, j in 1:5)
```

For an independent `in` filter, `@variables` walks the membership set instead
of the full product of the named axes. The named axes still define the domain
used for `Zero()` lookup. The membership set selects only the coordinates that
store variables:

```julia
@variables data.model begin
    value[p = product, i = industry, t = years; (p, i) in pairs]
    price[p = product, i = industry, t = years; (p, i, t) in keys(value)]
end
```

Thus, `value[p, i, t]` returns `Zero()` when `p`, `i`, and `t` belong to their
named axes but `(p, i)` is not in `pairs`. It causes an error only when an index
is outside its named axis.

JuMP evaluates other filter forms, including filters whose membership set
depends on an axis, over the named axes.

Put index names in parentheses to state stored coordinates directly:

```julia
@variables data.model begin
    use[(product, industry) = product_industry_pairs, year = years]
end
```

Each item in `product_industry_pairs` must contain a product and an industry.
This form creates variables only for `product_industry_pairs × years`; it does
not scan separate product and industry sets. For an ordinary coordinate source,
the unique values found in each coordinate position define that axis domain.
A missing combination of those values returns `Zero()`. A value that does not
occur in the applicable coordinate position is outside the domain and causes
an error.

Without parentheses, each tuple remains one value on one axis:

```julia
@variables data.model begin
    flow[edge = edges, year = years]
end
```

Use a `SparseZeroArray` as the source to reuse its stored coordinates and full
axis domain. The declaration creates new JuMP variables; it does not copy cell
values. `keys(array)` supplies only the stored coordinates and derives each
axis domain from those keys.

[`select_axes`](@ref) projects stored coordinates. It keeps the selected axis
domains when its source has domain data. [`merge_indices`](@ref) unions the
stored coordinates and axis domains of sparse arrays. A one-axis unpack needs
a trailing comma:

```julia
@variables data.model begin
    price[(product, industry, year) = value]
    from_keys[(product, industry, year) = keys(value)]
    share[(product, year) = select_axes(value, 1, 3)]
    output[(industry,) = select_axes(value, 2), year = years]
    use[(product, use, origin, year) = merge_indices(purchaser, margin)]
end
```

Disable the wrapper with [`use_sparse_zero_array!`](@ref) if you prefer standard
JuMP `SparseAxisArray` behavior.
