# Plotting and Printing

SquareModels delegates plotting to Makie. Load a backend such as CairoMakie to
activate the package extension:

```julia
using CairoMakie
using SquareModels
```

The extension teaches Makie how to plot `Window` and `LabeledSeries` values, so
model slices keep their axes and labels.

## Plotting Variables

`plotvar` builds a complete figure for one model variable or slice:

```@example plotting
using CairoMakie
import JuMP
using JuMP: Model
using SquareModels

model = Model()
t = 2020:2024
regions = [:north, :south]

@variables model begin
    qGDP[t], "Real GDP"
    qC[t], "Real consumption"
    emissions[regions, t], "Emissions"
end

data = ModelDictionary(model)
data[qGDP] = [100, 103, 106, 110, 114]
data[qC] = [80, 82, 84, 87, 90]
data[emissions] = [10 9 8 8 7; 6 6 5 5 4]

fig = plotvar(data, qGDP[t]; ylabel="Index")
save("qgdp.png", fig)
nothing
```

![](qgdp.png)

Multi-dimensional variables fan out into one line per leading-index combination;
the last dimension is used as the x-axis.

```@example plotting
fig = plotvar(data, emissions[regions, t]; ylabel="Mt CO2")
save("emissions.png", fig)
nothing
```

![](emissions.png)

## Plotting Expressions

[`@plot`](@ref) resolves bare variable names against a model dictionary and labels
each plotted expression with the source text:

```@example plotting
fig = @plot data [qGDP / qGDP[2020], qC / qC[2020]]
save("normalised.png", fig)
nothing
```

![](normalised.png)

Arithmetic operators are broadcast implicitly, so `qGDP / qGDP[2020]` works
elementwise. Named calls are left as written, so reductions stay reductions;
write explicit dots for elementwise functions and explicit generators for sums:

```julia
@plot data log.(qGDP)
@plot data sum(qX[s, :] for s in sectors)
```

For programmatic workflows, build series explicitly with [`labeled`](@ref) and
draw them with `plotseries`:

```julia
plotseries([
    labeled(data[qGDP] ./ data[qGDP[2020]], "qGDP / qGDP[2020]"),
    labeled(data[qC] ./ data[qC[2020]], "qC / qC[2020]"),
])
```

## Printing and Expression Evaluation

[`@evalexpr`](@ref) evaluates expressions without plotting:

```@example plotting
normalised_gdp = @evalexpr data qGDP / qGDP[2020]
round.(normalised_gdp, digits=3)
```

[`@prt`](@ref) prints values and transformations in a table-oriented format. An
optional operator symbol selects the transformation, e.g. `:p` for percent
growth and `:q` for percent deviation from a reference:

```julia
@prt data qGDP
@prt :p data qGDP[2020:2060]
@prt :q baseline=>scenario qGDP[2020:2060]
@prt 2020:2060 qGDP                         # default source, selected periods
```

In a limited display context such as the REPL, long `@prt` and `Window` tables
fit the available height. To print all rows, assign the result and show it with
`:limit => false` in the `IOContext`.

### Print operators

Operators transform the expression result along its final dimension. In the
definitions below, ``x_t`` is the source value, ``b_t`` is the reference value,
and ``\Delta x_t = x_t - x_{t-1}``.

Sparse operators transform stored cells only; gaps remain unstored and print as
blank cells. Lag operators use the prior displayed period for the same leading
index combination. If that prior cell is not stored, the result is `NaN`.

Source transformations:

- `:n`, `:abs` — level, ``x_t`` (no transformation).
- `:d`, `:dif` — difference, ``\Delta x_t``.
- `:p`, `:pch` — percent change, ``100(x_t/x_{t-1}-1)``.
- `:dp`, `:gdif` — change in the percent growth rate.
- `:l` — natural logarithm, ``\log(x_t)``.
- `:dl` — log difference, ``\log(x_t)-\log(x_{t-1})``.

Comparisons with a reference:

- `:m` — absolute deviation, ``x_t-b_t``.
- `:q` — percent deviation, ``100(x_t/b_t-1)``.
- `:mp` — difference between the source and reference percent growth rates.

Reference transformations:

- `:r`, `:rn` — reference level, ``b_t``.
- `:rd` — reference difference, ``\Delta b_t``.
- `:rp` — reference percent change.
- `:rdp` — change in the reference percent growth rate.
- `:rl` — natural logarithm of the reference.
- `:rdl` — reference log difference.

The following bundle operators return several transformations together:

- `:a` → `[:n, :p, :r, :rp]`
- `:an` → `[:n, :r]`
- `:ad` → `[:d, :rd]`
- `:ap` → `[:p, :rp]`
- `:adp` → `[:dp, :rdp]`
- `:al` → `[:l, :rl]`
- `:adl` → `[:dl, :rdl]`

Pass an explicit operator vector to request another combination, for example
`@prt [:n, :p] data qGDP`. The reference operators, comparison operators, and
all bundle operators require a `reference => source` pair.

A `reference => source` pair supplies the reference for operators that need one,
like `:q` above. Without such an operator (or with a `Tuple` of sources/pairs),
the values from each database print side by side instead, one column per
database — a reference shared by several pairs (like a common baseline) is only
shown once:

```julia
@prt baseline=>scenario qGDP[2020:2060]
@prt (baseline=>shock1, baseline=>shock2) qGDP[2020:2060]
#        baseline:qGDP    shock1:qGDP    shock2:qGDP
# 2020        1.2             1.3            1.4
# 2021        1.3             1.4            1.5
```

Multi-dimensional results print as a table via PrettyTables.jl. The last axis
supplies the rows, and each leading-index combination supplies a value column:

```julia
@prt data emissions
# year    emissions[north]    emissions[south]
# 2020                10.0                 6.0
# 2021                 9.0                 6.0
# 2024                 7.0                 4.0
```

Multiple expressions in a tuple print together as columns of one table (rows are
the shared index, e.g. periods) instead of a plain Julia `Tuple`:

```julia
@prt data (qGDP, qC)
#        qGDP    qC
# 2020    100    80
# 2021    103    82
```

For interactive work, set defaults once and omit the source:

```julia
set_default_source!(baseline => scenario)
set_default_operator!(:q)
set_default_periods!(2020:2030)
@prt qGDP
@plot qGDP
reset_print_defaults!()
```

## Theming and customization

`plotvar`, `plotseries`, and `@plot` inherit Makie theme defaults for colors,
fonts, grid, and figure size. By default a native legend is added with
`axislegend(ax; position=:rb)`; pass `legend=false` to suppress it,
`legend=true` for `axislegend(ax)` with default placement, or a NamedTuple like
`legend=(position=:cb,)` to customise it.

For organisation-specific layout (coloured labels below the chart, alternating
dashes, extra annotations), register a **finalize hook** — a function
`f(fig, ax, series)` called after the lines are drawn. When a hook is set, the
default native legend is skipped so the hook can supply its own (an explicit
`legend=true`/NamedTuple still applies):

```julia
using CairoMakie
using SquareModels
using MyOrgMakieTheme

MyOrgMakieTheme.activate!()
set_plot_finalize!(MyOrgMakieTheme.colored_text_legend!)

@plot data qGDP
reset_plot_finalize!()
```

A finalize function receives the figure, the axis, and the expanded line list
(a `Vector` of `AbstractSeries`) and can add legends or annotations and adjust
the layout.

### Alternating dash for repeated variables

When the same variable is drawn several times — from multiple default sources,
or as value/reference pairs with operators like `:an` — the lines share a base
label (the label with any ` <op>` suffix stripped). In that case the plot
builders automatically give each such group a single color and distinguish the
lines by linestyle (solid, dot, dash, ...). Control this with the
`alternating_dash` keyword: `false` disables it, `true` forces it (pairing
consecutive lines when all labels are unique), and `alternating_dash!(ax, series)`
applies it manually.
