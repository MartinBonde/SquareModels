module TestModelPlotting

using Test
using JuMP
using SquareModels

@testset "Plotting without Makie" begin
	if Base.get_extension(SquareModels, :SquareModelsMakieExt) === nothing
		err = try
			plotseries([labeled([1.0, 2.0], "demo")])
			nothing
		catch err
			err
		end
		@test err isa ErrorException
		@test occursin("using CairoMakie", sprint(showerror, err))
	end
end

using Makie

@testset "Makie extension plotseries methods" begin
	series = [labeled([1.0, 2.0], "demo")]
	@test plotseries(series) isa Makie.Figure
	@test plotseries(only(series)) isa Makie.Figure
end

@testset "Makie extension legend and finalize hook" begin
	series = [labeled([1.0, 2.0], "demo"), labeled([2.0, 3.0], "demo2")]
	has_legend(fig) = any(c isa Makie.Legend for c in fig.content)

	@test has_legend(plotseries(series))  # default axislegend
	@test !has_legend(plotseries(series; legend=false))
	@test has_legend(plotseries(series; legend=(position=:cb,)))

	calls = []
	set_plot_finalize!((fig, ax, s) -> (push!(calls, length(s)); fig))
	@test !has_legend(plotseries(series))  # hook suppresses default legend
	@test calls == [2]
	@test has_legend(plotseries(series; legend=true))  # explicit legend still applies
	reset_plot_finalize!()
	@test plot_finalize() === nothing
end

@testset "default y-axis label depends on operator" begin
	m_series = SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :m)
	fig = plotseries([m_series])
	ax = fig.content[1]
	@test ax.ylabel[] == "Difference from baseline"
	line = only(p for p in ax.scene.plots if p isa Makie.Lines)
	@test line.label[] == "qGDP"

	n_series = SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :n)
	@test plotseries([n_series]).content[1].ylabel[] == "Value"

	@test plotseries([n_series]; ylabel="Custom").content[1].ylabel[] == "Custom"

	mixed = [SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :n), SquareModels.LabeledSeries([1.0, 2.0], [3.0, 4.0], "qGDP", :p)]
	fig = plotseries(mixed)
	@test fig.content[1].ylabel[] == "Value"
	lines = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test Set(l.label[] for l in lines) == Set(["qGDP <n>", "qGDP <p>"])
end

@testset "alternating_dash for repeated variables" begin
	# Same base label twice (e.g. same variable from two sources): same color, different dash.
	series = [labeled([1.0, 2.0], "qGDP"), SquareModels.LabeledSeries([1.0, 2.0], [2.0, 3.0], "qGDP", :r), labeled([3.0, 4.0], "qC")]
	fig = plotseries(series; legend=false)
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] == plots[2].color[]
	@test plots[1].color[] != plots[3].color[]
	@test plots[1].linestyle[] === nothing        # :solid
	@test plots[2].linestyle[] isa AbstractVector # dashed
	@test plots[3].linestyle[] === nothing

	# Unique labels: no restyling unless forced.
	series = [labeled([1.0, 2.0], "a"), labeled([2.0, 3.0], "b")]
	fig = plotseries(series; legend=false)
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] != plots[2].color[]

	fig = plotseries(series; legend=false, alternating_dash=true)  # forced: consecutive pairs
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] == plots[2].color[]
	@test plots[2].linestyle[] isa AbstractVector
end

SquareModels.ModelPlotting.plotseries(series::Vector{SquareModels.LabeledSeries}; kwargs...) = series

model = Model()
JuMP.@variables model begin
	x
	p[[:hh, :firm], 2020:2021]
	q[[:hh, :firm], 2020:2021]
	L[[:cognitive, :physical], 2020:2021]
end

baseline = ModelDictionary(model)
baseline[x] = 3
for h in [:hh, :firm], t in 2020:2021
	baseline[p[h, t]] = h == :hh ? t - 2019 : 10 * (t - 2019)
	baseline[q[h, t]] = h == :hh ? 2 : 3
end
labor = [:cognitive, :physical]
l = labor
t = 2020:2021
for l in labor, t in 2020:2021
	baseline[L[l, t]] = l == :cognitive ? t - 2019 : 10 * (t - 2019)
end

shock = ModelDictionary(model)
shock[x] = 4
for h in [:hh, :firm], t in 2020:2021
	shock[p[h, t]] = 2 * baseline[p[h, t]]
	shock[q[h, t]] = baseline[q[h, t]]
end
for l in labor, t in 2020:2021
	shock[L[l, t]] = 2 * baseline[L[l, t]]
end

@testset "Model expression evaluation" begin
	@test @evalexpr(baseline, x) == 3
	@test @evalexpr(baseline, p[:hh, 2020] * q[:hh, 2020]) == 2
	@test (@evalexpr baseline p[:hh, 2020] * q[:hh, 2020]) == 2
	@test Array(@evalexpr(baseline, p[:firm, :] * q[:firm, :])) == [30, 60]
	@test Array(@evalexpr(baseline, p[:firm] * q[:firm])) == [30, 60]
	@test Array(@evalexpr(baseline, p * q)) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, p[:firm, :] / p[:firm, 2020])) == [1.0, 2.0]
	@test Array(@evalexpr(baseline, p[:firm, :] .* q[:firm, :])) == [30, 60]
	@test Array(@evalexpr(baseline, p .* q)) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, (@. p * q))) == [2 4; 30 60]
	@test Array(@prt(baseline, (@. p * q))) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, sum([L[l, :] for l in l]))) == [11, 22]
	@test Array(@evalexpr(baseline, sum(L[l, :] for l in l))) == [11, 22]
	@test Array(@evalexpr(baseline, sum(L[l] for l in l))) == [11, 22]
	local_vector = [7, 8]
	@test @evalexpr(baseline, local_vector[1]) == 7
	@test @prt(baseline, [x, p[:hh, 2021] * q[:hh, 2021]]) == [3, 4]
	multi_vector = @prt(:m, baseline=>shock, [p[:hh, :], q[:hh, :]])
	@test multi_vector.names == ["p[:hh, :]", "q[:hh, :]"]
	@test count(==("year"), split(sprint(show, MIME"text/plain"(), multi_vector))) == 1
	mixed_dims_print = sprint(show, MIME"text/plain"(), @prt(:m, baseline=>shock, [p[:hh, :], q]))
	@test count(==("year"), split(mixed_dims_print)) == 1
	@test all(label -> occursin(label, mixed_dims_print), ["p[:hh, :]", "q[hh]", "q[firm]"])
	fq = 2
	@test @prt(baseline, p[:hh, 2021] * q[:hh, 2021] / fq) == 2.0
	@test @prt(baseline, (p[:hh, 2021] * q[:hh, 2021], p[:firm, 2021] * q[:firm, 2020] / fq)) == (4, 30.0)
	multi = @prt(baseline, (p[:hh, :], p[:firm, :]))
	@test multi == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(baseline, p[:firm, :])))
	@test occursin("p[:hh, :]", sprint(show, MIME"text/plain"(), multi))
	@test occursin("p[:firm, :]", sprint(show, MIME"text/plain"(), multi))
	@test Array(@prt(baseline, p)) == [1.0 2.0; 10.0 20.0]
	@test occursin("2020", sprint(show, MIME"text/plain"(), @prt(baseline, p)))
	@test isequal(@prt(:p, baseline, p[:hh, :]), [NaN, 100.0])
	@test @prt(:m, baseline=>shock, p[:hh, :]) == [1.0, 2.0]
	@test @prt(:q, baseline=>shock, p[:hh, :]) == [100.0, 100.0]
	@test isequal(map(Array, @evalexpr([:n, :p], baseline, p[:hh, :])), [[1, 2], [NaN, 100.0]])
	an = @prt(:an, baseline=>shock, p)
	@test all(v -> v isa LabeledArray, an)
	@test occursin("year", sprint(show, MIME"text/plain"(), an[2]))
	@test (@evalexpr :q baseline=>shock p[:hh, :]) == [100.0, 100.0]
	op = :q
	@test @prt(op, baseline=>shock, p[:hh, :]) == [100.0, 100.0]
	pair_print = @prt(baseline=>shock, p[:hh, :])
	@test pair_print == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(shock, p[:hh, :])))
	printed_pair = sprint(show, MIME"text/plain"(), pair_print)
	@test !occursin("baseline:p[:hh, :]", printed_pair)
	@test !occursin("shock:p[:hh, :]", printed_pair)
	@test occursin("baseline", printed_pair)
	@test occursin("shock", printed_pair)
	@test occursin("p[:hh, :]", printed_pair)
	# Narrow width forces the 34-char label to wrap across two rows of 24 chars.
	set_column_label_total_width!(48)
	long_print = sprint(show, MIME"text/plain"(), @prt((baseline=>shock, baseline), p[:hh, :] * q[:hh, :] + p[:hh, :]))
	set_column_label_total_width!(72)
	@test !occursin("p[:hh, :] * q[:hh, :] + p[:hh, :]", long_print)
	@test occursin("p[:hh, :] *", long_print)
	@test occursin("q[:hh, :] +", long_print)
	@test SquareModels._column_label_width(1) == 72
	@test SquareModels._column_label_width(2) > SquareModels._column_label_width(8)
	set_column_label_total_width!(100)
	@test SquareModels._column_label_width(1) == 100
	set_column_label_total_width!(72)
	multi_db = @prt((baseline=>shock, baseline), p[:hh, :])
	@test multi_db.names == ["baseline\np[:hh, :]", "shock\np[:hh, :]"]
	@test multi_db == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(shock, p[:hh, :])))
	multi_q = @prt(:q, (baseline=>baseline, baseline=>shock), p[:hh, :])
	@test multi_q.names == ["baseline\np[:hh, :]", "shock\np[:hh, :]"]
	@test multi_q == ([0.0, 0.0], [100.0, 100.0])
	printed_p = sprint(show, MIME"text/plain"(), @prt(baseline, p))
	@test occursin("year", printed_p)
	@test occursin("hh", printed_p)
	@test occursin("firm", printed_p)
	printed_slice = sprint(show, MIME"text/plain"(), @prt(baseline, p[:hh, :]))
	@test occursin("p[:hh, :]", printed_slice)
	long_print = sprint(show, MIME"text/plain"(), @prt(:m, baseline => shock, p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :]))
	@test !occursin("p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :]", long_print)
	@test occursin("p[:hh, :]", long_print)
	@test !occursin("(", long_print)
	@test SquareModels.ModelExpressions._expr_label(:(a + b - c + d)) == "a + b - c + d"
	series = @plot :q baseline=>shock p[:hh, :]
	@test length(series) == 1
	@test series[1].label == "p[:hh, :]"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]
	series = @plot :p baseline p
	@test length(series) == 2
	@test series[1].label == "p[hh]"
	@test series[1].op == :p
	@test series[1].x == [2020, 2021]
	@test isequal(series[1].y, [NaN, 100.0])
	@test series[2].label == "p[firm]"
	@test series[2].x == [2020, 2021]
	@test isequal(series[2].y, [NaN, 100.0])
	series = @plot baseline p .* q
	@test length(series) == 2
	@test series[1].label == "p .* q[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "p .* q[firm]"
	@test series[2].x == [2020, 2021]
	@test series[2].y == [30.0, 60.0]
	series = @plot baseline p * q
	@test length(series) == 2
	@test series[1].label == "p * q[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "p * q[firm]"
	@test series[2].y == [30.0, 60.0]
	series = @plot baseline (@. p * q)
	@test length(series) == 2
	@test series[1].label == "(*).(p, q)[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "(*).(p, q)[firm]"
	@test series[2].y == [30.0, 60.0]
	series = @plot :q baseline=>shock p
	@test length(series) == 2
	@test series[1].y == [100.0, 100.0]
	@test series[2].y == [100.0, 100.0]
	series = @plot :q baseline=>shock sum([L[l, t] for l in l])
	@test length(series) == 1
	@test series[1].label == "sum([L[l, t] for l = l])"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]
	series = @plot :q baseline=>shock sum(L[l, t] for l in l)
	@test length(series) == 1
	@test series[1].label == "sum((L[l, t] for l = l))"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]

	set_default_source!(baseline)
	@test Array(@prt(p[:hh, :])) == [1, 2]
	@test isequal(@prt(:p, p[:hh, :]), [NaN, 100.0])
	@test isequal((@prt :p p[:hh, :]), [NaN, 100.0])

	set_default_source!(baseline => shock)
	@test @prt(:q, p[:hh, :]) == [100.0, 100.0]

	set_default_operator!(:q)
	@test @prt(p[:hh, :]) == [100.0, 100.0]

	set_default_source!(baseline)
	@test @prt(:q, p[:hh, :]) == [0.0, 0.0]

	set_default_source!(baseline, baseline => shock)
	default_multi = @prt(:q, p[:hh, :])
	@test default_multi.names == ["baseline1\np[:hh, :]", "s2\np[:hh, :]"]
	@test default_multi == ([0.0, 0.0], [100.0, 100.0])
	series = @plot(:q, p[:hh, :])
	@test length(series) == 2
	@test series[1].y == [0.0, 0.0]
	@test series[2].y == [100.0, 100.0]

	set_default_source!(baseline => baseline, baseline => shock)
	default_pairs = @evalexpr(:q, p[:hh, :])
	@test default_pairs.names == ["s1\np[:hh, :]", "s2\np[:hh, :]"]
	@test default_pairs == ([0.0, 0.0], [100.0, 100.0])

	@test_throws ErrorException set_default_source!([baseline => baseline, baseline => shock])
	@test_throws ErrorException set_default_source!((baseline, shock))

	set_default_source!(baseline)
	set_default_operator!(:n)
	set_default_periods!(2021:2021)
	set_default_source!(baseline => shock)
	@test @prt(:m, p[:hh, :]) == [2.0]
	series = @plot :m p[:hh, :]
	@test length(series) == 1
	@test series[1].x == [2021]
	@test series[1].y == [2.0]
	series = @plot :q p[:hh, :]
	@test series[1].y == [100.0]
	set_default_source!(baseline)
	@test Array(@prt(p[:hh, :])) == [2]
	@test Array(@prt(p[:hh])) == [2]
	@test @prt(p[:hh, 2020]) == 1
	@test @prt(2020, p[:hh]) == 1
	@test Array(@prt 2020:2020 p[:hh, :]) == [1]
	@test Array(@prt 2020:2020 p[:hh]) == [1]
	@test Array(@prt 2021:2021 baseline p[:hh, :]) == [2]
	@test Array(@prt(sum(L[l, :] for l in l))) == [22]
	@test Array(@prt(sum(L[l] for l in l))) == [22]
	series = @plot p
	@test length(series) == 2
	@test series[1].x == [2021]
	@test series[1].y == [2.0]
	@test series[2].x == [2021]
	@test series[2].y == [20.0]
	series = @plot p[:hh]
	@test only(series).x == [2021]
	@test only(series).y == [2.0]
	series = @plot 2020:2020 p
	@test series[1].x == [2020]
	@test series[1].y == [1.0]
	reset_print_defaults!()
end

@testset "nothing propagates through expressions" begin
	m2 = Model()
	JuMP.@variables m2 begin
		y
		a[[:hh], 2020:2022]
		b[[:hh], 2020:2022]
	end
	db = ModelDictionary(m2)
	for t in 2021:2022
		db[a[:hh, t]] = 1.0
	end
	for t in 2020:2022
		db[b[:hh, t]] = 2.0
	end
	# y and a[:hh, 2020] stay nothing
	@test @evalexpr(db, y) === nothing
	@test @evalexpr(db, y * b[:hh, 2020]) === nothing
	@test isequal(Array(@evalexpr(db, a[:hh, :] * b[:hh, :])), [nothing, 2.0, 2.0])
	@test isequal(Array(@evalexpr(db, b[:hh, :] / a[:hh, :])), [nothing, 2.0, 2.0])
	@test @evalexpr(db, sum(a[:hh, t] for t in 2020:2022)) === nothing
	@test occursin("nothing", sprint(show, MIME"text/plain"(), @prt(db, a[:hh, :] * b[:hh, :])))
	@test isequal(@evalexpr(:p, db, a[:hh, :]), [NaN, NaN, 0.0])
end

end
