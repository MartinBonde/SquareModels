# TableDisplay - Shared PrettyTables layout and rendering.
# Handles dense arrays and stored sparse cells.
# Does not materialize implicit sparse cells.

using PrettyTables: pretty_table
using Dates: TimeType

const _SparseTableArray = Union{SparseZeroArray,SparseAxisArray}
const _COLUMN_LABEL_MIN_WIDTH = 10
const DEFAULT_COLUMN_LABEL_TOTAL_WIDTH = Ref(72)

struct _TableLayout{T,P,C}
	data::Matrix{T}
	periods::P
	combos::C
end

_sparse_axis_array(data::SparseZeroArray) = data.data
_sparse_axis_array(data::SparseAxisArray) = data

_leading_combos(dims) = length(dims) == 1 ? [()] : vec(collect(Iterators.product(dims[1:end-1]...)))
_order_periods(periods) =
	(all(period -> period isa Real, periods) || all(period -> period isa TimeType, periods)) ? sort(periods) : periods

function _table_layout(data::AbstractArray, dims)
	periods = collect(dims[end])
	combos = _leading_combos(dims)
	matrix = permutedims(reshape(collect(data), length(combos), length(periods)))
	return _TableLayout(matrix, periods, combos)
end

function _sparse_table_layout(keys, values)
	@assert length(keys) == length(values)
	isempty(keys) && return _TableLayout(Matrix{Any}(undef, 0, 0), Any[], Tuple[])
	# Sparse containers have no declared axis order. Sort numeric and date-like
	# periods; keep first-seen order for labels that need not define `isless`.
	periods = _order_periods(unique(key[end] for key in keys))
	combos = unique(Base.front(key) for key in keys)
	period_index = Dict(period => i for (i, period) in enumerate(periods))
	combo_index = Dict(combo => i for (i, combo) in enumerate(combos))
	matrix = fill!(Matrix{Any}(undef, length(periods), length(combos)), "")
	for (key, value) in zip(keys, values)
		matrix[period_index[key[end]], combo_index[Base.front(key)]] = value
	end
	return _TableLayout(matrix, periods, combos)
end

function _table_layout(data::_SparseTableArray)
	sparse = _sparse_axis_array(data)
	return _sparse_table_layout(collect(keys(sparse.data)), collect(values(sparse.data)))
end

_column_label(name, combo) = isempty(combo) ? name : (isempty(name) ? join(combo, ", ") : "$name[$(join(combo, ", "))]")

function _source_expr_parts(name)
	parts = split(name, '\n', limit=2)
	length(parts) == 1 && return nothing
	return (parts[1], parts[2])
end

_column_label_width(ncols) = max(_COLUMN_LABEL_MIN_WIDTH, DEFAULT_COLUMN_LABEL_TOTAL_WIDTH[] ÷ max(ncols, 1))

function _wrap_label(label, width)
	isempty(label) && return [""]
	characters = collect(label)
	n = length(characters)
	n <= width && return [label]
	return [String(characters[i:min(i + width - 1, n)]) for i in 1:width:n]
end

# PrettyTables crops long labels instead of adding label rows.
function _column_labels(labels)
	isempty(labels) && return Vector{Vector{String}}()
	splits = _source_expr_parts.(labels)
	width = _column_label_width(length(labels))
	wrapped = [_wrap_label(isnothing(s) ? label : s[2], width) for (label, s) in zip(labels, splits)]
	n = maximum(length, wrapped)
	rows = [[i <= length(w) ? w[i] : "" for w in wrapped] for i in 1:n]
	any(!isnothing, splits) && pushfirst!(rows, [isnothing(s) ? "" : s[1] for s in splits])
	return rows
end

function _print_table(
	io::IO,
	data;
	fit_table_in_display_vertically=get(io, :limit, false),
	kwargs...,
)
	return pretty_table(io, data; fit_table_in_display_vertically, kwargs...)
end

function _print_period_table(io::IO, data, periods, labels)
	_print_table(io, data;
		column_labels=_column_labels(labels),
		row_labels=string.(periods),
		stubhead_label="year")
end

function _period_row_table(io::IO, layout::_TableLayout, name="")
	isempty(layout.combos) && return print(io, "0-element table")
	labels = [_column_label(name, combo) for combo in layout.combos]
	return _print_period_table(io, layout.data, layout.periods, labels)
end
