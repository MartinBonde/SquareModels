module GAMSExt

using SquareModels: SquareModels, ModelDictionary, parse_variable_name, _var_to_key, _build_slice_key
using GAMS: read_gdx
using JuMP: all_variables
using DataFrames: DataFrame, names, eachrow

"""Load from a GDX file using GAMS.jl's read_gdx."""
function SquareModels._load_gdx(path::AbstractString, model, rename_dict::Dict{String, String}, slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}})
	gdx = read_gdx(path)

	# Build index for O(1) lookup: (variable, indices) => value
	data_index = Dict{Tuple{String, String}, Float64}()

	for sym_name in keys(gdx.symbols)
		sym = gdx.symbols[sym_name]
		df = sym.records
		isempty(df) && continue

		# Get the value column name (differs by symbol type)
		value_col = if hasproperty(df, :value)
			:value
		elseif hasproperty(df, :level)
			:level
		else
			continue
		end

		# Get domain columns (all columns except value/level/marginal/etc.)
		domain_cols = [n for n in names(df) if n ∉ ("value", "level", "marginal", "lower", "upper", "scale")]

		for row in eachrow(df)
			indices_str = join([string(row[col]) for col in domain_cols], ",")
			data_index[(string(sym_name), indices_str)] = row[value_col]
		end
	end

	d = ModelDictionary(model)
	for var in all_variables(model)
		base, indices = _var_to_key(var)
		
		# Check for slice mapping first
		if haskey(slice_dict, base)
			gdx_symbol, fixed_indices, wildcard_positions = slice_dict[base]
			lookup_key = _build_slice_key(indices, fixed_indices, wildcard_positions)
			key = (gdx_symbol, lookup_key)
		else
			# Use renamed base if specified, otherwise use original
			lookup_base = get(rename_dict, base, base)
			key = (lookup_base, indices)
		end
		
		if haskey(data_index, key)
			d[var] = data_index[key]
		end
	end
	return d
end

end
