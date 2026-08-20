function _duplicate_variables(vars)
	seen = Set{VariableRef}()
	dups = Set{VariableRef}()
	for var in vars
		var in seen ? push!(dups, var) : push!(seen, var)
	end
	return collect(dups)
end

_is_swap_index(expr) =
	isexpr(expr, :kw) ||
	(isexpr(expr, :call) && !isempty(expr.args) && expr.args[1] in (:in, :∈))

_is_indexed_swap(expr) =
	isexpr(expr, :typed_vcat) ||
	(isexpr(expr, :ref) && any(_is_swap_index, expr.args[2:end]))

"""Helper function for endo_exo_swap! macro — single-pair swap (O(N) scan, no allocation)"""
function _endo_exo_swap!(block::Block, endo::AbstractVariableRef, exo::AbstractVariableRef, error_msg)
	@assert isa(block, Block)

	if !is_endogenous(exo, block)
	    block_vars_preview = join(string.(block.endogenous[1:min(10, length(block.endogenous))]), ", ")
	    if length(block.endogenous) > 10
	        block_vars_preview *= ", ..."
	    end

	    error_parts = ["$exo is not endogenous and cannot be made exogenous: $error_msg"]
	    push!(error_parts, "  Endogenous variables in block ($(length(block.endogenous))): $block_vars_preview")

	    if is_endogenous(endo, block) && exo ∈ block
	        push!(error_parts, "  Did you swap the arguments? Try: @endo_exo_swap!(block, $exo, $endo)")
	    end

	    error(join(error_parts, "\n"))
	end

	if endo ∉ block.variables
	    error("$endo does not appear in the block's constraints and cannot be endogenized: $error_msg")
	end

	idx = findfirst(==(exo), block.endogenous)
	if endo != exo && is_endogenous(endo, block)
	    error("$endo is already endogenous and cannot replace $exo. " *
	          "The swap would create a non-unique equation mapping: $error_msg")
	end
	block.endogenous[idx] = endo
	delete!(block._endogenous_set, exo)
	push!(block._endogenous_set, endo)
end

"""Helper function for endo_exo_swap! macro — batch swap with Dict-based O(1) lookup"""
function _endo_exo_swap!(block::Block, endos, exos, error_msg)
	@assert isa(block, Block)

	if length(endos) != length(exos)
	    endo_names = join(string.(endos), ", ")
	    exo_names = join(string.(exos), ", ")
	    error("Number of variables do not match in endo-exo: $error_msg\n" *
	          "  endo variables ($(length(endos))): $endo_names\n" *
	          "  exo variables ($(length(exos))): $exo_names")
	end

	# Validate the full swap before mutation so errors leave the block unchanged.
	idx_map = Dict{VariableRef, Int}(v => i for (i, v) in enumerate(block.endogenous))
	length(Set(exos)) == length(exos) ||
	    error("Exogenous variables in an endo-exo swap must be unique: $error_msg")
	indices = Int[]

	for (endo, exo) in zip(endos, exos)
	    if !is_endogenous(exo, block)
	        block_vars_preview = join(string.(block.endogenous[1:min(10, length(block.endogenous))]), ", ")
	        if length(block.endogenous) > 10
	            block_vars_preview *= ", ..."
	        end

	        error_parts = ["$exo is not endogenous and cannot be made exogenous: $error_msg"]
	        push!(error_parts, "  Endogenous variables in block ($(length(block.endogenous))): $block_vars_preview")

	        if is_endogenous(endo, block) && exo ∈ block
	            push!(error_parts, "  Did you swap the arguments? Try: @endo_exo_swap!(block, $exo, $endo)")
	        end

	        error(join(error_parts, "\n"))
	    end

	    if endo ∉ block.variables
	        error("$endo does not appear in the block's constraints and cannot be endogenized: $error_msg")
	    end
	    push!(indices, idx_map[exo])
	end

	candidate = copy(block.endogenous)
	for (idx, endo) in zip(indices, endos)
	    candidate[idx] = endo
	end
	duplicates = _duplicate_variables(candidate)
	if !isempty(duplicates)
	    error("Endo-exo swap would create a non-unique equation mapping.\n" *
	          "Duplicate endogenous variables:\n$(format_variables(duplicates))\n" *
	          "Swap: $error_msg")
	end

	copyto!(block.endogenous, candidate)
	empty!(block._endogenous_set)
	union!(block._endogenous_set, candidate)
end

macro _endo_exo_swap_indexed!(block, endos, exos, error_msg)
	jump_expression = GlobalRef(JuMP, Symbol("@expression"))
	all_keys_ref = GlobalRef(@__MODULE__, :_all_keys)
	index_var_ref = GlobalRef(@__MODULE__, :_index_var)
	swap_ref = GlobalRef(@__MODULE__, :_endo_exo_swap!)
	base = _get_name(endos)
	indices = Expr(isexpr(endos, :ref) ? :vect : :vcat, endos.args[2:end]...)
	exos_call = Expr(:macrocall, jump_expression, __source__, :_m, indices, exos)

	return esc(quote
		let _block = $block
			let _m = _block.model
				_exos = $exos_call
				_keys = $all_keys_ref(_exos)
				_endos = [$index_var_ref($base, key) for key in _keys]
				_exo_vars = [_exos[key...] for key in _keys]
				$swap_ref(_block, _endos, _exo_vars, $error_msg)
			end
		end
	end)
end

"""
Macro used to change which variables are matched to the constraints in a Block.
Example:
  @endo_exo_swap!(my_block, MPC, C[t₁])
"""
macro endo_exo_swap!(block, endos, exos)
	error_msg = string(:($endos => $exos))
	if _is_indexed_swap(endos)
		indexed_swap_ref = GlobalRef(@__MODULE__, Symbol("@_endo_exo_swap_indexed!"))
		return esc(Expr(
			:macrocall,
			indexed_swap_ref,
			__source__,
			block,
			endos,
			exos,
			error_msg,
		))
	end
	swap_ref = GlobalRef(@__MODULE__, :_endo_exo_swap!)
	esc(quote
	    $swap_ref($block, $endos, $exos, $error_msg)
	end)
end

"""
Macro used to change which variables are matched to the constraints in a Block.
Example:
  @endo_exo_swap! my_block begin
	    MPC, C[t₁]
	    δ, K[t₁]
	    share[(i, t) in keys(output); t == t₁], output[i, t]
  end
"""
macro endo_exo_swap!(block, expr)
	@assert isa(expr.args[1], LineNumberNode)
	swap_ref = GlobalRef(@__MODULE__, :_endo_exo_swap!)
	indexed_swap_ref = GlobalRef(@__MODULE__, Symbol("@_endo_exo_swap_indexed!"))
	code = Expr(:block)
	line_number = expr.args[1]
	for it in expr.args
		if isa(it, LineNumberNode)
			line_number = it
		elseif _is_indexed_swap(it.args[1])
			push!(code.args, Expr(
				:macrocall,
				indexed_swap_ref,
				line_number,
				block,
				it.args...,
				string(it),
			))
		else
			push!(code.args, :($swap_ref($block, $(it.args[1]), $(it.args[2]), $it)))
		end
	end
	return esc(code)
end
