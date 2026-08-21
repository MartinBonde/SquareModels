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

_swap_binding(axis) = begin
	parsed = _named_constraint_axis(axis)
	parsed === nothing ? nothing : first(parsed)
end
function _swap_binding(axis::Expr)
	if (isexpr(axis, :kw) || isexpr(axis, :(=)))
		return axis.args[1]
	elseif isexpr(axis, :call) && length(axis.args) == 3 && axis.args[1] in (:in, :∈)
		return axis.args[2]
	end
	return nothing
end

function _swap_selector_bindings(selector)
	isexpr(selector, :ref) || isexpr(selector, :typed_vcat) || return nothing
	axes = Any[selector.args[2:end]...]
	if isexpr(selector, :typed_vcat)
		length(axes) == 2 && pop!(axes)
	else
		filter!(axis -> !isexpr(axis, :parameters), axes)
	end
	bindings = map(_swap_binding, axes)
	any(isnothing, bindings) && return nothing
	return bindings
end

function _swap_selector_body(selector)
	bindings = _swap_selector_bindings(selector)
	bindings === nothing && return nothing
	key = Expr(:tuple, bindings...)
	return :($(_index_var)($(_get_name(selector)), $key))
end

function _swap_selection_from_container(container)
	keys = _all_keys(container)
	return [_index_var(container, key) for key in keys]
end

_swap_selection(value::AbstractVariableRef) = [value]
_swap_selection(value::AbstractArray) = _swap_selection_from_container(value)

function _rewrite_swap_selection_keys(expr, jump_expression, source, setup)
	if isexpr(expr, :call) && length(expr.args) == 2 && expr.args[1] == :keys &&
	   _is_indexed_swap(expr.args[2])
		selector = expr.args[2]
		container = gensym(:selection)
		body = _swap_selector_body(selector)
		push!(setup, _named_index_expression_code(
			selector, jump_expression, source, container => body,
		))
		keys = gensym(:keys)
		bindings = _swap_selector_bindings(selector)
		if all(binding -> binding isa Symbol, bindings)
			push!(setup, :($keys = [$(_flatten_key)(key) for key in $(_all_keys)($container)]))
		else
			push!(setup, :($keys = $(_all_keys)($container)))
		end
		return keys
	elseif expr isa Expr
		return Expr(expr.head, (
			_rewrite_swap_selection_keys(arg, jump_expression, source, setup)
			for arg in expr.args
		)...)
	end
	return expr
end

function _swap_selection_code(selector, jump_expression, source)
	setup = Any[]
	selector = _rewrite_swap_selection_keys(selector, jump_expression, source, setup)
	if _is_indexed_swap(selector)
		container = gensym(:selection)
		body = _swap_selector_body(selector)
		push!(setup, _named_index_expression_code(
			selector, jump_expression, source, container => body,
		))
		result = :($(_swap_selection_from_container)($container))
	else
		result = :($(_swap_selection)($selector))
	end
	return Expr(:block, setup..., result)
end

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
	swap_ref = GlobalRef(@__MODULE__, :_endo_exo_swap!)
	endos_code = _swap_selection_code(endos, jump_expression, __source__)
	exos_code = _swap_selection_code(exos, jump_expression, __source__)

	return esc(quote
		let _block = $block
			let _m = _block.model
				_endos = $endos_code
				_exos = $exos_code
				$swap_ref(_block, _endos, _exos, $error_msg)
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
	if _is_indexed_swap(endos) || _is_indexed_swap(exos)
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
	    share[i = I, t = t₁], output[i = I, t = t₁]
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
		elseif _is_indexed_swap(it.args[1]) || _is_indexed_swap(it.args[2])
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
