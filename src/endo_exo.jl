"""Helper function for endo_exo! macro"""
function _endo_exo!(block::Block, endos, exos, error_msg)
	@assert isa(block, Block)

	if length(endos) != length(exos)
	    endo_names = join(string.(endos), ", ")
	    exo_names = join(string.(exos), ", ")
	    error("Number of variables do not match in endo-exo: $error_msg\n" *
	          "  endo variables ($(length(endos))): $endo_names\n" *
	          "  exo variables ($(length(exos))): $exo_names")
	end

	for (endo, exo) in zip(endos, exos)
	    if exo ∉ block
	        block_vars_preview = join(string.(block.endogenous[1:min(10, length(block.endogenous))]), ", ")
	        if length(block.endogenous) > 10
	            block_vars_preview *= ", ..."
	        end

	        error_parts = ["$exo is not endogenous and cannot be used in endo-exo: $error_msg"]
	        push!(error_parts, "  Variables in block ($(length(block.endogenous))): $block_vars_preview")

	        if endo ∈ block
	            push!(error_parts, "  NOTE: $endo is already in the block. Did you mean @endo_exo!(block, $exo, $endo)?")
	        end

	        error(join(error_parts, "\n"))
	    end

	    # Find index of exo variable and replace with endo
	    idx = findfirst(==(exo), block.endogenous)
	    block.endogenous[idx] = endo

	    # Update the set
	    delete!(block._endogenous_set, exo)
	    push!(block._endogenous_set, endo)
	end
end
function _endo_exo!(block::Block, endos::AbstractVariableRef, exos::AbstractVariableRef, error_msg)
	_endo_exo!(block, [endos], [exos], error_msg)
end

"""
Macro used to change which variables are matched to the constraints in a Block.
Example:
  @endo_exo!(my_block, MPC, C[t₁])
"""
macro endo_exo!(block, endos, exos)
	error_msg = string(:($endos => $exos))
	esc(quote
	    SquareModels._endo_exo!($block, $endos, $exos, $error_msg)
	end)
end

"""
Macro used to change which variables are matched to the constraints in a Block.
Example:
  @endo_exo! my_block begin
	    MPC, C[t₁]
	    δ, K[t₁]
  end
"""
macro endo_exo!(block, expr)
	@assert isa(expr.args[1], LineNumberNode)
	code = Expr(:block)
	for it in expr.args
	    if !isa(it, LineNumberNode)
	        call = :(SquareModels._endo_exo!($block, $(it.args[1]), $(it.args[2]), $it))
	        push!(code.args, call)
	    end
	end
	return esc(code)
end
