"""
Zip two lists and return any pairs where at least one of the elements appears
more than once in one of the lists
"""
function non_unqiue_pairs(indices, values)
  i_count = countmap(indices)
  v_count = countmap(values)
  result = []
  for (i, v) in zip(indices, values)
	if i_count[i] > 1 || v_count[v] > 1
	  push!(result, (i => v))
	end
  end
  return result
end

"""Replace all variables in `expr` with their values in `d` where possible"""
macro replace_vars(expr, d)
  esc(replace_vars_helper(expr, d))
end

function replace_vars_helper(sym::Symbol, d)
  key = QuoteNode(sym) # QuoteNode is so that we lookup d[:x] instead of d[x] if expr==:x
  return :(haskey($d, $key) ? $d[$key] : $sym)
end
replace_vars_helper(expr::Expr, d) = Expr(expr.head, replace_vars_helper.(expr.args, Ref(d))...)
replace_vars_helper(expr, d) = expr

"""
Compare two expressions evaluated using `first` and `ref` to look up variables
using the supplied compare_function
"""
function call_compare_function(compare_function, expr, first, ref)
  esc(:($compare_function(@replace_vars($expr, $first), @replace_vars($expr, $ref))))
end

# Compare f
_q(a, b) = a ./ b .- 1
_pq(a, b) = (a ./ b .- 1) * 100
_m(a, b) = a .- b

"""Returns multiplier between expression evaluated using `first` and `ref` to look up variables"""
macro q(expr, first, ref)
  call_compare_function(_q, expr, first, ref)
end

"""Returns percentage multiplier between expression evaluated using `first` and `ref` to look up variables"""
macro pq(expr, first, ref)
  call_compare_function(_pq, expr, first, ref)
end

"""Returns difference between expression evaluated using `first` and `ref` to look up variables"""
macro m(expr, first, ref)
  call_compare_function(_m, expr, first, ref)
end
