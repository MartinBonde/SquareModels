module InputOutputPerfFixture

using SquareModels
import JuMP
using JuMP: Model, all_variables
using Ipopt

const scale = parse(Int, get(ENV, "SQUAREMODELS_IO_PERF_SCALE", "40"))
@assert scale >= 2 "SQUAREMODELS_IO_PERF_SCALE must be at least 2"

# Keep the four axes used by InputOutput. Grow product, use, and time with one
# setting. InputOutput has two origins, so keep that axis at its true size.
const product = [Symbol("p", index) for index in 1:scale]
const use = [Symbol("u", index) for index in 1:(scale + 5)]
const origin = [:domestic, :import]
const t = 0:scale
const t1 = 1
const T = last(t)

# Keep one origin for each product-use pair. This gives a large sparse
# four-axis domain without a large data fixture.
const active_p_u_o = Set(
    (p, u, o)
    for (p_index, p) in enumerate(product)
    for (u_index, u) in enumerate(use)
    for (origin_index, o) in enumerate(origin)
    if isodd(p_index + u_index + origin_index)
)

const model = Model(Ipopt.Optimizer)
JuMP.set_silent(model)

# InputOutput creates many linked arrays. Four arrays are enough to keep its
# main costs: a sparse four-axis array, arrays made from its axes, and a second
# sparse four-axis array with the same keys.
@variables model begin
    qUse_p_u_o[p=product, u=use, o=origin, period=t; (p, u, o) in active_p_u_o]
    qPurchaserUse_p_u[(p, u, period)=select_axes(qUse_p_u_o, 1, 2, 4)]
    qSupply_p_o[(p, o, period)=select_axes(qUse_p_u_o, 1, 3, 4)]
    rOriginShare[(p, u, o, period)=qUse_p_u_o]
end

# Make three equation arrays with the same four-axis sums as InputOutput.
function define_equations()
    return @block model begin
        qUse_p_u_o[p=product, u=use, o=origin, period=t1:T; (p, u, o) in active_p_u_o],
        qUse_p_u_o[p,u,o,period] ==
            rOriginShare[p,u,o,period] * qPurchaserUse_p_u[p,u,period]

        qPurchaserUse_p_u[p=product, u=use, period=t1:T],
        qPurchaserUse_p_u[p,u,period] ==
            ∑(qUse_p_u_o[p,u,o,period] for o in origin if (p, u, o) in active_p_u_o)

        qSupply_p_o[p=product, o=origin, period=t1:T],
        qSupply_p_o[p,o,period] ==
            ∑(qUse_p_u_o[p,u,o,period] for u in use if (p, u, o) in active_p_u_o)
    end
end

const variable_count = length(all_variables(model))
const forecast_period_count = length(t1:T)
const equation_count =
    length(active_p_u_o) * forecast_period_count +
    length(product) * length(use) * forecast_period_count +
    length(product) * length(origin) * forecast_period_count

end # module
