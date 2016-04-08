
using HDF5
using Gadfly
using Colors

################
# Cov function #
################

abstract Cov
immutable Cov1 <: Cov end
immutable Cov2 <: Cov end
immutable Cov3 <: Cov end

call(::Cov1, x, y) = exp(-norm(x-y))
call(::Cov2, x, y) = (norm(x)^0.85 + norm(y).^0.85 - norm(x-y).^0.85)
function call(::Cov3, x, y)
    ν    = 1.2
    ρ    = 0.01
    σ²   = 1.0
    arg  = √(2ν/ρ) * norm(x-y)
    if arg == 0.0
        return σ²
    else
        rtn  = arg^ν
        rtn *= besselk(ν, arg)
        rtn *= σ² * 2^(1-ν) / gamma(ν)
        return rtn
    end
end



######
# 1D #
######
function grf1d(cov::Cov, nx, x1d_obs, fx1d_obs, nsim)
    n         = length(fx1d_obs)
    x1d_pre    = linspace(-.1, 1.1, nx)

    Σobs       = zeros(Float64, length(x1d_obs), length(x1d_obs))
    Σcross     = zeros(Float64, length(x1d_pre), length(x1d_obs))
    Σpre       = zeros(Float64, length(x1d_pre), length(x1d_pre))

    covmat1d!(Σobs, Σcross, Σpre, cov, x1d_pre, x1d_obs)

    μcond = Σcross * ( Σobs \ fx1d_obs )
    Σcond = Σpre - Σcross * ( Σobs \ (Σcross).' )
    Lcond = chol(Σcond, Val{:L})
    fx1d_pre = [μcond + Lcond * randn(nx) for k in 1:nsim]
    return x1d_pre, fx1d_pre, μcond, diag(Σcond)
end


function covmat1d!(Σobs, Σcross, Σpre, cov::Cov, x1d_pre, x1d_obs)
    for i in eachindex(x1d_obs), j in eachindex(x1d_obs)
        Σobs[i,j] = cov(x1d_obs[i], x1d_obs[j])
    end
    for i in eachindex(x1d_obs)
        Σobs[i,i] += (0.15)^2
    end
    for i in eachindex(x1d_pre), j in eachindex(x1d_obs)
        Σcross[i, j] = cov(x1d_pre[i], x1d_obs[j])
    end
    for i in eachindex(x1d_pre), j in eachindex(x1d_pre)
        Σpre[i, j] = cov(x1d_pre[i], x1d_pre[j])
    end
    nothing
end


function plot1d(cov::Cov, nx::Integer, x1d_obs::AbstractVector, fx1d_obs, nsim::Integer)
    x1d_pre, fx1d_pre, μcond, σcond = grf1d(cov, nx, x1d_obs, fx1d_obs, nsim)
    plot1d(cov, x1d_pre, fx1d_pre, μcond, σcond)
end

function plot1d(cov::Cov, x1d_pre::AbstractVector, fx1d_pre, μcond, σcond)
    pobs = layer(x = x1d_obs, y = fx1d_obs, Geom.point, Theme(default_color=colorant"red", default_point_size = 0.06cm) )
    pl = [layer(x = x1d_pre, y = fx1d_pre[k], Geom.line, Theme(default_color=RGBA{Float32}(0, 0, 1, 0.2))) for k in 1:nsim]
    pribbon = layer(x = x1d_pre, y = μcond, ymin = μcond - 2σcond, ymax = μcond + 2σcond, Geom.ribbon, 
        Theme(lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2)))
    plot(pribbon, pl..., pobs, Guide.title("$(typeof(cov))"))
end

######
# 2D #
######
function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end
meshgrid(v)  = meshgrid(v,v)

function grf2d(cov::Cov, mesh_side::Integer, x2d_obs::Matrix, fx2d_obs::Vector, nsim::Integer)
    xmesh, ymesh = meshgrid(linspace(-.1, 1.1, mesh_side))

    n = size(x2d_obs, 1)
    m = length(xmesh)

    Σobs = Array(Float64, n, n)
    Σcross = Array(Float64, m, n)
    Σpre   = Array(Float64, m, m)

    covmat2d!(Σobs, Σcross, Σpre, cov, xmesh, ymesh, x2d_obs)    

    μcond = Σcross * ( Σobs \ fx2d_obs )
    Σcond = Σpre - Σcross * ( Σobs \ (Σcross).' )
    Lcond = chol(Σcond, Val{:L})
    fx2d_pre = [μcond + Lcond * randn(m) for k in 1:nsim]
    return xmesh, ymesh, fx2d_pre
end

function covmat2d!(Σobs, Σcross::Matrix, Σpre::Matrix, cov::Cov, xmesh, ymesh, x2d_obs)
    n = size(x2d_obs, 1)
    m = length(xmesh)

    @inbounds for i in 1:n, j in 1:n
        Σobs[i,j] = cov(vec(x2d_obs[i,:]), vec(x2d_obs[j,:]))
    end
    @inbounds for i in 1:n
        Σobs[i,i] += (0.1)^2
    end

    @inbounds for col in 1:n, row in 1:m
        Σcross[row, col] = cov([xmesh[row], ymesh[row]], vec(x2d_obs[col,:]))
    end
    @inbounds for col in 1:m, row in 1:m
        Σpre[row, col] = cov([xmesh[row], ymesh[row]], [xmesh[col], ymesh[col]])
    end
    nothing
end

function spyplot(cov::Cov, xmesh, ymesh, fx2d_pre, x2d_obs, fx2d_obs)
    spylayer = layer(x=xmesh, y=ymesh, color=fx2d_pre,
        Geom.rectbin)
    obslayer = layer(x = x2d_obs[:,1], y = x2d_obs[:,2], order = 2, Geom.point, 
        Theme(highlight_width=0.4mm, discrete_highlight_color=c->colorant"black", default_color=colorant"transparent")
    )
    plot(spylayer, 
        obslayer,
        Scale.ContinuousColorScale(Scale.lab_gradient(colorant"blue",
                                                   colorant"yellow",
                                                   colorant"red")),
        Coord.cartesian(yflip=true, fixed=true, xmin=minimum(xmesh), xmax=maximum(xmesh), ymin=minimum(ymesh), ymax=maximum(ymesh)),
        Scale.x_continuous,
        Scale.y_continuous,
        Guide.title("$(typeof(cov))"))
end

function spyplot(cov::Cov, mesh_side::Integer, x2d_obs::Matrix, fx2d_obs::Vector)
    xmesh, ymesh, fx2d_pre = grf2d(cov, mesh_side, x2d_obs, fx2d_obs, 1)
    spyplot(cov, xmesh, ymesh, fx2d_pre[1], x2d_obs, fx2d_obs)
end

##############
# likelihood #
##############

function calculateΣ(cov::Cov, x1d_obs::Vector)
    n = length(x1d_obs)
    Σ         = Float64[cov(xi,yi) for xi in x1d_obs, yi in x1d_obs]
    Σobs      = Σ +  (0.15)^2 * eye(n)
    return Σobs
end

function calculateΣ(cov::Cov, x2d_obs::Matrix)
    n = size(x2d_obs, 1)
    Σ         = Float64[cov(vec(x2d_obs[i,:]), vec(x2d_obs[j,:])) for i in 1:n, j in 1:n]
    Σobs      = Σ +  (0.1)^2 * eye(n)
    return Σobs
end

calculateΣ(cov::Vector{Cov}, x_obs) = Matrix{Float64}[calculateΣ(cov[k], x_obs) for k in eachindex(cov)]

function loglikelihood(Σ::Matrix{Float64}, x)
    k = size(Σ, 1)
    return -0.5 * log(det(Σ)) - 0.5 * dot(x, Σ \ x) - 0.5 * k * log(2π)
end

loglikelihood(Σ::Vector{Matrix{Float64}}, x) = [ loglikelihood(Σ[k], x) for k in eachindex(Σ)]

