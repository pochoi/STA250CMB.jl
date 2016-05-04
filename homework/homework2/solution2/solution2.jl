
using HDF5
bandpowers = h5read("../bandpowers.h5", "bandpowers")

# beam and spectral density are set to somewhat resemble WMAP
const σ² = (10.0/3437.75)^2    # <---10μkarcmin noise level converted to radian pixels (1 radian = 3437.75 arcmin)
const b² = (0.0035)^2          # <-- pixel width 0.2ᵒ ≈ 12.0 armin ≈ 0.0035 radians

ell  = 0:length(bandpowers)-1
clεε = σ² * exp(b² .* ell .* (ell + 1) ./ (8log(2)))

hat_clTT = bandpowers - clεε;

picodata_path = "pico3_tailmonty_v34.dat"
using PyCall
@pyimport pypico
const picoload = pypico.load_pico(picodata_path)

# --------  wrap pico
function pico(x)
    omega_b     = x[1]
    omega_cdm   = x[2]
    tau_reio    = x[3]
    theta_s     = x[4]
    A_s_109     = x[5]
    n_s         = x[6]
    plout::Dict{ASCIIString, Array{Float64,1}} = picoload[:get](;
        :re_optical_depth => tau_reio,
        symbol("scalar_amp(1)") =>  1e-9*A_s_109,
        :theta => theta_s,
        :ombh2 => omega_b,
        :omch2 => omega_cdm,
        symbol("scalar_spectral_index(1)") => n_s,
        :massive_neutrinos => 3.04,
        :helium_fraction => 0.25,
        :omnuh2 => 0.0,
        symbol("scalar_nrun(1)") => 0.0,
        :force     => true
    )
    clTT::Array{Float64,1} = plout["cl_TT"]
    ells   = 0:length(clTT)-1
    clTT .*= 2π ./ ells ./ (ells + 1)
    clTT[1] = 0.0
    return clTT
end

abstract Prior
immutable GaussianCMBPrior <: Prior
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    L::LowerTriangular{Float64,Array{Float64,2}}
    det::Float64
    k::Int64
    GaussianCMBPrior(μ, Σ) = new(μ, Σ, chol(Σ, Val{:L}), det(Σ), size(Σ,1))
end
function call(p::GaussianCMBPrior, θ::Vector{Float64})
    v = p.L\(θ - p.μ)
    return -0.5 * dot(v,v)
end

abstract PosteriorDensity
immutable CMBPosterior <: PosteriorDensity
    π::Prior
    σℓ::Vector{Float64}
    b²::Float64
    σ²::Float64
    CMBPosterior(π, σℓ, b², σ²) = new(π, σℓ, b², σ²)
end
function call(p::CMBPosterior, θ::Vector{Float64})
    ell = 0:length(p.σℓ)-1
    cldd = pico(θ) + p.σ² * exp(p.b² .* ell .* (ell + 1) ./ (8log(2)))
    rtn = 0.0
    @inbounds for l in ell[2:end]
      rtn -= log(cldd[l+1]) * (2l+1) / 2
      rtn -= (p.σℓ[l+1] / cldd[l+1]) * (2l+1) / 2
    end
    return p.π(θ) + rtn
end

abstract ProposalDensity
immutable GaussianProposalDensity <: ProposalDensity
    g::Float64
    Σ::Matrix{Float64}
    L::LowerTriangular{Float64,Array{Float64,2}}
    det::Float64
    k::Int64
    GaussianProposalDensity(g, Σ) = new(g, Σ, chol(Σ, Val{:L}), det(Σ), size(Σ,1))
end
function call(p::GaussianProposalDensity, θprop, θcurr)
    v = p.L\(θprop - θcurr)
    return -0.5 * dot(v,v)
end
sampling(p::GaussianProposalDensity, θcurr) = θcurr + √p.g * p.L * randn(p.k)


wmap_path = "/Users/po/Downloads/wmap_lcdm_wmap9_chains_v5"

# run(`tar -xf wmap_lcdm_wmap9_chains_v5.tar --directory $wmap_path`)
# run(`rm wmap_lcdm_wmap9_chains_v5.tar`)
nchain = 100_000
omega_b_chain     = readdlm("$wmap_path/omegabh2")[1:nchain,2]
omega_cdm_chain   = readdlm("$wmap_path/omegach2")[1:nchain,2]
tau_reio_chain    = readdlm("$wmap_path/tau")[1:nchain,2]
theta_s_chain     = readdlm("$wmap_path/thetastar")[1:nchain,2]
A_s_109_chain     = readdlm("$wmap_path/a002")[1:nchain,2]  # <-- 10⁹ * A_s
n_s_chain         = readdlm("$wmap_path/ns002")[1:nchain,2]
# note: kstar here is 0.002

full_chain    = hcat(omega_b_chain, omega_cdm_chain, tau_reio_chain, theta_s_chain, A_s_109_chain, n_s_chain)
names_chain   = [:omega_b, :omega_cdm, :tau_reio, :theta_s, :A_s_109, :n_s]
wmap_best_fit = vec(mean(full_chain,1))
Σwmap         = cov(full_chain)

function loglike(x::Vector{Float64}, bandpowers::Vector{Float64}, σ²::Float64, b²::Float64)
    ell = 0:length(bandpowers)-1
    cldd = pico(x) + σ² * exp(b² .* ell .* (ell + 1) ./ (8log(2)))
    rtn = 0.0
    @inbounds for l in ell[2:end]
      rtn -= log(cldd[l+1]) * (2l+1) / 2.0
      rtn -= (bandpowers[l+1] / cldd[l+1]) * (2l+1) / 2.0
    end
    return rtn::Float64
end

using NLopt
algm = [:LN_BOBYQA, :LN_COBYLA, :LN_PRAXIS, :LN_NELDERMEAD, :LN_SBPLX]
llmin(x, grad)  = loglike(x, bandpowers, σ², b²)

opt = Opt(algm[1], 6)
#initial_step!(opt, dx)
upper_bounds!(opt, [0.034, 0.2,  0.55,  .0108, exp(4.0)/10,  1.25])
lower_bounds!(opt, [0.018, 0.06, 0.01,  .0102, exp(2.75)/10, 0.85])  # <-- pico training bounds
maxtime!(opt, 5*60.0)   # <--- max time in seconds
max_objective!(opt, llmin)

optf, optx, ret = optimize(opt, wmap_best_fit);
hcat(names_chain, optx, wmap_best_fit)

mle_sim = copy(optx)

using ProgressMeter
function mh(post::PosteriorDensity, prop::ProposalDensity, θₒ, n::Integer)
    θcurr = zero(θₒ)
    θprop = zero(θₒ)
    θchain = [zero(θₒ) for k in 1:n]
    θchain[1] = θₒ
    logα = zeros(Float64, n)
    if n == 1
        return θchain
    end
    @showprogress 1 "MH MCMC... " for k in 2:n
        θcurr[:] = θchain[k-1]
        θprop[:] = sampling(prop, θcurr)
        logα[k] = min((post(θprop)-post(θcurr))+(prop(θcurr, θprop) - prop(θprop, θcurr)), 0)
        if log(rand()) <= logα[k]
            θchain[k][:] = θprop[:]
        else
            θchain[k][:] = θcurr[:]
        end
    end
    return θchain, logα
end

πθ = GaussianCMBPrior(mle_sim, eye(Float64, 6, 6) * 1e14);
postθcmb = CMBPosterior(πθ, bandpowers, b², σ²);
propθ = GaussianProposalDensity(0.02, Σwmap);

nnmh = 20000
mhθchain, α = mh(postθcmb, propθ, mle_sim, nnmh);

mean(exp(α))

using Gadfly
plot(x = 1:10:nnmh, y = [mhθchain[k][1] for k in 1:10:nnmh], Geom.line)

mhθsample = Vector{Float64}[]
for i in 100:5:nnmh
    push!(mhθsample, mhθchain[i])
end
mhθresult = zeros(Float64, length(mhθsample), 6)
for i in 1:length(mhθsample)
    mhθresult[i,:] = copy(mhθsample[i])
end

Σmh = cov(mhθresult)

nnmh_again = 30000
πθ = GaussianCMBPrior(mle_sim, eye(Float64, 6, 6) * 1e12);
postθcmb = CMBPosterior(πθ, bandpowers, b², σ²);
propθ_again = GaussianProposalDensity(0.2, Σmh);
mhθchain_again, α_again = mh(postθcmb, propθ_again, mle_sim, nnmh_again);

mean(exp(α_again))

plot(x = 1:10:nnmh_again, y = [mhθchain_again[k][1] for k in 1:10:nnmh_again], Geom.line)

mhθsample_again = Vector{Float64}[]
for i in 100:5:nnmh_again
    push!(mhθsample_again, mhθchain_again[i])
end
mhθresult_again = zeros(Float64, length(mhθsample_again), 6)
for i in 1:length(mhθsample_again)
    mhθresult_again[i,:] = copy(mhθsample_again[i])
end

function affine_invariant(post::PosteriorDensity, θwalkers::Vector{Vector{Float64}}, nitr::Int64, a::Float64)
    d = length(θwalkers[1])
    nwalkers = length(θwalkers)
    θchain = Array(Vector{Vector{Float64}}, nitr)
    for i in 1:nitr
        θchain[i] = Vector{Float64}[zeros(Float64, d) for k in 1:nwalkers]
    end
    
    θprop = zeros(Float64, d)
    
    θchain[1] = copy(θwalkers)
    @showprogress 2 "Affine Invariant... " for i in 1:nitr-1
        θchain[i+1][:] = deepcopy(θchain[i])
        zj = (rand(nwalkers) .* (√a - 1/√a) + 1/√a).^2
        selectj = rand(1:nwalkers-1, nwalkers)
        selectj[selectj .>= 1:nwalkers] += 1
        
        for j in 1:nwalkers
            θprop[:] = θchain[i][selectj[j]] + zj[j] * (θchain[i][j] - θchain[i][selectj[j]])
            logα = min((d-1) * log(zj[j]) + post(θprop) - post(θchain[i][j]), 0)
            if log(rand()) <= logα
                θchain[i+1][j] = copy(θprop)
            end
        end
    end
    
    return θchain
end


θwalkers = Vector{Float64}[mle_sim + diagm( 2 * sqrt(diag(Σwmap))) * randn(6) for k in 1:50];

nnai = 500
@time aiθchain = affine_invariant(postθcmb, θwalkers, nnai, 2.0);

plot(
layer(x = omega_b_chain[1:length(θwalkers)], 
y = omega_cdm_chain[1:length(θwalkers)], 
Geom.point, 
Theme(default_color = colorant"blue")),
layer(x = Float64[θwalkers[k][1] for k in 1:length(θwalkers)], 
y = Float64[θwalkers[k][2] for k in 1:length(θwalkers)], 
Geom.point, 
Theme(default_color = colorant"red")),
layer(x = Float64[aiθchain[nnai][k][1] for k in 1:length(θwalkers)], 
y = Float64[aiθchain[nnai][k][2] for k in 1:length(θwalkers)], 
Geom.point, 
Theme(default_color = colorant"green")),
Guide.xlabel("omega_b"),  Guide.ylabel("omega_cdm"),
Guide.title("End Walkers"),
Guide.manual_color_key("Legend", ["WMAP MCMC", "Initial walkers", "End walkers"], ["blue", "red", "green"])
)

plot(
layer(x = Float64[aiθchain[nnai][k][1] for k in 1:length(θwalkers)], 
y = Float64[aiθchain[nnai][k][2] for k in 1:length(θwalkers)], 
Geom.point, 
Theme(default_color = colorant"green")),
layer(x = Float64[aiθchain[100][k][1] for k in 1:length(θwalkers)], 
y = Float64[aiθchain[100][k][2] for k in 1:length(θwalkers)], 
Geom.point, 
Theme(default_color = colorant"blue")),
Guide.xlabel("omega_b"),  Guide.ylabel("omega_cdm"),
Guide.title("End Walkers"),
Guide.manual_color_key("Legend", ["walkers at itr 500", "walkers at itr 100"], ["blue", "green"])
)

aiθsample = Vector{Float64}[]
for i in 50:5:nnai
    append!(aiθsample, aiθchain[i])
end
ai_result = zeros(Float64, length(aiθsample), 6)
for i in 1:length(aiθsample)
    ai_result[i,:] = copy(aiθsample[i])
end

plot(x = 1:length(aiθsample), y = [aiθsample[k][6] for k in 1:length(aiθsample) ], Geom.line)

using PyCall, PyPlot

@pyimport getdist
@pyimport getdist.plots as plots
samples = getdist.MCSamples(samples=mhθresult_again, names=names_chain)
g = plots.getSubplotPlotter()
g[:triangle_plot](samples, filled=true, legend_labels = ["MH"])

samples = getdist.MCSamples(samples=ai_result, names=names_chain)
g = plots.getSubplotPlotter()
g[:triangle_plot](samples, filled=true, legend_labels = ["AI"])

samples1 = getdist.MCSamples(samples=mhθresult_again, names=names_chain)
samples2 = getdist.MCSamples(samples=ai_result, names=names_chain)
g = plots.getSubplotPlotter()
g[:triangle_plot]([samples1, samples2], filled=true, legend_labels = ["MH", "AI"])

samples0 = getdist.MCSamples(samples=full_chain, names=names_chain)
samples1 = getdist.MCSamples(samples=mhθresult_again, names=names_chain)
samples2 = getdist.MCSamples(samples=ai_result, names=names_chain)
g = plots.getSubplotPlotter()
g[:triangle_plot]([samples0, samples1, samples2], filled=true, legend_labels = ["WMAP", "MH", "AI"])
