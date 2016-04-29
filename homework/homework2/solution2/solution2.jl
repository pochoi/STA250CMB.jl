
using HDF5
bandpowers = h5read("../bandpowers.h5", "bandpowers")

# beam and spectral density are set to somewhat resemble WMAP
const σ² = (10.0/3437.75)^2    # <---10μkarcmin noise level converted to radian pixels (1 radian = 3437.75 arcmin)
const b² = (0.0035)^2          # <-- pixel width 0.2ᵒ ≈ 12.0 armin ≈ 0.0035 radians

ell  = 0:length(bandpowers)-1
clεε = σ² * exp(b² .* ell .* (ell + 1) ./ (8log(2)))

hat_clTT = bandpowers - clεε



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
    exp(-0.5 * norm( p.L\(θ - p.μ) )^2 ) / √((2π)^p.k * p.det)
end

πθ = GaussianCMBPrior(zeros(Float64, 6), eye(Float64, 6, 6)* 5)


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
    factor_ell = -(2*ell + 1)/2
    cldd = pico(θ) + p.σ² * exp(p.b² .* ell .* (ell + 1) ./ (8log(2)))
    pdθ = prod( cldd.^factor_ell .* exp(factor_ell .* p.σℓ ./ cldd) )
    #return p.π(θ) * pdθ
    return cldd,  cldd.^factor_ell, exp(factor_ell .* p.σℓ ./ cldd)
end


post = CMBPosterior(πθ, bandpowers, b², σ²)

histart = [0.0224567, 
0.118489, 
0.128312, 
0.0104098, 
26.857899869292677, 
0.968602]
pico(histart)
hi = post(histart)
find(isinf(hi[3]))



abstract ProposalDensity
immutable GaussainProposal <: ProposalDensity
    g::Float64
    Σ::Matrix{Float64}
    L::LowerTriangular{Float64,Array{Float64,2}}
    det::Float64
    k::Int64
    GaussainProposalDensity(g, Σ) = new(g, Σ, chol(Σ, Val{:L}), det(Σ), size(Σ,1))
end
function call(p::GaussainProposalDensity, θprop, θcur)
    exp(-0.5 * norm( p.L\(θprop - θcur) )^2 ) / √((2π)^p.k * p.det)
end
sampling(p::GaussainProposalDensity, θcur) = θcur + √p.g * p.L * randn(p.k)







function mh(post::PosteriorDensity, prop::ProposalDensity, θₒ, n::Integer)
    θcur = zero(θₒ)
    θprop = zero(θₒ)
    θchain = [zero(θₒ) for k in 1:n]
    θchain[1] = θₒ
    if n == 1
        return θchain
    end
    for k in 2:n
        θcur[:] = θchain[k-1]
        θprop[:] = sampling(prop, θcur)

        if rand() <= min((post(θprop)/post(θcurr))*(prop(θcurr, θprop)/prop(θprop, θcurr)), 1)
            θchain[k][:] = θprop[:]
        else
            θchain[k][:] = θcur[:]
        end
    end
    return θchain
end



function affine_invariant(post::PosteriorDensity ,θwalkers, nitr, a)
    d = length(θwalkers[1])
    nwalkers = length(θwalkers)
    θchain = Array(typeof(θwalkers), nitr)

    θchain[1] = copy(θwalkers)
    for i in 1:nitr-1
        θchain[i+1] = θchain[i]
        for j in 1:nwalkers
            z = (rand() * (√a - 1/√a) + 1/√a)^2
            selectθ = rand(1:nwalkers-1)
            if selectθ >= j
                selectθ += 1
            end
            θprop = θwalkers[selectθ] + z * (θwalkers[j] - θwalkers[selectθ])
            if rand() <= min(z^(d-1) *  post(θprop)/post(θwalkers[j]), 1)
                θchain[i][j] = θprop
            end
        end
    end
end



