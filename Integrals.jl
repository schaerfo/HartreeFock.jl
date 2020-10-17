module Integrals

using LinearAlgebra: norm
using StaticArrays

@enum OrbitalType s=0b0 px=0b01 py=0b10 pz=0b11 #dx²=0b101 dxy=0b110 dxz=0b111 dy²=0b1010 dyz=0b1011 dz²=0b1111

struct Orbital
    center::SVector{3, Float64}
    primitives::Vector{Tuple{Float64, Float64}}
    type::OrbitalType
end

is_s_orbital(o::Orbital) = o.type == s
function is_p_orbital(o::Orbital)
    compare_type = Base.Enums.basetype(OrbitalType)
    bitmask = compare_type(0b11)
    t = compare_type(o.type)
    (~bitmask & t) == 0 && (bitmask & t) != 0
end

one_electron_indices(n) = [(i, j) for i in 1:n for j in i:n]

function add_one_electron_integrals(orbital_a, orbital_b, mol, integral_func)
    res = 0.
    for (α, a) in orbital_a.primitives
        for (β, b) in orbital_b.primitives
            res += a * b * integral_func(α, orbital_a.center, Int(orbital_a.type), β, orbital_b.center, Int(orbital_b.type), mol)
        end
    end
    res
end

function one_electron_matrix(orbitals, indices, mol, integral_functions)
    @assert length(indices) == length(orbitals) * (length(orbitals) + 1) / 2
    res = zeros(Float64, (length(orbitals), length(orbitals)))
    for (i, j) in indices
        orbital_a = orbitals[i]
        orbital_b = orbitals[j]

        if is_s_orbital(orbital_a) && is_p_orbital(orbital_b)
            orbital_a, orbital_b = orbital_b, orbital_a
        end

        func = if is_s_orbital(orbital_a)
                   integral_functions[1]
               elseif is_s_orbital(orbital_b)
                   integral_functions[2]
               else
                   integral_functions[3]
               end

        res[j, i] = res[i, j] = add_one_electron_integrals(orbital_a, orbital_b, mol, func)

    end
    res
end

erf(x::Float64) = ccall("erf", Float64, (Float64,), x)
exp(x::Float64) = ccall("exp", Float64, (Float64,), x)

g(α, a, β, b) = exp(-α * β / (α + β) * norm(a - b)^2)
s1(t) = abs(t) < 1e-15 ? 2. / sqrt(π) : erf(t)/t
s2(t) = abs(t) < 1e-5  ? -4. / (3sqrt(π)) : (2π^-.5 * t * exp(-t^2) - erf(t)) / t^3
s3(t) = abs(t) < 1e-3  ? 8 / (5sqrt(π)) : (3erf(t) - 2π^-.5 * (3t + 2t^3) * exp(-t*t)) / (t*t*t*t*t)
s4(t) = abs(t) < 1e-15^(1/7) ? -16 / 7sqrt(π) : (2π^-.5 * (15t + 10t^3 + 4t^5) * exp(-t^2) - 15erf(t)) / t^7
s5(t) = abs(t) < 1e-15^(1/9) ? 32 / 9sqrt(π) : (105erf(t) - 2π^-.5 * (105t + 70t^3 + 28t^5 + 8t^7) * exp(-t^2)) / t^9

overlap_ss(α, a, _, β, b, _, _) = (π / (α + β))^1.5 * g(α, a, β, b)
overlap_ps(α, a, i, β, b, _, _) = -g(α, a, β, b) * β * π^1.5 / (α + β)^2.5 * (a[i] - b[i])
overlap_pp(α, a, i, β, b, j, _) = g(α, a, β, b) * π^1.5 / (α + β)^2.5 * (0.5 * (i == j) - α * β / (α + β) * (a[i] - b[i]) * (a[j] - b[j]))
overlap_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (overlap_ss, overlap_ps, overlap_pp))

kinetic_ss(α, a, _, β, b, _, _) = g(α, a, β, b) * α * β * π^1.5 / (α + β)^2.5 * (3 - 2α * β / (α + β) * norm(a - b)^2)
kinetic_ps(α, a, i, β, b, _, _) = g(α, a, β, b) * α * β^2 * π^1.5 / (α + β)^3.5 * (2α * β / (α + β) * norm(a - b)^2 - 5) * (a[i] - b[i])
kinetic_pp(α, a, i, β, b, j, _) = g(α, a, β, b) * α * β * π^1.5 / (α + β)^3.5 * ((2.5 - α * β / (α + β) * norm(a - b)^2) * (i == j) + α * β / (α + β) * (2α * β / (α + β) * norm(a - b)^2 - 7) * (a[i] - b[i]) * (a[j] - b[j]))
kinetic_energy_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (kinetic_ss, kinetic_ps, kinetic_pp))

function nuclear_loop(mol, func)
    res = 0.
    for curr_atom in mol.atoms
        res += curr_atom.charge * func(curr_atom.pos)
    end
    res
end

function nuclear_ss(α, a, _, β, b, _, mol)
    η = α + β
    return nuclear_loop(mol, c -> begin
        r = (α * a + β * b) / η - c
        t = sqrt(η) * norm(r)
        -g(α, a, β, b) * π^1.5 / η * s1(t)
    end)
end

function nuclear_ps(α, a, i, β, b, _, mol)
    η = α + β
    return nuclear_loop(mol, c -> begin
        r = (α * a + β * b) / η - c
        t = sqrt(η) * norm(r)
        -g(α, a, β, b) * π^1.5 / 2η * (r[i] * s2(t) - 2β * (a[i] - b[i]) / η * s1(t))
    end)
end

function nuclear_pp(α, a, i, β, b, j, mol)
    η = α + β
    δᵢⱼ = i == j
    return nuclear_loop(mol, c -> begin
        r = (α * a + β * b) / η - c
        t = sqrt(η) * norm(r)
        -g(α, a, β, b) * π^1.5 / 4η^2 * (η * r[i] * r[j] * s3(t) + (δᵢⱼ + 2α * (a[j] - b[j]) * r[i] - 2β * (a[i] - b[i]) * r[j]) * s2(t) + (2δᵢⱼ - 4α * β * (a[i] - b[i]) * (a[j] - b[j]) / η) * s1(t))
    end)
end
nuclear_potential_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (nuclear_ss, nuclear_ps, nuclear_pp))

function two_electron_indices(n::UInt)
    res = Vector{Tuple{UInt16, UInt16, UInt16, UInt16}}(undef, (n^4 + 2n^3 + 3n^2 + 2n) >> 3)
    n = UInt16(n)
    i = UInt(0)
    for μ ∈ 0x1:n
        for ν ∈ 0x1:μ
            for λ ∈ 0x1:μ
                for σ ∈ 0x1:(λ == μ ? ν : λ)
                    res[i += 1] = (μ, ν, λ, σ)
                end
            end
        end
    end
    res
end

two_electron_indices(n) = two_electron_indices(UInt(n))

function coulomb_ssss(α, a, _, β, b, _, γ, c, _, δ, d, _)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (η * θ)^1.5 * s1(t)
end

function coulomb_psss(α, a, i, β, b, _, γ, c, _, δ, d, _)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (2η^2.5 * θ^1.5) * (q^2 * r[i] * s2(t) - 2β * (a[i] - b[i]) * s1(t))
end

function coulomb_ppss(α, a, i, β, b, j, γ, c, _, δ, d, _)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    δᵢⱼ = i == j
    δabᵢ = a[i] - b[i]
    δabⱼ = a[j] - b[j]
    g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (4η^3.5 * θ^1.5) * (q^4 * r[i] * r[j] * s3(t) + q^2 * (δᵢⱼ + 2α * δabⱼ * r[i] - 2β * δabᵢ * r[j]) * s2(t) + (2η * δᵢⱼ - 4α * β * δabᵢ * δabⱼ) * s1(t))
end

function coulomb_psps(α, a, i, β, b, _, γ, c, k, δ, d, _)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    δabᵢ = a[i] - b[i]
    δcdₖ = c[k] - d[k]
    g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (4η^2.5 * θ^2.5) * (-q^4 * r[i] * r[k] * s3(t) + q^2 * (2β * δabᵢ * r[k] - 2δ * δcdₖ * r[i] - (i == k)) * s2(t) + 4β * δ * δabᵢ * δcdₖ * s1(t))
end

function coulomb_ppps(α, a, i, β, b, j, γ, c, k, δ, d, _)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    δᵢⱼ = i == j
    δⱼₖ = j == k
    δabᵢ = a[i] - b[i]
    δabⱼ = a[j] - b[j]
    δcdₖ = c[k] - d[k]
    u1 = -4η * δ * δcdₖ * δᵢⱼ
    u2 = 2 * (2β * δ * δabᵢ * δcdₖ * r[j] + β * δabᵢ * δⱼₖ - η * r[k] * δᵢⱼ - δ * δcdₖ * δᵢⱼ)
    u3 = 2β * δabᵢ * r[j] * r[k] - 2δ * δcdₖ * r[i] * r[j] - r[k] * δᵢⱼ - r[i] * δⱼₖ - r[j] * (i == k)
    u4 = -r[i] * r[j] * r[k]
    α / η * δabⱼ * coulomb_psps(α, a, i, β, b, j, γ, c, k, δ, d, 0) + g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (8η^3.5 * θ^2.5) * (q^6 * u4 * s4(t) +q^4 * u3 * s3(t) + q^2 * u2 * s2(t) + u1 * s1(t))
end

function coulomb_pppp(α, a, i, β, b, j, γ, c, k, δ, d, l)
    η = α + β
    θ = γ + δ
    q = sqrt(η * θ / (η + θ))
    r = (α * a + β * b) / η - (γ * c + δ * d) / θ
    t = q * norm(r)
    δᵢⱼ = i == j
    δᵢₗ = i == l
    δⱼₖ = j == k
    δⱼₗ = j == l
    δₖₗ = k == l
    δabᵢ = a[i] - b[i]
    δabⱼ = a[j] - b[j]
    δcdₖ = c[k] - d[k]
    δcdₗ = c[l] - d[l]
    u1 = -4η * δ * δcdₖ * δᵢⱼ
    u2 = 2 * (2β * δ * δabᵢ * δcdₖ * r[j] + β * δabᵢ * δⱼₖ - η * r[k] * δᵢⱼ - δ * δcdₖ * δᵢⱼ)
    u3 = 2β * δabᵢ * r[j] * r[k] - 2δ * δcdₖ * r[i] * r[j] - r[k] * δᵢⱼ - r[i] * δⱼₖ - r[j] * (i == k)
    u4 = -r[i] * r[j] * r[k]
    v1 = 4η * θ * δᵢⱼ * δₖₗ
    v2 = 2 * ((η + θ) * δᵢⱼ * δₖₗ - 2β * θ * δabᵢ * r[j] * δₖₗ - 2β * δ * δabᵢ * δcdₖ * δⱼₗ)
    v3 = 2θ * r[i] * r[j] * δₖₗ + 2δ * δcdₖ * (r[i] * δⱼₗ + r[j] * δᵢₗ) - 2β * δabᵢ * (r[k] * δⱼₗ + r[j] * δₖₗ) + δᵢⱼ * δₖₗ + δᵢₗ * δⱼₖ + (i == k) * δⱼₗ
    v4 = r[i] * r[k] * δⱼₗ + r[i] * r[j] * δₖₗ + r[j] * r[k] * δᵢₗ
    α / η * δabⱼ * coulomb_ppps(γ, c, k, δ, d, l, α, a, i, β, b, j) + g(α, a, β, b) * g(γ, c, δ, d) * q * π^3 / (16η^3.5 * θ^3.5) *
        (-q^8 * u4 * r[l] * s5(t) +
         q^6 * (v4 + 2γ * δcdₗ * u4 - u3 * r[l]) * s4(t) +
         q^4 * (v3 + 2γ * δcdₗ * u3 - u2 * r[l]) * s3(t) +
         q^2 * (v2 + 2γ * δcdₗ * u2 - u1 * r[l]) * s2(t) +
               (v1 + 2γ * δcdₗ * u1) * s1(t))
end

function add_two_electron_integrals(r, s, t, u, func)
    res = 0.
    for (α, a) in r.primitives
        for (β, b) in s.primitives
            for (γ, c) in t.primitives
                for (δ, d) in u.primitives
                    res += a * b * c * d * func(α, r.center, Int(r.type),
                                                β, s.center, Int(s.type),
                                                γ, t.center, Int(t.type),
                                                δ, u.center, Int(u.type))
                end
            end
        end
    end
    res
end

function dispatch_two_electron_integral(r, s, t, u, ssss, psss, psps, ppss, ppps, pppp)
    if is_p_orbital(u)
        @assert is_p_orbital(r) && is_p_orbital(s) && is_p_orbital(t) && is_p_orbital(u)
        return coulomb_pppp
    elseif is_p_orbital(t) && is_p_orbital(s)
        @assert is_p_orbital(r) && is_p_orbital(s) && is_p_orbital(t) && is_s_orbital(u)
        return coulomb_ppps
    elseif is_p_orbital(s)
        @assert is_p_orbital(r) && is_p_orbital(s) && is_s_orbital(t) && is_s_orbital(u)
        return coulomb_ppss
    elseif is_p_orbital(t)
        @assert is_p_orbital(r) && is_s_orbital(s) && is_p_orbital(t) && is_s_orbital(u)
        return coulomb_psps
    elseif is_p_orbital(r)
        @assert is_p_orbital(r) && is_s_orbital(s) && is_s_orbital(t) && is_s_orbital(u)
        return coulomb_psss
    else
        @assert is_s_orbital(r) && is_s_orbital(s) && is_s_orbital(t) && is_s_orbital(u)
        return coulomb_ssss
    end
end

function order_orbital_tuple(r, s, t, u)
    if is_s_orbital(r) && is_p_orbital(s)
        r, s = s, r
    end
    if is_s_orbital(t) && is_p_orbital(u)
        t, u = u, t
    end

    if (is_s_orbital(s) && is_p_orbital(t) && is_p_orbital(u)) ||
       (is_s_orbital(r) && is_s_orbital(s) && is_p_orbital(t))
       r, s, t, u = t, u, r, s
   end
   return r, s, t, u
end

function two_electron_integrals(orbitals, indices, mol)
    n = length(orbitals)
    @assert length(indices) == (n^4 + 2n^3 + 3n^2 + 2n) / 8
    res = zeros(Float64, length(indices))
    for i = 1:length(indices)
        μ, ν, λ, σ = indices[i]
        r = orbitals[μ]
        s = orbitals[ν]
        t = orbitals[λ]
        u = orbitals[σ]

        r, s, t, u = order_orbital_tuple(r, s, t, u)
        res[i] = add_two_electron_integrals(r, s, t, u, dispatch_two_electron_integral(r, s, t, u, coulomb_ssss, coulomb_psss, coulomb_psps, coulomb_ppss, coulomb_ppps, coulomb_pppp))
    end
    res
end

end  # module Integrals
