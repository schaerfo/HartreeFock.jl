module Integrals

using LinearAlgebra: norm
using StaticArrays
using SpecialFunctions: erf

@enum OrbitalType s=0b0 px=0b01 py=0b10 pz=0b11 #dx²=0b101 dxy=0b110 dxz=0b111 dy²=0b1010 dyz=0b1011 dz²=0b1111

struct Orbital
    center::SVector{3, Float64}
    primitives::Vector{Tuple{Float64, Float64}}
    type::OrbitalType
end

is_s_orbital(t::OrbitalType) = t == s
function is_p_orbital(t::OrbitalType)
    compare_type = Base.Enums.basetype(OrbitalType)
    bitmask = compare_type(0b11)
    t = compare_type(t)
    (~bitmask & t) == 0 && (bitmask & t) != 0
end

one_electron_indices(n) = [(i, j) for i in 1:n for j in i:n]

function one_electron_matrix(orbitals, indices, mol, integral_functions)
    @assert length(indices) == length(orbitals) * (length(orbitals) + 1) / 2
    res = zeros(Float64, (length(orbitals), length(orbitals)))
    for (i, j) in indices
        orbital_a = orbitals[i]
        orbital_b = orbitals[j]

        func = if is_s_orbital(orbital_a.type) && is_s_orbital(orbital_b.type)
            integral_functions[1]
        elseif is_s_orbital(orbital_a.type) || is_s_orbital(orbital_b.type)
            if is_s_orbital(orbital_a.type)
                orbital_a, orbital_b = orbital_b, orbital_a
            end
            integral_functions[2]
        else
            integral_functions[3]
        end

        for (α, a) in orbital_a.primitives
            for (β, b) in orbital_b.primitives
                curr_integral::Float64 = func(α, orbital_a.center, Int(orbital_a.type), β, orbital_b.center, Int(orbital_b.type), mol)
                res[i, j] += a * b * curr_integral
            end
        end
        res[j, i] = res[i, j]
    end
    res
end

g(α, a, β, b) = exp(-α * β / (α + β) * norm(a - b)^2)
s1(t) = abs(t) < 1e-15 ? 2. / sqrt(π) : erf(t)/t
s2(t) = abs(t) < 1e-5  ? -4. / (3sqrt(π)) : (2π^-.5 * t * exp(-t^2) - erf(t)) / t^3
s3(t) = abs(t) < 1e-3  ? 8 / (5sqrt(π)) : (3erf(t) - 2 * π^(-.5) * (3t + 2t^3) * exp(-t^2)) / t^5

overlap_ss(α, a, _, β, b, _, _) = (π / (α + β))^1.5 * g(α, a, β, b)
overlap_ps(α, a, i, β, b, _, _) = -g(α, a, β, b) * β * π^1.5 / (α + β)^2.5 * (a[i] - b[i])
overlap_pp(α, a, i, β, b, j, _) = g(α, a, β, b) * π^1.5 / (α + β)^2.5 * (0.5 * (i == j) - α * β / (α + β) * (a[i] - b[i]) * (a[j] - b[j]))
overlap_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (overlap_ss, overlap_ps, overlap_pp))

kinetic_ss(α, a, _, β, b, _, _) = g(α, a, β, b) * α * β * π^1.5 / (α + β)^2.5 * (3 - 2α * β / (α + β) * norm(a - b)^2)
kinetic_ps(α, a, i, β, b, _, _) = g(α, a, β, b) * α * β^2 * π^1.5 / (α + β)^3.5 * (2α * β / (α + β) * norm(a - b)^2 - 5) * (a[i] - b[i])
kinetic_pp(α, a, i, β, b, j, _) = g(α, a, β, b) * α * β * π^1.5 / (α + β)^3.5 * ((2.5 - α * β / (α + β) * norm(a - b)^2) * (i == j) + α * β / (α + β) * (2α * β / (α + β) * norm(a - b)^2 - 7) * (a[i] - b[i]) * (a[j] - b[j]))
kinetic_energy_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (kinetic_ss, kinetic_ps, kinetic_pp))

function nuclear_ss(α, a, _, β, b, _, mol)
    η = α + β
    res = 0.
    for curr_atom in mol.atoms
        r = (α * a + β * b) / η - curr_atom.pos
        t = sqrt(η) * norm(r)
        res -= curr_atom.charge * g(α, a, β, b) * π^1.5 / η * s1(t)
    end
    res
end

function nuclear_ps(α, a, i, β, b, _, mol)
    η = α + β
    res = 0.
    for curr_atom in mol.atoms
        r = (α * a + β * b) / η - curr_atom.pos
        t = sqrt(η) * norm(r)
        res -= curr_atom.charge * g(α, a, β, b) * π^1.5 / 2η * (r[i] * s2(t) - 2β * (a[i] - b[i]) / η * s1(t))
    end
    res
end

function nuclear_pp(α, a, i, β, b, j, mol)
    η = α + β
    δᵢⱼ = i == j
    res = 0.
    for curr_atom in mol.atoms
        r = (α * a + β * b) / η - curr_atom.pos
        t = sqrt(η) * norm(r)
        res -= curr_atom.charge * g(α, a, β, b) * π^1.5 / 4η^2 * (η * r[i] * r[j] * s3(t) + (δᵢⱼ + 2α * (a[j] - b[j]) * r[i] - 2β * (a[i] - b[i]) * r[j]) * s2(t) + (2δᵢⱼ - 4α * β * (a[i] - b[i]) * (a[j] - b[j]) / η) * s1(t))
    end
    res
end
nuclear_potential_matrix(orbitals, indices, mol) = one_electron_matrix(orbitals, indices, mol, (nuclear_ss, nuclear_ps, nuclear_pp))

end  # module Integrals
