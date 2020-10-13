module MoleculeUtils

import JSON
using StaticArrays

include("Integrals.jl")

struct Atom
    symbol::String
    pos::SVector{3, Float64}
    charge::UInt
end

struct Molecule
    atoms::Vector{Atom}
    electron_count::UInt
end

const atom_charges = Dict("H"  => 1,
                         "He" => 2,
                         "Li" => 3,
                         "Be" => 4,
                         "B"  => 5,
                         "C"  => 6,
                         "N"  => 7,
                         "O"  => 8,
                         "F"  => 9,
                         "Ne" => 10,
                         "Na" => 11,
                         "Mg" => 12,
                         "Al" => 13,
                         "Si" => 14,
                         "P"  => 15,
                         "S"  => 16,
                         "Cl" => 17,
                         "Ar" => 18)

const ang2bohr = 1.89

function get_atom(line)
    s = split(line)
    Atom(s[1], collect(map(p->parse(Float64, p), s[2:4])) .* ang2bohr, atom_charges[s[1]])
end

function read_xyz_file(filename)
    content = readlines(filename)
    atom_vec = [get_atom(l) for l in content[3:end]]
    Molecule(atom_vec, sum(a.charge for a in atom_vec))
end

function Base.show(io::IO, m::Molecule)
    println(io, "Molecule information")
    println(io, "Number of electrons: ", Int(m.electron_count))
    for a in m.atoms
        println(io, "Atom", " ", a.symbol, " ", a.pos[1], " ", a.pos[2], " ", a.pos[3], ", nuclear charge: ", Int(a.charge))
    end
end

norm_coeff_s(exp) = (2 * exp / π) ^ 0.75
norm_coeff_p(exp) = (128 * exp^5 / π^3) ^ 0.25

function norm_primitives(primitives, l)
    @assert l ∈ (0, 1)
    norm_func = if l == 0 norm_coeff_s else norm_coeff_p end
    [(e, norm_func(e) * coeff) for (e, coeff) in primitives]
end

function get_basisfunctions(shells)
    res = Vector{Integrals.Orbital}()
    for curr_shell in shells
        for (curr_l, curr_coeffs) in zip(curr_shell["angular_momentum"], curr_shell["coefficients"])
            primitives = collect(zip(map(s->parse(Float64, s), curr_shell["exponents"]), map(s->parse(Float64, s), curr_coeffs)))
            primitives = norm_primitives(primitives, curr_l)
            if curr_l == 0
                push!(res, Integrals.Orbital([0, 0, 0], primitives, Integrals.s))
            elseif curr_l == 1
                for curr_type in (Integrals.px, Integrals.py, Integrals.pz)
                    push!(res, Integrals.Orbital([0, 0, 0], primitives, curr_type))
                end
            else
                error("Unknown angular quantum number: $curr_l")
            end
        end
    end
    res
end

function construct_basisset(mol::Molecule, basisset_file)
    basisset = JSON.parsefile(basisset_file)
    basisfunctions = Dict{UInt, Vector{Integrals.Orbital}}()
    res = Vector{Integrals.Orbital}()
    for a in mol.atoms
        if !haskey(basisfunctions, a.charge)
            basisfunctions[a.charge] = get_basisfunctions(basisset["elements"][string(a.charge)]["electron_shells"])
        end
        #=curr_orbitals = basisfunctions[a.charge]
        for orb in curr_orbitals
            orb.center = [a.x, a.y, a.z]
        end
        append!(res, curr_orbitals)=#
        append!(res, [Integrals.Orbital(a.pos, o.primitives, o.type) for o in basisfunctions[a.charge]])
    end
    res
end

end  # module MoleculeS
