#!/usr/bin/julia

include("Molecule.jl")
include("FockMatrix.jl")

using LinearAlgebra: norm
using Random: shuffle!

const ϵₑ = 1e-9
const ϵₚ = 1e-5

function main(inputfile, basisset_file, n_damp = nothing, α = nothing)
    m = MoleculeUtils.read_xyz_file(inputfile)
    b = MoleculeUtils.construct_basisset(m, basisset_file)
    println("Number of orbitals: ", length(b))
    indices = MoleculeUtils.Integrals.one_electron_indices(length(b))
    @time s = MoleculeUtils.Integrals.overlap_matrix(b, indices, m)
    @time t = MoleculeUtils.Integrals.kinetic_energy_matrix(b, indices, m)
    @time v = MoleculeUtils.Integrals.nuclear_potential_matrix(b, indices, m)
    two_el_indices = MoleculeUtils.Integrals.two_electron_indices(length(b))
    shuffle!(two_el_indices)
    @time coulomb_integrals = MoleculeUtils.Integrals.two_electron_integrals(b, two_el_indices)

    c = FockMatrix.initial_coefficients(s)
    p = FockMatrix.get_density_matrix(length(b), n_damp, α)
    h = t + v
    f = similar(c)

    f_mo = FockMatrix.transform_matrix(c, f)
    old_energy = MoleculeUtils.electronic_energy(FockMatrix.transform_matrix(c, h), f_mo, m)

    step_count = 0
    @time while true
        step_count += 1
        FockMatrix.update_density!(p, c, Int(m.electron_count / 2))
        f .= t .+ v .+ FockMatrix.electron_repulsion_matrix(two_el_indices, coulomb_integrals, p.curr_density)

        f_mo = FockMatrix.transform_matrix(c, f)
        energy = MoleculeUtils.electronic_energy(FockMatrix.transform_matrix(c, h), f_mo, m)

        Δe = abs(energy - old_energy)
        Δp = FockMatrix.density_difference_norm(p) * 2 / m.electron_count

        println(Δp, " ", Δe)

        if Δe < ϵₑ && Δp < ϵₚ
            println("HF converged after $step_count steps. Total energy: $(energy + MoleculeUtils.coulomb_energy(m))")
            break
        end

        c = FockMatrix.update_coefficients(c, f_mo)
        old_energy = energy
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
