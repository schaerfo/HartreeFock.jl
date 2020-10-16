#!/usr/bin/julia

include("Molecule.jl")
include("FockMatrix.jl")

using LinearAlgebra: norm

const ϵₑ = 1e-9
const ϵₚ = 1e-5

function main(inputfile, basisset_file)
    m = MoleculeUtils.read_xyz_file(inputfile)
    b = MoleculeUtils.construct_basisset(m, basisset_file)
    println("Number of orbitals: ", length(b))
    indices = MoleculeUtils.Integrals.one_electron_indices(length(b))
    s = MoleculeUtils.Integrals.overlap_matrix(b, indices, m)
    t = MoleculeUtils.Integrals.kinetic_energy_matrix(b, indices, m)
    v = MoleculeUtils.Integrals.nuclear_potential_matrix(b, indices, m)
    two_el_indices = MoleculeUtils.Integrals.two_electron_indices(length(b))
    coulomb_integrals = MoleculeUtils.Integrals.two_electron_integrals(b, two_el_indices, m)

    c = FockMatrix.initial_coefficients(s)
    p = similar(c)
    h = t + v
    old_p = similar(p)
    f = similar(c)

    f_mo = FockMatrix.transform_matrix(c, f)
    old_energy = MoleculeUtils.electronic_energy(FockMatrix.transform_matrix(c, h), f_mo, m)

    step_count = 0
    while true
        step_count += 1
        FockMatrix.update_density!(p, c, Int(m.electron_count / 2))
        f .= t .+ v .+ FockMatrix.electron_repulsion_matrix(two_el_indices, coulomb_integrals, p)

        f_mo = FockMatrix.transform_matrix(c, f)
        energy = MoleculeUtils.electronic_energy(FockMatrix.transform_matrix(c, h), f_mo, m)

        Δe = abs(energy - old_energy)
        Δp = norm(old_p - p) / length(b)

        println(Δp, " ", Δe)

        if Δe < ϵₑ && Δp < ϵₚ
            println("HF converged after $step_count steps. Total energy: $(energy + MoleculeUtils.coulomb_energy(m))")
            break
        end

        c = FockMatrix.update_coefficients(c, f_mo)
        old_p .= p
        old_energy = energy
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
