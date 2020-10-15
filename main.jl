#!/usr/bin/julia

include("Molecule.jl")

function main(inputfile, basisset_file)
    m = MoleculeUtils.read_xyz_file(inputfile)
    b = MoleculeUtils.construct_basisset(m, basisset_file)
    println("Number of orbitals: ", length(b))
    indices = MoleculeUtils.Integrals.one_electron_indices(length(b))
    @time s = MoleculeUtils.Integrals.overlap_matrix(b, indices, m)
    @time t = MoleculeUtils.Integrals.kinetic_energy_matrix(b, indices, m)
    @time v = MoleculeUtils.Integrals.nuclear_potential_matrix(b, indices, m)
    display(s)
    display(t)
    display(v)
    @time two_el_indices = MoleculeUtils.Integrals.two_electron_indices(length(b))
    @time coulomb_integrals = MoleculeUtils.Integrals.two_electron_integrals(b, two_el_indices, m)
    display(coulomb_integrals)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
