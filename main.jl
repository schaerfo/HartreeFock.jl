#!/usr/bin/julia

include("Molecule.jl")

function main(inputfile, basisset_file)
    m = MoleculeUtils.read_xyz_file(inputfile)
    b = MoleculeUtils.construct_basisset(m, basisset_file)
    display(b)
    println("Number of orbitals: ", length(b))
    indices = MoleculeUtils.Integrals.one_electron_indices(length(b))
    @time s = MoleculeUtils.Integrals.overlap_matrix(b, indices, m)
    @time t = MoleculeUtils.Integrals.kinetic_energy_matrix(b, indices, m)
    @time v = MoleculeUtils.Integrals.nuclear_potential_matrix(b, indices, m)
    display(s)
    display(t)
    display(v)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
