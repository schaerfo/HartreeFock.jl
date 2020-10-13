#!/usr/bin/julia

include("Molecule.jl")

function main(inputfile, basisset_file)
    m = MoleculeUtils.read_xyz_file(inputfile)
    b = MoleculeUtils.construct_basisset(m, basisset_file)
    #=for f in b
        println(f)
    end=#
    println("Number of orbitals: ", length(b))
    indices = MoleculeUtils.Integrals.one_electron_indices(length(b))
    @time s = MoleculeUtils.Integrals.overlap_matrix(b, indices)
    @time t = MoleculeUtils.Integrals.kinetic_energy_matrix(b, indices)
    display(s)
    display(t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
