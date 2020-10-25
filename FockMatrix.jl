module FockMatrix

using LinearAlgebra
using Printf

function get_a(i, j, k, l)
    if (k == i || k == j) && l == k
      0.5
   elseif j != i && (k == i || k == j || l == i || l == j) && l != k
      1.5
   elseif k != i && k != j && l != i && l != j && l != k
      2.
   else
      1.
   end
end

function get_b(i, j, k, l)
    if k == i && k != j && l != i && l != j
       -1.
    elseif (j == i || k == j ) && l == j
       0.5
    elseif k == i && (k == j || l == i) && l != j
       1.
    elseif k != i && (k==j || l == i ) && l != j
       1.5
    else
       -0.5
    end
end

function electron_repulsion_matrix_impl!(res, integrals, density, μ)
    i = ((μ-1)^4 + 2(μ-1)^3 + 3(μ-1)^2 + 2(μ-1)) >> 3

    for ν ∈ 1:μ
        for λ ∈ 1:μ
            for σ ∈ 1:(λ == μ ? ν : λ)
                curr_integral = integrals[i += 1]

                res[μ, ν] += get_a(μ, ν, λ, σ) * density[λ, σ] * curr_integral # 1

                if λ ≠ ν
                    res[μ, λ] += get_b(μ, ν, λ, σ) * density[ν, σ] * curr_integral # 2
                end

                if σ ≠ ν && σ ≠ λ
                    res[μ, σ] += get_b(μ, ν, σ, λ) * density[ν, λ] * curr_integral # 3
                end

                if ν ≥ λ && ν ≠ μ
                    res[ν, λ] += get_b(ν, μ, λ, σ) * density[μ, σ] * curr_integral # 4
                end

                if ν < λ && λ ≠ μ
                    res[λ, ν] += get_b(λ, σ, ν, μ) * density[μ, σ] * curr_integral # 5
                end

                if ν ≥ σ && σ ≠ λ && ν ≠ μ
                    res[ν, σ] += get_b(ν, μ, σ, λ) * density[μ, λ] * curr_integral # 6
                end

                if ν < σ && σ ≠ λ
                    res[σ, ν] += get_b(σ, λ, ν, μ) * density[μ, λ] * curr_integral # 7
                end

                if λ ≠ μ && λ ≠ ν && σ ≠ ν
                    res[λ, σ] += get_a(λ, σ, μ, ν) * density[μ, ν] * curr_integral # 8
                end
            end
        end
    end
end

function electron_repulsion_matrix_worker!(res, integrals, density, index_counter::Threads.Atomic{T}) where T <: Integer
    while true
        curr_index = Threads.atomic_sub!(index_counter, 1)
        if curr_index < 1 break end
        electron_repulsion_matrix_impl!(res, integrals, density, curr_index)
    end
end

function electron_repulsion_matrix(integrals, density)
    n = Threads.Atomic{Int}(size(density, 1))
    n_threads = min(Threads.nthreads(), n[])
    tasks = Vector{Task}()
    results = zeros(eltype(density), size(density)..., n_threads)
    for i = 1:n_threads
        push!(tasks, Threads.@spawn @views electron_repulsion_matrix_worker!(results[:, :, i], integrals, density, n))
    end

    for t in tasks wait(t) end

    res = zero(density)
    for i = 1:n_threads
        res .+= results[:, :, i]
    end

    for i = 1:size(density, 1)
        for j = 1:i
            res[j, i] = res[i, j]
        end
    end
    res
end

struct DensityMatrix
    curr_density::Matrix{Float64}
    prev_density::Matrix{Float64}
end

DensityMatrix(n) = DensityMatrix(zeros(n, n), zeros(n, n))

mutable struct DampingDensityMatrix
    curr_density::Matrix{Float64}
    prev_density::Matrix{Float64}
    n::Int
    α::Float64
    i::Int
end

DampingDensityMatrix(n, n_damp, α) = DampingDensityMatrix(zeros(n, n), zeros(n, n), n_damp, α, 0)

function update_density!(p::AbstractArray, c, n_occ::Integer)
    @assert size(p) == size(c)
    for μ = 1:size(p, 1)
        for ν = 1:size(p, 2)
            curr_val = 0.
            for m = 1:n_occ
                curr_val += c[μ, m] * c[ν, m]
            end
            p[μ, ν] = 2 * curr_val
        end
    end
end

function update_density!(p::DensityMatrix, c, n_occ::Integer)
    p.prev_density .= p.curr_density
    update_density!(p.curr_density, c, n_occ)
end

function update_density!(p::DampingDensityMatrix, c, n_occ::Integer)
    p.prev_density .= p.curr_density
    update_density!(p.curr_density, c, n_occ)
    if p.i < p.n
        @. p.curr_density = (1 - p.α) * p.curr_density + p.α * p.prev_density
        p.i += 1
    end
end

density_difference_norm(p::Union{DensityMatrix, DampingDensityMatrix}) = norm(p.prev_density - p.curr_density)

get_density_matrix(n, _::Nothing, _::Nothing) = DensityMatrix(n)
get_density_matrix(n, n_damp::Int, α::Float64) = DampingDensityMatrix(n, n_damp, α)

function initial_coefficients(overlap)
    res = similar(overlap)
    @assert ishermitian(overlap)
    h = Hermitian(overlap)

    eval, evec = eigen(h)

    for r = 1:size(res, 1)
        for i = 1:size(res, 2)
            res[r, i] = evec[r, i] / sqrt(eval[i])
        end
    end
    res
end

transform_matrix(c, m) = transpose(c) * m * c

function update_coefficients(c, f_mo)
    # assert failes due to numerical zero
    #@assert ishermitian(f_mo)
    c * eigvecs(Hermitian(f_mo))
end

end
