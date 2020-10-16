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

function electron_repulsion_matrix(indices, integrals, density)
    res = similar(density)
    res .= 0
    for i = 1:length(indices)
        μ, ν, λ, σ = indices[i]
        curr_integral = integrals[i]

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

    for i = 1:size(density, 1)
        for j = 1:i
            res[j, i] = res[i, j]
        end
    end
    res
end

function update_density!(p, c, n_occ::Integer)
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
