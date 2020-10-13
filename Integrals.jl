module Integrals

using StaticArrays

@enum OrbitalType s=0b0 px=0b01 py=0b11 pz=0b10

struct Orbital
    center::SVector{3, Float64}
    primitives::Vector{Tuple{Float64, Float64}}
    type::OrbitalType
end

end  # module Integrals
