function atomic_cloud(N, distribution, fiber)
    r = zeros(3, N)
    for i in 1:N
        while sqrt(r[1, i]^2 + r[2, i]^2) ≤ fiber.radius
            r[:, i] = rand(distribution)
        end
    end
    return r
end

function radial_gaussian(r, σ)
    return r / (σ^2) * exp(-r^2 / (2 * σ^2))
end

function linear_gaussian(x, σ)
    return 1 / sqrt(2 * π * σ^2) * exp(-x^2 / (2 * σ^2))
end
