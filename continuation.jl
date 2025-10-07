##################################################################################################################################
# Rigorous continuation of period-2 orbits and its period-doubling for a predator-prey model
# -----------------------------------------------------------
# This file proves the period-doubling bifurcation curves consisting of three steps:
# 1. We define the model and its derivatives.
# 2. We use Newton’s method to find a discrete number of bifurcation points and interpolate.
# 3. We compute the necessary bounds Y, Z1 and Z2 to rigorously validate solutions.
# Afterwards, we plot the outcome.
##################################################################################################################################

using RadiiPolynomial

# Degree of Chebyshev expansion and number of points
K = 2^6

# Parameter interval and Chebyshev sampling grid
κ₁ = -16 # change accordingly
κ₂ = -13 # change accordingly

# ================================================================================================================================
# Step 1: Model definition and derivatives
# ================================================================================================================================

println("\n*************************** STEP 1: REFORMULATE THE PROBLEM ***************************")

# The main function, 1st and 2nd derivatives (wrt x) of the predator-prey model
f(x, β, κ) = β + x - κ / (1 + exp(x))
Df(x, κ) = 1 + κ * exp(x) / (1 + exp(x))^2
D²f(x, κ) = κ * exp(x) * (1 - exp(x)) / (1 + exp(x))^3

# Reformulate the problem (period-2 orbit and a period-doubling candidate) as a zero-finding problem
function F(w, κ)
    x0, x1, u, β = w
    [f(x1, β, κ) - x0 ;
     f(x0, β, κ) - x1 ;
     Df(x1, κ) * u + 1 ;
     Df(x0, κ) - u]
end

# Jacobian matrix of F with respect to w
function DF(w, κ)
    x0, x1, u, β = w
    [-1              Df(x1, κ)          0            1
      Df(x0, κ)     -1                  0            1
      0              D²f(x1, κ) * u     Df(x1, κ)    0
      D²f(x0, κ)     0                 -1            0]
end

println("Succesfully defined the function f, F, and their derivatives")


# ================================================================================================================================
# Step 2: Numerical approximation (Newton at Chebyshev nodes + interpolation)
# ================================================================================================================================

println("\n*************************** STEP 2: NUMERICAL APPROXIMATION ***************************")

# Total number of Chebyshev nodes
npts = K + 1

# Values of the Chebyshev nodes (i.e. sampling grid)
κ_grid = [(κ₁ + κ₂)/2 - cospi(j/K) * (κ₂ - κ₁)/2 for j ∈ 0:K]

# Storage for Branch 1 left and initial guess
w_grid_branch_1_left = Vector{Vector{Float64}}(undef, npts)
A_grid_branch_1_left = Vector{Matrix{Float64}}(undef, npts)
w_bar_branch_1_left = [-3.1483596176706583, -0.5491917828634445, 0.3686056118167106, -12.742336765005236]

# Storage for Branch 1 right and initial guess
w_grid_branch_1_right = Vector{Vector{Float64}}(undef, npts)
A_grid_branch_1_right = Vector{Matrix{Float64}}(undef, npts)
w_bar_branch_1_right = [-5.010276944633445, 1.6806717313168913, 0.89470423246783, -9.203053333137658]

# Storage for Branch 2 left and initial guess
w_grid_branch_2_left = Vector{Vector{Float64}}(undef, npts)
A_grid_branch_2_left = Vector{Matrix{Float64}}(undef, npts)
w_bar_branch_2_left = [5.010276944633555, -1.6806717313171953, 0.8947042324678495, -6.796946666862736]

# Storage for Branch 2 right and initial guess
w_grid_branch_2_right = Vector{Vector{Float64}}(undef, npts)
A_grid_branch_2_right = Vector{Matrix{Float64}}(undef, npts)
w_bar_branch_2_right = [0.5491917828634446, 3.1483596176706587, -2.712926683539617, -3.257663234994763]

# Run Newton’s method at each Chebyshev node
default_newton = (w, κ) -> (F(w, κ), DF(w, κ))

for i ∈ 1:npts
    global w_bar_branch_1_left, success1 = newton(w -> default_newton(w, κ_grid[i]), w_bar_branch_1_left; verbose = false)
    global w_bar_branch_1_right, success2 = newton(w -> default_newton(w, κ_grid[i]), w_bar_branch_1_right; verbose = false)
    success1 && success2 || error("Newton failed")
    w_grid_branch_1_left[i] = w_bar_branch_1_left
    A_grid_branch_1_left[i] = inv(DF(w_bar_branch_1_left, κ_grid[i]))
    w_grid_branch_1_right[i] = w_bar_branch_1_right
    A_grid_branch_1_right[i] = inv(DF(w_bar_branch_1_right, κ_grid[i]))
    #
    global w_bar_branch_2_left, success1 = newton(w -> default_newton(w, κ_grid[i]), w_bar_branch_2_left; verbose = false)
    global w_bar_branch_2_right, success2 = newton(w -> default_newton(w, κ_grid[i]), w_bar_branch_2_right; verbose = false)
    success1 && success2 || error("Newton failed")
    w_grid_branch_2_left[i] = w_bar_branch_2_left
    A_grid_branch_2_left[i] = inv(DF(w_bar_branch_2_left, κ_grid[i]))
    w_grid_branch_2_right[i] = w_bar_branch_2_right
    A_grid_branch_2_right[i] = inv(DF(w_bar_branch_2_right, κ_grid[i]))
end

println("Succesfully computed the numerical zero and approximate inverse for each Chebyshev node k=0,...,$K.")

# Chebyshev interpolation of the parameter κ
κ_cheb = Sequence(Chebyshev(1), [(interval(κ₁) + interval(κ₂))/interval(2),
                                 (interval(κ₂) - interval(κ₁))/interval(4)])

# Interpolate vector-valued data on a grid of Chebyshev nodes to Chebyshev series (i.e. sequences of Chebyshev coefficientes)
function grid2cheb(x_grid::Vector{<:Vector}, K)
    x_fft = [reverse(x_grid) ; x_grid[begin+1:end-1]]
    return [rifft!(complex(getindex.(x_fft, i)), Chebyshev(K)) for i ∈ eachindex(x_fft[1])]
end

# Interpolate matrix-valued data on a grid of Chebyshev nodes to Chebyshev series (i.e. sequences of Chebyshev coefficientes)
function grid2cheb(x_grid::Vector{<:Matrix}, K)
    x_fft = [reverse(x_grid) ; x_grid[begin+1:end-1]]
    return [rifft!(complex(getindex.(x_fft, i, j)), Chebyshev(K)) for i ∈ axes(x_fft[1], 1), j ∈ axes(x_fft[1], 2)]
end

# Chebyshev interpolation of the Chebyshev nodes for each branch
w_cheb_branch_1_left = interval.(grid2cheb(w_grid_branch_1_left, K))
A_cheb_branch_1_left = interval.(grid2cheb(A_grid_branch_1_left, K))
w_cheb_branch_1_right = interval.(grid2cheb(w_grid_branch_1_right, K))
A_cheb_branch_1_right = interval.(grid2cheb(A_grid_branch_1_right, K))

w_cheb_branch_2_left = interval.(grid2cheb(w_grid_branch_2_left, K))
A_cheb_branch_2_left = interval.(grid2cheb(A_grid_branch_2_left, K))
w_cheb_branch_2_right = interval.(grid2cheb(w_grid_branch_2_right, K))
A_cheb_branch_2_right = interval.(grid2cheb(A_grid_branch_2_right, K))

println("Succesfully interpolated the Chebyshev nodes.")


# ================================================================================================================================
# Step 3: Rigorous computation using interval arithmetic
# ================================================================================================================================

println("\n*************************** STEP 3: RIGOROUS VALIDATION ***************************")

# Taylor expansion of the nonlinearity
function coeffs(χ)
    E = exp(χ)
    den = (E + exact(1))
    a0  = - exact(1) / den
    a1  =   E / den^2
    a2  = - E * (- exact(1) + E) / (exact(2) * den^3)
    a3  =   E * (  exact(1) - exact(4)    * E +                E^2) / (exact(6) * den^4)
    a4  = - E * (- exact(1) + exact(11)   * E - exact(11)    * E^2 +                 E^3) / (exact(24) * den^5)
    a5  =   E * (  exact(1) - exact(26)   * E + exact(66)    * E^2 - exact(26)     * E^3 +                  E^4) / (exact(120) * den^6)
    a6  = - E * (- exact(1) + exact(57)   * E - exact(302)   * E^2 + exact(302)    * E^3 - exact(57)      * E^4 +                  E^5) / (exact(720) * den^7)
    a7  =   E * (  exact(1) - exact(120)  * E + exact(1191)  * E^2 - exact(2416)   * E^3 + exact(1191)    * E^4 -  exact(120)    * E^5 +                 E^6) / (exact(5040) * den^8)
    a8  = - E * (- exact(1) + exact(247)  * E - exact(4293)  * E^2 + exact(15619)  * E^3 - exact(15619)   * E^4 + exact(4293)    * E^5 - exact(247)    * E^6 +                E^7) / (exact(40320) * den^9)
    a9  =   E * (  exact(1) - exact(502)  * E + exact(14608) * E^2 - exact(88234)  * E^3 + exact(156190)  * E^4 - exact(88234)   * E^5 + exact(14608)  * E^6 - exact(502)   * E^7 +               E^8) / (exact(362880) * den^10)
    a10 = - E * (- exact(1) + exact(1013) * E - exact(47840) * E^2 + exact(455192) * E^3 - exact(1310354) * E^4 + exact(1310354) * E^5 - exact(455192) * E^6 + exact(47840) * E^7 - exact(1013) * E^8 + E^9) / (exact(3628800) * den^11)
    return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
end

function h_poly(x; χ)
    a = coeffs(χ)
    N = length(a) - 1
    y = x - χ
    s = a[N+1] * one(y)
    for n ∈ N-1:-1:0
        s = s * y + a[n+1]
    end
    return s
end

function Dh_poly(x; χ)
    a = coeffs(χ)
    N = length(a) - 1
    y = x - χ
    s = exact(N) * a[N+1] * one(y)
    for n ∈ N-1:-1:1
        s = s * y + exact(n) * a[n+1]
    end
    return s
end

function D²h_poly(x; χ)
    a = coeffs(χ)
    N = length(a) - 1
    y = x - χ
    s = exact(N * (N-1)) * a[N+1] * one(y)
    for n ∈ N-1:-1:2
        s = s * y + exact(n * (n-1)) * a[n+1]
    end
    return s
end

function h_Np1(x)
    E = exp(x)
    den = E + exact(1)
    E² = E^2; den² = den^2
    E³ = E² * E; den³ = den² * den
    E⁴ = E² * E²; den⁴ = den² * den²
    E⁵ = E⁴ * E; den⁵ = den⁴ * den
    E⁶ = E³ * E³; den⁶ = den³ * den³
    E⁷ = E⁶ * E; den⁷ = den⁶ * den
    E⁸ = E⁴ * E⁴; den⁸ = den⁴ * den⁴
    E⁹ = E⁸ * E; den⁹ = den⁸ * den
    E¹⁰ = E⁵ * E⁵; den¹⁰ = den⁵ * den⁵
    E¹¹ = E¹⁰ * E; den¹¹ = den¹⁰ * den
    den¹² = den⁶ * den⁶
    return E / den² +
        - exact(2046)      * E²  / den³  +
          exact(171006)    * E³  / den⁴  +
        - exact(3498000)   * E⁴  / den⁵  +
          exact(29607600)  * E⁵  / den⁶  +
        - exact(129230640) * E⁶  / den⁷  +
          exact(322494480) * E⁷  / den⁸  +
        - exact(479001600) * E⁸  / den⁹  +
          exact(419126400) * E⁹  / den¹⁰ +
        - exact(199584000) * E¹⁰ / den¹¹ +
          exact(39916800)  * E¹¹ / den¹²
end

f_poly(x, β, κ; χ) = β + x + κ * h_poly(x; χ)
Df_poly(x, κ; χ)   = exact(1) + κ * Dh_poly(x; χ)
D²f_poly(x, κ; χ)  = κ * D²h_poly(x; χ)

function F_poly(w, κ; χ₀, χ₁)
    x0, x1, u, β = w
    [f_poly(x1, β, κ; χ = χ₁) - x0 ;
     f_poly(x0, β, κ; χ = χ₀) - x1 ;
     Df_poly(x1, κ; χ = χ₁) * u + exact(1) ;
     Df_poly(x0, κ; χ = χ₀) - u]
end

function DF_poly(w, κ; χ₀, χ₁)
    x0, x1, u, β = w
    [-exact(1)                Df_poly(x1, κ; χ = χ₁)          exact(0)              exact(1)
      Df_poly(x0, κ; χ = χ₀)     -exact(1)                    exact(0)              exact(1)
      exact(0)                D²f_poly(x1, κ; χ = χ₁) * u     Df_poly(x1, κ; χ = χ₁)    exact(0)
      D²f_poly(x0, κ; χ = χ₀)     exact(0)                   -exact(1)              exact(0)]
end

# Compute the bounds Y, Z₁ and Z₂ (using Taylor expansion) for the usage of the uniform contraction Theorem
function proof(w_grid, w_cheb, A_cheb, κ_cheb, K)
    # Extract approximate orbits
    x₀_bar = w_cheb[1]
    x₁_bar = w_cheb[2]
    u_bar  = w_cheb[3]

    # Compute centers χ₀, χ₁ of the Taylor expansion of the nonlinearity
    χ₀ = interval(sum([w_grid[j][1] for j ∈ 1:length(w_grid)]) / length(w_grid))
    χ₁ = interval(sum([w_grid[j][2] for j ∈ 1:length(w_grid)]) / length(w_grid))

    # Compute rigorous remainder bounds for both coordinates
    h̃₀   = interval(691) / interval(8) / interval(factorial(11)) * norm((x₀_bar - χ₀)^11, 1)
    h̃′₀  = interval(691) / interval(8) / interval(factorial(10)) * norm((x₀_bar - χ₀)^10, 1)
    h̃′′₀ = interval(691) / interval(8) / interval(factorial(9))  * norm((x₀_bar - χ₀)^9, 1)

    h̃₁   = interval(691) / interval(8) / interval(factorial(11)) * norm((x₁_bar - χ₁)^11, 1)
    h̃′₁  = interval(691) / interval(8) / interval(factorial(10)) * norm((x₁_bar - χ₁)^10, 1)
    h̃′′₁ = interval(691) / interval(8) / interval(factorial(9))  * norm((x₁_bar - χ₁)^9, 1)

    # Y bound
    Y_poly = norm(norm.(A_cheb * F_poly(w_cheb, κ_cheb; χ₀ = χ₀, χ₁ = χ₁), 1), 1)
    Y_remainder = opnorm(norm.(A_cheb, 1), 1) * norm(κ_cheb, 1) * (h̃₁ + h̃₀ + norm(u_bar, 1) * h̃′₁ + h̃′₀)
    Y_bound = Y_poly + Y_remainder

    # Z₁ bound
    Z₁_poly = opnorm(norm.(A_cheb * DF_poly(w_cheb, κ_cheb; χ₀ = χ₀, χ₁ = χ₁) - interval(I(4)), 1), 1)
    Z₁_remainder = opnorm(norm.(A_cheb, 1), 1) * norm(κ_cheb, 1) * max(h̃′₀ + h̃′′₀, h̃′₁ + norm(u_bar, 1) * h̃′′₁)
    Z₁_bound = Z₁_poly + Z₁_remainder

    # Z₂ bound
    R = exact(1e-3)
    C1 = inv(interval(6)*sqrt(interval(3)))
    C2 = inv(interval(8))
    Z₂_bound = opnorm(norm.(A_cheb, 1), 1) * norm(κ_cheb, 1) * (C1 + max(C1, max(interval(1), norm(u_bar, 1) + R) * C2))

    # Certify interval of existence of r>0 that admits a true zero of F
    interval_of_existence(Y_bound, Z₁_bound, Z₂_bound, R, verbose = true)
end

# Run the proof for each branch
proof(w_grid_branch_1_left, w_cheb_branch_1_left, A_cheb_branch_1_left, κ_cheb, K)
proof(w_grid_branch_1_right, w_cheb_branch_1_right, A_cheb_branch_1_right, κ_cheb, K)
proof(w_grid_branch_2_left, w_cheb_branch_2_left, A_cheb_branch_2_left, κ_cheb, K)
proof(w_grid_branch_2_right, w_cheb_branch_2_right, A_cheb_branch_2_right, κ_cheb, K)

println("\nFinished validating the existence of four branches of period-doubling bifurcations.")


# ================================================================================================================================
# Step 4: Plotting results
# ================================================================================================================================

println("\n*************************** STEP 4: PLOT RESULTS ***************************")

using GLMakie

function plot_branches(w_grid_branch_1_left, w_grid_branch_1_right,
                       w_grid_branch_2_left, w_grid_branch_2_right,
                       w_cheb_branch_1_left, w_cheb_branch_1_right,
                       w_cheb_branch_2_left, w_cheb_branch_2_right,
                       κ_grid, κ_cheb)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="β", ylabel="κ",
                limits = ((-20, 0), (-45, -5)))


    # Extract and plot the Chebyshev nodes: 4th coordinate (i.e. β) vs κ
    scatter!(ax, getindex.(w_grid_branch_1_left, 4), κ_grid)
    scatter!(ax, getindex.(w_grid_branch_1_right, 4), κ_grid)
    scatter!(ax, getindex.(w_grid_branch_2_left, 4), κ_grid)
    scatter!(ax, getindex.(w_grid_branch_2_right, 4), κ_grid)

    # Extract and plot the Chebyshev interpolation: 4th coordinate (i.e. β) vs κ
    lines!(ax, [Point2f(mid(w_cheb_branch_1_left[4](s)), mid(κ_cheb(s)))
                for s in LinRange(-1,1,501)], label="branch 1 left")
    lines!(ax, [Point2f(mid(w_cheb_branch_1_right[4](s)), mid(κ_cheb(s)))
                for s in LinRange(-1,1,501)], label="branch 1 right")
    lines!(ax, [Point2f(mid(w_cheb_branch_2_left[4](s)), mid(κ_cheb(s)))
                for s in LinRange(-1,1,501)], label="branch 2 left")
    lines!(ax, [Point2f(mid(w_cheb_branch_2_right[4](s)), mid(κ_cheb(s)))
                for s in LinRange(-1,1,501)], label="branch 2 right")

    # Add legend in lower left
    axislegend(ax; position = :lb)

    display(fig)
    save("branches.png", fig)
    return fig
end

plot_branches(w_grid_branch_1_left, w_grid_branch_1_right,
              w_grid_branch_2_left, w_grid_branch_2_right,
              w_cheb_branch_1_left, w_cheb_branch_1_right,
              w_cheb_branch_2_left, w_cheb_branch_2_right,
              κ_grid, κ_cheb)


println("Succesfully plotted and saved the outcome.")
