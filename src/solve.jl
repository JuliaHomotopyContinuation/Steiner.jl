using HomotopyContinuation, LinearAlgebra
const HC = HomotopyContinuation

function steiner_system()
    @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5]
    #tangential conics
    f = a[1] * x[1]^2 + a[2] * x[1] * x[2] + a[3] * x[2]^2 + a[4] * x[1] + a[5] * x[2] + 1
    ∇ = differentiate(f, x)
    #5 conics
    g =
        c[1] * x[1]^2 +
        c[2] * x[1] * x[2] +
        c[3] * x[2]^2 +
        c[4] * x[1] +
        c[5] * x[2] +
        c[6]
    ∇_2 = differentiate(g, x)
    #the general system
    #f_a_0 is tangent to g_b₀ at x₀
    function Incidence(f, a₀, g, b₀, x₀)
        fᵢ = f(x => x₀, a => a₀)
        ∇ᵢ = ∇(x => x₀, a => a₀)
        Cᵢ = g(x => x₀, c => b₀)
        ∇_Cᵢ = ∇_2(x => x₀, c => b₀)
        [fᵢ; Cᵢ; det([∇ᵢ ∇_Cᵢ])]
    end
    @var v[1:6, 1:5]
    F = vcat(map(i -> Incidence(f, a, g, v[:, i], y[:, i]), 1:5)...)
    System(F, [a; vec(y)], vec(v))
end

const startconics_startsolutions = begin
    start_conics = read_parameters(joinpath(@__DIR__, "../data/start_parameters.txt"))
    start_solutions = read_solutions(joinpath(@__DIR__, "../data/start_solutions.txt"))
    start_conics, start_solutions
end

function assemble_trackers()
    startconics, startsolutions = startconics_startsolutions
    F = steiner_system()
    H = ParameterHomotopy(F, startconics, randn(ComplexF64, 30); compile = true)
    tracker = EndgameTracker(H)
    [deepcopy(tracker) for _ in 1:Threads.nthreads()]
end

const trackers = Base.RefValue{Any}(nothing)

function setup_parameters!(homotopy, p₁, p₀)
    H = basehomotopy(homotopy)
    if !(H isa ParameterHomotopy)
       error("Base homotopy is not a ParameterHomotopy")
    end
    set_parameters!(H, (p₁, p₀))
end

function count_ellipses_hyperbolas(tangential_conics)
    nellipses = nhyperbolas = 0
    for conic in tangential_conics
        a, b, c = conic[1], conic[2], conic[3]
        if b^2 - 4 * a * c < 0
            nellipses += 1
        else
            nhyperbolas += 1
        end
    end

    nellipses, nhyperbolas
end

function find_circle(tangential_conics)
    isempty(tangential_conics) && return nothing
    e = map(conic -> (conic[1]-conic[3])^2 + conic[2]^2, tangential_conics)
    f, i = findmin(e)
    return tangential_conics[i]
end

function find_most_complex(complex_solutions)
    e = map(conic -> sum(abs2.(imag(conic))), complex_solutions)
    f, i = findmax(e)
    return complex_solutions[i]
end

function find_most_nondeg(complex_solutions, tangential_conics)
    e1 = map(c -> cond([2c[1] c[2] c[4]; c[2] 2c[3] c[5]; c[4] c[5] 2]), complex_solutions)
    f1, i1 = findmin(e1)
    if !isempty(tangential_conics)
        e2 = map(c -> cond([2c[1] c[2] c[4]; c[2] 2c[3] c[5]; c[4] c[5] 2]), tangential_conics)
        f2, i2 = findmin(e1)

        f2 ≤ f1 && return complex_solutions[i1]
    end

    return complex_solutions[i1]
end

"""
    solve_conics(M::Matrix)

The conics are described as 6 × 5 matrix.

Output is a vector of normalized conics.
"""
function solve_conics(M::Matrix; kwargs...)
    if isnothing(trackers[])
        trackers[] = assemble_trackers()
    end
    solve_conics(M, trackers[]; kwargs...)
end

function solve_conics(M::Matrix, trackers; threading=true)
    tangential_conics = Vector{Float64}[]
    tangential_points = Vector{Float64}[] # Stores [x1,y1,x2,y2,...,x5,y5]
    complex_solutions = Vector{ComplexF64}[]
    tangential_indices = Int[]
    compute_time = @elapsed solve_conics(M, trackers; threading=threading) do x, k
        if maximum(abs ∘ imag, x) < 1e-8
            push!(tangential_conics, real.(x[1:5]))
            push!(tangential_points, real.(x[6:15]))
            push!(tangential_indices, k)
        end
        push!(complex_solutions, x[1:5])
    end
    nreal = length(tangential_conics)
    if !iseven(nreal)
        nreal -= 1
        pop!(tangential_conics)
    end
    nellipses, nhyperbolas = count_ellipses_hyperbolas(tangential_conics)
    is_most_complex = find_most_complex(complex_solutions)
    looks_most_like_a_circle = find_circle(tangential_conics)
    C_nondeg = find_most_nondeg(complex_solutions, tangential_conics)

    Dict("tangential_conics" => tangential_conics,
         "tangential_points" => tangential_points,
         "tangential_indices" => tangential_indices,
         "nreal" => nreal,
         "nellipses" => nellipses,
         "nhyperbolas" => nhyperbolas,
         "compute_time" => round(compute_time; digits=2),
         "complex_solutions" => Dict("real" => real.(complex_solutions),
                                     "imag" => imag.(complex_solutions)),
         "is_most_complex" => Dict("real" => real(is_most_complex),
                                   "imag" => imag(is_most_complex)),
         "looks_most_like_a_circle" => looks_most_like_a_circle,
         "most_nondeg" => Dict("real" => real(C_nondeg),
                               "imag" => imag(C_nondeg)))
end

function partition_work(N)
    k = Threads.nthreads()

    ls = range(1, stop=N, length=k+1)
    map(1:k) do i
        a = round(Int, ls[i])
        if i > 1
            a += 1
        end
        b = round(Int, ls[i+1])
        a:b
    end
end

function solve_conics(output!::F, M::Matrix, trackers; threading=false) where {F<:Function}
    p₁, startsolutions = startconics_startsolutions
    p₀ = vec(complex.(M))
    for tracker in trackers
        HC.parameters!(tracker, p₁, p₀)
    end
    if length(trackers) > 1 && threading
        results = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, 3264)
        ranges = partition_work(3264)
        Threads.@threads for range in ranges
            tid = Threads.threadid()
            track_batch!(results, trackers[tid], range, startsolutions)
        end
        for (k, x) in enumerate(results)
            if x !== nothing
                output!(x, k)
            end
        end
    else
        pathtracker = trackers[1]
        for (k, s) in enumerate(startsolutions)
            ret = track(pathtracker, s)
            if is_success(ret)
                output!(solution(ret), k)
            end
        end
    end
    nothing
end

function track_batch!(results, pathtracker, range, starts)
    for k in range
        s = starts[k]
        ret = track(pathtracker, s)
        if is_success(ret)
            results[k] = solution(ret)
        else
            results[k] = nothing
        end
    end
    nothing
end


function handle_solve_conics(conics_input)
    M = Float64[conics_input[j][i] for i ∈ ["a", "b", "c", "d", "e", "f"], j ∈ 1:5]
    @info "Compute conics:"
    display(M)
    solve_conics(M)
end
