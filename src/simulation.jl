using Random
using JSON
using Statistics
using ArgParse
using LinearAlgebra
using SparseArrays

const is_directed = Dict("twitter" => true, "gplus" => true, "facebook" => false)

const TASK_ALLOWED_DATASETS = Dict(
    "writing"           => Set(["ukp"]),
    "improvement"       => Set(["semeval"]),
    "contextualization" => Set(["semeval"]),
)

"""
Return `true` if `task` is allowed to operate on `dataset`. Unknown tasks
return `true` (no constraint registered) so new tasks aren't silently dropped
before their mapping is added.
"""
function task_dataset_compatible(task::String, dataset::String)
    allowed = get(TASK_ALLOWED_DATASETS, task, nothing)
    return allowed === nothing || dataset in allowed
end


"""
Precomputed Nadaraya-Watson estimator for the LLM transformation function f,
loaded from `outputs/transformation/transformation__*.json` (produced by
`notebooks/transformation.ipynb`). `bandwidth` is the LOO-CV-selected Gaussian
kernel bandwidth h; `f_grid[i]` is f̂ at the i-th point of a uniform grid over
[0, 1]. Queries use linear interpolation.
"""
struct OpinionStatements
    bandwidth::Float64
    f_grid::Vector{Float64}
end


"""
Load all ego networks from .edges/.feat files and combine into a single sparse adjacency matrix.
"""
function load_ego_network(data_dir; directed::Bool=false)

    nodes = Set{UInt128}()
    edges = Set{Tuple{UInt128,UInt128}}()

    feat_files = filter(f -> endswith(f, ".feat"), readdir(data_dir))
    println("Found $(length(feat_files)) ego networks")

    # Add nodes and edges from ego nodes to other nodes
    for file in feat_files
        ego_node = parse(UInt128, split(file, ".")[1])
        push!(nodes, ego_node)
        for line in eachline(joinpath(data_dir, file))
            parts = split(line)
            if length(parts) >= 1
                neighbor = parse(UInt128, parts[1])
                push!(nodes, neighbor)
                if neighbor != ego_node
                    push!(edges, (ego_node, neighbor))
                    if !directed
                        push!(edges, (neighbor, ego_node))
                    end
                end
            end
        end
    end

    edge_files = filter(f -> endswith(f, ".edges"), readdir(data_dir))

    # Add edges between neighbors of ego nodes
    for file in edge_files
        for line in eachline(joinpath(data_dir, file))
            parts = split(line)
            if length(parts) >= 2
                u, v = parse.(UInt128, parts[1:2])
                if u != v
                    # make sure nodes exist
                    push!(nodes, u, v)
                    # add edge
                    push!(edges, (u, v))
                    if !directed
                        push!(edges, (v, u))
                    end
                end
            end
        end
    end

    println("Loaded $(length(edges)) edges and $(length(nodes)) unique nodes")

    node_list = sort(collect(nodes))
    node_to_idx = Dict(node => i for (i, node) in enumerate(node_list))
    n = length(node_list)

    rows = Int[]
    cols = Int[]
    for (u, v) in edges
        i, j = node_to_idx[u], node_to_idx[v]
        push!(rows, i)
        push!(cols, j)
    end
    A = sparse(rows, cols, ones(Float64, length(rows)), n, n)

    return A
end

"""
Convert adjacency matrix to row-stochastic influence matrix W.
"""
function normalize_adjacency(A)
    row_sums = vec(sum(A, dims=2))
    inv_row_sums = [s > 0 ? 1.0 / s : 0.0 for s in row_sums]
    D_inv = Diagonal(inv_row_sums)
    W = D_inv * A
    return W
end

"""
Randomly assign each agent to community 1 (with probability `community_ratio`) or 2.
"""
function assign_communities(community_ratio, n, rng)

    community_assignments = [rand(rng) < community_ratio ? 1 : 2 for _ in 1:n]

    return community_assignments
end

"""
Sample initial opinions from per-community normals, rejection-sampled to [0, 1].
"""
function generate_initial_opinions(community_assignments,
                                   comm1_mean, comm1_std,
                                   comm2_mean, comm2_std, rng)
    n = length(community_assignments)
    opinions = zeros(Float64, n)

    for i in 1:n
        while true
            if community_assignments[i] == 1
                val = comm1_mean + comm1_std * randn(rng)
            else
                val = comm2_mean + comm2_std * randn(rng)
            end
            if val >= 0 && val <= 1
                opinions[i] = val
                break
            end
        end
    end

    return opinions
end

"""
Sample stubbornness values from Normal(mean, std), rejection-sampled to (0, 1).
"""
function generate_stubbornness(n, mean, std, rng)
    stubbornness = zeros(Float64, n)
    for i in 1:n
        while true
            val = mean + std * randn(rng)
            if val > 0 && val < 1
                stubbornness[i] = val
                break
            end
        end
    end
    return stubbornness
end

"""
Load the precomputed Nadaraya-Watson lookup for one combination from
`<transformation_dir>/transformation__*.json`. The bandwidth and `f_grid` are
fit by `notebooks/transformation.ipynb` (LOO-CV bandwidth selection); this
function only reads them. Returns `nothing` if upstream pipeline steps did not
produce a predict_opinion file for this combination (e.g. unsupported topic).
Hard-errors if the predict_opinion file exists but the transformation JSON
does not — that means the notebook is out of date and the simulation must not
silently fall back.
"""
function load_opinion_statements(transformation_dir::String, dataset::String, task::String,
                                 model::String, topic::String,
                                 quantification_method::String)
    # predict_opinion.py sanitizes '/' to '_' in its output filenames; the
    # transformation notebook reuses the same convention.
    model_sanitized = replace(model, "/" => "_")
    filename = "transformation__dataset=$(dataset)__task=$(task)__model=$(model_sanitized)__topic=$(topic)__quantification_method=$(quantification_method).json"
    filepath = joinpath(transformation_dir, filename)

    if !isfile(filepath)
        # Distinguish "this combo was skipped upstream" (no predict_opinion
        # file either) from "the transformation notebook hasn't been re-run"
        # (predict_opinion exists, transformation JSON missing).
        predict_filename = "predict_opinion__dataset=$(dataset)__task=$(task)__model=$(model_sanitized)__topic=$(topic)__quantification_method=$(quantification_method).tsv"
        predict_filepath = joinpath("outputs/predict_opinion", predict_filename)
        if isfile(predict_filepath)
            error("Transformation file not found: $filepath. The predict_opinion " *
                  "input exists, so the transformation notebook is out of date — " *
                  "rerun notebooks/transformation.ipynb before launching simulations.")
        end
        println("Input file not found: $filepath (and no upstream predict_opinion either). Skipping.")
        return nothing
    end

    println("Loading transformation lookup from $filepath...")
    payload = JSON.parsefile(filepath)
    h = Float64(payload["bandwidth"])
    f_grid = Vector{Float64}(payload["f_grid"])
    if length(f_grid) != F_GRID_SIZE
        error("Transformation file $filepath has f_grid of length $(length(f_grid)); " *
              "expected $F_GRID_SIZE. Regenerate via notebooks/transformation.ipynb.")
    end
    println("Nadaraya-Watson estimator: h=$h (LOO-CV), $(length(f_grid))-point grid, " *
            "n_statements=$(payload["n_statements"])")

    return OpinionStatements(h, f_grid)
end

const F_GRID_SIZE = 4001    # uniform grid over [0, 1]; spacing 2.5e-4

"""
Expressed opinion for one agent: identity if not transformed, else f̂(opinion)
via linear interpolation on `data.f_grid`.
"""
function compute_expressed_opinion_agent(opinion::Float64, is_transformed::Bool, data::OpinionStatements)
    if !is_transformed
        return opinion
    end
    n_grid = length(data.f_grid)
    x = clamp(opinion, 0.0, 1.0)
    pos = x * (n_grid - 1) + 1.0
    i_lo = floor(Int, pos)
    i_hi = min(i_lo + 1, n_grid)
    α = pos - i_lo
    return (1.0 - α) * data.f_grid[i_lo] + α * data.f_grid[i_hi]
end

"""
Run opinion dynamics with data-driven expressed opinions.

Supported `dynamics`:
- `"fj"`      Friedkin-Johnsen: `x ← (I-Λ) W x_expressed + Λ x0`.
- `"degroot"` DeGroot: `x ← W x_expressed` (FJ with `λ = 0`).
- `"hk"`      Hegselmann-Krause: each agent averages expressed opinions of
              all agents within `ε`; stays put if none qualify.
- `"hkn"`     HK restricted to graph in-neighbors (`{j : A[i,j] ≠ 0}`).
- `"dw"`      Deffuant-Weisbuch: each step picks one directed edge `(i, j)`
              uniformly; if `|x[i] - x_exp[j]| ≤ ε`, set
              `x[i] ← x[i] + μ (x_exp[j] - x[i])`.

Returns `(internal_stats, expressed_stats, max_change_per_step, mean_change_per_step)`
with `max_change_per_step[t] = maximum(abs.(x_t - x_{t-1}))` and
`mean_change_per_step[t] = abs(mean(x_t) - mean(x_{t-1}))` for t in 1:T.
"""
function simulate_realistic(x0, A, W, lambda, T, rng, transformed_pct,
                            data::OpinionStatements, community_assignments,
                            dynamics::String;
                            epsilon::Float64=0.0, mu::Float64=0.0,
                            trajectory_file=nothing)
    n = length(x0)

    # Precompute community masks
    comm1_mask = community_assignments .== 1
    comm2_mask = community_assignments .== 2

    # Initialize stats arrays
    internal_stats = Vector{Dict{String,Float64}}(undef, T + 1)
    expressed_stats = Vector{Dict{String,Float64}}(undef, T + 1)
    max_change_per_step = Vector{Float64}(undef, T)
    mean_change_per_step = Vector{Float64}(undef, T)

    # Helper to compute stats for a single timestep
    function compute_stats(opinions)
        Dict(
            "overall_mean" => mean(opinions),
            "overall_std" => std(opinions),
            "community_1_mean" => mean(opinions[comm1_mask]),
            "community_1_std" => std(opinions[comm1_mask]),
            "community_2_mean" => mean(opinions[comm2_mask]),
            "community_2_std" => std(opinions[comm2_mask])
        )
    end

    # Helper function to compute expressed opinions using data-driven interpolation
    function compute_expressed_data_driven(x, transformed_mask)
        x_expressed = similar(x)
        for i in 1:n
            x_expressed[i] = compute_expressed_opinion_agent(x[i], transformed_mask[i], data)
        end
        return x_expressed
    end

    Lambda_x0 = lambda .* x0
    I_minus_Lambda = I - Diagonal(lambda)

    # Edge list for HKN/DW. A[i,j] ≠ 0 means j influences i.
    edge_rows = Int[]
    edge_cols = Int[]
    if dynamics in ("hkn", "dw")
        edge_rows, edge_cols, _ = findnz(A)
        dynamics == "dw" && isempty(edge_rows) && error("DW dynamics requires a non-empty edge list")
    end

    # Open trajectory file if specified
    traj_io = nothing
    if trajectory_file !== nothing
        traj_io = open(trajectory_file, "w")
        write(traj_io, Int32(n))
        write(traj_io, Int32(T + 1))
        write(traj_io, Float32.(x0))
    end

    # Fix the transformed mask for the whole simulation
    transformed_mask = rand(rng, n) .< transformed_pct

    # Store initial state
    x_expressed_0 = compute_expressed_data_driven(x0, transformed_mask)
    internal_stats[1] = compute_stats(x0)
    expressed_stats[1] = compute_stats(x_expressed_0)

    # Run dynamics
    x = copy(x0)
    for t in 1:T
        x_prev = copy(x)
        println("Time step: ", t)
        x_expressed = compute_expressed_data_driven(x, transformed_mask)

        if dynamics == "fj"
            x = I_minus_Lambda * W * x_expressed + Lambda_x0
        elseif dynamics == "degroot"
            x = W * x_expressed
        elseif dynamics == "hk"
            # n×n confidence BitMatrix: M[i, j] = |x[i] - x_expressed[j]| ≤ ε.
            # Each agent averages x_expressed over the j's where M[i, j] holds.
            M = abs.(x .- x_expressed') .<= epsilon
            counts = vec(sum(M, dims=2))
            sums = M * x_expressed
            x = ifelse.(counts .> 0, sums ./ counts, x)
        elseif dynamics == "hkn"
            # Like HK but restricted to graph edges: build a sparse confidence
            # matrix on edges within ε and row-average x_expressed over them.
            diffs = abs.(x[edge_rows] .- x_expressed[edge_cols])
            keep = diffs .<= epsilon
            C = sparse(edge_rows[keep], edge_cols[keep], 1.0, n, n)
            counts = vec(sum(C, dims=2))
            sums = C * x_expressed
            x = ifelse.(counts .> 0, sums ./ counts, x)
        elseif dynamics == "dw"
            # One pair meets per timestep
            e = rand(rng, 1:length(edge_rows))
            i, j = edge_rows[e], edge_cols[e]
            if i != j && abs(x[i] - x_expressed[j]) <= epsilon
                x[i] += mu * (x_expressed[j] - x[i])
            end
        else
            error("Unknown dynamics: $dynamics")
        end

        internal_stats[t + 1] = compute_stats(x)
        expressed_stats[t + 1] = compute_stats(compute_expressed_data_driven(x, transformed_mask))
        max_change_per_step[t] = maximum(abs.(x .- x_prev))
        mean_change_per_step[t] = abs(mean(x) - mean(x_prev))

        # Write current opinions to trajectory file
        if traj_io !== nothing
            write(traj_io, Float32.(x))
        end
    end

    # Close trajectory file
    if traj_io !== nothing
        close(traj_io)
    end

    return internal_stats, expressed_stats, max_change_per_step, mean_change_per_step
end

"""
Generate initial conditions and run one simulation seed.
"""
function single_seed_experiment_realistic(rng, A, timesteps, community_ratio,
                                          stubbornness_mean, stubbornness_std,
                                          transformed_pct, data::OpinionStatements,
                                          comm1_opinion_mean, comm1_opinion_std,
                                          comm2_opinion_mean, comm2_opinion_std,
                                          dynamics::String, epsilon::Float64, mu::Float64;
                                          trajectory_file::Union{String,Nothing}=nothing)

    n_agents = size(A)[1]
    W = normalize_adjacency(A)
    println("Network has $n_agents agents")

    println("Generating initial conditions...")
    community_assignments = assign_communities(community_ratio, n_agents, rng)
    x0 = generate_initial_opinions(community_assignments, comm1_opinion_mean, comm1_opinion_std,
                                   comm2_opinion_mean, comm2_opinion_std, rng)
    lambda = generate_stubbornness(n_agents, stubbornness_mean, stubbornness_std, rng)

    println("Running $dynamics dynamics for $timesteps timesteps...")
    internal_stats, expressed_stats, max_change_per_step, mean_change_per_step = simulate_realistic(
        x0, A, W, lambda, timesteps, rng,
        transformed_pct, data, community_assignments, dynamics;
        epsilon=epsilon, mu=mu,
        trajectory_file=trajectory_file)

    println("Max |x_T - x_{T-1}|_∞ = $(max_change_per_step[end])")
    println("|mean(x_T) - mean(x_{T-1})| = $(mean_change_per_step[end])")

    result = Dict{String, Any}(
        "n_agents" => n_agents,
        "n_community_1" => sum(community_assignments .== 1),
        "n_community_2" => sum(community_assignments .== 2),
        "internal_trajectory" => internal_stats,
        "expressed_trajectory" => expressed_stats,
        "max_change_per_step" => max_change_per_step,
        "mean_change_per_step" => mean_change_per_step
    )

    if trajectory_file !== nothing
        result["trajectory_file"] = trajectory_file
    end

    return result
end

function experiment(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        # Standard executor arguments
        "--exp_name"
            help = "Name of the experiment"
            required = true
            arg_type = String
        "--output_dir"
            help = "Output directory"
            required = true
            arg_type = String
        "--output_filename"
            help = "Output filename"
            required = true
            arg_type = String
        # Network parameter
        "--network"
            help = "Network name: 'twitter', 'facebook', or 'gplus'"
            required = true
            arg_type = String
        # Dynamics selection
        "--dynamics"
            help = "Opinion dynamics: 'fj', 'degroot', 'hk', 'hkn', or 'dw'"
            required = false
            default = "fj"
            arg_type = String
        "--epsilon"
            help = "Confidence threshold for HK and DW dynamics"
            required = false
            default = 0.2
            arg_type = Float64
        "--mu"
            help = "Convergence rate (in (0, 1]) for DW pair updates"
            required = false
            default = 0.5
            arg_type = Float64
        # Simulation parameters
        "--timesteps"
            help = "Number of simulation timesteps"
            required = true
            arg_type = Int
        "--community_ratio"
            help = "Fraction of agents in community 1"
            required = false
            default = 0.5
            arg_type = Float64
        "--stubbornness_mean"
            help = "Mean of stubbornness distribution"
            required = false
            default = 0.5
            arg_type = Float64
        "--stubbornness_std"
            help = "Standard deviation of stubbornness distribution"
            required = false
            default = 0.1
            arg_type = Float64
        # Initial opinion distribution parameters (opinions are in [0, 1])
        "--comm1_opinion_mean"
            help = "Mean of initial opinion distribution for community 1"
            required = false
            default = 0.75
            arg_type = Float64
        "--comm1_opinion_std"
            help = "Standard deviation of initial opinion distribution for community 1"
            required = false
            default = 0.1
            arg_type = Float64
        "--comm2_opinion_mean"
            help = "Mean of initial opinion distribution for community 2"
            required = false
            default = 0.25
            arg_type = Float64
        "--comm2_opinion_std"
            help = "Standard deviation of initial opinion distribution for community 2"
            required = false
            default = 0.1
            arg_type = Float64
        # Transformed agents
        "--transformed_pct"
            help = "Fraction of agents who are transformed (0.0 to 1.0)"
            required = false
            default = 0.0
            arg_type = Float64
        # Data-driven opinion parameters
        "--dataset"
            help = "Dataset for statement data (e.g., 'ukp', 'semeval')"
            required = true
            arg_type = String
        "--task"
            help = "Task for statement data (e.g., 'writing', 'improvement')"
            required = true
            arg_type = String
        "--topic"
            help = "Topic for statement data (e.g., 'abortion', 'gun_control')"
            required = true
            arg_type = String
        "--model"
            help = "Model name for statement data (e.g., 'meta-llama_Llama-3.1-8B-Instruct')"
            required = true
            arg_type = String
        "--quantification_method"
            help = "Quantification method used in predict_opinion (e.g., 'centroid')"
            required = false
            default = "centroid"
            arg_type = String
        "--transformation_dir"
            help = "Directory containing precomputed transformation JSONs from notebooks/transformation.ipynb"
            required = false
            default = "outputs/transformation"
            arg_type = String
        # Seeding
        "--master_seed"
            help = "Master seed for generating seeds"
            required = false
            default = 42
            arg_type = Int
        "--num_seeds"
            help = "Number of derived seeds (0 = use master_seed directly)"
            required = false
            default = 0
            arg_type = Int
        # Trajectory saving
        "--save_trajectories"
            help = "Save full opinion trajectories to binary files"
            action = :store_true
    end

    # Parse arguments
    parsed_args = parse_args(args, s)
    exp_name = parsed_args["exp_name"]
    output_dir = parsed_args["output_dir"]
    output_filename = parsed_args["output_filename"]

    # Skip if this config has already completed (idempotent reruns)
    if isfile(joinpath(output_dir, output_filename))
        println("Output already exists at $(joinpath(output_dir, output_filename)); skipping.")
        return
    end

    network = parsed_args["network"]
    dynamics = parsed_args["dynamics"]
    epsilon = parsed_args["epsilon"]
    mu = parsed_args["mu"]
    timesteps = parsed_args["timesteps"]
    community_ratio = parsed_args["community_ratio"]
    stubbornness_mean = parsed_args["stubbornness_mean"]
    stubbornness_std = parsed_args["stubbornness_std"]
    comm1_opinion_mean = parsed_args["comm1_opinion_mean"]
    comm1_opinion_std = parsed_args["comm1_opinion_std"]
    comm2_opinion_mean = parsed_args["comm2_opinion_mean"]
    comm2_opinion_std = parsed_args["comm2_opinion_std"]
    transformed_pct = parsed_args["transformed_pct"]
    dataset = parsed_args["dataset"]
    task = parsed_args["task"]
    topic = parsed_args["topic"]
    model = parsed_args["model"]
    quantification_method = parsed_args["quantification_method"]
    transformation_dir = parsed_args["transformation_dir"]
    master_seed = parsed_args["master_seed"]
    num_seeds = parsed_args["num_seeds"]
    save_trajectories = parsed_args["save_trajectories"]

    # Skip combinations where the task is not designed for this dataset
    # (mirrors transform_text.py so the executor treats it as a no-op).
    if !task_dataset_compatible(task, dataset)
        allowed = sort(collect(TASK_ALLOWED_DATASETS[task]))
        println("Task '$task' is not compatible with dataset '$dataset' (expected one of $allowed). Skipping.")
        return
    end

    # Validate parameters
    if !(network in ["twitter", "facebook", "gplus"])
        error("network must be 'twitter', 'facebook', or 'gplus'")
    end
    network_data_dir = "data/original/$network"
    if !isdir(network_data_dir)
        error("Network data directory does not exist: $network_data_dir")
    end
    edge_files = filter(f -> endswith(f, ".edges"), readdir(network_data_dir))
    if isempty(edge_files)
        error("No .edges files found in $network_data_dir")
    end
    println("Found $(length(edge_files)) ego networks in $network_data_dir")
    if timesteps < 1
        error("timesteps must be at least 1")
    end
    if community_ratio < 0 || community_ratio > 1
        error("community_ratio must be in [0, 1]")
    end
    if stubbornness_std <= 0
        error("stubbornness_std must be positive")
    end
    if transformed_pct < 0 || transformed_pct > 1
        error("transformed_pct must be in [0, 1]")
    end
    if !(dynamics in ("fj", "degroot", "hk", "hkn", "dw"))
        error("dynamics must be 'fj', 'degroot', 'hk', 'hkn', or 'dw'")
    end
    if dynamics in ("hk", "hkn", "dw") && epsilon <= 0
        error("epsilon must be positive for $dynamics dynamics")
    end
    if dynamics == "dw" && (mu <= 0 || mu > 1)
        error("mu must be in (0, 1] for DW dynamics")
    end

    # Ensure output directory exists (executor only creates log_dir)
    mkpath(output_dir)

    # Load statement data (once, shared across all seeds). Missing input is
    # not an error: it just means this combo (e.g. topic not in dataset) was
    # skipped upstream. Mirrors predict_opinion.py.
    statement_data = load_opinion_statements(transformation_dir, dataset, task, model, topic,
                                             quantification_method)
    if statement_data === nothing
        return
    end

    # Load network adjacency matrix (once, shared across all seeds)
    println("Loading network: $network...")
    A = load_ego_network("data/original/$network", directed=is_directed[network])

    # Determine seeds to use
    if num_seeds <= 0
        seeds = [master_seed]
    else
        master_rng = Random.Xoshiro(master_seed)
        seeds = rand(master_rng, UInt32, num_seeds)
    end

    # Collect seed results
    seed_results = []
    for (i, seed) in enumerate(seeds)
        println("Running experiment with seed $seed...")
        flush(stdout)
        rng = Xoshiro(seed)

        # Generate trajectory file path if saving trajectories
        trajectory_file = nothing
        if save_trajectories
            base_name = replace(output_filename, r"\.json$" => "")
            trajectory_file = joinpath(output_dir, "$(base_name)_seed$(i)_trajectory.bin")
        end

        result = single_seed_experiment_realistic(
            rng, A, timesteps, community_ratio,
            stubbornness_mean, stubbornness_std,
            transformed_pct, statement_data,
            comm1_opinion_mean, comm1_opinion_std,
            comm2_opinion_mean, comm2_opinion_std,
            dynamics, epsilon, mu;
            trajectory_file=trajectory_file
        )
        result["seed"] = Int(seed)
        push!(seed_results, result)
    end

    output = Dict(
        "exp_name" => exp_name,
        "network" => network,
        "dynamics" => dynamics,
        "epsilon" => epsilon,
        "mu" => mu,
        "n_agents" => seed_results[1]["n_agents"],
        "timesteps" => timesteps,
        "community_ratio" => community_ratio,
        "stubbornness_mean" => stubbornness_mean,
        "stubbornness_std" => stubbornness_std,
        "comm1_opinion_mean" => comm1_opinion_mean,
        "comm1_opinion_std" => comm1_opinion_std,
        "comm2_opinion_mean" => comm2_opinion_mean,
        "comm2_opinion_std" => comm2_opinion_std,
        "transformed_pct" => transformed_pct,
        "dataset" => dataset,
        "task" => task,
        "topic" => topic,
        "model" => model,
        "quantification_method" => quantification_method,
        "transformation_dir" => transformation_dir,
        "bandwidth" => statement_data.bandwidth,
        "master_seed" => master_seed,
        "num_seeds" => num_seeds,
        "save_trajectories" => save_trajectories,
        "seed_results" => seed_results
    )

    println("Saving results to $output_dir/$output_filename...")
    flush(stdout)
    open(joinpath(output_dir, output_filename), "w") do f
        write(f, JSON.json(output, 2))
    end

    println("Done!")
end

experiment(ARGS)
