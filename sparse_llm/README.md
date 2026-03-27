# Sparse LLM Sparse Architecture Search Guide

## 1. Purpose of This Document

This document explains how sparse architecture search is currently organized in `sparse_llm/` and provides a consistent integration guide for future contributors.

It focuses on two things:

1. How different models should integrate sparse architecture search within `sparse_llm`
2. Using the current `Qwen3` implementation as an example to explain how search works today, what it optimizes for, and which search methods are used

This document is mainly intended for two types of readers:

- Developers: people who need to integrate support for new models, new sparse kernels, or new search strategies
- Experimenters: people who need to understand the assumptions, objectives, and execution flow of the current `Qwen3` search

---

## 2. Current Project Structure

The code directly related to sparse architecture search is currently split into two layers.

### 2.1 Common Layer

Location:

- `sparse_llm/common/sparse_architecture_search/`

Responsibilities:

- Define search protocols
- Provide the shared search main loop
- Provide shared objectives
- Provide shared strategies
- Provide result export, Pareto computation, and plotting utilities

Key files:

- `contracts.py`
- `runner.py`
- `objectives.py`
- `results.py`
- `plotting.py`
- `strategies/random_search.py`
- `strategies/bayesian_search.py`

### 2.2 Model Layer

Location:

- `sparse_llm/<model>/search_adapter.py`
- `sparse_llm/<model>/sparse_architecture_search/search.py`

Responsibilities:

- Describe which targets in the model are searchable
- Define how candidate parameters are materialized into executable model configs
- Define how a materialized candidate is applied to benchmark arguments
- Preserve the model-specific CLI entry point

---

## 3. Recommended Development Pattern for Different Models

If you want to add sparse architecture search for a new model in the future, the following division of responsibilities is recommended.

### 3.1 Integrate the BenchmarkAdapter First

Reason:

- Search is fundamentally repeated benchmark execution
- If the benchmark interface is unstable, the search layer cannot be stable either

Recommendation:

- Make sure the model already has a usable `BenchmarkAdapter`
- At minimum, it should be able to run the target benchmark for both dense and sparse modes
- It should return structured metrics instead of scattered logs

### 3.2 Then Implement a SearchAdapter

Each model should have its own `SearchAdapter`.

Suggested location:

- `sparse_llm/<model>/search_adapter.py`

It should be responsible for:

- Defining the search context `SearchContext`
- Listing targets where sparse attention can be inserted
- Defining the default search space
- Validating candidates
- Materializing candidates into executable model configs
- Generating candidate signatures
- Applying materialized candidates to runtime arguments

The following should not be placed in `common`:

- How layers are counted for a specific model
- How head count or KV group count should be interpreted
- How layer/group sparsity maps to model configuration
- Whether a specific backend is accepted by the model

### 3.3 Keep the Boundary Between Common and Model Layers Clear

Recommendation:

- `common/sparse_architecture_search` should contain only protocols and reusable search flow
- `<model>/search_adapter.py` should contain only model-specific mapping logic
- `<model>/sparse_architecture_search/search.py` should remain a thin CLI entry point

Not recommended:

- Directly importing a specific model class inside `common`
- Rewriting a separate full search loop inside `<model>/search.py`

### 3.4 Do Not Start With a Very High-Dimensional Search Space

This is a very important engineering recommendation.

For a new model, do not start by directly searching over:

- Every layer
- Every head
- Every block
- Multiple kernels
- Multiple patterns

This creates a very high-dimensional search space, leads to poor search efficiency, and makes Bayesian methods especially fragile.

A staged approach is usually better:

1. Start with a low-dimensional structured search space
2. Gradually increase the degrees of freedom

For example:

- First search global group sparsity
- Then search tail-segment sparsity
- Finally consider per-layer search

### 3.5 Keep Candidate and MaterializedCandidate Separate

This is one of the core ideas in the current framework.

Recommendation:

- Keep `candidate` strategy-friendly
- Keep `materialized_candidate` execution-friendly

Benefits:

- `RandomSearchStrategy` and `BayesianSearchStrategy` do not need to understand internal model details
- The model layer can change its internal implementation without breaking the strategy layer

### 3.6 Decouple Search Objectives From Search Methods

At minimum, the search layer should be split into three parts:

- `SearchAdapter`: how a model represents and executes a candidate
- `SearchObjective`: what counts as "better"
- `SearchStrategy`: how new candidates are proposed

Recommendation:

- Do not hardcode the objective function inside the strategy
- Do not hardcode the candidate representation inside the objective
- Do not hardcode model details inside the runner

### 3.7 Recommendation for Bayesian Search

If you want to add Bayesian search for a new model later, it is recommended to start with:

- Optuna TPE

Instead of immediately implementing:

- Gaussian-process BO
- Multi-objective GP

Reason:

- Sparse search spaces are usually high-dimensional, discrete, and mixed
- TPE is more robust for this kind of space and easier to integrate into the current codebase

---

## 4. How Qwen3 Sparse Architecture Search Currently Works

This section describes the actual implementation for `Qwen3`.

### 4.1 Relevant Files

The code directly related to `Qwen3` sparse architecture search includes:

- `sparse_llm/qwen3/search_adapter.py`
- `sparse_llm/qwen3/sparse_architecture_search/config.py`
- `sparse_llm/qwen3/sparse_architecture_search/user_config.py`
- `sparse_llm/qwen3/sparse_architecture_search/search.py`

The shared dependencies are:

- `sparse_llm/common/sparse_architecture_search/`
- `sparse_llm/common/benchmark/`

### 4.2 What Is Being Searched

The current `Qwen3` search targets:

- `self_attn` inside the `Qwen3` decoder layers

More specifically, the current search focuses on:

- The sparse configuration of grouped-query sparse attention

In the current search context, the `self_attn` module of each layer is treated as a searchable target.

### 4.3 What the Current Search Objective Is

The overall goal of the current `Qwen3` search is:

- Improve sparse runtime speed while keeping quality acceptable

Quality and speed are currently measured mainly through the perplexity benchmark, so this is not a fully task-level objective. It is an engineering proxy objective built around "PPL + throughput."

The core metrics actually used are:

- `perplexity`
- `tokens_per_second`

From these, two derived metrics are computed:

- `baseline_over_ppl = dense_ppl / sparse_ppl`
- `speedup = sparse_tok_s / dense_tok_s`

Where:

- `baseline_over_ppl` being closer to 1 means quality has not degraded significantly
- A larger `speedup` means the sparse configuration provides higher throughput

### 4.4 What the Current Search Space Is

The default `Qwen3` search space is currently "fixed structure, variable sparsity values."

In other words:

- `prefix_dense_layers` is fixed by configuration
- `layer_share_span` is fixed by configuration
- What is actually searched under that structure is `group_sparsities`

There are currently two main candidate forms:

1. Global group sparsity
2. Layer-wise or block-wise `layer_group_sparsities`

The candidate values usually come from a discrete grid such as:

- `(0.0, 0.2, 0.4, 0.6, 0.8)`

This means the current search is closer to:

- Structured search over a discrete sparsity grid

Rather than:

- Infinite-precision optimization over continuous sparsity parameters

### 4.5 What the Current Search Methods Are

`Qwen3` currently supports two strategies.

#### Method A: Random Search

Characteristics:

- Randomly sample candidates from the discrete search space
- Materialize each candidate through `SearchAdapter`, then run the benchmark
- Suitable for a first-pass coarse search

The default objective used with it is:

- `ParetoSpeedVsQualityObjective`

That means it is primarily intended to observe:

- The Pareto relationship between quality and speed

#### Method B: Bayesian Search

Characteristics:

- The current implementation uses Optuna TPE-style Bayesian optimization
- It is more suitable for achieving better sample efficiency in an already structured, low-dimensional search space

The default objective used with it is:

- `WeightedScalarObjective`

That means quality and speed are combined into a single scalar score for optimization.

The reasons are:

- The current common runner and result structure are easier to support first with single-objective Bayesian search
- Multi-objective Bayesian optimization is possible, but the engineering complexity is higher

### 4.6 How Objectives Are Used in Current Qwen3 Search

The default mapping between strategy and objective is:

- `random` -> default `pareto`
- `bayesian` -> default `weighted_scalar`

Where:

- `pareto` is better for inspecting the global trade-off
- `weighted_scalar` is better when the Bayesian optimizer should directly maximize a single numeric score

The current `weighted_scalar` takes the following into account:

- Speed improvement
- Quality retention
- Penalties when the allowed PPL degradation threshold is exceeded

### 4.7 Current Qwen3 Execution Flow

The current `Qwen3` search flow is approximately:

1. Read `user_config.py` or CLI arguments
2. Create `Qwen3SearchAdapter`
3. Create `SearchStrategy`
4. Create `SearchObjective`
5. Enter `common/sparse_architecture_search/run_search(...)`
6. Run the dense baseline
7. Run the shared benchmark for each sparse candidate
8. Record results, update the payload, and recompute the Pareto front
9. Output JSON and plots

In other words:

- Model logic lives in `Qwen3SearchAdapter`
- The main search loop lives in the common runner
- Benchmark execution also reuses the common benchmark layer

### 4.8 Current Limitations of Qwen3 Search

Although `Qwen3` sparse architecture search has already been generalized into common components, there are still some clear limitations:

1. The evaluation path still defaults to the perplexity benchmark
2. The current search space still mainly revolves around group sparsity
3. `prefix_dense_layers` and `layer_share_span` are still treated as fixed structural parameters rather than being searched by default
4. Bayesian search currently prioritizes single-objective scalar optimization

These do not indicate a design mistake. They are simply engineering trade-offs at the current stage.

---

## 5. Recommended Integration Steps for Future Models

If you want to add sparse architecture search for a new model in the future, the recommended order is:

### Step 1

Implement the `BenchmarkAdapter`

### Step 2

Implement `<model>/search_adapter.py`

### Step 3

Define a low-dimensional structured search space first

### Step 4

Run through the flow with `RandomSearchStrategy` first

### Step 5

Then add `BayesianSearchStrategy`

### Step 6

Finally consider:

- Higher-dimensional search spaces
- Multi-objective optimization
- Generation-based objectives
- Task-specific objectives

---

## 6. Suggestions for Experimenters

If your focus is on experiments rather than code changes, it is helpful to think about the current `Qwen3` search in the following way:

- It is not performing unconstrained search over arbitrary model structures
- It is searching within a predefined sparse-attention parameterization space
- Its current benchmark proxy objective is PPL and throughput
- Random Search is suitable for rough exploration
- Bayesian Search is suitable for improving sample efficiency further in a structured, discrete search space

If your experiment goal is:

- To inspect a trade-off curve

Then the more recommended setup is:

- `random + pareto`

If your experiment goal is:

- To let the optimizer automatically find a balanced "best overall" configuration

Then the more recommended setup is:

- `bayesian + weighted_scalar`

---

## 7. Currently Recommended Qwen3 Usage

### 7.1 Random Search

Suitable for:

- First-round exploration
- Inspecting the Pareto front

### 7.2 Bayesian Search

Suitable for:

- Cases where the search-space structure is already determined
- Cases where you want faster convergence within a limited number of trials

The current default template is in:

- `sparse_llm/qwen3/sparse_architecture_search/user_config.py`

Before using Bayesian search for the first time, make sure the environment includes:

- `optuna`

Recommended commands:

```bash
uv sync
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search
```

---

## 8. Summary

In one sentence, the current design is:

- `common` is responsible for "how to search"
- `<model>/search_adapter.py` is responsible for "how this model represents and executes a candidate"
- `Qwen3` is currently the first complete example of this structure

If the framework is expanded to more models in the future, the most important thing is not to copy the `Qwen3` search script itself, but rather to:

1. Reuse the common runner
2. Write a proper `SearchAdapter` for the new model
3. Choose an appropriate objective and strategy

This is how `sparse_llm` can remain extensible instead of drifting back toward a state where each model maintains its own separate search script.
