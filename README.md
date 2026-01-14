# Network Security Game - AGT Project

**Topic:** Game Theoretic approaches to Minimal Network Security Sets (Vertex Cover).
**Modules:** Strategic Games, Coalitional Games, Market Allocation, VCG Path Auctions.

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install networkx numpy matplotlib sympy
    ```
2.  **Run Simulation:**
    ```bash
    python main.py
    ```
    *Generates random graphs, executes all algorithms (Game Theory, Coalitional, Market, VCG, Exact SAT), and saves plots to `results/`.*
3.  **View Dashboard:**
    Open **`index.html`** in your browser to inspect the results interactively.

## Configuration
Modify `main.py` (under `if __name__ == "__main__":`) to change simulation parameters:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `num_nodes` | `200` | Graph size. Decrease (e.g., 50) for clearer plots; increase for stress tests. (Note: SAT Solver auto-switches to subgraphs for N>50) |
| `k` | `3` | Degree for **Regular Graph**. |
| `p` | `0.05` | Edge probability for **Erdős-Rényi**. |
| `m` | `2` | Attachment edges for **Barabasi-Albert**. |
| `max_iter` | `100` | Max iterations for Strategic Algorithms. |
| `num_permutations`| `500` | Monte Carlo samples for Shapley Value estimation. |

## Implementation Details

### 1. Strategic Game (`SecurityGame.py`)
* **Goal:** Find Pure Nash Equilibria (PNE) representing Minimal Security Sets.
* **Payoff:** $\alpha - c$ (secure), $\alpha$ (covered by neighbor), $0$ (uncovered).
* **Algorithms:** **Best Response Dynamics** (Deterministic), **Fictitious Play** (Belief-based), **Regret Matching** (Probabilistic).

### 2. Coalitional Game (`CoalitionalSecurityGame.py`)
* **Method:** **Monte Carlo Shapley Value** approximation to measure node importance (marginal contribution to coverage).
* **Selection:** **Reverse Greedy** heuristic (pruning lowest Shapley nodes first) to construct a minimal set.
* **Visualization:** **Shapley Heatmap** (`shapley_heatmap.png`) to visually identify critical nodes (red/warm colors) vs. non-critical ones (blue/cool colors).

### 3. Market Allocation (`SecurityMarketplace.py`)
Matches Security Set nodes (Buyers) to Vendors based on Budget/Price/Quality.
* **Infinite Capacity:** Buyers select vendors maximizing $U = (\text{Security} \times 12) + \text{Savings}$. Result: Pareto-optimality.
* **Limited Capacity:** **Global Greedy** allocation to maximize Total Social Welfare. Result: High-budget buyers may be "crowded out" to suboptimal vendors.
* **Visualization:**
    *   **Budget Histogram:** Shows matched vs. unmatched buyers by budget range.
    *   **Vendor Bubble Chart:** Plot of Price vs. Security Level, where bubble size = sales volume and color = average match utility.

### 4. VCG Path Auction (`VCGPathAuction.py`)
Truthful mechanism to buy a secure path ($S \to T$).
* **Cost:** Node Bid + Penalty (if node is insecure).
* **Payment:** VCG Rule ($P_i = \text{SocialCost without } i - \text{SocialCost with } i$). Internalizes the insecurity externality.

### 5. Exact Minimal Vertex Cover (`SatVertexCover.py`)
Finds the **Exact Global Minimal** Vertex Cover using a Symbolic SAT Solver.
* **Logic:** Encodes Edge Coverage and Cardinality Constraints (via Sequential Counter) into Boolean Logic.
* **Objective:** Validates the "Ground Truth" to benchmark how close the Game Theoretic Nash Equilibria are to the true optimum.
* **Usage:** Runs on full graph for $N \le 50$, or on a subgraph for larger networks to ensure performance.

## Documentation
For a deeper dive into the theoretical background, methodology, and detailed analysis of results, please refer to:
*   [**Project Report**](report_agt.pdf)
*   [**Presentation Slides**](slides_agt.pdf)

### 6. High-Performance Simulation (`onlyBestResp.py`)
Optimized implementation for large-scale networks ($N > 100,000$).
*   **Tech Stack:** `scipy.sparse` for memory efficiency and `numba` (JIT compilation) for C-like speed.
*   **Algorithm:** **Best Response Dynamics** (Asynchronous).
*   **Usage:** Run independent of `main.py` for stress testing.
    ```bash
    python logic/onlyBestResp.py
    ```

## File Structure
* `main.py`: Entry point & Orchestrator.
* `index.html`: Visualization Dashboard.
* `logic/`: Core logic modules.
    *   `onlyBestResp.py`: High-performance standalone simulation.
* `results/`: Output directory for generated plots.
* `assets/`: Assets for the dashboard.

## Authors
* **Di Franco Federico**
* **Serratore Francesco**
* **Visciglia Domenico**