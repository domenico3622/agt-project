import numpy as np
import random
import scipy.sparse as sp
import time
from numba import njit

# --- Regular Graph Generator (Pure Numpy/Scipy) ---
def generate_random_regular_sparse(n, k, max_attempts=20):
    """
    Generates a random k-regular graph directly in sparse format (CSR).
    Uses the Configuration Model algorithm with restart in case of collisions (self-loops/multi-edges).
    
    :param n: Number of nodes
    :param k: Degree of each node
    :return: scipy.sparse.csr_matrix
    """
    if (n * k) % 2 != 0:
        raise ValueError("n * k deve essere pari.")
    
    # Create "stubs": each node appears k times in the list
    # Example n=3, k=2 -> [0, 0, 1, 1, 2, 2]
    stubs = np.repeat(np.arange(n), k)
    
    for attempt in range(max_attempts):
        # Shuffle stubs to create random connections
        np.random.shuffle(stubs)
        
        # Pair stubs 2 by 2 to form edges
        # Reshape into (num_edges, 2)
        edges = stubs.reshape(-1, 2)
        
        # --- Validity Checks ---
        
        # 1. Self-loops: check if edges[:,0] == edges[:,1]
        if np.any(edges[:, 0] == edges[:, 1]):
            continue # Retry, found self-loop
            
        # 2. Multi-edges (duplicate edges)
        # Sort each edge so that u < v to facilitate comparison
        edges.sort(axis=1)
        
        # Use numpy.unique to find duplicates. 
        # axis=0 checks rows (edges)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        
        if np.any(counts > 1):
            continue # Retry, found parallel edges
            
        # If we are here, the graph is valid (Simple)
        print(f"Graph generated successfully at attempt {attempt + 1}")
        
        # Build the symmetric sparse matrix
        # An edge (u, v) must become two entries: (u, v) and (v, u)
        row_indices = np.concatenate([edges[:, 0], edges[:, 1]])
        col_indices = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(row_indices), dtype=np.int8)
        
        # Create the CSR matrix
        adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return adj_matrix

    raise RuntimeError("Unable to generate a simple regular graph after multiple attempts. "
                       "Try increasing n or reducing k.")

# --- Optimized SecurityGame Class (Sparse Matrix) ---
class SecurityGameSparse:
    def __init__(self, adj_matrix, alpha=10, c=4):
        self.adj_matrix = adj_matrix
        self.num_players = adj_matrix.shape[0]
        self.alpha = alpha
        self.c = c

        if not (self.alpha > self.c > 0):
            raise ValueError("Parameters must satisfy alpha > c > 0")

    def check_coverage(self, player_id, strategies):
        start_idx = self.adj_matrix.indptr[player_id]
        end_idx = self.adj_matrix.indptr[player_id + 1]
        
        if start_idx == end_idx:
            return False
            
        neighbor_indices = self.adj_matrix.indices[start_idx:end_idx]
        neighbor_strats = strategies[neighbor_indices]
        return np.all(neighbor_strats == 1)

# --- Numba JIT Compiled Function ---
@njit(fastmath=True)
def run_dynamics_numba(indptr, indices, strategies, num_players, alpha, c, max_iterations):
    """
    JIT-compiled version of Best Response Dynamics.
    Maintains sequential (asynchronous) logic but runs at C speed.
    """
    for iteration in range(max_iterations):
        changes_count = 0
        for player_id in range(num_players):
            # Check coverage (manual loop optimized for Numba)
            start_idx = indptr[player_id]
            end_idx = indptr[player_id + 1]
            
            is_covered = True
            if start_idx == end_idx:
                is_covered = False
            else:
                for k in range(start_idx, end_idx):
                    neighbor = indices[k]
                    if strategies[neighbor] == 0:
                        is_covered = False
                        break
            
            # Payoff Calculation
            payoff_1 = alpha - c
            payoff_0 = alpha if is_covered else 0
            
            current_s = strategies[player_id]
            new_s = current_s

            if payoff_1 > payoff_0:
                new_s = 1
            elif payoff_0 > payoff_1:
                new_s = 0
            else:
                # Random tie-break
                if np.random.random() < 0.5:
                    new_s = 1 - current_s 
            
            if new_s != current_s:
                strategies[player_id] = new_s
                changes_count += 1
                
        if changes_count == 0:
            return strategies, iteration + 1
            
    return strategies, max_iterations

# --- Optimized BestResponseDynamics ---
class BestResponseDynamicsSparse:
    def __init__(self, game, max_iterations=1000):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        # Initial random strategies
        self.current_strategies = np.random.choice([0, 1], size=self.num_players).astype(np.int8)

    def run(self):
        print(f"Starting dynamic simulation on {self.num_players} nodes (JIT Optimized)...")
        start_time = time.time()

        # Data preparation for Numba (NumPy arrays underlying the sparse matrix)
        indptr = self.game.adj_matrix.indptr
        indices = self.game.adj_matrix.indices
        strategies = self.current_strategies
        
        # Scalar parameters
        n = self.num_players
        alpha = self.game.alpha
        c = self.game.c
        max_iter = self.max_iterations

        # Call to compiled function
        # Note: The first run will include compilation time (small initial overhead)
        final_strategies, used_iterations = run_dynamics_numba(
            indptr, indices, strategies, n, alpha, c, max_iter
        )
        
        self.current_strategies = final_strategies

        if used_iterations < max_iter:
            print(f"Convergence reached at iteration {used_iterations}.")
        else:
            print(f"Stop: Max iterations ({max_iter}) reached.")
        
        print(f"Execution time: {time.time() - start_time:.4f} seconds")
        return self.current_strategies

# --- Validation Functions ---
def is_minimal_security_set_sparse(adj_matrix, strategies_array):
    indices_in_set = np.where(strategies_array == 1)[0]
    indices_out_set = np.where(strategies_array == 0)[0]
    
    # 1. Verify global coverage
    for node in indices_out_set:
        start = adj_matrix.indptr[node]
        end = adj_matrix.indptr[node+1]
        if start == end: continue 
        neighbors = adj_matrix.indices[start:end]
        if not np.all(strategies_array[neighbors] == 1):
            return False 
            
    # 2. Verify Minimality
    for node_to_remove in indices_in_set:
        strategies_array[node_to_remove] = 0
        is_still_valid = True
        
        # Check only the removed node (which is now 0 and must be covered)
        start = adj_matrix.indptr[node_to_remove]
        end = adj_matrix.indptr[node_to_remove+1]
        neighbors = adj_matrix.indices[start:end]
        
        # If removing it makes the node uncovered, then the removal was invalid -> OK
        # If instead removing it it is still covered (or has no neighbors), the original set was NOT minimal.
        if start != end and np.all(strategies_array[neighbors] == 1):
             is_still_valid = True # It is still covered by neighbors
        else:
             is_still_valid = False # It is uncovered
        
        strategies_array[node_to_remove] = 1 # Restore
        
        if is_still_valid:
            return False 

    return True

# --- Main Execution ---
def run_simulation_no_nx():
    # Parameters
    num_nodes = 10_000_000
    k = 3
    max_iter = 200

    print(f"--- GRAPH GENERATION (No NetworkX) ---")
    print(f"N={num_nodes}, K={k}")
    
    # 1. Direct Sparse Graph Creation
    try:
        adj_matrix = generate_random_regular_sparse(num_nodes, k)
    except Exception as e:
        print(e)
        return

    # 2. Game Configuration
    game = SecurityGameSparse(adj_matrix, alpha=10, c=4)
    algo = BestResponseDynamicsSparse(game, max_iterations=max_iter)
    
    # 3. Execution
    final_strategies = algo.run()
    
    set_size = np.sum(final_strategies)
    print(f"\nFinal Security Set Size: {set_size} ({set_size/num_nodes:.2%})")
    
    # 4. Verification (only if small enough)
    if num_nodes <= 10000:
        print("Verifying minimality...")
        is_min = is_minimal_security_set_sparse(adj_matrix, final_strategies.copy())
        print(f"Is it a Minimal Security Set? {is_min}")
    else:
        print("Verification skipped (N too large).")

if __name__ == "__main__":
    run_simulation_no_nx()