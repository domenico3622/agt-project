import sympy
from sympy.logic.inference import satisfiable
from sympy import symbols, Or, And, Not
import networkx as nx
import math

class SatVertexCover:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        # Create a symbol for each node: True if node is in Vertex Cover
        self.node_vars = {node: symbols(f'x_{node}') for node in self.nodes}

    def _encode_vertex_cover_constraint(self):
        """
        Constraint 1: For every edge (u, v), at least one endpoint must be in the cover.
        (x_u OR x_v)
        """
        clauses = []
        for u, v in self.graph.edges():
            clauses.append(Or(self.node_vars[u], self.node_vars[v]))
        return And(*clauses)

    def _encode_at_most_k_naive(self, k):
        """
        Naive encoding for AtMostK is combinatorial (n choose k+1) clauses.
        Too slow for n > 20. Sticking to sequential counter or similar is better,
        but for sympy which handles general expressions, we can try to be clever or just rely on a simpler counting network.
        
        Actually, for 'symbolic' focus, let's use a bit counter (adder) constraint 
        or simply use the 'Sequential Counter' encoding which is O(n*k) variables.
        """
        if k >= self.n:
            return sympy.true
        if k < 0:
            return sympy.false

        # Sequential Counter Encoding (Sinze, 2005)
        # s_{i, j} is true if the sum of x_1...x_i is >= j (or similar, definitions vary)
        # Let's use: R_{i, j} means "sum of first i variables is >= j"
        # We want to forbid current sum > k. 
        
        # Variables: x_1 to x_n (mapped from self.node_vars)
        # Aux vars: s_{i, j} for 1 <= i <= n-1, 1 <= j <= k
        
        # Using a slightly easier version for implementation:
        # Sum inputs <= k.
        
        # Let's use a simpler recursive definition supported by SymPy if possible? 
        # No, SymPy SAT solver works best with CNF-like structures.
        
        # Let's implement the standard Sequential Counter to be safe and "algorithmic".
        # Variables s_{i,j}: sum of first i vars is at least j.
        
        # Optimization: We only need to check if count > k.
        
        # Let's map nodes to 0..n-1 indices
        x = [self.node_vars[self.nodes[i]] for i in range(self.n)]
        
        # auxiliary variables s[i][j] : "Sum of x_0...x_i is >= j"
        # We need j up to k+1 to detect overflow.
        # If s[n-1][k+1] is true, then we have > k, which is invalid.
        
        # Wait, if we want "At Most K", we want to FORBID "At Least K+1".
        # So we need to encode "Count(x) >= K+1" and negate it? 
        # Or just encode "Count(x) <= K" directly.
        
        # Let's use the standard "S_{i,j} <-> Sum(x_1...x_i) >= j" encoding.
        # Then we enforce NOT S_{n, k+1}.
        
        s = {}
        clauses = []
        
        # Range of i: 0 to n-1
        # Range of j: 1 to k+1
        
        for i in range(self.n):
            for j in range(1, k + 2):
                s[(i, j)] = symbols(f's_{i}_{j}')

        # Base case for x_0
        # s_{0,1} <-> x_0
        clauses.append(Equivalent(s[(0, 1)], x[0]))
        # s_{0,j} is false for j > 1
        for j in range(2, k + 2):
            clauses.append(Not(s[(0, j)]))

        # Recursive step
        for i in range(1, self.n):
            for j in range(1, k + 2):
                # s_{i, j} <-> (s_{i-1, j} OR (s_{i-1, j-1} AND x_i))
                # Note: s_{i-1, 0} is essentially "sum >= 0", which is True.
                
                term1 = s[(i-1, j)]
                if j == 1:
                    term2 = x[i] # effectively (True AND x_i)
                else:
                    term2 = And(s[(i-1, j-1)], x[i])
                
                clauses.append(Equivalent(s[(i, j)], Or(term1, term2)))

        # Constraint: The sum must NOT be >= k+1
        # So NOT s_{n-1, k+1}
        clauses.append(Not(s[(self.n - 1, k + 1)]))

        return And(*clauses)

    def get_at_most_k_formula(self, k):
        # Implementation of Sequential Counter directly using And/Or/Not
        # Variable: s_i_j is True if sum(x_0...x_i) >= j
        
        if k >= self.n:
            return sympy.true
        
        x = [self.node_vars[self.nodes[i]] for i in range(self.n)]
        
        # We need auxiliary variables. 
        # To avoid name collisions, let's store them in valid python vars
        # s[i][j]
        s = [[None for _ in range(k + 2)] for _ in range(self.n)]
        
        aux_vars_defs = []
        
        # Create symbols
        for i in range(self.n):
            for j in range(1, k + 2):
                s[i][j] = symbols(f's_{i}_{j}')

        constraints = []

        # i = 0
        # s[0][1] <-> x[0]
        constraints.append( Equivalent(s[0][1], x[0]) )
        for j in range(2, k + 2):
            constraints.append( Not(s[0][j]) )

        # i > 0
        for i in range(1, self.n):
            for j in range(1, k + 2):
                # s[i][j] <-> s[i-1][j] OR (s[i-1][j-1] AND x[i])
                prev_same = s[i-1][j]
                if j == 1:
                    prev_minus = x[i]
                else:
                    prev_minus = And(s[i-1][j-1], x[i])
                
                constraints.append( Equivalent(s[i][j], Or(prev_same, prev_minus)) )

        # Final constraint: NOT s[n-1][k+1]
        constraints.append( Not(s[self.n - 1][k + 1]) )
        
        return And(*constraints)

    def solve(self):
        """
        Finds the Minimum Vertex Cover using Binary Search on k.
        """
        # Upper bound: Total nodes (trivial VC)
        # Lower bound: 0
        low = 0
        high = self.n
        min_cover = None
        
        # 1. Base Constraints (Edge Coverage)
        base_formula = self._encode_vertex_cover_constraint()
        
        # Optimization: Check if graph is empty
        if self.n == 0:
            return set()
            
        print(f"SAT Solver: Finding exact MVC for graph with {self.n} nodes. Constraints: {self.graph.number_of_edges()} edges.")
        
        # Binary Search
        while low <= high:
            mid = (low + high) // 2
            print(f"  Checking k={mid}...", end='', flush=True)
            
            # Construct formula: VC_Constraint AND AtMostK(mid)
            # Re-generating cardinality formula each time is fine.
            cardinality_formula = self.get_at_most_k_formula(mid)
            full_formula = And(base_formula, cardinality_formula)
            
            # SymPy solve
            # Note: satisfiable can be slow. 
            model = satisfiable(full_formula)
            
            if model:
                print(" SAT")
                # Found a cover of size mid. Try smaller.
                # Extract solution to ensure we have a valid set to return
                current_cover = set()
                for node in self.nodes:
                    sym = self.node_vars[node]
                    if model.get(sym):
                        current_cover.add(node)
                
                min_cover = current_cover
                high = mid - 1
            else:
                print(" UNSAT")
                # Need larger k
                low = mid + 1
        
        return min_cover

def Equivalent(a, b):
    # Helper because sympy.Equivalent might not be directly exposed or behaves differently in some versions
    return And(Or(Not(a), b), Or(Not(b), a))

if __name__ == "__main__":
    # Simple Test
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    # This is a square with one diagonal. 
    # Nodes: 1,2,3,4.
    # MVC should be size 2? {1, 3} covers (1,2),(1,4),(1,3),(2,3),(3,4). Yes.
    # {2,4} also covers.
    
    solver = SatVertexCover(g)
    result = solver.solve()
    print("Found MVC:", result)
