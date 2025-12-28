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
        # Create a symbol (logic variable) for each node:
        # True if the node is part of the Vertex Cover, False otherwise.
        self.node_vars = {node: symbols(f'x_{node}') for node in self.nodes}

    def _encode_vertex_cover_constraint(self):
        """
        Constraint 1: Edge Coverage.
        For every edge (u, v) of the graph, at least one of the two endpoints must be chosen.
        Logic: (x_u OR x_v) must be TRUE for every edge.
        """
        clauses = []
        for u, v in self.graph.edges():
            clauses.append(Or(self.node_vars[u], self.node_vars[v]))
        return And(*clauses)

    def get_at_most_k_formula(self, k):
        """
        Generates the logical formula for the cardinality constraint: "The sum of chosen nodes must be <= k".
        Uses the "Sequential Counter" technique. Since the sum operation does not exist in Boolean logic, 
        a sequential counter is used to count the number of chosen nodes.
        
        We cannot count as we don't have numbers (1, 2, 3...), but only switches (ON/OFF) 
        and we need to keep track of a game score. The Sequential Counter is a way to 
        build a "counter" using only logic. Here is how it works, explained with a practical example.
        The Goal:
        You have 3 bulbs (x_1, x_2, x_3). I want to know if at least 2 are on. The computer cannot do 1+0+1=2. 
        It has to get there step by step.
        The "Counter" (The S variables)
        The counter creates a grid of registers (temporary variables called S). Each register S_{i,j} answers a 
        specific question: "Having reached bulb i, have I found at least j on?"
        - i --> Where I am in the row (bulb 1, 2 or 3).
        - j --> The total I am checking (total 1, total 2...).
        Step-by-Step Example: Imagine your bulbs are:
        - x_1: ON (1)   
        - x_2: OFF (0)
        - x_3: ON (1)
        Let's see how the sequential counter registers are filled.
        1. Look at the first bulb (x_1 = 1)
        - Question: Have I found at least 1 bulb on?
            - YES, because x_1 is on. (So register S_{1,1} becomes TRUE).
        - Question: Have I found at least 2 bulbs on?
            - NO, I've only seen one so far. (So register S_{1,2} is FALSE).

        2. Move to the second bulb (x_2 = 0)
        The counter must update totals based on what it knew before PLUS the new bulb.
        - Question: Have I found at least 1 on so far? 
            - The computer reasons: "Either I already had 1 before, OR I had 0 before and this one is on".
            - Did I have 1 before? YES. Result: YES (total 1 is "conserved").
        - Question: Have I found at least 2 on so far?
            - The computer reasons: "Either I already had 2 before, OR I had 1 before and this one is on".
            - Did I have 2 before? NO.
            - Did I have 1 before (S_{1,1} was True) AND is this one (x_2) on? NO, because x_2 is off.
            - Result: NO (We stay stuck at 1).

        3. Move to the third bulb (x_3 = 1)
        - Question: Have I found at least 1 on?
            - Yes, I already knew that. Remains YES.
        - Question: Have I found at least 2 on? 
            - Logic reasoning: "Either I already had 2 before (S_{2,2}), OR I had at least 1 before (S_{2,1}) AND the 
            current bulb (x_3) is on".
            - Analysis:
                - Did I have 2? No.
                - Did I have 1? YES. 
                - Is x_3 on? YES.
                - So: YES + YES = I reached quota 2!
                - Result: YES (S_{3,2} becomes TRUE).
        
        The Logic Rule (The Formula):
        Here is the complex formula in the code explained: S[i][j] = S[i-1][j] OR (S[i-1][j-1] AND x[i]) 
        The current total reaches quota J if:
            Case A: I had already reached quota J in the previous step (I don't need the current bulb, I already "won").
            OR
            Case B: I had reached quota J-1 (I needed 1 more) AND the current bulb gives exactly that missing point.
        
        How do I say "At Most K"?
        If I want to enforce that there are at most 2 bulbs on (K=2), I do all this calculation and at the end I check the register 
        for total 3 (K+1). If the register "I found at least 3 bulbs" (S_{n, 3}) turns on, it means I exceeded.
        Therefore, the final rule I give to the SAT Solver is: NOT (S_{n, 3}).
        Basically: "Do whatever combinations you want, but make sure the total 3 box remains OFF".
        """
    
        # If k is greater than or equal to n, the constraint is always satisfied (trivial).
        if k >= self.n:
            return sympy.true
        
        # Map nodes to an ordered list x_0, x_1, ...
        x = [self.node_vars[self.nodes[i]] for i in range(self.n)]
        
        # Auxiliary variables s[i][j]: indicate if "the sum of the first i+1 variables reaches at least j".
        # We need to check up to k+1 to detect if we exceed k.
        s = [[None for _ in range(k + 2)] for _ in range(self.n)]
        
        # Creation of symbols for auxiliary variables
        for i in range(self.n):
            for j in range(1, k + 2):
                s[i][j] = symbols(f's_{i}_{j}')

        constraints = []

        # Base Case (i = 0):
        # s[0][1] is true if and only if x[0] is true.
        constraints.append( Equivalent(s[0][1], x[0]) )
        # s[0][j] for j > 1 is impossible (cannot have sum > 1 with a single element).
        for j in range(2, k + 2):
            constraints.append( Not(s[0][j]) )

        # Recursive Step (i > 0):
        # The partial sum at step i reaches j if:
        # 1. It reached j already at step i-1 (without counting x[i]).
        # 2. It reached j-1 at step i-1 AND x[i] is true (so adding 1).
        for i in range(1, self.n):
            for j in range(1, k + 2):
                # s[i][j] <-> s[i-1][j] OR (s[i-1][j-1] AND x[i])
                prev_same = s[i-1][j]
                if j == 1:
                    prev_minus = x[i]
                else:
                    prev_minus = And(s[i-1][j-1], x[i])
                
                constraints.append( Equivalent(s[i][j], Or(prev_same, prev_minus)) )

        # Final Constraint:
        # It must NOT be true that the total sum reaches k+1.
        # If s[n-1][k+1] were true, we would have chosen more than k nodes, violating the constraint.
        constraints.append( Not(s[self.n - 1][k + 1]) )
        
        return And(*constraints)

    def solve(self):
        """
        This is the function that executes the search for the optimal solution.
        The problem asks for the Minimum Vertex Cover. To find it, it uses Binary Search on the number k (from 0 to N).
        
        While loop low <= high:
        1. Choose a middle number mid.
        2. Construct the complete formula:
           - base_formula: Edges must be covered.
           - cardinality_formula: Must use at most mid nodes.
           - full_formula: And(base_formula, cardinality_formula).
        3. satisfiable(full_formula): Ask sympy if there is a way to make this formula true.
        
        - If SAT (solution found): It means it is possible to cover the graph with mid nodes. 
          Ok, but maybe it can be done with fewer. So save the solution and try with a lower k 
          (high = mid - 1).
        
        - If UNSAT (impossible): It means mid nodes are too few. 
          Need to aim higher (low = mid + 1).
          
        Finally, returns the set of nodes (min_cover) corresponding to the smallest solution found.
        """
        # Upper bound: All nodes (trivially a valid cover)
        # Lower bound: 0
        low = 0
        high = self.n
        min_cover = None
        
        # 1. Base Formula (Edges must always be covered)
        base_formula = self._encode_vertex_cover_constraint()
        
        # Optimization: If the graph has no nodes, return empty
        if self.n == 0:
            return set()
            
        print(f"SAT Solver: Searching for exact MVC for graph with {self.n} nodes. Constraints: {self.graph.number_of_edges()} edges.")
        
        # Binary Search
        while low <= high:
            mid = (low + high) // 2
            print(f"  Testing k={mid}...", end='', flush=True)
            
            # Construct the formula: (Edge Coverage) AND (Node Sum <= mid)
            cardinality_formula = self.get_at_most_k_formula(mid)
            full_formula = And(base_formula, cardinality_formula)
            
            # Call SymPy logic solver
            model = satisfiable(full_formula)
            
            if model:
                print(" SAT (Possible)")
                # Found a cover of size 'mid'.
                # Save nodes found in the model.
                current_cover = set()
                for node in self.nodes:
                    sym = self.node_vars[node]
                    if model.get(sym):
                        current_cover.add(node)
                
                min_cover = current_cover
                # Try to see if a smaller solution exists
                high = mid - 1
            else:
                print(" UNSAT (Impossible)")
                # 'mid' nodes are not enough, more are needed
                low = mid + 1
        
        return min_cover

def Equivalent(a, b):
    # Helper Function: Implements logical equivalence (A <-> B)
    # SymPy might not have Equivalent directly exposed simply for logic.
    # A <-> B is equivalent to (A -> B) AND (B -> A), or (A AND B) OR (NOT A AND NOT B).
    return And(Or(Not(a), b), Or(Not(b), a))

if __name__ == "__main__":
    # Simple Test
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    # This is a square with a diagonal (1-3).
    # Nodes: 1, 2, 3, 4.
    # A Minimum Vertex Cover should be of size 2. 
    # Example: {1, 3} covers (1,2), (1,4), (1,3), (2,3), (3,4). Correct.
    
    solver = SatVertexCover(g)
    result = solver.solve()
    print("MVC Found:", result)
