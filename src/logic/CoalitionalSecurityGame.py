import random
import networkx as nx

class CoalitionalSecurityGame:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        
        # We precalculate the degree (number of neighbors needed) for each node
        self.degrees = {n: len(list(graph.neighbors(n))) for n in self.nodes}
        self.neighbors_map = {n: list(graph.neighbors(n)) for n in self.nodes}

    def calculate_shapley_monte_carlo(self, num_permutations=1000):
        """
        Shapley Values​ ​for the ALL Rule (Strict Cooperative Game).

        Marginal Contribution Logic with the ALL rule:
        When I add node U to coalition S, U generates a value (+1) if:
            1. U becomes safe (because it enters S).
            2. A neighbor V becomes safe THANKS to U.
                - V is not in S.
                - V needed ALL neighbors in S.
                - U was the last neighbor missing to complete V's protection.
        """
        shapley_values = {node: 0.0 for node in self.nodes}
        
        for _ in range(num_permutations):
            perm = list(self.nodes)
            random.shuffle(perm)
            
            # Simulation state for this permutation
            nodes_in_S = set()           # Nodes entered into the coalition
            is_secured = {n: False for n in self.nodes} # Who is secure?
            
            # Counter: how many neighbors of V are currently in S?
            current_neighbors_in_S = {n: 0 for n in self.nodes}
            
            for node in perm:
                marginal_contribution = 0
                
                # 1. Direct Contribution: The node itself becomes secure by entering S
                # (If it wasn't already, but with the ALL rule it is impossible to be secure from outside 
                # unless the permutation has already inserted all neighbors. 
                # Checking for rigor).
                if not is_secured[node]:
                    is_secured[node] = True
                    marginal_contribution += 1
                
                # Add the node to the coalition
                nodes_in_S.add(node)
                
                # 2. Indirect Contribution: Do I help neighbors complete their coverage?
                for neighbor in self.neighbors_map[node]:
                    # Signal to the neighbor that 'node' has arrived
                    current_neighbors_in_S[neighbor] += 1
                    
                    # CRUCIAL Check ALL Rule:
                    # Does the neighbor 'neighbor' become secure NOW thanks to me?
                    # Conditions:
                    # A. Must not be already secure (e.g. must not be in S).
                    # B. Must have reached the full quota of neighbors (ALL present).
                    
                    if not is_secured[neighbor]:
                        # Have I completed its list of neighbors?
                        if current_neighbors_in_S[neighbor] == self.degrees[neighbor]:
                            # I was the last missing piece.
                            # The neighbor is now covered by its neighbors.
                            is_secured[neighbor] = True
                            marginal_contribution += 1
                            
                # Register the contribution
                shapley_values[node] += marginal_contribution
                
                # Early stop if all are secure
                if len(nodes_in_S) == self.num_nodes: 
                    # With ALL rule often the whole set is needed, so this break triggers late
                    break
                    
        # Average
        for node in shapley_values:
            shapley_values[node] /= num_permutations
            
        return shapley_values

    def build_security_set_from_shapley(self, shapley_values):
        """
        "Reverse Greedy" construction guided by Shapley.
        Instead of adding the best ones (which fails with ALL rule),
        we start from the complete set and remove the worst ones (low Shapley).
        """
        # 1. Start from the TOTAL set (All nodes on)
        security_set = set(self.nodes)
        
        # 2. Sort nodes by Shapley ASCENDING (from most "useless" to most "precious")
        sorted_nodes_ascending = sorted(shapley_values, key=shapley_values.get)
        
        
        # 3. Removal Cycle (Smart Pruning)
        for node in sorted_nodes_ascending:
            neighbors = self.neighbors_map[node]

            if not neighbors:
                continue 
            
            can_be_removed = True
            
            for neighbor in neighbors:
                if neighbor not in security_set:
                    can_be_removed = False
                    break
            
            if can_be_removed:                
                security_set.remove(node)
                
        return list(security_set)
