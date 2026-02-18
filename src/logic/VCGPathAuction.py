import networkx as nx
import random

class VCGPathAuction:
    def __init__(self, graph, security_set, penalty_weight=10):
        """
        VCG Auction where NODES are agents.
        
        graph: Network graph.
        security_set: Safe nodes.
        penalty_weight: Additional 'social' cost to pass through an insecure node.
        """
        self.graph = graph
        self.security_set = set(security_set)
        self.alpha = penalty_weight
        self.node_bids = {}
        
        # Private cost generation (bids) for each NODE
        # Each node has a cost to process/forward traffic
        for node in graph.nodes():
            true_cost = random.randint(1, 20)
            self.node_bids[node] = true_cost

    def get_transit_weight(self, target_node):
        """
        Calculate the traversal cost of a specific node.
        Weight = Node Bid + Penalty (if node is not secure).
        """
        bid = self.node_bids.get(target_node, float('inf'))
        is_secure = target_node in self.security_set
        security_cost = 0 if is_secure else self.alpha
        return bid + security_cost

    def run_vcg_mechanism(self, source, target):
        """
        Executes Node-Based VCG mechanism.
        """
        # 1. Construction of DIRECTED weighted graph
        # In a node-weighted graph, edge weight (u, v) is the cost to traverse v.
        temp_G = nx.DiGraph()
        
        for u, v in self.graph.edges():
            # Cost to go to v (pay v)
            weight_to_v = self.get_transit_weight(v)
            temp_G.add_edge(u, v, weight=weight_to_v)
            
            # Cost to go to u (pay u)
            weight_to_u = self.get_transit_weight(u)
            temp_G.add_edge(v, u, weight=weight_to_u)
            
        try:
            optimal_path = nx.shortest_path(temp_G, source, target, weight='weight')
            optimal_cost = nx.shortest_path_length(temp_G, source, target, weight='weight')
        except nx.NetworkXNoPath:
            return None, None, None
        
        # 2. Calculate VCG Payments for winning nodes
        payments = {}
        
        # Winners are intermediate nodes + target.
        # Source does not pay itself to start the packet.
        winning_nodes = [n for n in optimal_path if n != source and n != target]
        
        for node in winning_nodes:
            my_bid = self.node_bids[node]
            my_total_weight = self.get_transit_weight(node)
            
            # Total cost of optimal path WITHOUT my specific weight
            # (We subtract all weight I added to system, including penalty,
            # because VCG calculates the social 'damage' of my absence)
            cost_with_me_excluded = optimal_cost - my_total_weight
            
            # Calculate alternative path removing THE NODE
            temp_G_minus_n = temp_G.copy()
            temp_G_minus_n.remove_node(node)
            
            try:
                # What is the best path if 'node' went on strike/died?
                cost_without_me = nx.shortest_path_length(temp_G_minus_n, source, target, weight='weight')
                
                # VCG Formula:
                # Payment = (Social Cost without me) - (Social Cost with me, excluding my weight)
                vcg_payment = cost_without_me - cost_with_me_excluded
                
                payments[node] = {'bid': my_bid, 'payment': vcg_payment}
                
            except nx.NetworkXNoPath:
                # If removing the node there is no path, the node is a "Bridge"
                # It has infinite monopoly power.
                payments[node] = {'bid': my_bid, 'payment': float('inf')}
                
        return optimal_path, optimal_cost, payments
