import random
import networkx as nx
from time import time

class SecurityMarketplace:
    def __init__(self, buyers_nodes, num_vendors=8):
        """
        Buyers: Security set nodes with budget.
        Vendors: Sellers with price and quality.
        """
        self.buyers = [{'id': n, 'budget': random.randint(1, 100)} for n in buyers_nodes]
        self.vendors = [{'id': v, 
                         'price': random.randint(1, 100), 
                         'security_level': random.randint(1, 10),
                         'capacity': random.randint(25, 100)} for v in range(num_vendors)]

    def calculate_utility(self, buyer, vendor):
        """Calculate Utility: Welfare = (Security * 10) + Savings """
        if buyer['budget'] < vendor['price']:
            return -float('inf') # Incompatible
        return (vendor['security_level'] * 10) + (buyer['budget'] - vendor['price'])

    def run_scenario_infinite_capacity(self):
        """Infinite capacity scenario: each buyer can match with the vendor maximizing their utility"""
        matches = []
        total_welfare = 0
        
        for buyer in self.buyers:
            best_vendor = None
            best_utility = -float('inf')
            
            # Find the vendor maximizing utility for this buyer
            for vendor in self.vendors:
                util = self.calculate_utility(buyer, vendor)
                if util > best_utility:
                    best_utility = util
                    best_vendor = vendor
            
            if best_vendor is not None and best_utility >= -float('inf'):
                matches.append((buyer['id'], best_vendor['id'], best_utility))
                total_welfare += best_utility
            else:
                matches.append((buyer['id'], None, 0))
        
        return matches, total_welfare
    
    def run_scenario_limited_capacity(self):
        """Limited capacity scenario (Greedy Global Maximization) """
        start_time_greedy = time()
        possible_matches = []
        for buyer in self.buyers:
            for vendor in self.vendors:
                util = self.calculate_utility(buyer, vendor)
                if util > -float('inf'):
                    possible_matches.append({'buyer': buyer, 'vendor': vendor, 'util': util})
        
        # Sort by utility to maximize social welfare 
        possible_matches.sort(key=lambda x: x['util'], reverse=True)
        
        matches = []
        total_welfare = 0
        matched_buyers = set()
        vendor_sales = {v['id']: 0 for v in self.vendors}
        
        for m in possible_matches:
            b_id = m['buyer']['id']
            v_id = m['vendor']['id']

            max_items = m['vendor']['capacity']
            
            if b_id not in matched_buyers and vendor_sales[v_id] < max_items:
                matched_buyers.add(b_id)
                vendor_sales[v_id] += 1
                total_welfare += m['util']
                matches.append((b_id, v_id, m['util']))
        
        # Add unmatched
        for buyer in self.buyers:
            if buyer['id'] not in matched_buyers:
                matches.append((buyer['id'], None, 0))

        end_time_greedy = time()
        print("Total welfare (limited, greedy):", total_welfare)
        print("Time taken (limited capacity, greedy):", end_time_greedy - start_time_greedy, "seconds")
        return matches, total_welfare
    
    def run_scenario_optimal_capacity(self):
            """
            Limited capacity scenario solved with GLOBAL OPTIMIZATION.
            Uses the Min-Cost Max-Flow algorithm (Network Flow).
            """
            # Create a directed graph
            start_time_optimal = time()
            G = nx.DiGraph()
            source_node = 'SOURCE'
            sink_node = 'SINK'
            
            # 1. Add nodes and edges from SOURCE to BUYERS
            # Capacity 1 (each buyer purchases max 1 service)
            # Cost 0
            for buyer in self.buyers:
                b_node = f"buyer_{buyer['id']}"
                G.add_edge(source_node, b_node, capacity=1, weight=0)
                
            # 2. Add nodes and edges from VENDORS to SINK
            # Capacity = Vendor availability (Capacity)
            # Cost 0
            for vendor in self.vendors:
                v_node = f"vendor_{vendor['id']}"
                G.add_edge(v_node, sink_node, capacity=vendor['capacity'], weight=0)
                
            # 3. Add MATCHING edges (Buyer -> Vendor)
            # Capacity 1
            # Weight = -UTILITY (Negative because the algorithm minimizes cost)
            for buyer in self.buyers:
                for vendor in self.vendors:
                    util = self.calculate_utility(buyer, vendor)
                    
                    # Add edge only if purchase is possible (budget >= price)
                    if util > -float('inf'):
                        b_node = f"buyer_{buyer['id']}"
                        v_node = f"vendor_{vendor['id']}"
                        
                        # We could multiply by -100 or similar to avoid float issues,
                        # but networkx handles floats well. Here we use the exact opposite.
                        G.add_edge(b_node, v_node, capacity=1, weight=-util)

            # 4. Run the Min-Cost Max-Flow algorithm
            # This finds the configuration that moves the maximum number of users
            # at the lowest possible cost (i.e., maximum utility).
            try:
                flow_dict = nx.max_flow_min_cost(G, source_node, sink_node, weight='weight')
            except nx.NetworkXUnfeasible:
                print("No feasible solution found.")
                return [], 0

            # 5. Reconstruct results from calculated flow
            matches = []
            total_welfare = 0
            matched_buyer_ids = set()

            # Analyze the outgoing flow from Buyer nodes
            for buyer in self.buyers:
                b_node = f"buyer_{buyer['id']}"
                
                if b_node in flow_dict:
                    # Check which vendor the flow went to (if any)
                    for potential_vendor, flow_amount in flow_dict[b_node].items():
                        if flow_amount > 0 and potential_vendor != source_node:
                            # Found a match!
                            # Extract numerical ID of the vendor from string "vendor_X"
                            v_id = int(potential_vendor.split('_')[1])
                            
                            # Recalculate original utility (positive)
                            # Note: we must retrieve the original vendor object
                            original_vendor = next(v for v in self.vendors if v['id'] == v_id)
                            real_util = self.calculate_utility(buyer, original_vendor)
                            
                            matches.append((buyer['id'], v_id, real_util))
                            total_welfare += real_util
                            matched_buyer_ids.add(buyer['id'])
                            break # A buyer has max 1 match

            # Add unmatched buyers
            for buyer in self.buyers:
                if buyer['id'] not in matched_buyer_ids:
                    matches.append((buyer['id'], None, 0))

            end_time_optimal = time()

            print("Total welfare (limited, optimal):", total_welfare)
            print("Time taken (limited capacity, optimal):", end_time_optimal - start_time_optimal, "seconds")

            return matches, total_welfare