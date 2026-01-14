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
            Scenario a capacità limitata risolto con OTTIMO GLOBALE.
            Usa l'algoritmo Min-Cost Max-Flow (Network Flow).
            """
            # Creiamo un grafo diretto
            start_time_optimal = time()
            G = nx.DiGraph()
            source_node = 'SOURCE'
            sink_node = 'SINK'
            
            # 1. Aggiungiamo nodi e archi dalla SORGENTE agli ACQUIRENTI
            # Capacità 1 (ogni acquirente compra max 1 servizio)
            # Costo 0
            for buyer in self.buyers:
                b_node = f"buyer_{buyer['id']}"
                G.add_edge(source_node, b_node, capacity=1, weight=0)
                
            # 2. Aggiungiamo nodi e archi dai VENDITORI al POZZO (SINK)
            # Capacità = Disponibilità del venditore (Capacity)
            # Costo 0
            for vendor in self.vendors:
                v_node = f"vendor_{vendor['id']}"
                G.add_edge(v_node, sink_node, capacity=vendor['capacity'], weight=0)
                
            # 3. Aggiungiamo gli archi di MATCHING (Acquirente -> Venditore)
            # Capacità 1
            # Peso (Weight) = -UTILITÀ (Negativo perché l'algo minimizza il costo)
            for buyer in self.buyers:
                for vendor in self.vendors:
                    util = self.calculate_utility(buyer, vendor)
                    
                    # Aggiungiamo l'arco solo se l'acquisto è possibile (budget >= price)
                    if util > -float('inf'):
                        b_node = f"buyer_{buyer['id']}"
                        v_node = f"vendor_{vendor['id']}"
                        
                        # Moltiplichiamo per -100 o simile se vogliamo evitare problemi con float,
                        # ma networkx gestisce bene anche i float. Qui usiamo l'opposto esatto.
                        G.add_edge(b_node, v_node, capacity=1, weight=-util)

            # 4. Eseguiamo l'algoritmo Min-Cost Max-Flow
            # Questo trova la configurazione che sposta il massimo numero di utenti
            # al minor costo possibile (cioè alla massima utilità).
            try:
                flow_dict = nx.max_flow_min_cost(G, source_node, sink_node, weight='weight')
            except nx.NetworkXUnfeasible:
                print("Nessuna soluzione fattibile trovata.")
                return [], 0

            # 5. Ricostruiamo i risultati dal flusso calcolato
            matches = []
            total_welfare = 0
            matched_buyer_ids = set()

            # Analizziamo il flusso uscente dai nodi Buyer
            for buyer in self.buyers:
                b_node = f"buyer_{buyer['id']}"
                
                if b_node in flow_dict:
                    # Vediamo verso quale venditore è andato il flusso (se c'è)
                    for potential_vendor, flow_amount in flow_dict[b_node].items():
                        if flow_amount > 0 and potential_vendor != source_node:
                            # Abbiamo trovato un match!
                            # Estrarre l'ID numerico del venditore dalla stringa "vendor_X"
                            v_id = int(potential_vendor.split('_')[1])
                            
                            # Ricalcoliamo l'utilità originale (positiva)
                            # Nota: dobbiamo recuperare l'oggetto vendor originale
                            original_vendor = next(v for v in self.vendors if v['id'] == v_id)
                            real_util = self.calculate_utility(buyer, original_vendor)
                            
                            matches.append((buyer['id'], v_id, real_util))
                            total_welfare += real_util
                            matched_buyer_ids.add(buyer['id'])
                            break # Un buyer ha max 1 match

            # Aggiungiamo gli acquirenti rimasti senza match
            for buyer in self.buyers:
                if buyer['id'] not in matched_buyer_ids:
                    matches.append((buyer['id'], None, 0))

            end_time_optimal = time()

            print("Total welfare (limited, optimal):", total_welfare)
            print("Time taken (limited capacity, optimal):", end_time_optimal - start_time_optimal, "seconds")

            return matches, total_welfare