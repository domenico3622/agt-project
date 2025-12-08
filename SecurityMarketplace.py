import random

class SecurityMarketplace:
    def __init__(self, buyers_nodes, num_vendors=5):
        """
        Buyers: Nodi del security set con budget.
        Vendors: Venditori con prezzo e qualità.
        """
        self.buyers = [{'id': n, 'budget': random.randint(1, 100)} for n in buyers_nodes]
        self.vendors = [{'id': v, 
                         'price': random.randint(1, 100), 
                         'security_level': random.randint(1, 10),
                         'capacity': random.randint(20, 150)} for v in range(num_vendors)]

    def calculate_utility(self, buyer, vendor):
        """Calcola Utilità: Welfare = (Sicurezza * 10) + Risparmio """
        if buyer['budget'] < vendor['price']:
            return -float('inf') # Incompatibile
        return (vendor['security_level'] * 10) + (buyer['budget'] - vendor['price'])

    def run_scenario_infinite_capacity(self):
        """Scenario con capacità infinita: ogni buyer può matchare con il vendor che massimizza la sua utilità"""
        matches = []
        total_welfare = 0
        
        for buyer in self.buyers:
            best_vendor = None
            best_utility = -float('inf')
            
            # Trova il vendor che massimizza l'utilità per questo buyer
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
        """Scenario con capacità limitata (Greedy Global Maximization) """
        possible_matches = []
        for buyer in self.buyers:
            for vendor in self.vendors:
                util = self.calculate_utility(buyer, vendor)
                if util > -float('inf'):
                    possible_matches.append({'buyer': buyer, 'vendor': vendor, 'util': util})
        
        # Ordiniamo per utilità per massimizzare il welfare sociale 
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
        
        # Aggiungiamo i non matchati
        for buyer in self.buyers:
            if buyer['id'] not in matched_buyers:
                matches.append((buyer['id'], None, 0))
                
        return matches, total_welfare
