import networkx as nx
import random

class VCGPathAuction:
    def __init__(self, graph, security_set, penalty_weight=10):
        """
        VCG Auction dove i NODI sono gli agenti.
        
        graph: Grafo della rete.
        security_set: I nodi sicuri.
        penalty_weight: Costo aggiuntivo 'sociale' per passare da un nodo insicuro.
        """
        self.graph = graph
        self.security_set = set(security_set)
        self.alpha = penalty_weight
        self.node_bids = {}
        
        # Generazione dei costi privati (bids) per ogni NODO
        # Ogni nodo ha un costo per processare/inoltrare il traffico
        for node in graph.nodes():
            true_cost = random.randint(1, 20)
            self.node_bids[node] = true_cost

    def get_transit_weight(self, target_node):
        """
        Calcola il costo di attraversamento di un nodo specifico.
        Peso = Bid del Nodo + Penalità (se il nodo non è sicuro).
        """
        bid = self.node_bids.get(target_node, float('inf'))
        is_secure = target_node in self.security_set
        security_cost = 0 if is_secure else self.alpha
        return bid + security_cost

    def run_vcg_mechanism(self, source, target):
        """
        Esegue il meccanismo VCG Node-Based.
        """
        # 1. Costruzione grafo pesato DIRETTO
        # In un grafo pesato sui nodi, il peso dell'arco (u, v) è il costo di attraversare v.
        temp_G = nx.DiGraph()
        
        for u, v in self.graph.edges():
            # Costo per andare verso v (pago v)
            weight_to_v = self.get_transit_weight(v)
            temp_G.add_edge(u, v, weight=weight_to_v)
            
            # Costo per andare verso u (pago u)
            weight_to_u = self.get_transit_weight(u)
            temp_G.add_edge(v, u, weight=weight_to_u)
            
        try:
            optimal_path = nx.shortest_path(temp_G, source, target, weight='weight')
            optimal_cost = nx.shortest_path_length(temp_G, source, target, weight='weight')
        except nx.NetworkXNoPath:
            return None, None, None
        
        # 2. Calcolo Pagamenti VCG per i nodi vincitori
        payments = {}
        
        # I vincitori sono i nodi intermedi + il target.
        # La source non si paga da sola per iniziare il pacchetto.
        winning_nodes = [n for n in optimal_path if n != source and n != target]
        
        for node in winning_nodes:
            my_bid = self.node_bids[node]
            my_total_weight = self.get_transit_weight(node)
            
            # Costo totale del percorso ottimo SENZA il mio peso specifico
            # (Nota: sottraiamo tutto il peso che ho aggiunto al sistema, inclusa la penalità,
            # perché il VCG calcola il 'danno' sociale della mia assenza)
            cost_with_me_excluded = optimal_cost - my_total_weight
            
            # Calcolo percorso alternativo rimuovendo IL NODO
            temp_G_minus_n = temp_G.copy()
            temp_G_minus_n.remove_node(node)
            
            try:
                # Qual è il miglior percorso se il nodo 'node' scioperasse/morisse?
                cost_without_me = nx.shortest_path_length(temp_G_minus_n, source, target, weight='weight')
                
                # Formula VCG:
                # Pagamento = (Costo Sociale senza di me) - (Costo Sociale con me, escluso il mio peso)
                vcg_payment = cost_without_me - cost_with_me_excluded
                
                payments[node] = {'bid': my_bid, 'payment': vcg_payment}
                
            except nx.NetworkXNoPath:
                # Se rimuovendo il nodo non c'è più percorso, il nodo è un "Ponte" (Bridge)
                # Ha un potere di monopolio infinito.
                payments[node] = {'bid': my_bid, 'payment': float('inf')}
                
        return optimal_path, optimal_cost, payments
