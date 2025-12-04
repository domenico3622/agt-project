import random
import networkx as nx

class CoalitionalSecurityGame:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        
        # Pre-calcoliamo il grado (numero di vicini necessari) per ogni nodo
        self.degrees = {n: len(list(graph.neighbors(n))) for n in self.nodes}
        self.neighbors_map = {n: list(graph.neighbors(n)) for n in self.nodes}

    def calculate_shapley_monte_carlo(self, num_permutations=1000):
        """
        Shapley Values per Regola ALL (Strict Cooperative Game).
        
        Logica del Contributo Marginale con regola ALL:
        Quando aggiungo il nodo U alla coalizione S, U genera valore (+1) se:
        1. U diventa sicuro (perché entra in S).
        2. Un vicino V diventa sicuro GRAZIE a U.
           - V non è in S.
           - V aveva bisogno di TUTTI i vicini in S.
           - U era l'ultimo vicino mancante per completare la protezione di V.
        """
        shapley_values = {node: 0.0 for node in self.nodes}
        
        for _ in range(num_permutations):
            perm = list(self.nodes)
            random.shuffle(perm)
            
            # Stato della simulazione per questa permutazione
            nodes_in_S = set()           # Nodi entrati nella coalizione
            is_secured = {n: False for n in self.nodes} # Chi è sicuro?
            
            # Contatore: quanti vicini di V sono attualmente in S?
            current_neighbors_in_S = {n: 0 for n in self.nodes}
            
            for node in perm:
                marginal_contribution = 0
                
                # 1. Contributo Diretto: Il nodo stesso diventa sicuro entrando in S
                # (Se non lo era già, ma con regola ALL è impossibile essere sicuri da fuori 
                # a meno che la permutazione non abbia già inserito tutti i vicini. 
                # Controlliamo per rigore).
                if not is_secured[node]:
                    is_secured[node] = True
                    marginal_contribution += 1
                
                # Aggiungiamo il nodo alla coalizione
                nodes_in_S.add(node)
                
                # 2. Contributo Indiretto: Aiuto i vicini a completare la loro copertura?
                for neighbor in self.neighbors_map[node]:
                    # Segnaliamo al vicino che 'node' è arrivato
                    current_neighbors_in_S[neighbor] += 1
                    
                    # Controllo CRUCIALE Regola ALL:
                    # Il vicino 'neighbor' diventa sicuro ORA grazie a me?
                    # Condizioni:
                    # A. Non deve essere già sicuro (es. non deve essere in S).
                    # B. Deve aver raggiunto la quota piena di vicini (TUTTI presenti).
                    
                    if not is_secured[neighbor]:
                        # Ho completato la sua lista di vicini?
                        if current_neighbors_in_S[neighbor] == self.degrees[neighbor]:
                            # BINGO! Ero l'ultimo pezzo mancante.
                            # Il vicino ora è coperto dai suoi vicini.
                            is_secured[neighbor] = True
                            marginal_contribution += 1
                            
                # Registro il contributo
                shapley_values[node] += marginal_contribution
                
                # Stop anticipato se tutti sicuri
                if len(nodes_in_S) == self.num_nodes: 
                    # Nota: con regola ALL spesso serve tutto il set, quindi questo break scatta tardi
                    break
                    
        # Media
        for node in shapley_values:
            shapley_values[node] /= num_permutations
            
        return shapley_values

    def build_security_set_from_shapley(self, shapley_values):
        """
        Costruzione "Reverse Greedy" guidata da Shapley.
        Invece di aggiungere i migliori (che fallisce con regola ALL),
        partiamo dal set completo e rimuoviamo i peggiori (basso Shapley).
        """
        # 1. Partiamo dal set TOTALE (Tutti i nodi accesi)
        security_set = set(self.nodes)
        
        # 2. Ordiniamo i nodi per Shapley CRESCENTE (dal più "inutile" al più "prezioso")
        sorted_nodes_ascending = sorted(shapley_values, key=shapley_values.get)
        
        
        # 3. Ciclo di Rimozione (Pruning Intelligente)
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
