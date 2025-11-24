import networkx as nx
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


# Variabile centralizzata per il numero massimo di iterazioni
MAX_ITERATIONS = 100
# Variabile centralizzata per il numero di nodi nei grafi
NUM_NODES = 1000

# =============================================================================
# TASK 1: STRATEGIC GAME (Non-Cooperative)
# Modellazione: Giochi Strategici, Nash Equilibrium, Best Response, Fictitious Play
# =============================================================================


class NetworkSecurityGame:
    def __init__(self, graph, alpha=10, c=9):
        """
        Inizializza il gioco strategico.
        Ogni nodo è un giocatore che sceglie se proteggersi (1) o no (0).
        """ 
        assert c > 0 and alpha >= 0

        self.graph = graph
        self.nodes = list(graph.nodes())
        self.alpha = alpha
        self.c = c
        self.num_players = graph.number_of_nodes()
        self.strategies = {node: 0 for node in self.nodes} # 0 non protetto, 1 protetto
        self.node_tolerances = {n: random.uniform(-0.01, 0.01) for n in self.nodes}

    def get_payoff(self, player_id, strategies):
        """
        Calculates the payoff for a given player based on the current strategy profile.
        strategies: a dictionary mapping player_id to their chosen strategy (0 or 1).
        """
        s_i = strategies[player_id]

        # Check if the player is covered
        is_covered = False
        if s_i == 1:
            # Player secures themselves
            is_covered = True
        else:
            # Player is covered only if ALL of its neighbors have played 1.
            neighbors = list(self.graph.neighbors(player_id))
            if not neighbors:
                # No neighbors -> cannot be covered by neighbors
                is_covered = False
            else:
                is_covered = all(strategies.get(neighbor) == 1 for neighbor in neighbors)
        
        # Determine payoff based on strategy and coverage
        if s_i == 1:
            return self.alpha - self.c + self.node_tolerances[player_id]
        elif s_i == 0 and is_covered:
            return self.alpha
        elif s_i == 0 and not is_covered:
            return 0
        else:
            raise ValueError("Invalid strategy or state encountered.")

    def get_all_payoffs(self, strategies):
        """
        Calculates payoffs for all players.
        """
        all_payoffs = {}
        for player_id in range(self.num_players):
            all_payoffs[player_id] = self.get_payoff(player_id, strategies)
        return all_payoffs
    
    def is_nash_equilibrium(self, strategies):
        """
        Checks if the given strategy profile is a Pure Nash Equilibrium.
        """
        for player_id in range(self.num_players):
            original_payoff = self.get_payoff(player_id, strategies)

            # Try changing player_id's strategy
            alt_strategies = strategies.copy()
            alt_strategies[player_id] = 1 - strategies[player_id] # Flip strategy

            alt_payoff = self.get_payoff(player_id, alt_strategies)

            if alt_payoff > original_payoff:
                # Player_id has an incentive to deviate, so it's not a PNE
                return False
        return True

    def get_security_set(self):
        return [n for n, s in self.strategies.items() if s == 1]
        

    def best_response(self, player_id):
        """
        Massimizza il payoff per il nodo 'player_id' utilizzando la funzione get_payoff.
        """
        current_strategies = self.strategies.copy()
        
        # 1. Calcola payoff per s_i = 0
        current_strategies[player_id] = 0
        payoff_0 = self.get_payoff(player_id, current_strategies)
        
        # 2. Calcola payoff per s_i = 1
        current_strategies[player_id] = 1
        payoff_1 = self.get_payoff(player_id, current_strategies)
        
        # 3. Scegli la strategia con il payoff maggiore (con s=0 come tie-breaker)
        if payoff_0 >= payoff_1:
            return 0
        else:
            return 1

    def best_response_dynamics(self, max_iters=10000, random_seed=None):
        """
        BRD teoricamente corretto: converge sempre a un minimal network security set.
        """
        convergence_history = []
        if random_seed is not None:
            random.seed(random_seed)

        for it in range(max_iters):
            changes = 0
            nodes_order = list(range(self.num_players))
            random.shuffle(nodes_order)

            for node in nodes_order:
                old = self.strategies[node]
                # Usa il Best Response basato sul nuovo payoff
                new = self.best_response(node) 
                if new != old:
                    self.strategies[node] = new
                    changes += 1
            
            current_set_size = sum(self.strategies.values())
            convergence_history.append({'iteration': it, 'changes': changes, 'set_size': current_set_size})
            
            if changes == 0:
                break # Convergenza e ottimizzazione completate

        self.convergence_history = convergence_history
        pne = self.is_nash_equilibrium(self.strategies)
        return self.strategies, pne
    

    
    def get_utility(self, node, action, other_strategies):
        """
        Calcola l'utilità: U_i = - (Costo * az) - (Penalità * archi_scoperti)
        Rif: 
        """
        current_cost = self.c * action
        uncovered_edges = 0
        
        for neighbor in self.graph.neighbors(node):
            neighbor_action = other_strategies[neighbor]
            # Un arco è scoperto se nessuno dei due estremi lo protegge
            if action == 0 and neighbor_action == 0:
                uncovered_edges += 1
        
        penalty_cost = uncovered_edges * self.alpha
        return -(current_cost + penalty_cost)
    

    def fictitious_play(self, max_iters=100, decay=0.9): # Ora max_iters sono "Round", quindi bastano meno (es. 100-200)
        """
        Algoritmo True Fictitious Play Asincrono organizzato in Round (Epoche).
        Integra la logica della regola ALL e della Soglia di convenienza.
        """
        # INIZIALIZZAZIONE (Identica a prima)
        # Laplace Smoothing: {0: 1, 1: 1}
        # self.play_counts = {n: {0: 1, 1: 1} for n in self.nodes}
        self.play_counts = {n: {0: 0.01, 1: 0.01} for n in self.nodes}

        self.strategies = {n: random.choice([0, 1]) for n in self.nodes}
        
        convergence_history = []
        
        # Log stato iniziale
        initial_size = sum(self.strategies.values())
        convergence_history.append({'iteration': 0, 'set_size': initial_size})

        # --- MODIFICA: La pazienza ora si conta in Round ---
        # 5 Round stabili consecutivi sono sufficienti (equivalgono a 5*N aggiornamenti singoli)
        patience = 5 
        stable_rounds = 0
        
        # --- Pre-calcolo la soglia di convenienza ---
        # Se P(coperto) > threshold, allora EU(0) > EU(1) -> Conviene giocare 0
        # Formula derivata da: alpha * P > alpha - c
        threshold = (self.alpha - self.c) / self.alpha

        # --- CICLO ESTERNO (I Round, paragonabili alle iterazioni di BRD) ---
        for round_idx in range(1, max_iters + 1):
            strategies_start_of_round = self.strategies.copy()
            
            # --- CICLO INTERNO (Il volume di lavoro) ---
            # Eseguiamo N aggiornamenti asincroni per simulare un "giro" completo
            for _ in range(self.num_players):
                
                # 1. ASINCRONO: Seleziona UN nodo a caso
                node = random.choice(list(self.nodes))
                
                # 2. Calcolo Probabilità di essere Coperto (Regola ALL)
                # Calcoliamo la probabilità congiunta che TUTTI i vicini giochino 1
                prob_covered = 1.0
                neighbors = list(self.graph.neighbors(node))
                
                if not neighbors:
                    prob_covered = 0.0 # Senza vicini, impossibile essere coperti
                else:
                    for neighbor in neighbors:
                        # Recupera la storia del vicino
                        cnt = self.play_counts[neighbor]
                        total_plays = cnt[0] + cnt[1]
                        
                        # Probabilità empirica che QUESTO vicino giochi 1
                        prob_neighbor_is_1 = cnt[1] / total_plays
                        
                        # Regola ALL: Moltiplicazione diretta (Intersezione eventi indipendenti)
                        prob_covered *= prob_neighbor_is_1
                
                # 3. Decisione basata sulla Soglia
                # Invece di ricalcolare eu_0 e eu_1 ogni volta, usiamo la soglia pre-calcolata.
                # È matematicamente identico a: if (alpha * prob) > (alpha - c)
                
                if prob_covered > threshold:
                    new_strat = 0 # Mi sento abbastanza sicuro per fare Free Riding
                else:
                    new_strat = 1 # Troppo rischioso, mi proteggo
                
                # 4. Aggiornamento
                self.strategies[node] = new_strat
                # Aggiorniamo la memoria storica del nodo
                self.play_counts[node][new_strat] += 1

                #(Fading Memory):
                # 1. Moltiplica TUTTO lo storico per il decadimento (dimentica un po' il passato)
                self.play_counts[node][0] *= decay
                self.play_counts[node][1] *= decay
                
                # 2. Aggiungi l'osservazione corrente
                self.play_counts[node][new_strat] += 1
            
            # --- FINE DEL ROUND ---
            
            # Logging: Registriamo solo alla fine del round
            current_size = sum(self.strategies.values())
            # Manteniamo la tua logica di logging (frequente all'inizio, poi diradata)
            if round_idx <= 20 or round_idx % 20 == 0:
                convergence_history.append({'iteration': round_idx, 'set_size': current_size})
            
            # Check Stabilità
            if self.strategies == strategies_start_of_round:
                stable_rounds += 1
            else:
                stable_rounds = 0
            
            if stable_rounds >= patience:
                break

        self.convergence_history = convergence_history
        pne = self.is_nash_equilibrium(self.strategies)
        return self.strategies, pne
    
    def regret_matching(self, max_iters=MAX_ITERATIONS, gamma=0.95):
        """
        Algoritmo Regret Matching Puro.
        Adattato alle variabili esistenti (self.strategies, regrets).
        """
        # Inizializzazione Strategie Pure (casuale)
        self.strategies = {n: random.choice([0, 1]) for n in self.nodes}
        
        # Inizializzazione Rimpianti Cumulativi (Regrets)
        regrets = {n: {0: 0.0, 1: 0.0} for n in self.nodes}
        
        convergence_history = []
        
        # Log stato iniziale (Iterazione 0)
        initial_size = sum(self.strategies.values())
        convergence_history.append({'iteration': 0, 'set_size': initial_size})

        nodes_order = list(self.nodes)
        random.shuffle(nodes_order)
        
        for iteration in range(1, max_iters + 1):
            # Creiamo una copia "congelata" delle strategie per calcolare i payoff simultanei
            current_strategies_snapshot = self.strategies.copy()
            
            # 1. CALCOLO RIMPIANTI (Fase di Apprendimento)
            for node in nodes_order:
                action_taken = current_strategies_snapshot[node]
                
                # Utilità ottenuta realmente
                util_taken = self.get_utility(node, action_taken, current_strategies_snapshot)
                
                # Utilità controfattuale (Se avessi giocato l'altra mossa?)
                other_action = 1 - action_taken
                
                # Creiamo uno scenario ipotetico dove SOLO 'node' cambia strategia
                # Nota: get_utility di solito accetta il dizionario completo, 
                # dobbiamo simulare il cambio
                hypothetical_strategies = current_strategies_snapshot.copy()
                hypothetical_strategies[node] = other_action
                util_other = self.get_utility(node, other_action, hypothetical_strategies)
                
                # Calcolo del Rimpianto (Regret)
                # "Quanto avrei guadagnato in più se avessi giocato l'altra?"
                # Non usiamo bonus (+0.2), ci fidiamo dei valori di alpha e c.
                regret = util_other - util_taken

                # Applica il decadimento ai rimpianti accumulati
                regrets[node][0] *= gamma
                regrets[node][1] *= gamma
                
                # Accumuliamo solo i rimpianti positivi
                # (Se util_other < util_taken, non ho rimpianti, ho fatto bene)
                if regret > 0:
                    regrets[node][other_action] += regret
            
            # 2. AGGIORNAMENTO STRATEGIE (Fase di Azione)
            # Scegliamo la mossa per il PROSSIMO turno in base ai rimpianti accumulati
            for node in self.nodes:
                # Consideriamo solo la parte positiva dei rimpianti cumulativi
                r_0 = max(0, regrets[node][0])
                r_1 = max(0, regrets[node][1])
                sum_r = r_0 + r_1
                
                if sum_r > 0:
                    # Probabilità proporzionale al rimpianto
                    prob_1 = r_1 / sum_r
                else:
                    # Se non ho rimpianti (o sono tutti negativi), 
                    # torno a una distribuzione uniforme per esplorare
                    prob_1 = 0.5
                
                # Campionamento della nuova strategia pura per il prossimo turno
                self.strategies[node] = 1 if random.random() < prob_1 else 0
            
            # Logging (Prime 20 iterazioni sempre, poi ogni 10)
            if iteration <= 20 or iteration % 10 == 0:
                convergence_history.append({'iteration': iteration, 'set_size': sum(self.strategies.values())})
        
        
        # Aggiunta log finale
        final_size = sum(self.strategies.values())
        if convergence_history[-1]['set_size'] != final_size:
            convergence_history.append({'iteration': max_iters, 'set_size': final_size})

        self.convergence_history = convergence_history
        pne = self.is_nash_equilibrium(self.strategies)
        return self.strategies, pne


# =============================================================================
# TASK 2: COALITIONAL GAME (Cooperative)
# Modellazione: Shapley Values, Giochi Cooperativi
# =============================================================================
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


# =============================================================================
# TASK 3: MARKET SIMULATION (Matching)
# Modellazione: Matching Compravendita, Welfare Maximization
# =============================================================================

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


# =============================================================================
# TASK 4: MECHANISM DESIGN (VCG Auction)
# Modellazione: Aste Veritiere, Shortest Path con Penalità
# =============================================================================

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


# =============================================================================
# VALIDATION & VERIFICATION FUNCTIONS
# =============================================================================

def is_valid_security_set(graph, security_set):
    """
    Checks if 'security_set' is a dominating set for the graph based on the 
    specific strict logic provided: ALL neighbors must be in the set to cover a node.
    """
    if not security_set: 
        return graph.number_of_nodes() == 0

    # Ensure all nodes in the security_set are actually in the graph
    if not all(node in graph.nodes for node in security_set):
        print("Warning: Security set contains nodes not in the graph.")
        return False
        
    for node in graph.nodes:
        if node in security_set:
            continue # Node is in the set, so it's covered

        neighbors = list(graph.neighbors(node))
        if not neighbors:
            # isolated node and not in security_set -> not covered
            return False

        # STRICT LOGIC FROM USER PROMPT:
        # Node is covered only if ALL its neighbors are in the security set
        if not all(neighbor in security_set for neighbor in neighbors):
            return False
    return True

def is_minimal_security_set_fast(graph, security_set):
    """
    Checks if 'security_set' is a minimal dominating set.
    """
    if not is_valid_security_set(graph, security_set):
        return False
    
    # Convert to set if it's a list
    security_set_as_set = set(security_set)
    
    for node_to_remove in security_set_as_set:
        temp_set = security_set_as_set - {node_to_remove}
        if is_valid_security_set(graph, temp_set):
            return False
            
    return True

def greedy_minimal_vertex_cover(graph):
    """Algoritmo greedy per trovare un vertex cover approssimato"""
    uncovered_edges = set(graph.edges())
    cover = set()
    
    # Greedy: prendi il nodo che copre più archi non coperti
    while uncovered_edges:
        # Conta quanti archi scoperti copre ogni nodo
        node_coverage = {}
        for u, v in uncovered_edges:
            node_coverage[u] = node_coverage.get(u, 0) + 1
            node_coverage[v] = node_coverage.get(v, 0) + 1
        
        # Scegli il nodo con massima copertura
        best_node = max(node_coverage, key=node_coverage.get)
        cover.add(best_node)
        
        # Rimuovi gli archi coperti da questo nodo
        uncovered_edges = {(u, v) for u, v in uncovered_edges if u != best_node and v != best_node}
    
    return list(cover)

def analyze_security_set_quality(graph, security_set, algorithm_name):
    """Analizza la qualità di un security set"""
    results = {
        'algorithm': algorithm_name,
        'size': len(security_set),
        'is_valid': is_valid_security_set(graph, security_set),
        'is_minimal': is_minimal_security_set_fast(graph, security_set),
        'nodes': sorted(security_set)
    }
    return results

# =============================================================================
# MAIN ORCHESTRATOR & VISUALIZATION
# =============================================================================

def create_output_directory():
    """Crea la cartella output con timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/graphs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    return output_dir

def generate_networks():
    """Genera i 3 tipi di reti richiesti dal progetto """
    if NUM_NODES < 101:
        return {
            'Regular-Like': nx.random_regular_graph(d=5, n=NUM_NODES, seed=42),
            'Erdos-Renyi': nx.erdos_renyi_graph(n=NUM_NODES, p=0.15, seed=42),
            'Barabasi-Albert': nx.barabasi_albert_graph(n=NUM_NODES, m=2, seed=42)
        }
    elif NUM_NODES < 500:
        return {
            'Regular-Like': nx.random_regular_graph(d=5, n=NUM_NODES, seed=42),
            'Erdos-Renyi': nx.erdos_renyi_graph(n=NUM_NODES, p=0.05, seed=42),
            'Barabasi-Albert': nx.barabasi_albert_graph(n=NUM_NODES, m=2, seed=42)
        }
    else:
        return {
            'Regular-Like': nx.random_regular_graph(d=3, n=NUM_NODES, seed=42),
            'Erdos-Renyi': nx.erdos_renyi_graph(n=NUM_NODES, p=0.005, seed=42),
            'Barabasi-Albert': nx.barabasi_albert_graph(n=NUM_NODES, m=2, seed=42)
        }

def visualize_graph(graph, security_set, output_path, title):
    """Visualizza il grafo con i nodi del security set evidenziati"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    # Colori: rosso per nodi sicuri, azzurro per gli altri
    node_colors = ['#FF6B6B' if node in security_set else '#4ECDC4' for node in graph.nodes()]
    node_sizes = [800 if node in security_set else 500 for node in graph.nodes()]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=2)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_comparison(algorithms_data, output_path, title):
    """Plot confronto convergenza degli algoritmi"""

    fig, ax = plt.subplots(figsize=(12, 6))
    
    br_data = None 
    
    for algo_name, history in algorithms_data.items():
        if history:
            iterations = [h['iteration'] for h in history]
            set_sizes = [h['set_size'] for h in history]
            
            #
            if 'Best Response' in algo_name:
                br_data = (iterations, set_sizes)
                
            ax.plot(iterations, set_sizes, marker='o', label=algo_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Iterazioni', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimensione Security Set', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if br_data:
        ax_inset = ax.inset_axes([0.7, 0.7, 0.28, 0.28]) # [left, bottom, width, height]
        
        ax_inset.plot(br_data[0], br_data[1], marker='o', color='blue')
        
        ax_inset.set_xlim(-0.5, 10) 
        
        min_y = min(br_data[1]) - 5
        max_y = max(br_data[1]) + 5
        ax_inset.set_ylim(min_y, max_y)
        
        ax_inset.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shapley_values(shapley_values, security_set, output_path, title):
    """Plot dei valori di Shapley come un grafico a linea (plot) ordinato"""
    plt.figure(figsize=(14, 6))
    
    # --- MODIFICA: Ordina i dati per valore decrescente ---
    sorted_data = sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    
    # Ricrea le liste ordinate
    nodes = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]
    
    # --- MODIFICA: Usa plt.plot invece di plt.bar ---
    x_axis = range(len(nodes))
    plt.plot(x_axis, values, color='#FF6B6B', linewidth=2, label='Valore Shapley')
    
    # --- MODIFICA: Colora l'area sotto la curva ---
    # Trova il punto in cui smettiamo di selezionare nodi
    # (assumendo che i nodi del set siano quelli con valore più alto)
    try:
        cutoff_index = max(i for i, node in enumerate(nodes) if node in security_set)
    except ValueError:
        cutoff_index = 0 # Nessun nodo nel set
        
    # Colora l'area dei nodi "selezionati"
    plt.fill_between(x_axis[:cutoff_index+1], values[:cutoff_index+1], 
                     color='#FF6B6B', alpha=0.5, label='Nodi nel Security Set')
    # Colora l'area dei nodi "non selezionati"
    plt.fill_between(x_axis[cutoff_index:], values[cutoff_index:], 
                     color='#95E1D3', alpha=0.5, label='Nodi non nel Set')
    # --- Fine Modifica ---
    
    plt.xlabel('Nodi (Ordinati per Shapley Value)', fontsize=12, fontweight='bold')
    plt.ylabel('Shapley Value', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shapley_boxplot(shapley_values, output_path, title):
    """Crea un box plot per la distribuzione dei Valori di Shapley"""
    plt.figure(figsize=(10, 6))
    
    # Estrai solo i valori POSITIVI (log(0) non è definito)
    values = [v for v in list(shapley_values.values()) if v > 0]
    
    if not values:
        print("Attenzione: nessun valore di Shapley positivo da plottare.")
        plt.close()
        return

    # Crea il box plot
    plt.boxplot(values, vert=False, patch_artist=True,
                boxprops=dict(facecolor='#95E1D3'),
                medianprops=dict(color='#FF6B6B', linewidth=2))
    
    # --- QUESTA È LA MODIFICA CHIAVE ---
    plt.xscale('log')
    # -----------------------------------
    
    plt.xlabel('Shapley Value (Scala Log)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks([]) 
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_market_allocation(buyers, vendors, matches, output_path, title):
    """Visualizza l'allocazione del marketplace (con Istogramma Buyer e Bubble Chart Vendor)"""
    # Crea la figura con 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # Reso un po' più largo
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # --- Subplot 1: Istogramma Impilato dei Budget (INVARIATO) ---
    matched_buyers_ids = {m[0] for m in matches if m[1] is not None}
    
    budgets_matched = [b['budget'] for b in buyers if b['id'] in matched_buyers_ids]
    budgets_unmatched = [b['budget'] for b in buyers if b['id'] not in matched_buyers_ids]
    
    bins = np.arange(0, 101, 10) 
    
    ax1.hist([budgets_matched, budgets_unmatched], 
             bins=bins, 
             stacked=True, 
             color=['#FF6B6B', '#95E1D3'], 
             label=['Matched', 'Unmatched'],
             edgecolor='black')

    ax1.set_xlabel('Fascia di Budget', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Numero di Buyer (Count)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribuzione Budget (Rosso=Matched)', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_xticks(bins) 
    ax1.legend()
    # --- Fine Subplot 1 ---

    
    # --- MODIFICA Subplot 2: Sostituito con Bubble Chart ---
    
    # 1. Raccogli i dati per ogni vendor
    vendor_data = {v['id']: {
        'price': v['price'],
        'security_level': v['security_level'],
        'sales_count': 0,
        'utility_sum': 0
    } for v in vendors}

    for buyer_id, vendor_id, utility in matches:
        if vendor_id is not None:
            vendor_data[vendor_id]['sales_count'] += 1
            vendor_data[vendor_id]['utility_sum'] += utility

    # 2. Estrai le liste per il plotting
    prices = [d['price'] for d in vendor_data.values()]
    security_levels = [d['security_level'] for d in vendor_data.values()]
    sales_counts = [d['sales_count'] for d in vendor_data.values()]
    
    # Calcola l'utilità media, gestendo la divisione per zero se un vendor non vende nulla
    avg_utilities = [
        (d['utility_sum'] / d['sales_count']) if d['sales_count'] > 0 else 0 
        for d in vendor_data.values()
    ]
    
    # 3. Disegna il Bubble Chart
    # La dimensione 's' è scalata per leggibilità (es. vendite^2 * 10)
    sizes = [(s**1.5 * 20) + 10 for s in sales_counts] # +10 per vedere anche chi non vende
    
    scatter = ax2.scatter(prices, security_levels, s=sizes, c=avg_utilities, 
                          cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Aggiungi etichette per i Vendor ID
    for v_id, data in vendor_data.items():
        ax2.text(data['price'] + 1, data['security_level'] + 0.1, str(v_id), 
                 fontsize=9, ha='left')

    # Aggiungi una Color Bar per l'utilità
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.05)
    cbar.set_label('Utilità Media del Match', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Prezzo (€)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Livello di Sicurezza', fontsize=12, fontweight='bold')
    ax2.set_title('Analisi Venditori (Dimensione = Vendite)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 11) # Limiti per livello di sicurezza 1-10
    ax2.set_xlim(0, 101) # Limiti per prezzo 1-100
    # --- Fine Modifica Subplot 2 ---
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta per il titolo generale
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_vcg_path(graph, security_set, optimal_path, source, target, output_path, title):
    """Visualizza il percorso trovato dal VCG (con colori di sicurezza sul percorso)"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    # --- 1. Mappatura Colori per TUTTI i nodi ---
    # Creiamo una mappa {nodo: colore} per un accesso facile
    node_color_map = {}
    for node in graph.nodes():
        if node == source:
            node_color_map[node] = "#4853B1"  # 1. Blu per source
        elif node == target:
            node_color_map[node] = "#DA9A1B"  # 2. Arancione per target
        
        # 3. Logica per i nodi SUL PERCORSO (non source/target)
        elif node in optimal_path:
            if node in security_set:
                node_color_map[node] = '#6BCB77'  # Verde (Sicuro sul percorso)
            else:
                node_color_map[node] = "#FF0000"  # Rosso (Insicuro sul percorso)
        
        # 4. Logica per i nodi NON SUL PERCORSO
        elif node in security_set:
            node_color_map[node] = '#95E1D3'  # Azzurro (Sicuro, non percorso)
        else:
            node_color_map[node] = '#E5E5E5'  # Grigio (Insicuro, non percorso)
    
    # Lista di colori nell'ordine di graph.nodes(), per lo sfondo
    all_node_colors = [node_color_map[node] for node in graph.nodes()]

    
    # --- 2. Disegna lo sfondo "fantasma" ---
    nx.draw_networkx_nodes(graph, pos, node_color=all_node_colors, node_size=500, alpha=0.2)
    nx.draw_networkx_edges(graph, pos, alpha=0.05, width=1)
    
    # --- 3. Evidenzia il percorso (disegnando sopra) ---
    path_edges = [(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)]
    path_nodes = list(optimal_path)
    
    # --- MODIFICA CHIAVE ---
    # Costruisci la lista dei colori del percorso NELLO STESSO ORDINE di path_nodes
    path_node_colors = [node_color_map[node] for node in path_nodes]
    # --- Fine Modifica ---
    
    # Ridisegna solo i nodi del percorso (opaco)
    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_color=path_node_colors, node_size=700, alpha=1.0, 
                           edgecolors='black', linewidths=0.5)
    
    # Ridisegna solo gli archi del percorso (opaco)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='#FF6B6B', width=4, alpha=1.0)
    
    # --- Etichette Selettive ---
    labels = {node: node for node in optimal_path}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_full_analysis(graph, network_name, output_dir):
    """Esegue l'analisi completa e salva tutti i risultati su file"""
    report = []
    report.append("="*70)
    report.append(f" ANALISI NETWORK: {network_name}")
    report.append(f" Nodi: {len(graph.nodes)} | Archi: {len(graph.edges)}")
    report.append("="*70)
    report.append("")

    # --- TASK 1: Strategic Games ---
    report.append("[TASK 1] STRATEGIC GAME ANALYSIS")
    report.append("-" * 70)
    
    # Algoritmo Greedy di riferimento
    greedy_set = greedy_minimal_vertex_cover(graph)
    report.append(f"RIFERIMENTO - Greedy Algorithm:")
    report.append(f"  - Security Set Size: {len(greedy_set)}")
    report.append(f"  - Security Set Nodes: {sorted(greedy_set)}")
    greedy_analysis = analyze_security_set_quality(graph, greedy_set, "Greedy")
    report.append(f"  - Valid: {greedy_analysis['is_valid']} | Minimal: {greedy_analysis['is_minimal']}")
    if not greedy_analysis['is_valid']:
        report.append(f"  ❌ ERRORE: Il set NON è valido")
    elif not greedy_analysis['is_minimal']:
        report.append(f"  ⚠️ WARNING: Valido ma NON minimale")
    else:
        report.append(f"  ✅ Il set è minimale")
    report.append("")
    
    # Best Response
    strat_game_br = NetworkSecurityGame(graph)
    _, pne_br = strat_game_br.best_response_dynamics()
    nash_set = strat_game_br.get_security_set()
    br_analysis = analyze_security_set_quality(graph, nash_set, "Best Response")
    report.append(f"Best Response Dynamics:")
    report.append(f"  - Security Set Size: {len(nash_set)}")
    report.append(f"  - Security Set Nodes: {sorted(nash_set)}")
    report.append(f"  - Is Nash Equilibrium: {pne_br}")
    if not br_analysis['is_valid']:
        report.append(f"  ❌ ERRORE: Il set NON è valido")
    elif not br_analysis['is_minimal']:
        report.append(f"  ⚠️ WARNING: Valido ma NON minimale")
    else:
        report.append(f"  - Convergenza in {len(strat_game_br.convergence_history)} iterazioni")
        report.append(f"  ✅ Il set è minimale")
        
    
    # Fictitious Play
    strat_game_fp = NetworkSecurityGame(graph)
    _, pne_fp = strat_game_fp.fictitious_play()
    fp_set = strat_game_fp.get_security_set()
    fp_analysis = analyze_security_set_quality(graph, fp_set, "Fictitious Play")
    report.append(f"\nFictitious Play:")
    report.append(f"  - Security Set Size: {len(fp_set)}")
    report.append(f"  - Security Set Nodes: {sorted(fp_set)}")
    report.append(f"  - Is Nash Equilibrium: {pne_fp}")
    if not fp_analysis['is_valid']:
        report.append(f"  ❌ ERRORE: Il set NON è valido")
    elif not fp_analysis['is_minimal']:
        report.append(f"  ⚠️ WARNING: Valido ma NON minimale")
    else:
        report.append(f"  - Convergenza in {len(strat_game_fp.convergence_history)} iterazioni")
        report.append(f"  ✅ Il set è minimale")
    
    # Regret Matching
    strat_game_rm = NetworkSecurityGame(graph)
    _, pne_rm = strat_game_rm.regret_matching()
    rm_set = strat_game_rm.get_security_set()
    rm_analysis = analyze_security_set_quality(graph, rm_set, "Regret Matching")
    report.append(f"\nRegret Matching:")
    report.append(f"  - Security Set Size: {len(rm_set)}")
    report.append(f"  - Security Set Nodes: {sorted(rm_set)}")
    report.append(f"  - Is Nash Equilibrium: {pne_rm}")
    if not rm_analysis['is_valid']:
        report.append(f"  ❌ ERRORE: Il set NON è valido")
    elif not rm_analysis['is_minimal']:
        report.append(f"  ⚠️ WARNING: Valido ma NON minimale")
    else:
        report.append(f"  - Convergenza in {len(strat_game_rm.convergence_history)} iterazioni")
        report.append(f"  ✅ Il set è minimale")
    
    # Confronto con Greedy
    report.append(f"\n📊 CONFRONTO CON GREEDY:")
    report.append(f"  - Best Response: {len(nash_set)} vs Greedy: {len(greedy_set)} ({'+' if len(nash_set) > len(greedy_set) else ''}{len(nash_set) - len(greedy_set)})")
    report.append(f"  - Fictitious Play: {len(fp_set)} vs Greedy: {len(greedy_set)} ({'+' if len(fp_set) > len(greedy_set) else ''}{len(fp_set) - len(greedy_set)})")
    report.append(f"  - Regret Matching: {len(rm_set)} vs Greedy: {len(greedy_set)} ({'+' if len(rm_set) > len(greedy_set) else ''}{len(rm_set) - len(greedy_set)})")
    report.append("")

    # Plot convergenza
    algorithms_data = {
        'Best Response': strat_game_br.convergence_history,
        'Fictitious Play': strat_game_fp.convergence_history,
        'Regret Matching': strat_game_rm.convergence_history
    }
    plot_convergence_comparison(
        algorithms_data,
        f"{output_dir}/plots/{network_name}_convergence.png",
        f"Convergenza Algoritmi - {network_name}"
    )

    # Visualizza grafo con Nash set
    visualize_graph(
        graph, nash_set,
        f"{output_dir}/graphs/{network_name}_nash_equilibrium.png",
        f"{network_name} - Nash Equilibrium (Best Response)"
    )

    # --- TASK 2: Coalitional Games ---
    report.append("[TASK 2] COALITIONAL GAME ANALYSIS (Shapley Values)")
    report.append("-" * 70)
    coal_game = CoalitionalSecurityGame(graph)
    shapley_vals = coal_game.calculate_shapley_monte_carlo(num_permutations=200)
    shapley_set = coal_game.build_security_set_from_shapley(shapley_vals)
    shapley_analysis = analyze_security_set_quality(graph, shapley_set, "Shapley")
    
    report.append(f"Shapley-based Security Set:")
    report.append(f"  - Size: {len(shapley_set)}")
    report.append(f"  - Nodes: {sorted(shapley_set)}")
    if not shapley_analysis['is_valid']:
        report.append(f"  ❌ ERRORE: Il set NON è valido")
    elif not shapley_analysis['is_minimal']:
        report.append(f"  ⚠️ WARNING: Valido ma NON minimale")
    else:
        report.append(f"  ✅ Il set è minimale")
    report.append(f"  - Confronto con Greedy: {len(shapley_set)} vs {len(greedy_set)} ({'+' if len(shapley_set) > len(greedy_set) else ''}{len(shapley_set) - len(greedy_set)})")
    report.append(f"\nTop 5 Shapley Values:")
    sorted_shapley = sorted(shapley_vals.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, value in sorted_shapley:
        report.append(f"  - Node {node}: {value:.4f}")
    report.append("")
    
    # Plot Shapley values
    plot_shapley_values(
        shapley_vals, shapley_set,
        f"{output_dir}/plots/{network_name}_shapley_values.png",
        f"Shapley Values - {network_name}"
    )

    plot_shapley_boxplot(
        shapley_vals,
        f"{output_dir}/plots/{network_name}_shapley_boxplot.png",
        f"Distribuzione Valori di Shapley - {network_name}"
    )
    
    # Visualizza grafo con Shapley set
    visualize_graph(
        graph, shapley_set,
        f"{output_dir}/graphs/{network_name}_shapley_set.png",
        f"{network_name} - Shapley-based Security Set"
    )
    
    # Confronto
    final_set = nash_set if len(nash_set) <= len(shapley_set) else shapley_set
    report.append(f"Set Finale Scelto: {'Nash (Best Response)' if final_set == nash_set else 'Shapley'}")
    report.append(f"  - Size: {len(final_set)}")
    report.append("")

    # --- TASK 3: Market ---
    report.append("[TASK 3] MARKET SIMULATION (Matching)")
    report.append("-" * 70)
    market = SecurityMarketplace(final_set)

    # --- Scenario 1: Capacità Infinita ---
    report.append("SCENARIO 1: Capacità Infinita")
    matches_inf, welfare = market.run_scenario_infinite_capacity()
    
    matched_count = sum(1 for m in matches_inf if m[1] is not None)
    report.append(f"Social Welfare Totale: {welfare}")
    # # report.append(f"Buyers Matched: {matched_count}/{len(final_set)}")
    if len(final_set) > 0:
        tasso_match = (matched_count / len(final_set) * 100)
        report.append(f"Tasso di Match: {tasso_match:.1f}%")
    else:
        # Gestisci il caso in cui il set è vuoto (0 nodi protetti)
        report.append("Tasso di Match: N/A (Il set di sicurezza è vuoto)")
        report.append("AVVISO: La dinamica ha prodotto un set di sicurezza vuoto, il che è un risultato anomalo se il grafo ha archi.")
        # report.append(f"Tasso di Match: {(matched_count/len(final_set)*100):.1f}%")

    # Plot market INFINITO
    plot_market_allocation(
        market.buyers, market.vendors, matches_inf,
        f"{output_dir}/plots/{network_name}_market_infinite.png", # Nuovo nome file
        f"Market Allocation (Infinite Capacity) - {network_name}"
    )

# --- Scenario 2: Capacità Limitata ---
    report.append("")
    report.append("SCENARIO 2: Capacità Limitata (max_items=2)")
    matches_lim, welfare_lim = market.run_scenario_limited_capacity()
    
    matched_count_lim = sum(1 for m in matches_lim if m[1] is not None)
    report.append(f"  Social Welfare Totale: {welfare_lim:.2f}")
    report.append(f"  Buyers Matched: {matched_count_lim}/{len(final_set)}")
    if len(final_set) > 0:
        tasso_match = (matched_count_lim / len(final_set) * 100)
        report.append(f"  Tasso di Match: {tasso_match:.1f}%")
    else:
        # Gestisci il caso in cui il set è vuoto (0 nodi protetti)
        report.append("Tasso di Match: N/A (Il set di sicurezza è vuoto)")
        report.append("AVVISO: La dinamica ha prodotto un set di sicurezza vuoto, il che è un risultato anomalo se il grafo ha archi.")

    report.append(f"\nDettagli Matches (Limited):")
    for buyer_id, vendor_id, utility in matches_lim[:10]:  # Prime 10
        if vendor_id is not None:
            report.append(f"  - Buyer {buyer_id} -> Vendor {vendor_id} (Utility: {utility:.2f})")
        else:
            report.append(f"  - Buyer {buyer_id} -> Non assegnato")
    if len(matches_lim) > 10:
        report.append(f"  ... e altri {len(matches_lim)-10} buyers")
    report.append("")
    
    # Plot market LIMITATO
    plot_market_allocation(
        market.buyers, market.vendors, matches_lim,
        f"{output_dir}/plots/{network_name}_market_limited.png", # Nuovo nome file
        f"Market Allocation (Limited Capacity) - {network_name}"
    )

    # --- TASK 4: Mechanism Design ---
    report.append("[TASK 4] MECHANISM DESIGN (VCG Auction)")
    report.append("-" * 70)
    
    if len(graph.nodes) > 1:
        source, target = random.sample(list(graph.nodes), 2)
        report.append(f"Percorso Richiesto: Nodo {source} -> Nodo {target}")
        report.append("")
        
        vcg = VCGPathAuction(graph, final_set)
        optimal_path, optimal_cost, payments = vcg.run_vcg_mechanism(source, target)
        
        if optimal_path:
            report.append(f"Path Ottimale: {' -> '.join(map(str, optimal_path))}")
            report.append(f"Costo Totale: {optimal_cost:.2f}")
            report.append(f"\nPagamenti VCG per gli archi vincitori:")
            total_payment = 0
            for edge, data in payments.items():
                payment_str = f"{data['payment']:.2f}" if data['payment'] != float('inf') else "INF (Bridge)"
                report.append(f"  - Arco {edge}: Bid={data['bid']}, Pagamento={payment_str}")
                if data['payment'] != float('inf'):
                    total_payment += data['payment']
            report.append(f"\nPagamento Totale: {total_payment:.2f}")
            
            # Visualizza path
            visualize_vcg_path(
                graph, final_set, optimal_path, source, target,
                f"{output_dir}/graphs/{network_name}_vcg_path.png",
                f"{network_name} - VCG Path: {source} → {target}"
            )
        else:
            report.append("ERRORE: Nessun percorso trovato tra source e target")
    else:
        report.append("SKIP: Grafo troppo piccolo per VCG auction")
    
    report.append("")
    report.append("="*70)
    
    return "\n".join(report)

def create_html_index(output_dir, networks):
    """Crea un file HTML per navigare facilmente i risultati"""
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGT Project - Network Security Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .network-section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-card h3 {{
            margin-top: 10px;
            color: #2c3e50;
            font-size: 16px;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .report-link {{
            display: inline-block;
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border-radius: 5px;
            margin: 10px 5px;
        }}
        .report-link:hover {{
            background: #2980b9;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <h1>🔒 Network Security Set Analysis - AGT Project</h1>
    <p><strong>Data generazione:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="network-section">
        <h2>📄 Report Testuali</h2>
        <a href="report_completo.txt" class="report-link">Report Completo</a>
"""    
    for name in networks.keys():
        html_content += f'        <a href="{name}_report.txt" class="report-link">{name} Report</a>\n'
    
    html_content += """    </div>
    
"""
    
    for name in networks.keys():
        html_content += f"""    <div class="network-section">
        <h2>🌐 {name}</h2>
        
        <h3>Grafi</h3>
        <div class="image-grid">
            <div class="image-card">
                <img src="graphs/{name}_nash_equilibrium.png" alt="Nash Equilibrium">
                <h3>Nash Equilibrium (Best Response)</h3>
            </div>
            <div class="image-card">
                <img src="graphs/{name}_shapley_set.png" alt="Shapley Set">
                <h3>Shapley-based Security Set</h3>
            </div>
            <div class="image-card">
                <img src="graphs/{name}_vcg_path.png" alt="VCG Path">
                <h3>VCG Optimal Path</h3>
            </div>
        </div>
        
        <h3>Analisi e Metriche</h3>
        <div class="image-grid">
            <div class="image-card">
                <img src="plots/{name}_convergence.png" alt="Convergence">
                <h3>Convergenza Algoritmi</h3>
            </div>
            <div class="image-card">
                <img src="plots/{name}_shapley_boxplot.png" alt="Shapley Boxplot">
                <h3>Distribuzione Shapley Values</h3>
            </div>
            <div class="image-card">
                <img src="plots/{name}_shapley_values.png" alt="Shapley Values">
                <h3>Shapley Values</h3>
            </div>
            <div class="image-card">
                <img src="plots/{name}_market_infinite.png" alt="Market Infinite">
                <h3>Market Allocation (Infinite Capacity)</h3>
            </div>
            <div class="image-card">
                <img src="plots/{name}_market_limited.png" alt="Market Limited">
                <h3>Market Allocation (Limited Capacity)</h3>
            </div>
        </div>
    </div>
    
"""
    
    html_content += """</body>
</html>"""
    
    with open(f"{output_dir}/index.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    # Crea directory di output
    output_dir = create_output_directory()
    
    # Genera networks
    networks = generate_networks()
    
    # File report principale
    main_report_path = f"{output_dir}/report_completo.txt"
    
    with open(main_report_path, 'w', encoding='utf-8') as main_report:
        main_report.write("#" * 80 + "\n")
        main_report.write("# ALGORITHMIC GAME THEORY PROJECT - NETWORK SECURITY SET\n")
        main_report.write("# Analisi Completa su Grafi Multipli\n")
        main_report.write(f"# Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        main_report.write("#" * 80 + "\n\n")
        
        # Summary iniziale
        main_report.write("NETWORKS ANALIZZATI:\n")
        for name, G in networks.items():
            main_report.write(f"  - {name}: {len(G.nodes)} nodi, {len(G.edges)} archi\n")
        main_report.write("\n" + "="*80 + "\n\n")
        
        # Analisi di ogni network
        comparison_data = {}
        for name, G in networks.items():
            report_text = run_full_analysis(G, name, output_dir)
            main_report.write(report_text + "\n\n")
            
            # Salva report individuale
            with open(f"{output_dir}/{name}_report.txt", 'w', encoding='utf-8') as individual_report:
                individual_report.write(report_text)
        
        # Summary finale
        main_report.write("\n" + "#" * 80 + "\n")
        main_report.write("# SUMMARY\n")
        main_report.write("#" * 80 + "\n\n")
        main_report.write("Output generati:\n")
        main_report.write(f"  - Report principale: {main_report_path}\n")
        main_report.write(f"  - Visualizzazioni grafi: {output_dir}/graphs/\n")
        main_report.write(f"  - Plot e grafici: {output_dir}/plots/\n")
        main_report.write(f"  - Report individuali: {output_dir}/*_report.txt\n")
        main_report.write("\nAnalisi completata con successo!\n")
    
    # Crea un file indice HTML per navigazione facile
    create_html_index(output_dir, networks)
    
    print(f"\n{'='*70}")
    print("ANALISI COMPLETATA!")
    print(f"{'='*70}")
    print(f"\nTutti i risultati sono stati salvati in: {output_dir}/")
    print(f"\nFile generati:")
    print(f"  📄 Report completo: {output_dir}/report_completo.txt")
    print(f"  📄 Index HTML: {output_dir}/index.html")
    print(f"  📊 Grafici: {output_dir}/plots/")
    print(f"  🌐 Visualizzazioni grafi: {output_dir}/graphs/")
    print(f"\n{'='*70}\n")