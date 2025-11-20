import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# --- FUNZIONE AGGIUNTA PER STAMPARE LA CONVERGENZA ---
def plot_convergence(history, algorithm_name, filename="convergence.png"):
    """
    Traccia l'andamento del numero di nodi che giocano '1' nel tempo.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, label='Size of Security Set (Strategy = 1)')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Nodes playing 1')
    plt.title(f'Convergence Analysis: {algorithm_name}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Salva l'immagine
    plt.savefig(filename, dpi=300)
    plt.close() # Chiude la figura per liberare memoria
    print(f"Grafico di convergenza salvato come: {filename}")

# --- SecurityGame Class (The Strategic Game) ---
class SecurityGame:
    def __init__(self, graph, alpha=10, c=4):
        self.graph = graph
        self.num_players = graph.number_of_nodes()
        self.alpha = alpha
        self.c = c

        if not (self.alpha > self.c > 0):
            raise ValueError("Parameters must satisfy alpha > c > 0")

    def get_payoff(self, player_id, strategies):
        s_i = strategies[player_id]
        is_covered = False
        if s_i == 1:
            is_covered = True
        else:
            neighbors = list(self.graph.neighbors(player_id))
            if not neighbors:
                is_covered = False
            else:
                is_covered = all(strategies.get(neighbor) == 1 for neighbor in neighbors)
        
        if s_i == 1:
            return self.alpha - self.c
        elif s_i == 0 and is_covered:
            return self.alpha
        elif s_i == 0 and not is_covered:
            return 0
        else:
            raise ValueError("Invalid strategy or state encountered.")

    def is_nash_equilibrium(self, strategies):
        for player_id in range(self.num_players):
            original_payoff = self.get_payoff(player_id, strategies)
            alt_strategies = strategies.copy()
            alt_strategies[player_id] = 1 - strategies[player_id]
            alt_payoff = self.get_payoff(player_id, alt_strategies)
            if alt_payoff > original_payoff:
                return False
        return True

# --- BestResponseDynamics Class (Updated with History) ---
class BestResponseDynamics:
    def __init__(self, game, max_iterations=1000):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        self.strategy_counts = {player_id: {0: 0.0, 1: 0.0} for player_id in range(self.num_players)}
        self.current_strategies = {player_id: random.choice([0, 1]) for player_id in range(self.num_players)}
        for player_id, strategy in self.current_strategies.items():
            self.strategy_counts[player_id][strategy] += 1.0

    def run(self):
        equilibrium_strategies = None
        history = [] # NEW: Track convergence
        
        for iteration in range(self.max_iterations):
            # NEW: Record current set size
            history.append(sum(self.current_strategies.values()))

            previous_strategies = self.current_strategies.copy()
            
            for player_id in range(self.num_players):
                temp_strategies_0 = self.current_strategies.copy()
                temp_strategies_0[player_id] = 0
                payoff_if_0 = self.game.get_payoff(player_id, temp_strategies_0)
                
                temp_strategies_1 = self.current_strategies.copy()
                temp_strategies_1[player_id] = 1
                payoff_if_1 = self.game.get_payoff(player_id, temp_strategies_1)
                
                if payoff_if_1 > payoff_if_0:
                    self.current_strategies[player_id] = 1
                elif payoff_if_0 > payoff_if_1:
                    self.current_strategies[player_id] = 0
                else:
                    self.current_strategies[player_id] = random.choice([0,1])
                                    
                self.strategy_counts[player_id][self.current_strategies[player_id]] += 1.0
            
            if self.current_strategies == previous_strategies:
                print(f"Convergence detected at iteration {iteration+1}.")
                equilibrium_strategies = self.current_strategies
                # Fill remaining history for plotting consistency (flat line)
                history.extend([history[-1]] * (self.max_iterations - iteration - 1))
                break
        
        if equilibrium_strategies is None:
            print(f"Best Response Dynamics did not converge strictly after {self.max_iterations} iterations.")
            equilibrium_strategies = self.current_strategies

        print("Final Strategies found by Best Response Dynamics:")
        # print(equilibrium_strategies) # Commented out to reduce noise in console
        
        is_pne = self.game.is_nash_equilibrium(equilibrium_strategies)
        print(f"Is this a Pure Nash Equilibrium? {is_pne}")
        
        return equilibrium_strategies, is_pne, history # NEW: Return history

# --- FictitiousPlay Class (Updated with History) ---
class FictitiousPlay:
    def __init__(self, game, max_iterations=5000, update_fraction=0.7):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        self.graph = game.graph
        self.update_fraction = update_fraction
        self.neighbor_counts = {
            i: {v: 0 for v in self.graph.neighbors(i)} for i in range(self.num_players)
        }
        self.total_counts = {i: 0 for i in range(self.num_players)}
        self.current_strategies = {i: random.choice([0, 1]) for i in range(self.num_players)}

    def run(self):
        history = []
        no_change_counter = 0
        # La stabilità richiesta può essere abbassata se l'algoritmo è deterministico
        required_stability = self.num_players * 5 
        
        # Pre-calcolo la soglia di convenienza
        # Se la probabilità di essere coperti supera questa soglia, conviene giocare 0.
        threshold = (self.game.alpha - self.game.c) / self.game.alpha

        for t in range(self.max_iterations):
            current_set_size = sum(self.current_strategies.values())
            history.append(current_set_size)
            
            changed = False
            
            # Selezione batch dei nodi
            num_to_update = max(1, int(self.num_players * self.update_fraction))
            nodes_to_update = random.sample(range(self.num_players), num_to_update)
            
            # Fase 1: Aggiornamento delle credenze (Belief Update)
            # Nota: In Asynchronous FP, spesso si aggiornano le credenze solo quando si viene attivati,
            # oppure si aggiornano globalmente. Qui manteniamo la tua logica:
            # aggiorniamo i conteggi solo per i nodi che "pensano" in questo turno.
            for i in nodes_to_update:
                self.total_counts[i] += 1
                for v in self.graph.neighbors(i):
                    if self.current_strategies[v] == 1:
                        self.neighbor_counts[i][v] += 1

            # Fase 2: Calcolo Best Response
            for i in nodes_to_update:
                neighbors = list(self.graph.neighbors(i))
                old_strategy = self.current_strategies[i]
                new_strategy = old_strategy

                if not neighbors:
                    # Se non ho vicini, non posso essere coperto -> Devo giocare 1
                    new_strategy = 1
                else:
                    # Calcolo le probabilità empiriche che ogni vicino giochi 1
                    probs = []
                    for v in neighbors:
                        if self.total_counts[i] > 0:
                            p = self.neighbor_counts[i][v] / self.total_counts[i]
                        else:
                            p = 0.5 # Prior uniforme se prima iterazione
                        probs.append(p)
                    
                    # Calcolo Probabilità Congiunta (PRODOTTO, non media)
                    prob_all_neighbors_1 = math.prod(probs)
                    
                    
                    # --- BEST RESPONSE DETERMINISTICA ---
                    # Se la probabilità che tutti mi coprano è alta, rischio e gioco 0.
                    # Altrimenti, mi proteggo e gioco 1.
                    if prob_all_neighbors_1 > threshold:
                        new_strategy = 0
                    else:
                        new_strategy = 1
                    """
                    # --- BEST RESPONSE STOCASTICA ---
                    if random.random() <= prob_all_neighbors_1:
                        new_strategy = 0
                    else:
                        new_strategy = 1
                    """

                # Applica il cambiamento
                if new_strategy != old_strategy:
                    self.current_strategies[i] = new_strategy
                    changed = True
            
            # Controllo convergenza
            if not changed:
                no_change_counter += 1
            else:
                no_change_counter = 0
                
            if no_change_counter >= required_stability:
                print(f"Convergence detected at step {t}.")
                history.extend([history[-1]] * (self.max_iterations - t - 1))
                break
        
        print(f"Final Strategies (Size S: {sum(self.current_strategies.values())})")
        return self.current_strategies, self.game.is_nash_equilibrium(self.current_strategies), history
    
# --- 5. RegretMatching Class (Updated with History) ---
class RegretMatching:
    def __init__(self, game, max_iterations=1000):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        self.cumulative_regrets = {i: {0: 0.0, 1: 0.0} for i in range(self.num_players)}
        self.strategy_probs = {i: {0: 0.5, 1: 0.5} for i in range(self.num_players)}
        self.current_strategies = {i: random.choice([0, 1]) for i in range(self.num_players)}

    def _get_action(self, player_id):
        probs = self.strategy_probs[player_id]
        return 1 if random.random() < probs[1] else 0

    def run(self):
        history = [] # NEW: Track convergence

        for t in range(self.max_iterations):
            # NEW: Record current set size
            history.append(sum(self.current_strategies.values()))

            self.current_strategies = {i: self._get_action(i) for i in range(self.num_players)}
            
            for i in range(self.num_players):
                u_actual = self.game.get_payoff(i, self.current_strategies)
                
                actual_strat = self.current_strategies[i]
                other_strat = 1 - actual_strat
                
                temp_profile = self.current_strategies.copy()
                temp_profile[i] = other_strat
                u_counterfactual = self.game.get_payoff(i, temp_profile)
                
                regret = u_counterfactual - u_actual
                self.cumulative_regrets[i][other_strat] += regret

            for i in range(self.num_players):
                r_0_pos = max(0, self.cumulative_regrets[i][0])
                r_1_pos = max(0, self.cumulative_regrets[i][1])
                sum_r = r_0_pos + r_1_pos
                
                if sum_r > 0:
                    self.strategy_probs[i][0] = r_0_pos / sum_r
                    self.strategy_probs[i][1] = r_1_pos / sum_r
                else:
                    self.strategy_probs[i][0] = 0.5
                    self.strategy_probs[i][1] = 0.5
        
        print(f"Regret Matching finished after {self.max_iterations} iterations.")
        
        final_pure_strategies = {}
        for i in range(self.num_players):
            prob_1 = self.strategy_probs[i][1]
            if prob_1 > 0.5:
                final_pure_strategies[i] = 1
            elif prob_1 < 0.5:
                final_pure_strategies[i] = 0
            else:
                final_pure_strategies[i] = self.current_strategies[i]

        print("Final Strategies found by Regret Matching (from probabilities):")
        is_pne = self.game.is_nash_equilibrium(final_pure_strategies)
        print(f"Is this a Pure Nash Equilibrium? {is_pne}")
        
        return final_pure_strategies, is_pne, history # NEW

# --- 6. Validation Functions (Unchanged) ---
def is_security_set(graph, security_set):
    if not security_set: 
        return graph.number_of_nodes() == 0
    if not all(node in graph.nodes for node in security_set):
        return False
    for node in graph.nodes:
        if node in security_set:
            continue
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return False
        if not all(neighbor in security_set for neighbor in neighbors):
            return False
    return True

def is_minimal_security_set(graph, security_set):
    if not is_security_set(graph, security_set):
        print("Error: The set is not a security set.")
        return False
    for node_to_remove in security_set:
        temp_set = security_set - {node_to_remove}
        if is_security_set(graph, temp_set):
            return False
    return True

# --- Main Execution ---
def create_regular_graph(n: int, k: int):
    return nx.random_regular_graph(k, n)

def create_erdos_renyi(n: int, p: float):
    return nx.gnp_random_graph(n, p)

def run_and_report(game, algo_class, algo_name, plot_file, max_iter=5000, **algo_kwargs):
    print(f"\n----- RUNNING {algo_name} -----")
    algo = algo_class(game, max_iterations=max_iter, **algo_kwargs)
    strats, is_pne, history = algo.run()
    plot_convergence(history, algo_name, plot_file)
    result_set = {node for node, s in strats.items() if s == 1}
    print(f"Result Set Size: {len(result_set)}")
    if is_minimal_security_set(game.graph, result_set):
        print("The resulting set is a Minimal security Set.")
    else:
        print("The resulting set is NOT a Minimal security Set.")
    return result_set

if __name__ == "__main__":
    # --- Step 1: Create a k-regular Graph ---
    num_nodes = 1000 # Reduced node count for faster plot generation in example
    k = 3
    max_iter = 5000
    update_fraction_fictitious = 0.2

    G_reg = create_regular_graph(num_nodes, k)
    game_reg = SecurityGame(G_reg, alpha=10, c=4)
    run_and_report(game_reg, BestResponseDynamics, "Best Response Dynamics (Regular)", "brd_regular_convergence.png", max_iter)
    run_and_report(game_reg, FictitiousPlay, "Batch Fictitious Play (Regular)", "fictitious_play_regular_convergence.png", max_iter, update_fraction=update_fraction_fictitious)

    G_erdos = create_erdos_renyi(num_nodes, p=0.05)
    game_erdos = SecurityGame(G_erdos, alpha=10, c=4)
    run_and_report(game_erdos, BestResponseDynamics, "Best Response Dynamics (Erdős-Rényi)", "brd_erdos_convergence.png", max_iter)
    run_and_report(game_erdos, FictitiousPlay, "Batch Fictitious Play (Erdős-Rényi)", "fictitious_play_erdos_convergence.png", max_iter, update_fraction=update_fraction_fictitious)