import numpy as np
import random
import scipy.sparse as sp
import time

# --- Generatore Grafo Regolare (Numpy/Scipy puro) ---
def generate_random_regular_sparse(n, k, max_attempts=10):
    """
    Genera un grafo random k-regolare direttamente in formato sparso (CSR).
    Utilizza l'algoritmo del Configuration Model con riavvio in caso di collisioni (self-loops/multi-edges).
    
    :param n: Numero di nodi
    :param k: Grado di ogni nodo
    :return: scipy.sparse.csr_matrix
    """
    if (n * k) % 2 != 0:
        raise ValueError("n * k deve essere pari.")
    
    # Creiamo gli "stubs": ogni nodo appare k volte nella lista
    # Esempio n=3, k=2 -> [0, 0, 1, 1, 2, 2]
    stubs = np.repeat(np.arange(n), k)
    
    for attempt in range(max_attempts):
        # Mescoliamo gli stubs per creare collegamenti casuali
        np.random.shuffle(stubs)
        
        # Accoppiamo gli stubs a due a due per formare i lati
        # Reshape in (num_edges, 2)
        edges = stubs.reshape(-1, 2)
        
        # --- Controlli di validità ---
        
        # 1. Self-loops: controlla se edges[:,0] == edges[:,1]
        if np.any(edges[:, 0] == edges[:, 1]):
            continue # Riprova, trovato self-loop
            
        # 2. Multi-edges (lati duplicati)
        # Ordiniamo ogni lato in modo che u < v per facilitare il confronto
        edges.sort(axis=1)
        
        # Usiamo numpy.unique per trovare duplicati. 
        # axis=0 controlla le righe (lati)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        
        if np.any(counts > 1):
            continue # Riprova, trovati lati paralleli
            
        # Se siamo qui, il grafo è valido (Semplice)
        print(f"Grafo generato con successo al tentativo {attempt + 1}")
        
        # Costruiamo la matrice sparsa simmetrica
        # Un lato (u, v) deve diventare due entrate: (u, v) e (v, u)
        row_indices = np.concatenate([edges[:, 0], edges[:, 1]])
        col_indices = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(row_indices), dtype=np.int8)
        
        # Creiamo la matrice CSR
        adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return adj_matrix

    raise RuntimeError("Impossibile generare un grafo regolare semplice dopo vari tentativi. "
                       "Prova ad aumentare n o ridurre k.")

# --- SecurityGame Class Ottimizzata (Sparse Matrix) ---
class SecurityGameSparse:
    def __init__(self, adj_matrix, alpha=10, c=4):
        self.adj_matrix = adj_matrix
        self.num_players = adj_matrix.shape[0]
        self.alpha = alpha
        self.c = c

        if not (self.alpha > self.c > 0):
            raise ValueError("Parameters must satisfy alpha > c > 0")

    def check_coverage(self, player_id, strategies):
        start_idx = self.adj_matrix.indptr[player_id]
        end_idx = self.adj_matrix.indptr[player_id + 1]
        
        if start_idx == end_idx:
            return False
            
        neighbor_indices = self.adj_matrix.indices[start_idx:end_idx]
        neighbor_strats = strategies[neighbor_indices]
        return np.all(neighbor_strats == 1)

# --- BestResponseDynamics Ottimizzata ---
class BestResponseDynamicsSparse:
    def __init__(self, game, max_iterations=1000):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        # Strategie casuali iniziali
        self.current_strategies = np.random.choice([0, 1], size=self.num_players).astype(np.int8)

    def run(self):
        print(f"Inizio simulazione dinamica su {self.num_players} nodi...")
        start_time = time.time()

        for iteration in range(self.max_iterations):
            previous_strategies = self.current_strategies.copy()
            
            # Nota: Questo loop può essere ulteriormente vettorizzato, 
            # ma lo teniamo iterativo per chiarezza logica come richiesto.
            for player_id in range(self.num_players):
                is_covered = self.game.check_coverage(player_id, self.current_strategies)
                
                payoff_1 = self.game.alpha - self.game.c
                payoff_0 = self.game.alpha if is_covered else 0
                
                current_s = self.current_strategies[player_id]
                new_s = current_s

                if payoff_1 > payoff_0:
                    new_s = 1
                elif payoff_0 > payoff_1:
                    new_s = 0
                else:
                    if random.random() < 0.5:
                        new_s = 1 - current_s 
                
                self.current_strategies[player_id] = new_s

            if np.array_equal(self.current_strategies, previous_strategies):
                print(f"Convergenza raggiunta all'iterazione {iteration+1}.")
                break
        else:
            print(f"Stop: Massimo iterazioni ({self.max_iterations}) raggiunto.")
        
        print(f"Tempo esecuzione: {time.time() - start_time:.4f} secondi")
        return self.current_strategies

# --- Validation Functions ---
def is_minimal_security_set_sparse(adj_matrix, strategies_array):
    indices_in_set = np.where(strategies_array == 1)[0]
    indices_out_set = np.where(strategies_array == 0)[0]
    
    # 1. Verifica copertura globale
    for node in indices_out_set:
        start = adj_matrix.indptr[node]
        end = adj_matrix.indptr[node+1]
        if start == end: continue 
        neighbors = adj_matrix.indices[start:end]
        if not np.all(strategies_array[neighbors] == 1):
            return False 
            
    # 2. Verifica Minimalità
    for node_to_remove in indices_in_set:
        strategies_array[node_to_remove] = 0
        is_still_valid = True
        
        # Check solo il nodo rimosso (che ora è 0 e deve essere coperto)
        start = adj_matrix.indptr[node_to_remove]
        end = adj_matrix.indptr[node_to_remove+1]
        neighbors = adj_matrix.indices[start:end]
        
        # Se rimuovendolo il nodo diventa scoperto, allora la rimozione non era valida -> OK
        # Se invece rimuovendolo è ancora coperto (o non ha vicini), il set originale NON era minimale.
        if start != end and np.all(strategies_array[neighbors] == 1):
             is_still_valid = True # È ancora coperto dai vicini
        else:
             is_still_valid = False # È scoperto
        
        strategies_array[node_to_remove] = 1 # Ripristina
        
        if is_still_valid:
            return False 

    return True

# --- Main Execution ---
def run_simulation_no_nx():
    # Parametri
    num_nodes = 50000 
    k = 3
    max_iter = 200

    print(f"--- GENERAZIONE GRAFO (No NetworkX) ---")
    print(f"N={num_nodes}, K={k}")
    
    # 1. Creazione Grafo Sparso Diretta
    try:
        adj_matrix = generate_random_regular_sparse(num_nodes, k)
    except Exception as e:
        print(e)
        return

    # 2. Configurazione Gioco
    game = SecurityGameSparse(adj_matrix, alpha=10, c=4)
    algo = BestResponseDynamicsSparse(game, max_iterations=max_iter)
    
    # 3. Esecuzione
    final_strategies = algo.run()
    
    set_size = np.sum(final_strategies)
    print(f"\nDimensione finale Security Set: {set_size} ({set_size/num_nodes:.2%})")
    
    # 4. Verifica (solo se piccolo abbastanza)
    if num_nodes <= 10000:
        print("Verifica minimalità...")
        is_min = is_minimal_security_set_sparse(adj_matrix, final_strategies.copy())
        print(f"È un Minimal Security Set? {is_min}")
    else:
        print("Verifica saltata (N troppo grande).")

if __name__ == "__main__":
    run_simulation_no_nx()