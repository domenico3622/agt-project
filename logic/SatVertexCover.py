import sympy
from sympy.logic.inference import satisfiable
from sympy import symbols, Or, And, Not
import networkx as nx
import math

class SatVertexCover:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        # Crea un simbolo (variabile logica) per ogni nodo:
        # True se il nodo fa parte del Vertex Cover, False altrimenti.
        self.node_vars = {node: symbols(f'x_{node}') for node in self.nodes}

    def _encode_vertex_cover_constraint(self):
        """
        Vincolo 1: Copertura degli Archi.
        Per ogni arco (u, v) del grafo, almeno uno dei due estremi deve essere scelto.
        Logica: (x_u OR x_v) deve essere VERO per ogni arco.
        """
        clauses = []
        for u, v in self.graph.edges():
            clauses.append(Or(self.node_vars[u], self.node_vars[v]))
        return And(*clauses)

    def get_at_most_k_formula(self, k):
        """
        Genera la formula logica per il vincolo di cardinalità: "La somma dei nodi scelti deve essere <= k".
        Usa la tecnica "Sequential Counter" (contatore sequenziale). In quanto in logica booleana non esiste 
        l'operazione di somma, si utilizza un contatore sequenziale per contare il numero di nodi scelti.  
        
        Non possiamo contare non avendo numeri (1, 2, 3...), ma abbiamo a disposizione solo degli interruttori (ACCESO/SPENTO) 
        e dobbiamo tenere traccia del punteggio di una partita. Il Contatore Sequenziale (Sequential Counter) è un modo per 
        costruire un "contatore" usando solo la logica. Ecco come funziona, spiegato con un esempio pratico.
        L'Obiettivo:
        Hai 3 lampadine (x_1, x_2, x_3). Voglio sapere se almeno 2 sono accese. Il computer non sa fare 1+0+1=2. Deve 
        arrivarci passo dopo passo. 
        Il "Contatore" (Le variabili S)
        Il contatore crea una griglia di registri (variabili temporanee chiamate S). Ogni registro S_{i,j} risponde a una 
        domanda specifica: "Arrivato alla lampadina i, ne ho trovate accese almeno j?"
        - i --> A che punto sono della fila (lampadina 1, 2 o 3).
        - j --> Il totale che sto controllando (totale 1, totale 2...).
        Esempio Passo-Passo: Immaginiamo che le tue lampadine siano:
        - x_1: ACCESA (1)   
        - x_2: SPENTA (0)
        - x_3: ACCESA (1)
        Vediamo come si riempiono i registri del contatore sequenziale.
        1. Guardo la prima lampadina (x_1 = 1)
        - Domanda: Ho trovato almeno 1 lampadina accesa?
            - SÌ, perché x_1 è accesa. (Quindi il registro S_{1,1} diventa VERO).
        - Domanda: Ho trovato almeno 2 lampadine accese?
            - NO, ne ho vista solo una finora. (Quindi il registro S_{1,2} è FALSO).

        2. Passo alla seconda lampadina (x_2 = 0)
        Il contatore deve aggiornare i totali basandosi su quello che sapeva prima PIÙ la nuova lampadina.
        - Domanda: Ho trovato almeno 1 accesa finora? 
            - Il computer ragiona così: "O ne avevo già 1 prima, OPPURE ne avevo 0 prima e questa è accesa".
            - Ne avevo 1 prima? SÌ. Risultato: SÌ (il totale 1 "si conserva").
        - Domanda: Ho trovato almeno 2 accese finora?
            - Il computer ragiona: "O ne avevo già 2 prima, OPPURE ne avevo 1 prima e questa è accesa".
            - Ne avevo 2 prima? NO.
            - Ne avevo 1 prima (S_{1,1} era Vero) E questa (x_2) è accesa? NO, perché x_2 è spenta.
            - Risultato: NO (Restiamo fermi a 1).

        3. Passo alla terza lampadina (x_3 = 1)
        - Domanda: Ho trovato almeno 1 accesa?
            - Sì, lo sapevo già da prima. Resta SÌ.
        - Domanda: Ho trovato almeno 2 accese? 
            - Il ragionamento logico: "O ne avevo già 2 prima (S_{2,2}), OPPURE ne avevo almeno 1 prima (S_{2,1}) E la 
            lampadina attuale (x_3) è accesa".
            - Analisi:
                - Avevo già 2? No.
                - Avevo 1? SÌ. 
                - La x_3 è accesa? SÌ.
                - Quindi: SÌ + SÌ = Ho raggiunto quota 2!
                - Risultato: SÌ (S_{3,2} diventa VERO).
        
        La Regola Logica (La Formula):
        Ecco spiegata la formula complessa nel codice:S[i][j] = S[i-1][j] OR (S[i-1][j-1] AND x[i]) 
        Il totale attuale raggiunge quota J se:
            Caso A: Avevo già raggiunto quota J al passaggio precedente (non ho bisogno della lampadina attuale, ho già "vinto").
            OPPURE (OR)
            Caso B: Ero arrivato a quota J-1 (mi mancava 1) E (AND) la lampadina attuale mi dà proprio quel punto che mancava.
        
        Come faccio per dire "Al Massimo K"?
        Se voglio imporre che ci siano al massimo 2 lampadine accese (K=2), faccio tutto questo calcolo e alla fine guardo il registro 
        per il totale 3 (K+1). Se il registro "Ho trovato almeno 3 lampadine" (S_{n, 3}) si accende, significa che ho sforato.
        Quindi, la regola finale che do al SAT Solver è: NON (S_{n, 3}).
        In pratica: "Faccio tutte le combinazioni che vuoglio, ma assicurati che la casella del totale 3 rimanga SPENTA".
        """
    
        # Se k è maggiore o uguale a n, il vincolo è sempre soddisfatto (banale).
        if k >= self.n:
            return sympy.true
        
        # Mappa i nodi in una lista ordinata x_0, x_1, ...
        x = [self.node_vars[self.nodes[i]] for i in range(self.n)]
        
        # Variabili ausiliarie s[i][j]: indicano se "la somma dei primi i+1 variabili raggiunge almeno j".
        # Abbiamo bisogno di controllare fino a k+1 per rilevare se superiamo k.
        s = [[None for _ in range(k + 2)] for _ in range(self.n)]
        
        # Creazione dei simboli per le variabili ausiliarie
        for i in range(self.n):
            for j in range(1, k + 2):
                s[i][j] = symbols(f's_{i}_{j}')

        constraints = []

        # Caso Base (i = 0):
        # s[0][1] è vero se e solo se x[0] è vero.
        constraints.append( Equivalent(s[0][1], x[0]) )
        # s[0][j] per j > 1 è impossibile (non posso avere somma > 1 con un solo elemento).
        for j in range(2, k + 2):
            constraints.append( Not(s[0][j]) )

        # Passo Ricorsivo (i > 0):
        # La somma parziale al passo i raggiunge j se:
        # 1. Raggiungeva j già al passo i-1 (senza contare x[i]).
        # 2. Raggiungeva j-1 al passo i-1 E x[i] è vero (quindi aggiungo 1).
        for i in range(1, self.n):
            for j in range(1, k + 2):
                # s[i][j] <-> s[i-1][j] OR (s[i-1][j-1] AND x[i])
                prev_same = s[i-1][j]
                if j == 1:
                    prev_minus = x[i]
                else:
                    prev_minus = And(s[i-1][j-1], x[i])
                
                constraints.append( Equivalent(s[i][j], Or(prev_same, prev_minus)) )

        # Vincolo Finale:
        # NON deve essere vero che la somma totale raggiunga k+1.
        # Se s[n-1][k+1] fosse vero, avremmo scelto più di k nodi, violando il vincolo.
        constraints.append( Not(s[self.n - 1][k + 1]) )
        
        return And(*constraints)

    def solve(self):
        """
        Questa è la funzione che esegue la ricerca della soluzione ottima.
        Il problema chiede il Minimum Vertex Cover. Per trovarlo, usa la Ricerca Binaria sul numero k (da 0 a N).
        
        Ciclo while low <= high:
        1. Sceglie un numero medio mid.
        2. Costruisce la formula completa:
           - base_formula: Gli archi devono essere coperti.
           - cardinalità_formula: Devo usare al massimo mid nodi.
           - full_formula: And(base_formula, cardinalità_formula).
        3. satisfiable(full_formula): Chiede a sympy se esiste un modo per rendere vera questa formula.
        
        - Se SAT (trovata soluzione): Significa che è possibile coprire il grafo con mid nodi. 
          Ottimo! Ma forse si può fare con meno. Quindi salva la soluzione e prova con un k più basso 
          (high = mid - 1).
        
        - Se UNSAT (impossibile): Significa che mid nodi sono troppo pochi. 
          Bisogna alzare il tiro (low = mid + 1).
          
        Alla fine, restituisce l'insieme di nodi (min_cover) corrispondente alla soluzione più piccola trovata.
        """
        # Limite superiore: Tutti i nodi (banalmente una copertura valida)
        # Limite inferiore: 0
        low = 0
        high = self.n
        min_cover = None
        
        # 1. Formula Base (Gli archi devono sempre essere coperti)
        base_formula = self._encode_vertex_cover_constraint()
        
        # Ottimizzazione: Se il grafo non ha nodi, ritorna vuoto
        if self.n == 0:
            return set()
            
        print(f"SAT Solver: Cerco MVC esatto per grafo con {self.n} nodi. Vincoli: {self.graph.number_of_edges()} archi.")
        
        # Ricerca binaria
        while low <= high:
            mid = (low + high) // 2
            print(f"  Testo k={mid}...", end='', flush=True)
            
            # Costruisco la formula: (Copertura Archi) AND (Somma Nodi <= mid)
            cardinality_formula = self.get_at_most_k_formula(mid)
            full_formula = And(base_formula, cardinality_formula)
            
            # Chiamo il solver SymPy logic
            model = satisfiable(full_formula)
            
            if model:
                print(" SAT (Possibile)")
                # Ho trovato una copertura grande 'mid'.
                # Salvo i nodi trovati nel modello.
                current_cover = set()
                for node in self.nodes:
                    sym = self.node_vars[node]
                    if model.get(sym):
                        current_cover.add(node)
                
                min_cover = current_cover
                # Provo a vedere se esiste una soluzione più piccola
                high = mid - 1
            else:
                print(" UNSAT (Impossibile)")
                # 'mid' nodi non bastano, ne servono di più
                low = mid + 1
        
        return min_cover

def Equivalent(a, b):
    # Funzione Helper: Implementa l'equivalenza logica (A <-> B)
    # SymPy potrebbe non avere Equivalent esposto direttamente in modo semplice per logic.
    # A <-> B è equivalente a (A -> B) AND (B -> A), oppure (A AND B) OR (NOT A AND NOT B).
    return And(Or(Not(a), b), Or(Not(b), a))

if __name__ == "__main__":
    # Test Semplice
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    # Questo è un quadrato con una diagonale (1-3).
    # Nodi: 1, 2, 3, 4.
    # Un Minimum Vertex Cover dovrebbe essere di taglia 2. 
    # Esempio: {1, 3} copre (1,2), (1,4), (1,3), (2,3), (3,4). Corretto.
    
    solver = SatVertexCover(g)
    result = solver.solve()
    print("MVC Trovato:", result)
