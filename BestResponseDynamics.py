import random

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
