import random

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
        history = [] # Track convergence

        for t in range(self.max_iterations):
            # Record current set size
            history.append(sum(self.current_strategies.values()))

            self.current_strategies = {i: self._get_action(i) for i in range(self.num_players)}
            
            for i in range(self.num_players):
                u_actual = self.game.get_payoff(i, self.current_strategies)
                
                actual_strat = self.current_strategies[i]
                other_strat = 1 - actual_strat # What was the other option? (If I did 0, the other is 1. If I did 1, the other is 0).
                
                # Create a copy of the game where I play the other move (strategy), 
                # leaving other players' moves unchanged
                temp_profile = self.current_strategies.copy()
                temp_profile[i] = other_strat
                u_counterfactual = self.game.get_payoff(i, temp_profile)
                
                # calculate regret, if positive I regret not playing the other move
                regret = u_counterfactual - u_actual
                self.cumulative_regrets[i][other_strat] += regret

            for i in range(self.num_players):
                # if regret is negative, consider it 0 (no regrets for not playing that move)
                r_0_pos = max(0, self.cumulative_regrets[i][0])
                r_1_pos = max(0, self.cumulative_regrets[i][1])
                sum_r = r_0_pos + r_1_pos
                
                # normalization: transform regrets into percentages.
                if sum_r > 0:
                    self.strategy_probs[i][0] = r_0_pos / sum_r
                    self.strategy_probs[i][1] = r_1_pos / sum_r
                else:
                    self.strategy_probs[i][0] = 0.5
                    self.strategy_probs[i][1] = 0.5
        
        print(f"Regret Matching finished after {self.max_iterations} iterations.")
        
        final_pure_strategies = {}
        # Transform probabilities into a hard choice.
        # If the probability of playing 1 is greater than 50%, my final strategy is 1. Otherwise it is 0.
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
        
        return final_pure_strategies, is_pne, history