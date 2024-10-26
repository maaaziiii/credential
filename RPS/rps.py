import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

class RockPaperScissors:
    def __init__(self):
        self.actions = ['rock', 'paper', 'scissors']
        self.rewards = {
            ('rock', 'rock'): 0,
            ('rock', 'paper'): -1,
            ('rock', 'scissors'): 1,
            ('paper', 'rock'): 1,
            ('paper', 'paper'): 0,
            ('paper', 'scissors'): -1,
            ('scissors', 'rock'): -1,
            ('scissors', 'paper'): 1,
            ('scissors', 'scissors'): 0
        }

    def get_reward(self, action1, action2):
        return self.rewards.get((action1, action2), 0)

class Game:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rock-Paper-Scissors")
        self.rps = RockPaperScissors()
        
        self.q_table = np.random.uniform(size=(3, 3))
        self.action_to_index = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

        
        self.score = 0
        self.opponent_score = 0
        self.train(1000)
        self.create_widgets()

    def train(self, episodes=100):
        """Train the Q-table using simulated games."""
        for _ in range(episodes):
            state = random.choice(self.rps.actions)
            done = False

            while not done:
                if np.random.rand() < self.epsilon:
                    action = random.choice(self.rps.actions)  
                else:
                    action = self.rps.actions[np.argmax(self.q_table[self.action_to_index[state]])]  
                opponent_action = random.choice(self.rps.actions)
                reward = self.rps.get_reward(action, opponent_action)

                current_q = self.q_table[self.action_to_index[state], self.action_to_index[action]]
                max_future_q = np.max(self.q_table[self.action_to_index[opponent_action]])
                new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
                self.q_table[self.action_to_index[state], self.action_to_index[action]] = new_q

                state = opponent_action  # Move to the next state

                if np.random.rand() < 0.1:  
                    done = True

    def create_widgets(self):
        """Create buttons and labels for the game interface."""
        tk.Button(self.root, text="Rock", command=lambda: self.play('rock')).pack(pady=5)
        tk.Button(self.root, text="Paper", command=lambda: self.play('paper')).pack(pady=5)
        tk.Button(self.root, text="Scissors", command=lambda: self.play('scissors')).pack(pady=5)
        self.score_label = tk.Label(self.root, text="Your score: 0, Opponent's score: 0")
        self.score_label.pack(pady=5)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=5)

    def play(self, action):
        """Play a round and train after each competition."""
        # Opponent's action based on the current Q-table
        opponent_action = self.rps.actions[np.argmax(self.q_table[self.action_to_index[action]])]
        reward = self.rps.get_reward(action, opponent_action)
        if reward == 1:
            self.score += 1
            result_text = "You win this round!"
        elif reward == -1:
            self.opponent_score += 1
            result_text = "Opponent wins this round!"
        else:
            result_text = "It's a tie!"
            
        self.result_label.config(text=result_text)
        self.score_label.config(text=f"Your score: {self.score}, Opponent's score: {self.opponent_score}")
        self.train(10)

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    game = Game()
    game.run()
