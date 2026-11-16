class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_n = init

        for i in range(iterations):
            x_n = x_n - learning_rate * 2 * x_n
        
        return round(x_n, 5)
