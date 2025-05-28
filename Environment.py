import random

class NumberStabilizer:
    def __init__(self, start_number=None):
        self.done = False
        self.state = start_number
        
    def set_state(self):
        return random.randint(-10, 10)
             
    def reset(self, value=None):
        self.state = value if value is not None else self.set_state()
        self.done = False
        return self.state
    
    def step(self, action, reward=0):
        """
        Actions: +1 or -1
        Return: (New_State, reward, done=True)
        """

        self.state += action

        if -1<=self.state<=1:
            reward = 1
            self.done = True
        else:
            reward = -1

        return self.state, reward, self.done

    def render(self):
        print(f"Number: {self.state}")

