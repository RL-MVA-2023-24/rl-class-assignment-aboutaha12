from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import pickle

import numpy as np
from tqdm import tqdm


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    # dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        # dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1, 1))
    R = np.array(R)
    S2 = np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D


def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S, A, axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter == 0:
            value = R.copy()
        else:
            Q2 = np.zeros((nb_samples, nb_actions))
            for a2 in range(nb_actions):
                A2 = a2 * np.ones((S.shape[0], 1))
                S2A2 = np.append(S2, A2, axis=1)
                Q2[:, a2] = Qfunctions[-1].predict(S2A2)
            max_Q2 = np.max(Q2, axis=1)
            value = R + gamma * (1 - D) * max_Q2
        # Q = RandomForestRegressor()
        Q = ExtraTreesRegressor(n_estimators=100, random_state=0)
        Q.fit(SA, value)
        Qfunctions.append(Q)
    return Qfunctions


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, Qvalue=None):
        self.nactions = env.action_space.n
        self.Qvalue = Qvalue

    def act(self, observation, use_random=False):
        # Act greedily
        Qsa = []
        for a in range(self.nactions):
            sa = np.append(observation, a).reshape(1, -1)
            Qsa.append(self.Qvalue.predict(sa))
        return np.argmax(Qsa)

    def save(self, path):
        with open(path, "wb") as model:
            pickle.dump(self.Qvalue, model)

    def load(self):
        with open("src/model_extratrees.pkl", "rb") as model:
            self.Qvalue = pickle.load(model)


if __name__ == "__main__":
    gamma = 0.9
    nb_iter = 300
    nb_actions = env.action_space.n
    S, A, R, S2, D = collect_samples(env, int(1e4))
    Qfunctions = rf_fqi(S, A, R, S2, D, nb_iter, nb_actions, gamma)
    Qvalue = Qfunctions[-1]
    agent = ProjectAgent(Qvalue)
    agent.save("src/model_extratrees.pkl")
