from model import *


def train_dqn(agent, env, episodes=100, batch_size=32):
    scores = []
    for e in range(episodes):
        state, info = env.reset(seed=1)
        state = np.reshape(state, [agent.state_size])

        for time in range(200):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -1
            next_state = np.reshape(next_state, [agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
            if done or time == 199:
                print(f"episode: {e}/{episodes}, score: {time + 1}, e: {agent.epsilon:.2f}")
                scores.append(time + 1)
                break
    return scores
