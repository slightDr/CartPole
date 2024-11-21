from module import *


def train_dqn(agent, env, episodes=1000, batch_size=32):
    for e in range(episodes):
        state, info = env.reset()
        # print(state, agent.state_size)
        state = np.reshape(state, [agent.state_size])
        has_done = False
        for time in range(300):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
            if done:
                has_done = True
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2f}")
                break
        if not has_done:
            print(f"episode: {e}/{episodes}, score: {300}, e: {agent.epsilon:.2f}")