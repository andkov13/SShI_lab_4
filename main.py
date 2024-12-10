import numpy as np
import gym
import matplotlib.pyplot as plt

# Середовище Taxi-v3
env = gym.make('Taxi-v3')

# Параметри Q-learning
alpha = 0.1  # Швидкість навчання
gamma = 0.9  # Discount factor
epsilon = 0.1  # Коефіцієнт "випадковості"
num_epochs = 5000  # Кількість епох

# Винагороди (ДЛЯ ТЕСТІВ)
action_rewards = {
    -1: -1,  # Штраф за крок
    20: 20,  # Винагорода за доставку пасажира
    -10: -10  # Штраф за нелегальне виконання дій "pickup" та "drop-off"
}

def q_learning(env, alpha, gamma, epsilon, num_epochs, action_rewards):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    results = []
    penalties_per_epoch = []
    steps_per_epoch = []

    for epoch in range(num_epochs):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        penalties = 0
        steps = 0
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # Коригування винагороди для дії
            if reward == -10:
                penalties += 1
            reward = action_rewards.get(reward, reward)

            # Оновлення Q-таблиці
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            state = next_state
            total_reward += reward
            steps += 1

        results.append(total_reward)
        penalties_per_epoch.append(penalties)
        steps_per_epoch.append(steps)


    return q_table, results, penalties_per_epoch, steps_per_epoch

q_table, results, penalties_per_epoch, steps_per_epoch = q_learning(env, alpha, gamma, epsilon, num_epochs,
                                                                    action_rewards)

# Візуалізація результатів
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(results, label='Rewards per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Rewards(Epoch)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(penalties_per_epoch, label='Penalties per Epoch', color='red')
plt.xlabel('Epoch')
plt.ylabel('Penalties')
plt.title('Penalties(Epoch)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(steps_per_epoch, label='Steps per Epoch', color='green')
plt.xlabel('Epoch')
plt.ylabel('Steps')
plt.title('Steps(Epoch)')
plt.legend()

plt.tight_layout()
plt.show()


# Тестування на 1000 іграх
def test_q_learning(env, q_table, num_games=1000):
    total_rewards = []
    total_penalties = 0
    total_steps = 0

    for _ in range(num_games):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        rewards = 0
        penalties = 0
        steps = 0

        while not done:
            action = np.argmax(q_table[state])

            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            if reward == -10:
                penalties += 1

            rewards += reward
            state = next_state
            steps += 1

        total_rewards.append(rewards)
        total_penalties += penalties
        total_steps += steps

    avg_reward = np.mean(total_rewards)
    avg_penalties = total_penalties / num_games
    avg_steps = total_steps / num_games

    return avg_reward, avg_penalties, avg_steps


avg_reward, avg_penalties, avg_steps = test_q_learning(env, q_table)

print(f"Середня нагорода за 1000 ігор: {avg_reward}")
print(f"Середня к-сть штрафів за 1000 ігор: {avg_penalties}")
print(f"Середня кількість кроків за гру: {avg_steps}")