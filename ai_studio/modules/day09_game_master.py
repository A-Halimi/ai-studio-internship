"""Day 9 module - Game Master.

Unlocked by: solutions to Day 9 (reinforcement learning).
The notebook exports:
  ai_studio/models/game_master_qtable.npy   16x4 Q-table (FrozenLake, not slippery)
  ai_studio/models/game_master_dqn.zip      stable-baselines3 DQN (CartPole-v1)
"""
TITLE = "🎮 Game Master"
REQUIRES = ["game_master_qtable.npy", "game_master_dqn.zip"]


def _to_gif(frames, fps):
    import tempfile

    import imageio.v2 as imageio

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    imageio.mimsave(tmp.name, frames, fps=fps, loop=0)
    return tmp.name


def render(models_dir):
    import gradio as gr
    import numpy as np

    q_table = np.load(models_dir / "game_master_qtable.npy")

    def frozenlake_run():
        import gymnasium as gym

        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,
                       render_mode="rgb_array")
        state, _ = env.reset(seed=0)
        frames = [env.render()]
        total, done = 0.0, False
        for _ in range(30):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            total += reward
            if terminated or truncated:
                done = True
                break
        env.close()
        msg = ("🏆 **The agent reached the gift!** It follows the Q-table "
               "it learned — no luck involved."
               if total > 0 else
               "🕳️ Oops — it fell in a hole. Retrain it in the notebook!")
        return _to_gif(frames, fps=3), msg

    def cartpole_run(trained):
        import gymnasium as gym

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        model = None
        if trained:
            from stable_baselines3 import DQN
            model = DQN.load(str(models_dir / "game_master_dqn.zip"),
                             device="cpu")
        obs, _ = env.reset(seed=42)
        frames, total = [], 0.0
        for step in range(500):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if step % 2 == 0:            # every 2nd frame keeps the GIF small
                frames.append(env.render())
            if terminated or truncated:
                break
        env.close()
        who = "your trained DQN" if trained else "an untrained random agent"
        return (_to_gif(frames, fps=25),
                f"**{who}** balanced the pole for **{int(total)} steps** "
                f"(500 = perfect).")

    gr.Markdown(
        "Two agents **you trained on Day 9** with reinforcement learning — "
        "no examples, no labels, just rewards."
    )
    with gr.Tabs():
        with gr.Tab("🧊 FrozenLake (Q-table)"):
            gr.Markdown("A tiny brain: 16 states x 4 actions = a 64-number "
                        "Q-table. Watch it walk the ice without falling in.")
            b1 = gr.Button("Run my agent 🧊", variant="primary")
            g1 = gr.Image(label="Episode replay")
            m1 = gr.Markdown()
            b1.click(frozenlake_run, None, [g1, m1])

        with gr.Tab("🎪 CartPole (Deep Q-Network)"):
            gr.Markdown("A neural network learned to balance a pole by "
                        "playing thousands of episodes. Compare it to a "
                        "random agent!")
            with gr.Row():
                b_rand = gr.Button("Random agent 🎲")
                b_dqn = gr.Button("My trained DQN 🧠", variant="primary")
            g2 = gr.Image(label="Episode replay")
            m2 = gr.Markdown()
            b_rand.click(lambda: cartpole_run(False), None, [g2, m2])
            b_dqn.click(lambda: cartpole_run(True), None, [g2, m2])
