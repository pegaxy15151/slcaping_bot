import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scalping_binance_env import ScalpingBinanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ========== CONFIGURAÇÃO ==========
TIMESTEPS = 3_000_000
CHECKPOINT_DIR = "final_checkpoints"
REWARD_LOGS = "final_reward_logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REWARD_LOGS, exist_ok=True)

# ========== LER MELHOR CONFIG DO OPTUNA ==========
try:
    with open("optuna_best_model/best_config.txt", "r") as f:
        lines = f.readlines()

    indicators = ast.literal_eval(lines[0].split(":", 1)[1].strip())
    learning_rate = float(lines[1].split(":")[1].strip())
    gamma = float(lines[2].split(":")[1].strip())
    ent_coef = float(lines[3].split(":")[1].strip())
except Exception as e:
    print(f"⚠️ Erro ao carregar configuração: {e}")
    indicators = ["ema", "rsi", "macd", "bollinger", "vwap", "atr", "stoch_rsi"]
    learning_rate = 0.0003
    gamma = 0.99
    ent_coef = 0.01
    print(f"⚠️ Usando configuração padrão")

# Garantir que atr e stoch_rsi estejam incluídos nos indicadores
if "atr" not in indicators:
    indicators.append("atr")
if "stoch_rsi" not in indicators:
    indicators.append("stoch_rsi")

print(f"\n✅ Configuração carregada:")
print(f"Indicadores: {indicators}")
print(f"learning_rate: {learning_rate}, gamma: {gamma}, ent_coef: {ent_coef}")

# Salvar indicadores para uso posterior pelo bot de execução
with open(f"{CHECKPOINT_DIR}/training_indicators.txt", "w") as f:
    f.write(str(indicators))
print(f"✅ Indicadores salvos em {CHECKPOINT_DIR}/training_indicators.txt")

# ========== CALLBACK DE LOG ==========
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path="reward_log.npy", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    ep_rew = self.locals["infos"][i].get("episode", {}).get("r")
                    if ep_rew is not None:
                        self.rewards.append(ep_rew)
        return True

    def _on_training_end(self):
        np.save(self.log_path, np.array(self.rewards))

# ========== AMBIENTE + MODELO ==========
env = ScalpingBinanceEnv(indicators=indicators)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    ent_coef=ent_coef,
    verbose=1
)

# ========== TREINAMENTO ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
reward_log_path = f"{REWARD_LOGS}/rewards_final_{timestamp}.npy"
checkpoint_path = f"{CHECKPOINT_DIR}/ppo_final_{timestamp}"
latest_path = f"{CHECKPOINT_DIR}/ppo_final_latest"
callback = RewardLoggerCallback(log_path=reward_log_path)

print("\n🚀 Iniciando treinamento final...")
model.learn(total_timesteps=TIMESTEPS, callback=callback)
model.save(checkpoint_path)
model.save(latest_path)  # Salvar também com nome fixo para facilitar carregamento

# ========== GRÁFICO ==========
rewards = np.load(reward_log_path)
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Recompensa por Episódio", color="green")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.title("Treinamento Final")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_plot_final.png")
plt.show()

print(f"\n✅ Modelo salvo: {checkpoint_path}")
print(f"✅ Modelo salvo também como: {latest_path}")
print("📊 Gráfico salvo como 'reward_plot_final.png'")
