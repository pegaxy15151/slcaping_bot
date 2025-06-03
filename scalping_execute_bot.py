import time
import numpy as np
import pandas as pd
import ccxt
import ast
import os
from stable_baselines3 import PPO
from scalping_binance_env_fixed import ScalpingBinanceEnv
import warnings
import traceback

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === CONFIGURAÇÃO ===
API_KEY = '7a1cbcb710e41a5999ce0e01a7cf22666066849fed712f5ddd77bb130684ce70'
API_SECRET = 'e08378154a7a2e76f47daff2b448bf871802c4527da21a344e2f95846de94449'
SYMBOL = 'ETH/USDT'
TRADE_AMOUNT = 0.01  # em ETH
LEVERAGE = 10  # Alavancagem fixa de 10x conforme solicitado
SIMULATION_MODE = False  # Defina como False para conectar à Binance real
FORCE_DIM = 12  # Forçar dimensão da observação para compatibilidade com o modelo

# === CONEXÃO COM BINANCE FUTURES TESTNET ===
if not SIMULATION_MODE:
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        exchange.set_sandbox_mode(True)
        print("✓ Conectado à Binance Futures Testnet")

        # Configurar alavancagem - método 1 (API direta)
        try:
            # Formato correto para a API de Futures (sem a barra)
            symbol_for_leverage = SYMBOL.replace('/', '')
            exchange.fapiPrivatePostLeverage({
                'symbol': symbol_for_leverage,
                'leverage': LEVERAGE
            })
            print(f"✓ Alavancagem definida: {LEVERAGE}x")
        except Exception as e:
            print(f"! Erro ao configurar alavancagem (método 1): {e}")
            
            # Método 2 (método da biblioteca)
            try:
                exchange.set_leverage(LEVERAGE, SYMBOL)
                print(f"✓ Alavancagem definida (método 2): {LEVERAGE}x")
            except Exception as e2:
                print(f"! Erro ao configurar alavancagem (método 2): {e2}")
                
                # Método 3 (formato alternativo do símbolo)
                try:
                    exchange.set_leverage(LEVERAGE, symbol_for_leverage)
                    print(f"✓ Alavancagem definida (método 3): {LEVERAGE}x")
                except Exception as e3:
                    print(f"! Erro ao configurar alavancagem (método 3): {e3}")
                    print("! Continuando sem configurar alavancagem...")
    except Exception as e:
        print(f"! Erro ao conectar à Binance: {e}")
        print("! Ativando modo de simulação")
        SIMULATION_MODE = True
else:
    print("✓ Modo de simulação ativado (sem conexão com Binance)")

# === CARREGA MELHOR CONFIG ===
# Lista de caminhos para procurar configurações
config_paths = [
    "optuna_best_model/best_config.txt",
    "final_checkpoints/best_config.txt",
    "scalping_checkpoints/best_config.txt"
]

indicators = None
for path in config_paths:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            indicators = ast.literal_eval(lines[0].split(":", 1)[1].strip())
            print(f"✓ Configuração carregada de {path}: {indicators}")
            break
    except Exception as e:
        continue

# Se não encontrou configuração, tenta carregar diretamente dos indicadores de treinamento
if indicators is None:
    train_indicator_paths = [
        "final_checkpoints/training_indicators.txt",
        "scalping_checkpoints/training_indicators.txt",
        "optuna_best_model/training_indicators.txt"
    ]
    
    for train_path in train_indicator_paths:
        try:
            if os.path.exists(train_path):
                with open(train_path, "r") as f:
                    indicators = ast.literal_eval(f.read().strip())
                print(f"✓ Indicadores carregados de {train_path}: {indicators}")
                break
        except Exception as e:
            continue

# Se ainda não encontrou, usa indicadores padrão
if indicators is None:
    indicators = ["close", "volume", "atr", "stoch_rsi", "obv", "ichimoku"]
    print(f"! Usando indicadores padrão: {indicators}")

# === MODELO E AMBIENTE ===
# Verificar e criar diretórios se não existirem
os.makedirs("scalping_checkpoints", exist_ok=True)
os.makedirs("final_checkpoints", exist_ok=True)

# Tentar carregar o modelo em diferentes locais
model_paths = [
    "final_checkpoints/ppo_final_latest",  # Último modelo salvo
    "final_checkpoints/ppo_final",  # Caminho principal do treinamento
    "scalping_checkpoints/model",  # Caminho alternativo - sem extensão
    "scalping_checkpoints/model.zip",  # Caminho alternativo - com extensão
]

model = None
for path in model_paths:
    try:
        if os.path.exists(path) or os.path.exists(f"{path}.zip"):
            model = PPO.load(path)
            print(f"✓ Modelo carregado: {path}")
            
            # Extrair informações do modelo para debug
            observation_space = model.observation_space
            print(f"✓ Dimensão esperada pelo modelo: {observation_space.shape}")
            break
    except Exception as e:
        print(f"! Erro ao carregar modelo de {path}: {e}")
        continue

# Se nenhum modelo foi encontrado, criar um modelo dummy
if model is None:
    print("! Nenhum modelo encontrado. Criando modelo dummy para teste.")
    env = ScalpingBinanceEnv(indicators=indicators, force_dim=FORCE_DIM)
    model = PPO("MlpPolicy", env, verbose=0)
    print("! Modelo dummy criado. Recomendado executar o treinamento primeiro.")
else:
    # Criar ambiente com os mesmos indicadores usados no treinamento
    # e forçar a dimensão para corresponder ao modelo
    env = ScalpingBinanceEnv(indicators=indicators, force_dim=FORCE_DIM)
    print(f"✓ Ambiente criado com indicadores: {indicators}")

# Verificar dimensões
obs, _ = env.reset()
print(f"✓ Dimensão da observação do ambiente: {obs.shape}")

# Verificar compatibilidade
expected_shape = model.observation_space.shape[0]
actual_shape = obs.shape[0]

if expected_shape != actual_shape:
    print(f"! ALERTA: Incompatibilidade de dimensões! Modelo espera {expected_shape}, ambiente fornece {actual_shape}")
    print("! Recriando ambiente com dimensão forçada...")
    
    # Recriar ambiente com dimensão forçada
    env = ScalpingBinanceEnv(indicators=indicators, force_dim=expected_shape)
    obs, _ = env.reset()
    print(f"✓ Ambiente recriado. Nova dimensão: {obs.shape}")

# === ESTADO DO BOT ===
position = 0  # 0 = neutro, 1 = comprado, -1 = vendido
entry_price = 0
total_episodios = 0
episodios_lucrativos = 0
episodios_prejuizo = 0
step = 0
entry_step = 0

# === MÉTRICAS AVANÇADAS ===
total_gain = 0
total_loss = 0
max_drawdown = 0
current_drawdown = 0
peak_value = 0
current_value = 0
consecutive_wins = 0
consecutive_losses = 0
max_consecutive_wins = 0
max_consecutive_losses = 0
trades_duration = []  # Duração dos trades em steps

print("→ Iniciando execução contínua com ordens reais na Testnet...")
print("→ Estratégia: Scalping Puro (modelo decide entradas e saídas) com alavancagem 10x")

try:
    while True:
        step += 1
        
        # Verificar dimensão da observação antes de prever
        if obs.shape[0] != expected_shape:
            print(f"! Erro: Dimensão da observação ({obs.shape[0]}) não corresponde ao esperado pelo modelo ({expected_shape})")
            print("! Ajustando observação...")
            
            # Opção 1: Preencher com zeros se faltar dimensões
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
                print(f"✓ Observação preenchida para dimensão {obs.shape}")
            # Opção 2: Truncar se tiver dimensões demais
            elif obs.shape[0] > expected_shape:
                obs = obs[:expected_shape]
                print(f"✓ Observação truncada para dimensão {obs.shape}")
        
        # Obter previsão do modelo
        try:
            action, _states = model.predict(obs.reshape(1, -1), deterministic=False)
            
            # Obter probabilidades das ações para calcular confiança
            try:
                action_probs = model.policy.get_distribution(obs.reshape(1, -1)).distribution.probs.detach().numpy()[0]
                confidence = action_probs[action]
            except:
                confidence = 0.5  # Valor padrão se não conseguir obter a confiança
        except Exception as e:
            print(f"! Erro ao prever ação: {e}")
            print("! Usando ação padrão (hold)")
            action = 0
            confidence = 0.0
        
        price = env.data.iloc[env.current_step]['close']
        
        try:
            if action == 1 and position == 0:  # Comprar
                print(f"+ BUY @ {price:.2f} (confiança: {confidence:.2f})")
                
                if not SIMULATION_MODE:
                    try:
                        # Criar ordem de mercado para entrada
                        order = exchange.create_market_buy_order(SYMBOL, TRADE_AMOUNT)
                        entry_price = float(order['price']) if 'price' in order else price
                    except Exception as e:
                        print(f"! Erro ao executar ordem de compra: {e}")
                        entry_price = price
                else:
                    # Simulação
                    print(f"→ SIMULAÇÃO: Executando ordem de compra @ {price:.2f}")
                    entry_price = price
                
                entry_step = step
                position = 1
                
            elif action == 2 and position == 0:  # Vender
                print(f"- SELL @ {price:.2f} (confiança: {confidence:.2f})")
                
                if not SIMULATION_MODE:
                    try:
                        # Criar ordem de mercado para entrada
                        order = exchange.create_market_sell_order(SYMBOL, TRADE_AMOUNT)
                        entry_price = float(order['price']) if 'price' in order else price
                    except Exception as e:
                        print(f"! Erro ao executar ordem de venda: {e}")
                        entry_price = price
                else:
                    # Simulação
                    print(f"→ SIMULAÇÃO: Executando ordem de venda @ {price:.2f}")
                    entry_price = price
                
                entry_step = step
                position = -1
                
            elif action == 0 and position != 0:  # Fechar posição
                reward = 0
                trade_duration = step - entry_step
                trades_duration.append(trade_duration)
                
                if position == 1:  # Fechar compra
                    reward = price - entry_price
                    print(f"○ CLOSE BUY @ {price:.2f} | Lucro: {reward:.2f} | Duração: {trade_duration} steps")
                    
                    if not SIMULATION_MODE:
                        try:
                            exchange.create_market_sell_order(SYMBOL, TRADE_AMOUNT)
                        except Exception as e:
                            print(f"! Erro ao fechar posição comprada: {e}")
                    else:
                        print(f"→ SIMULAÇÃO: Fechando posição comprada @ {price:.2f}")
                        
                elif position == -1:  # Fechar venda
                    reward = entry_price - price
                    print(f"○ CLOSE SELL @ {price:.2f} | Lucro: {reward:.2f} | Duração: {trade_duration} steps")
                    
                    if not SIMULATION_MODE:
                        try:
                            exchange.create_market_buy_order(SYMBOL, TRADE_AMOUNT)
                        except Exception as e:
                            print(f"! Erro ao fechar posição vendida: {e}")
                    else:
                        print(f"→ SIMULAÇÃO: Fechando posição vendida @ {price:.2f}")

                # Estatísticas
                total_episodios += 1
                if reward > 0:
                    episodios_lucrativos += 1
                    total_gain += reward
                    consecutive_losses = 0
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                elif reward < 0:
                    episodios_prejuizo += 1
                    total_loss += abs(reward)
                    consecutive_wins = 0
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

                # Atualizar métricas de drawdown
                current_value += reward
                if current_value > peak_value:
                    peak_value = current_value
                    current_drawdown = 0
                else:
                    current_drawdown = peak_value - current_value
                    max_drawdown = max(max_drawdown, current_drawdown)

                position = 0
                entry_price = 0

        except Exception as e:
            print(f"! Erro ao executar ordem: {e}")
            print(traceback.format_exc())

        # Executar passo no ambiente e verificar dimensão da nova observação
        try:
            obs, reward, done, truncated, info = env.step(int(action))
        except Exception as e:
            print(f"! Erro ao executar passo no ambiente: {e}")
            print("! Tentando resetar o ambiente...")
            try:
                obs, _ = env.reset()
                done = False
                truncated = False
            except Exception as e2:
                print(f"! Erro ao resetar ambiente: {e2}")
                print("! Recriando ambiente...")
                env = ScalpingBinanceEnv(indicators=indicators, force_dim=expected_shape)
                obs, _ = env.reset()
                done = False
                truncated = False

        if done or truncated:
            try:
                obs, _ = env.reset()
            except Exception as e:
                print(f"! Erro ao resetar ambiente após done/truncated: {e}")
                env = ScalpingBinanceEnv(indicators=indicators, force_dim=expected_shape)
                obs, _ = env.reset()
            
            # Se ainda tiver posição aberta ao final dos dados, fechar
            if position != 0:
                try:
                    if position == 1:
                        print(f"! Fechando posição comprada no final dos dados @ {price:.2f}")
                        if not SIMULATION_MODE:
                            try:
                                exchange.create_market_sell_order(SYMBOL, TRADE_AMOUNT)
                            except Exception as e:
                                print(f"! Erro ao fechar posição comprada final: {e}")
                        else:
                            print(f"→ SIMULAÇÃO: Fechando posição comprada final @ {price:.2f}")
                    elif position == -1:
                        print(f"! Fechando posição vendida no final dos dados @ {price:.2f}")
                        if not SIMULATION_MODE:
                            try:
                                exchange.create_market_buy_order(SYMBOL, TRADE_AMOUNT)
                            except Exception as e:
                                print(f"! Erro ao fechar posição vendida final: {e}")
                        else:
                            print(f"→ SIMULAÇÃO: Fechando posição vendida final @ {price:.2f}")
                except Exception as e:
                    print(f"! Erro ao fechar posição final: {e}")
            
            position = 0
            entry_price = 0

        # Imprimir estatísticas a cada 50 passos
        if step % 50 == 0:
            print("\n→ ESTATÍSTICAS PARCIAIS:")
            if total_episodios > 0:
                pct_lucro = (episodios_lucrativos / total_episodios) * 100
                pct_preju = (episodios_prejuizo / total_episodios) * 100
                avg_duration = sum(trades_duration) / len(trades_duration) if trades_duration else 0
                profit_factor = total_gain / total_loss if total_loss > 0 else float('inf')
                
                print(f"→ Operações com fechamento: {total_episodios}")
                print(f"✓ Lucros: {episodios_lucrativos} ({pct_lucro:.2f}%)")
                print(f"! Prejuízos: {episodios_prejuizo} ({pct_preju:.2f}%)")
                print(f"+ Ganho total: {total_gain:.4f}")
                print(f"- Perda total: {total_loss:.4f}")
                print(f"→ Fator de lucro: {profit_factor:.2f}")
                print(f"→ Drawdown máximo: {max_drawdown:.4f}")
                print(f"→ Duração média dos trades: {avg_duration:.1f} steps")
                print(f"→ Sequência máxima de ganhos: {max_consecutive_wins}")
                print(f"→ Sequência máxima de perdas: {max_consecutive_losses}")
            
        time.sleep(1)  # Reduzido para simulação mais rápida

except KeyboardInterrupt:
    print("\n! Execução interrompida manualmente.")

    print("\n→ RESULTADO FINAL:")
    print(f"→ Operações com fechamento: {total_episodios}")
    print(f"✓ Lucros: {episodios_lucrativos}")
    print(f"! Prejuízos: {episodios_prejuizo}")

    if total_episodios > 0:
        pct_lucro = (episodios_lucrativos / total_episodios) * 100
        pct_preju = (episodios_prejuizo / total_episodios) * 100
        profit_factor = total_gain / total_loss if total_loss > 0 else float('inf')
        avg_duration = sum(trades_duration) / len(trades_duration) if trades_duration else 0

        print(f"→ % de lucro: {pct_lucro:.2f}%")
        print(f"→ % de prejuízo: {pct_preju:.2f}%")
        print(f"+ Ganho total: {total_gain:.4f}")
        print(f"- Perda total: {total_loss:.4f}")
        print(f"→ Fator de lucro: {profit_factor:.2f}")
        print(f"→ Drawdown máximo: {max_drawdown:.4f}")
        print(f"→ Duração média dos trades: {avg_duration:.1f} steps")
        print(f"→ Sequência máxima de ganhos: {max_consecutive_wins}")
        print(f"→ Sequência máxima de perdas: {max_consecutive_losses}")
    else:
        print("! Nenhuma operação foi encerrada.")

# Cancelar todas as ordens pendentes ao encerrar (por segurança)
if not SIMULATION_MODE:
    try:
        exchange.cancel_all_orders(SYMBOL)
        print("✓ Todas as ordens pendentes foram canceladas.")
    except Exception as e:
        print(f"! Erro ao cancelar ordens pendentes: {e}")
