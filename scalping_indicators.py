import os
import time
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import json
from datetime import datetime
from scalping_binance_env import ScalpingBinanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import warnings
import traceback
import signal
import sys

# Configuração de logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/optuna_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scalping_indicators")

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Criar diretórios para salvar resultados
os.makedirs("optuna_best_model", exist_ok=True)
os.makedirs("optuna_results", exist_ok=True)
os.makedirs("final_checkpoints", exist_ok=True)

# Lista de todos os indicadores disponíveis
ALL_INDICATORS = ["ema", "rsi", "macd", "bollinger", "vwap", "atr", "stoch_rsi", "obv", "momentum", "ichimoku"]

# Configurações globais
CONFIG = {
    "n_trials": 30,
    "timeout": None,  # Tempo máximo em segundos (None = sem limite)
    "n_jobs": 1,      # Número de jobs paralelos (1 = sequencial)
    "n_startup_trials": 5,  # Número de trials aleatórios antes de usar o modelo probabilístico
    "optimization_timesteps": 17000,  # Passos de treinamento para cada trial
    "evaluation_episodes": 10,  # Episódios para avaliação durante otimização
    "final_evaluation_episodes": 20,  # Episódios para avaliação final
    "pruning_enabled": True,  # Habilitar pruning de trials ruins
    "pruning_patience": 5,    # Número de trials ruins antes de considerar pruning
    "required_indicators": ["atr", "stoch_rsi"],  # Indicadores que devem estar sempre presentes
    "advanced_sampler": True,  # Usar amostrador avançado (CMA-ES após trials iniciais)
    "save_checkpoints": True,  # Salvar checkpoints intermediários
    "plot_all_visualizations": True,  # Gerar visualizações avançadas
    "auto_analyze_results": True,  # Analisar resultados automaticamente
    "indicator_penalty_factor": 0.01,  # Fator de penalidade para muitos indicadores
    "early_stopping_patience": 10000,  # Passos sem melhoria antes de parar treinamento
}

# Variáveis globais para controle de interrupção
interrupted = False

# Handler para interrupção segura
def signal_handler(sig, frame):
    global interrupted
    if not interrupted:
        logger.warning("Interrupcao detectada! Finalizando trials atuais e salvando resultados...")
        interrupted = True
    else:
        logger.warning("Forcando encerramento imediato!")
        sys.exit(1)

# Registrar handler para CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# Função para avaliar o modelo com um conjunto específico de indicadores
def evaluate_indicators(trial, indicators, n_eval_episodes=10, timesteps=None):
    """
    Avalia um modelo com um conjunto específico de indicadores.
    
    Args:
        trial: Trial do Optuna para pruning
        indicators: Lista de indicadores a serem usados
        n_eval_episodes: Número de episódios para avaliação
        timesteps: Número de passos de treinamento (usa CONFIG se None)
        
    Returns:
        mean_reward: Recompensa média
        std_reward: Desvio padrão da recompensa
        env: Ambiente usado para avaliação
    """
    if timesteps is None:
        timesteps = CONFIG["optimization_timesteps"]
    
    # Validar indicadores
    valid_indicators = ["close", "volume"]
    for ind in indicators:
        if ind in ALL_INDICATORS:
            valid_indicators.append(ind)
    
    # Garantir que os indicadores obrigatórios estejam presentes
    for req_ind in CONFIG["required_indicators"]:
        if req_ind not in valid_indicators and req_ind in ALL_INDICATORS:
            valid_indicators.append(req_ind)
    
    try:
        # Criar ambiente com os indicadores selecionados
        env = ScalpingBinanceEnv(indicators=valid_indicators)
        
        # Extrair hiperparâmetros do trial se disponíveis
        if trial and hasattr(trial, 'params'):
            learning_rate = trial.params.get('learning_rate', 3e-4)
            gamma = trial.params.get('gamma', 0.99)
            ent_coef = trial.params.get('ent_coef', 0.01)
            n_steps = trial.params.get('n_steps', 2048)
            clip_range = trial.params.get('clip_range', 0.2)
        else:
            learning_rate = 3e-4
            gamma = 0.99
            ent_coef = 0.01
            n_steps = 2048
            clip_range = 0.2
        
        # Criar modelo PPO com treinamento curto para otimização
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_range=clip_range,
            verbose=0
        )
        
        # Configurar callback para early stopping
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=CONFIG["early_stopping_patience"],
            min_evals=int(timesteps/10),
            verbose=0
        )
        
        eval_env = ScalpingBinanceEnv(indicators=valid_indicators)
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=stop_train_callback,
            eval_freq=max(int(timesteps/10), 1000),
            n_eval_episodes=5,
            verbose=0,
            deterministic=True
        )
        
        # Treinamento curto para otimização
        model.learn(
            total_timesteps=timesteps, 
            progress_bar=False,
            callback=eval_callback
        )
        
        # Avaliar modelo
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        
        # Pruning (parar trials ruins precocemente)
        if trial and CONFIG["pruning_enabled"]:
            trial.report(mean_reward, timesteps)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return mean_reward, std_reward, env
    
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Erro na avaliacao: {e}")
        logger.debug(traceback.format_exc())
        raise

# Função objetivo para otimização Optuna
def objective(trial):
    """
    Função objetivo para otimização Optuna.
    
    Args:
        trial: Trial do Optuna
        
    Returns:
        score: Pontuação do trial
    """
    global interrupted
    if interrupted:
        raise optuna.exceptions.TrialPruned()
    
    # Registrar início do trial
    trial_start_time = time.time()
    logger.info(f"Iniciando Trial {trial.number}")
    
    try:
        # Selecionar indicadores
        indicators = ["close", "volume"]
        
        # Decidir quais indicadores técnicos incluir
        for indicator in ALL_INDICATORS:
            if trial.suggest_categorical(f"use_{indicator}", [True, False]):
                indicators.append(indicator)
        
        # Garantir que pelo menos um indicador técnico seja selecionado
        if len(indicators) <= 2:  # Se só tiver close e volume
            indicators.append(ALL_INDICATORS[0])  # Adicionar pelo menos um indicador
        
        # Garantir que os indicadores obrigatórios estejam sempre incluídos
        for req_ind in CONFIG["required_indicators"]:
            if req_ind not in indicators:
                indicators.append(req_ind)
        
        # Hiperparâmetros do modelo
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
        
        # Parâmetros adicionais para PPO
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        
        # Avaliar modelo com os indicadores e hiperparâmetros selecionados
        mean_reward, std_reward, env = evaluate_indicators(
            trial, 
            indicators, 
            n_eval_episodes=CONFIG["evaluation_episodes"]
        )
        
        # Calcular métricas adicionais para otimização
        # Penalizar conjuntos muito grandes de indicadores (para evitar overfitting)
        indicator_penalty = CONFIG["indicator_penalty_factor"] * max(0, len(indicators) - 5)
        
        # Calcular pontuação final
        score = mean_reward - indicator_penalty
        
        # Salvar informações do trial
        trial.set_user_attr("indicators", indicators)
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("n_indicators", len(indicators) - 2)  # Excluindo close e volume
        
        # Registrar conclusão do trial
        trial_duration = time.time() - trial_start_time
        logger.info(f"Trial {trial.number} concluido em {trial_duration:.2f}s - Score: {score:.4f}, Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
        
        return score
    
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} interrompido por pruning")
        raise
    except Exception as e:
        logger.error(f"Erro no trial {trial.number}: {e}")
        logger.debug(traceback.format_exc())
        return -1000  # Valor muito baixo para trials com erro

# Função para analisar resultados
def analyze_results(study):
    """
    Analisa os resultados do estudo Optuna e gera insights.
    
    Args:
        study: Estudo Optuna concluído
    
    Returns:
        insights: Dicionário com insights sobre os resultados
    """
    insights = {}
    
    # Analisar importância dos indicadores
    indicator_importance = {}
    indicator_usage = {}
    
    for indicator in ALL_INDICATORS:
        param_name = f"use_{indicator}"
        importance = 0
        usage = 0
        count = 0
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
                count += 1
                if trial.params[param_name]:
                    usage += 1
                    # Calcular importância baseada na correlação com a recompensa
                    importance += trial.value / max(1, abs(study.best_value))
        
        if count > 0:
            indicator_importance[indicator] = importance / count
            indicator_usage[indicator] = usage / count * 100  # Porcentagem de uso
    
    # Ordenar indicadores por importância
    sorted_indicators = sorted(
        indicator_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Calcular correlações entre hiperparâmetros e recompensas
    param_values = {}
    rewards = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            rewards.append(trial.value)
            for param_name, param_value in trial.params.items():
                if not param_name.startswith("use_"):
                    if param_name not in param_values:
                        param_values[param_name] = []
                    param_values[param_name].append(param_value)
    
    param_correlations = {}
    for param_name, values in param_values.items():
        if len(values) == len(rewards):
            correlation = np.corrcoef(values, rewards)[0, 1]
            param_correlations[param_name] = correlation
    
    # Compilar insights
    insights["top_indicators"] = [ind for ind, _ in sorted_indicators[:5]]
    insights["indicator_usage"] = indicator_usage
    insights["param_correlations"] = param_correlations
    insights["best_reward"] = study.best_trial.user_attrs.get("mean_reward", 0)
    insights["best_indicators"] = study.best_trial.user_attrs.get("indicators", [])
    insights["n_complete_trials"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    insights["n_pruned_trials"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    insights["n_failed_trials"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    return insights

# Função para gerar visualizações avançadas
def generate_visualizations(study, output_dir="optuna_results"):
    """
    Gera visualizações avançadas dos resultados do estudo Optuna.
    
    Args:
        study: Estudo Optuna concluído
        output_dir: Diretório para salvar as visualizações
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Histórico de otimização
    fig = plot_optimization_history(study)
    fig.write_image(f"{output_dir}/optimization_history_{timestamp}.png")
    
    # Importância dos parâmetros
    fig = plot_param_importances(study)
    fig.write_image(f"{output_dir}/param_importances_{timestamp}.png")
    
    # Visualizações avançadas para hiperparâmetros contínuos
    try:
        fig = plot_contour(study, params=["learning_rate", "gamma"])
        fig.write_image(f"{output_dir}/contour_lr_gamma_{timestamp}.png")
    except:
        pass
    
    try:
        fig = plot_slice(study, params=["learning_rate", "gamma", "ent_coef"])
        fig.write_image(f"{output_dir}/slice_params_{timestamp}.png")
    except:
        pass
    
    # Visualização personalizada para indicadores
    plt.figure(figsize=(14, 8))
    
    # Gráfico de barras para uso de indicadores
    plt.subplot(2, 2, 1)
    indicator_usage = {}
    for indicator in ALL_INDICATORS:
        param_name = f"use_{indicator}"
        usage_count = sum(1 for t in study.trials if 
                          t.state == optuna.trial.TrialState.COMPLETE and 
                          t.params.get(param_name, False))
        total_count = sum(1 for t in study.trials if 
                          t.state == optuna.trial.TrialState.COMPLETE and 
                          param_name in t.params)
        if total_count > 0:
            indicator_usage[indicator] = usage_count / total_count * 100
    
    if indicator_usage:
        indicators = list(indicator_usage.keys())
        usage_pct = list(indicator_usage.values())
        
        # Ordenar por uso
        sorted_indices = np.argsort(usage_pct)[::-1]
        sorted_indicators = [indicators[i] for i in sorted_indices]
        sorted_usage = [usage_pct[i] for i in sorted_indices]
        
        colors = ['green' if ind in study.best_trial.user_attrs.get("indicators", []) else 'blue' 
                 for ind in sorted_indicators]
        
        plt.bar(sorted_indicators, sorted_usage, color=colors)
        plt.title("Uso de Indicadores (%)")
        plt.xticks(rotation=45)
        plt.ylabel("% de Uso nos Trials")
        plt.grid(axis='y')
    
    # Gráfico de distribuição de recompensas
    plt.subplot(2, 2, 2)
    rewards = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if rewards:
        sns.histplot(rewards, kde=True)
        plt.axvline(x=study.best_value, color='r', linestyle='--', 
                   label=f'Melhor valor: {study.best_value:.4f}')
        plt.title("Distribuicao de Recompensas")
        plt.xlabel("Valor Objetivo")
        plt.ylabel("Frequencia")
        plt.legend()
    
    # Gráfico de correlação entre número de indicadores e recompensa
    plt.subplot(2, 2, 3)
    n_indicators = [t.user_attrs.get("n_indicators", 0) for t in study.trials 
                   if t.state == optuna.trial.TrialState.COMPLETE]
    rewards = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if n_indicators and rewards:
        plt.scatter(n_indicators, rewards, alpha=0.6)
        plt.title("Relacao entre Numero de Indicadores e Recompensa")
        plt.xlabel("Numero de Indicadores")
        plt.ylabel("Valor Objetivo")
        plt.grid(True)
        
        # Adicionar linha de tendência
        if len(n_indicators) > 1:
            z = np.polyfit(n_indicators, rewards, 1)
            p = np.poly1d(z)
            plt.plot(sorted(set(n_indicators)), p(sorted(set(n_indicators))), "r--", 
                    label=f"Tendencia: y={z[0]:.4f}x+{z[1]:.4f}")
            plt.legend()
    
    # Gráfico de relação entre learning_rate e recompensa
    plt.subplot(2, 2, 4)
    lr_values = [t.params.get("learning_rate", 0) for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE]
    rewards = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if lr_values and rewards:
        plt.scatter(lr_values, rewards, alpha=0.6)
        plt.title("Relacao entre Learning Rate e Recompensa")
        plt.xlabel("Learning Rate")
        plt.ylabel("Valor Objetivo")
        plt.xscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/advanced_analysis_{timestamp}.png")
    
    # Gráfico de correlação entre parâmetros
    param_names = [p for p in study.best_trial.params.keys() if not p.startswith("use_")]
    if len(param_names) >= 2:
        param_values = {}
        for param in param_names:
            param_values[param] = [t.params.get(param) for t in study.trials 
                                  if t.state == optuna.trial.TrialState.COMPLETE and param in t.params]
        
        # Criar DataFrame para correlação
        param_df = pd.DataFrame(param_values)
        if not param_df.empty and param_df.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(param_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("Correlacao entre Hiperparametros")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/param_correlation_{timestamp}.png")

# Função principal
def main():
    """
    Função principal para otimização de indicadores.
    """
    global interrupted
    
    # Registrar início da execução
    start_time = time.time()
    logger.info(f"Iniciando otimizacao de indicadores para scalping puro...")
    logger.info(f"Indicadores disponiveis: {ALL_INDICATORS}")
    logger.info(f"Configuracao: {json.dumps(CONFIG, indent=2)}")
    
    try:
        # Configurar sampler e pruner
        if CONFIG["advanced_sampler"]:
            sampler = TPESampler(n_startup_trials=CONFIG["n_startup_trials"])
        else:
            sampler = TPESampler()
        
        if CONFIG["pruning_enabled"]:
            pruner = MedianPruner(n_startup_trials=CONFIG["n_startup_trials"], 
                                 n_warmup_steps=CONFIG["pruning_patience"])
        else:
            pruner = None
        
        # Criar estudo Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Otimizar com tratamento de interrupção
        try:
            study.optimize(
                objective, 
                n_trials=CONFIG["n_trials"],
                timeout=CONFIG["timeout"],
                n_jobs=CONFIG["n_jobs"],
                show_progress_bar=True,
                catch=(Exception,)
            )
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Otimizacao interrompida pelo usuario!")
        
        # Verificar se temos resultados válidos
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.error("Nenhum trial foi concluido com sucesso!")
            return None, None
        
        # Obter melhor trial
        best_trial = study.best_trial
        best_indicators = best_trial.user_attrs.get("indicators", [])
        best_mean_reward = best_trial.user_attrs.get("mean_reward", 0)
        best_std_reward = best_trial.user_attrs.get("std_reward", 0)
        
        # Registrar resultados
        logger.info("\nOtimizacao concluida!")
        logger.info(f"Melhores indicadores: {best_indicators}")
        logger.info(f"Recompensa media: {best_mean_reward:.4f} +/- {best_std_reward:.4f}")
        logger.info(f"Melhores hiperparametros:")
        for key, value in best_trial.params.items():
            if not key.startswith("use_"):
                logger.info(f"  {key}: {value}")
        
        # Salvar estudo para análise posterior
        joblib.dump(study, f"optuna_results/study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # Salvar melhores indicadores e hiperparâmetros
        with open("optuna_best_model/best_config.txt", "w") as f:
            f.write(f"Indicadores: {best_indicators}\n")
            f.write(f"learning_rate: {best_trial.params.get('learning_rate', 0.0003)}\n")
            f.write(f"gamma: {best_trial.params.get('gamma', 0.99)}\n")
            f.write(f"ent_coef: {best_trial.params.get('ent_coef', 0.01)}\n")
            f.write(f"n_steps: {best_trial.params.get('n_steps', 2048)}\n")
            f.write(f"clip_range: {best_trial.params.get('clip_range', 0.2)}\n")
        
        # Salvar indicadores para uso posterior pelo bot de execução
        os.makedirs("final_checkpoints", exist_ok=True)
        with open("final_checkpoints/training_indicators.txt", "w") as f:
            f.write(str(best_indicators))
        logger.info(f"Indicadores salvos em final_checkpoints/training_indicators.txt")
        
        # Gerar visualizações avançadas
        if CONFIG["plot_all_visualizations"]:
            logger.info("Gerando visualizacoes avancadas...")
            generate_visualizations(study)
        
        # Analisar resultados automaticamente
        if CONFIG["auto_analyze_results"]:
            logger.info("Analisando resultados...")
            insights = analyze_results(study)
            
            # Salvar insights
            with open("optuna_results/insights.json", "w") as f:
                json.dump(insights, f, indent=2)
            
            # Exibir insights principais
            logger.info("\nINSIGHTS DA OTIMIZACAO:")
            logger.info(f"Top 5 indicadores: {insights['top_indicators']}")
            logger.info(f"Correlacoes de parametros: {insights['param_correlations']}")
            logger.info(f"Trials completos: {insights['n_complete_trials']}")
            logger.info(f"Trials interrompidos: {insights['n_pruned_trials']}")
            logger.info(f"Trials com erro: {insights['n_failed_trials']}")
        
        # Testar os melhores indicadores
        logger.info("\nTestando os melhores indicadores...")
        mean_reward, std_reward, env = evaluate_indicators(
            None, 
            best_indicators, 
            n_eval_episodes=CONFIG["final_evaluation_episodes"],
            timesteps=CONFIG["optimization_timesteps"] * 2  # Mais passos para avaliação final
        )
        logger.info(f"Recompensa final: {mean_reward:.4f} +/- {std_reward:.4f}")
        
        # Registrar tempo total
        total_time = time.time() - start_time
        logger.info(f"\nTempo total de execucao: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
        
        return best_indicators, best_trial.params
    
    except Exception as e:
        logger.error(f"Erro durante a otimizacao: {e}")
        logger.debug(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    main()
