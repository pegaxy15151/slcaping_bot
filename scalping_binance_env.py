import gymnasium as gym
import numpy as np
import pandas as pd
import ccxt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.momentum import ROCIndicator
import os
import ast

class ScalpingBinanceEnv(gym.Env):
    def __init__(self, indicators=None, force_dim=12):
        super(ScalpingBinanceEnv, self).__init__()
        self.client = ccxt.binance()
        self.symbol = 'ETH/USDT'
        self.timeframe = '5m'
        self.indicators = indicators if indicators else ["close", "volume"]
        self.force_dim = force_dim  # Forçar dimensão específica
        self.data = self.load_data()

        self.action_space = gym.spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        
        # Definir observation_space com dimensão forçada se especificada
        if self.force_dim:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.force_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.data.columns),),
                dtype=np.float32
            )

        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        
        # Verificar se a dimensão atual corresponde à dimensão forçada
        actual_dim = len(self.data.columns)
        if self.force_dim and actual_dim != self.force_dim:
            print(f"⚠️ Aviso: Dimensão atual ({actual_dim}) não corresponde à dimensão forçada ({self.force_dim})")
            print(f"⚠️ Ajustando automaticamente para {self.force_dim} dimensões")

    def load_data(self):
        # === Timeframe principal (5m) ===
        ohlcv_5m = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=8640)
        df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Indicadores 5m
        if "ema" in self.indicators:
            df_5m['ema'] = EMAIndicator(close=df_5m['close'], window=9).ema_indicator()

        if "rsi" in self.indicators:
            df_5m['rsi'] = RSIIndicator(close=df_5m['close'], window=14).rsi()

        if "macd" in self.indicators:
            macd = MACD(close=df_5m['close'])
            df_5m['macd'] = macd.macd()
            df_5m['macd_signal'] = macd.macd_signal()

        if "bollinger" in self.indicators:
            bb = BollingerBands(close=df_5m['close'], window=20, window_dev=2)
            df_5m['bb_mavg'] = bb.bollinger_mavg()
            df_5m['bb_high'] = bb.bollinger_hband()
            df_5m['bb_low'] = bb.bollinger_lband()

        if "vwap" in self.indicators:
            vwap = VolumeWeightedAveragePrice(
                high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], volume=df_5m['volume']
            )
            df_5m['vwap'] = vwap.volume_weighted_average_price()
            
        # Adicionando indicador ATR (Average True Range)
        if "atr" in self.indicators:
            atr = AverageTrueRange(high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], window=14)
            df_5m['atr'] = atr.average_true_range()
            
        # Adicionando indicador Stochastic RSI
        if "stoch_rsi" in self.indicators:
            stoch = StochasticOscillator(
                high=df_5m['high'], 
                low=df_5m['low'], 
                close=df_5m['close'],
                window=14, 
                smooth_window=3
            )
            df_5m['stoch_k'] = stoch.stoch()
            df_5m['stoch_d'] = stoch.stoch_signal()
            
        # Adicionando indicador OBV (On-Balance Volume)
        if "obv" in self.indicators:
            obv = OnBalanceVolumeIndicator(close=df_5m['close'], volume=df_5m['volume'])
            df_5m['obv'] = obv.on_balance_volume()
            
        # Adicionando indicador Momentum
        if "momentum" in self.indicators:
            roc = ROCIndicator(close=df_5m['close'], window=10)
            df_5m['momentum'] = roc.roc()
            
        # Adicionando indicador Ichimoku (simplificado)
        if "ichimoku" in self.indicators:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = df_5m['high'].rolling(window=9).max()
            period9_low = df_5m['low'].rolling(window=9).min()
            df_5m['ichimoku_conv'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = df_5m['high'].rolling(window=26).max()
            period26_low = df_5m['low'].rolling(window=26).min()
            df_5m['ichimoku_base'] = (period26_high + period26_low) / 2

        # === Monta colunas finais ===
        final_cols = ['close', 'volume']
        if "ema" in self.indicators: final_cols.append('ema')
        if "rsi" in self.indicators: final_cols.append('rsi')
        if "macd" in self.indicators: final_cols.extend(['macd', 'macd_signal'])
        if "bollinger" in self.indicators: final_cols.extend(['bb_mavg', 'bb_high', 'bb_low'])
        if "vwap" in self.indicators: final_cols.append('vwap')
        if "atr" in self.indicators: final_cols.append('atr')
        if "stoch_rsi" in self.indicators: final_cols.extend(['stoch_k', 'stoch_d'])
        if "obv" in self.indicators: final_cols.append('obv')
        if "momentum" in self.indicators: final_cols.append('momentum')
        if "ichimoku" in self.indicators: final_cols.extend(['ichimoku_conv', 'ichimoku_base'])

        df_5m = df_5m[final_cols]
        df_5m.dropna(inplace=True)
        df_5m.reset_index(drop=True, inplace=True)

        return df_5m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Obter observação básica
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        
        # Se a dimensão forçada for especificada, ajustar a observação
        if self.force_dim:
            current_dim = len(obs)
            if current_dim < self.force_dim:
                # Preencher com zeros se a dimensão atual for menor
                padding = np.zeros(self.force_dim - current_dim, dtype=np.float32)
                obs = np.concatenate([obs, padding])
            elif current_dim > self.force_dim:
                # Truncar se a dimensão atual for maior
                obs = obs[:self.force_dim]
                
        return obs

    def step(self, action):
        reward = 0
        price = self.data.iloc[self.current_step]['close']

        # === Executa lógica de trade ===
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 0 and self.position != 0:
            if self.position == 1:
                reward = price - self.entry_price
            elif self.position == -1:
                reward = self.entry_price - price
            self.position = 0
            self.entry_price = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        pass
