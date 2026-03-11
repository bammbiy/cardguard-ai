"""
LSTM Autoencoder 기반 이상 탐지

정상 플레이어 시퀀스로만 학습시키고
재구성 오차가 크면 이상으로 판정

왜 Autoencoder인가:
- 정상 데이터 패턴을 압축/복원하는 걸 학습
- 어뷰저 패턴은 재구성을 잘 못함 → 오차 큼
- 새로운 유형의 어뷰저도 잡을 수 있음

threshold 설정이 제일 까다로운 부분
지금은 정상 훈련 데이터의 95th percentile로 잡음
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h, _) = self.lstm(x)
        # 마지막 타임스텝 hidden state 사용
        latent = self.fc(h[-1])   # (batch, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0
        )
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (batch, latent_dim)
        h = self.fc(z).unsqueeze(1)                  # (batch, 1, hidden_dim)
        h = h.repeat(1, self.seq_len, 1)             # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(h)
        recon = self.out_layer(out)                   # (batch, seq_len, output_dim)
        return recon


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 20,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, n_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """시퀀스별 MSE 재구성 오차"""
        with torch.no_grad():
            recon = self.forward(x)
        # (batch, seq_len, input_dim) → batch별 평균 MSE
        error = ((x - recon) ** 2).mean(dim=(1, 2))
        return error


class LSTMDetector:
    """
    LSTM Autoencoder 학습/추론 래퍼

    정상 데이터로만 학습하고, 재구성 오차로 이상 판정
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 20,
        lr: float = 1e-3,
        device: str = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, seq_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.threshold = None
        self.is_fitted = False

    def fit(
        self,
        normal_sequences: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
        verbose: bool = True,
    ):
        """
        정상 시퀀스로 학습
        normal_sequences: (N, seq_len, input_dim)
        """
        X = torch.FloatTensor(normal_sequences).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                # gradient clipping - LSTM에서 exploding gradient 방지
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            history.append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        # threshold 설정: 정상 데이터 재구성 오차의 p-th percentile
        self.model.eval()
        with torch.no_grad():
            recon_errors = []
            for (batch,) in DataLoader(dataset, batch_size=512):
                err = self.model.reconstruction_error(batch)
                recon_errors.extend(err.cpu().numpy())

        self.threshold = np.percentile(recon_errors, threshold_percentile)
        self.is_fitted = True

        print(f"\nLSTM 학습 완료")
        print(f"  Threshold ({threshold_percentile}th percentile): {self.threshold:.6f}")
        print(f"  최종 Loss: {history[-1]:.6f}")
        return history

    def predict_score(self, sequences: np.ndarray) -> np.ndarray:
        """재구성 오차 반환 (높을수록 이상)"""
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요")

        self.model.eval()
        X = torch.FloatTensor(sequences).to(self.device)
        errors = []

        with torch.no_grad():
            for i in range(0, len(X), 512):
                batch = X[i: i + 512]
                err = self.model.reconstruction_error(batch)
                errors.extend(err.cpu().numpy())

        return np.array(errors)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """0 (정상) / 1 (이상) 예측"""
        scores = self.predict_score(sequences)
        return (scores > self.threshold).astype(int)

    def evaluate(self, sequences: np.ndarray, true_labels: np.ndarray):
        """성능 평가"""
        from sklearn.metrics import classification_report, roc_auc_score

        scores = self.predict_score(sequences)
        preds = self.predict(sequences)

        print("\n=== LSTM Autoencoder 평가 ===")
        print(classification_report(true_labels, preds, target_names=['정상', '이상']))
        try:
            auc = roc_auc_score(true_labels, scores)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception:
            pass

        return scores, preds

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'seq_len': self.seq_len,
        }, path)
        print(f"LSTM 모델 저장 → {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'LSTMDetector':
        data = torch.load(path, map_location='cpu')
        obj = cls(seq_len=data.get('seq_len', 20), **kwargs)
        obj.model.load_state_dict(data['model_state'])
        obj.threshold = data['threshold']
        obj.is_fitted = True
        return obj
