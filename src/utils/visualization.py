"""
시각화 유틸

노트북이랑 리포트에서 공통으로 쓰는 플롯 함수 모음
matplotlib 설정도 여기서 통일

처음엔 노트북 안에 다 때려박았다가
02_model_training.ipynb 만들면서 중복이 너무 많아져서 분리함
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional


# 공통 스타일 설정
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

COLORS = {
    'normal': 'steelblue',
    'abuser': 'tomato',
    'bot': 'darkorange',
    'ensemble': '#2ecc71',
    'if': '#e74c3c',
    'lstm': '#3498db',
}


# ─────────────────────────────────────────────────────
#  피처 분포
# ─────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = 'label',
    save_path: Optional[str] = None,
):
    """
    정상 vs 이상 플레이어 피처 분포 비교

    논문 스타일로 그리려다가 너무 오버같아서 그냥 깔끔하게
    """
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    normal = df[df[label_col] == 0]
    abnormal = df[df[label_col] == 1]

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(normal[feat].dropna(), bins=30, alpha=0.6,
                label=f'정상 (n={len(normal)})', color=COLORS['normal'], density=True)
        ax.hist(abnormal[feat].dropna(), bins=30, alpha=0.6,
                label=f'이상 (n={len(abnormal)})', color=COLORS['abuser'], density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)

    # 남는 axes 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('피처별 정상 vs 이상 분포 (밀도 기준)', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────
#  ROC Curve
# ─────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    score_dict: dict,   # {'모델명': scores 배열}
    save_path: Optional[str] = None,
):
    """
    여러 모델 ROC 한 그래프에
    score_dict 예시: {'IF': if_scores, 'LSTM': lstm_scores, '앙상블': ensemble_scores}
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))

    color_cycle = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, (name, scores) in enumerate(score_dict.items()):
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color_cycle[i % len(color_cycle)],
                    linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"{name} ROC 계산 실패: {e}")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.02, color='gray')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve 비교')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────
#  Threshold 튜닝
# ─────────────────────────────────────────────────────

def plot_threshold_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    current_threshold: float = 0.55,
    save_path: Optional[str] = None,
):
    """
    threshold 바꾸면서 precision/recall/f1 변화 보기
    어느 값 쓸지 결정하는 용도
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.arange(0.2, 0.95, 0.025)
    results = {'threshold': [], 'precision': [], 'recall': [], 'f1': [], 'flagged_pct': []}

    for t in thresholds:
        preds = (scores > t).astype(int)
        results['threshold'].append(round(float(t), 3))
        results['precision'].append(precision_score(y_true, preds, zero_division=0))
        results['recall'].append(recall_score(y_true, preds, zero_division=0))
        results['f1'].append(f1_score(y_true, preds, zero_division=0))
        results['flagged_pct'].append(preds.mean() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results['threshold'], results['precision'], label='Precision',
             color='#e74c3c', linewidth=2, marker='o', markersize=3)
    ax1.plot(results['threshold'], results['recall'], label='Recall',
             color='#3498db', linewidth=2, marker='s', markersize=3)
    ax1.plot(results['threshold'], results['f1'], label='F1',
             color='#2ecc71', linewidth=2.5, marker='^', markersize=4)
    ax1.axvline(x=current_threshold, color='gray', linestyle='--',
                alpha=0.7, label=f'현재 threshold ({current_threshold})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Threshold별 Precision / Recall / F1')
    ax1.legend()
    ax1.set_ylim([0, 1.05])

    ax2.plot(results['threshold'], results['flagged_pct'],
             color='#9b59b6', linewidth=2, marker='o', markersize=3)
    ax2.axvline(x=current_threshold, color='gray', linestyle='--',
                alpha=0.7, label=f'현재 threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('플래그 비율 (%)')
    ax2.set_title('Threshold별 플래그 비율')
    ax2.legend()

    # 최적 F1 threshold 표시
    best_idx = np.argmax(results['f1'])
    best_t = results['threshold'][best_idx]
    best_f1 = results['f1'][best_idx]
    ax1.annotate(f'최대 F1={best_f1:.3f}\n@ {best_t}',
                 xy=(best_t, best_f1),
                 xytext=(best_t + 0.07, best_f1 - 0.12),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()
    print(f"\n최적 F1 threshold: {best_t} (F1={best_f1:.4f})")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────
#  LSTM 학습 곡선
# ─────────────────────────────────────────────────────

def plot_training_history(
    history: list,
    save_path: Optional[str] = None,
):
    """
    LSTM 학습 Loss 곡선
    history는 epoch별 loss 리스트
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, color='#3498db', linewidth=2, label='Train Loss')
    ax.fill_between(epochs, history,
                    alpha=0.1, color='#3498db')

    # 수렴 지점 대략 표시
    if len(history) > 5:
        smooth = pd.Series(history).rolling(3).mean().dropna().values
        conv_idx = np.argmin(np.abs(np.diff(smooth)))
        ax.axvline(x=conv_idx + 2, color='gray', linestyle='--',
                   alpha=0.6, label=f'수렴 근사 (epoch {conv_idx + 2})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM Autoencoder 학습 곡선')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────
#  이상 스코어 분포
# ─────────────────────────────────────────────────────

def plot_score_distribution(
    scores: np.ndarray,
    true_labels: np.ndarray,
    model_name: str = '앙상블',
    threshold: float = 0.55,
    save_path: Optional[str] = None,
):
    """
    정상/이상 그룹의 스코어 분포 + threshold 위치
    이걸 보면 두 그룹이 얼마나 잘 분리되는지 한눈에 보임
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    normal_scores = scores[true_labels == 0]
    abnormal_scores = scores[true_labels == 1]

    bins = np.linspace(0, 1, 50)
    ax.hist(normal_scores, bins=bins, alpha=0.6, density=True,
            color=COLORS['normal'], label=f'정상 (n={len(normal_scores)})')
    ax.hist(abnormal_scores, bins=bins, alpha=0.6, density=True,
            color=COLORS['abuser'], label=f'이상 (n={len(abnormal_scores)})')

    ax.axvline(x=threshold, color='black', linestyle='--',
               linewidth=1.5, label=f'Threshold = {threshold}')
    ax.axvspan(threshold, 1.0, alpha=0.05, color='red', label='플래그 영역')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('밀도')
    ax.set_title(f'{model_name} 스코어 분포')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────
#  피처 중요도
# ─────────────────────────────────────────────────────

def plot_feature_importance(
    importance_dict: dict,
    top_n: int = 12,
    save_path: Optional[str] = None,
):
    """
    Isolation Forest 피처 중요도 가로 바차트
    importance_dict: {feature_name: importance_score}
    """
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # 내림차순으로 아래에서 위로 그리기
    names = names[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#e74c3c' if v > np.mean(values) * 1.5 else '#3498db' for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor='white')

    # 값 레이블
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    high_patch = mpatches.Patch(color='#e74c3c', alpha=0.8, label='평균 대비 1.5배 이상')
    low_patch = mpatches.Patch(color='#3498db', alpha=0.8, label='기타')
    ax.legend(handles=[high_patch, low_patch], fontsize=8)

    ax.set_xlabel('Importance Score (Permutation 방식)')
    ax.set_title(f'Isolation Forest 피처 중요도 TOP {top_n}')
    ax.set_xlim([0, max(values) * 1.15])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 → {save_path}")

    plt.show()
