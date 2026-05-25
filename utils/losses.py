import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class BellLoss(nn.Module):
    """Bell-shaped exponential loss."""
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_p = torch.pow((y - p), 2)
        y_p_div = -1.0 * torch.div(y_p, 162.0)
        exp_y_p = torch.exp(y_p_div)
        loss = 300 * (1.0 - exp_y_p)
        return torch.mean(loss)


class LogCosh(nn.Module):
    """Log-cosh loss for smooth regression."""
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(p - y))
        return torch.mean(loss)


class RMSE(nn.Module):
    """Root Mean Squared Error loss."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(p, y))


class GL(nn.Module):
    """Generalized loss combining exponential and squared terms."""
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.sigma = sigma

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gl = self.eps / (self.lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (self.sigma ** 2)))
        return gl.mean()


class RMBell(nn.Module):
    """RMSE + BellLoss."""
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y)


class RMLCosh(nn.Module):
    """RMSE + LogCosh."""
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.logcosh(p, y)


class RMGL(nn.Module):
    """RMSE + GL."""
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.gl(p, y)


class RMBellLCosh(nn.Module):
    """RMSE + BellLoss + LogCosh."""
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.logcosh(p, y)


class RMBellGL(nn.Module):
    """RMSE + BellLoss + GL."""
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.gl(p, y)


class BellLCosh(nn.Module):
    """BellLoss + LogCosh."""
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y)


class BellGL(nn.Module):
    """BellLoss + GL."""
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.gl(p, y)


class BellLCoshGL(nn.Module):
    """BellLoss + LogCosh + GL."""
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()
        self.gl = GL()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y) + self.gl(p, y)


class LogCoshGL(nn.Module):
    """LogCosh + GL."""
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.logcosh = LogCosh()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logcosh(p, y) + self.gl(p, y)


class MAELoss(nn.Module):
    """Mean Absolute Error."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x - y))


class MSELoss(nn.Module):
    """Mean Squared Error."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(x - y, 2))


class CCCLoss(nn.Module):
    """
    Lin's Concordance Correlation Coefficient (CCC) loss.
    Measures agreement via precision (Pearson correlation) and accuracy (closeness to 45° line).

    Ref: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    """
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute 1 − CCC."""
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


def binarize_with_nan(x, threshold=0.5):
    """Binarize values, preserving NaN positions."""
    nan_mask = torch.isnan(x)
    binary = torch.zeros_like(x)
    binary[x > threshold] = 1.0
    binary[nan_mask] = float('nan')
    return binary


class MultiTaskLossWithNaN_v2(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        emo_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = 'BCE',
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
        ssl_weight_emotion: float = 0.0,
        ssl_weight_personality: float = 0.0,
        ssl_confidence_threshold_pt: float = 0.60,
        ssl_confidence_threshold_emo: float = 0.60
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality

        self.ssl_weight_emotion = ssl_weight_emotion or 0.0
        self.ssl_weight_personality = ssl_weight_personality or 0.0
        self.ssl_confidence_threshold_pt = ssl_confidence_threshold_pt or 0.60
        self.ssl_confidence_threshold_emo = ssl_confidence_threshold_emo or 0.6

        if emotion_loss_type == 'CE':
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        elif emotion_loss_type == 'BCE':
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        else:
            raise ValueError(f"Неизвестный emotion_loss_type: {emotion_loss_type}")

        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Неизвестный personality_loss_type: {personality_loss_type}. Доступные: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0
        emo_mask = labels.get('valid_emo', None)
        pred_emotion_all = outputs.get('emotion_logits')
        if pred_emotion_all is not None:
            if emo_mask is None:
                true_emotion = labels['emotion']
                pred_emotion = pred_emotion_all
                valid_any = True
            else:
                valid_any = emo_mask.any()
                if valid_any:
                    true_emotion = labels['emotion'][emo_mask]
                    pred_emotion = pred_emotion_all[emo_mask]

            if pred_emotion_all is not None and (emo_mask is None or valid_any):
                if self.emotion_loss_type == 'BCE':
                    true_emotion = binarize_with_nan(true_emotion, threshold=0)
                loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Полуконтролируемая потеря
        if self.ssl_weight_emotion > 0.0 and pred_emotion_all is not None:
            emo_mask = labels.get('valid_emo', None)
            if emo_mask is not None:
                unlabeled_mask = ~emo_mask
                if unlabeled_mask.any():
                    pred_emotion_unlabeled = pred_emotion_all[unlabeled_mask]
                    if self.emotion_loss_type == 'BCE':
                        probs = torch.sigmoid(pred_emotion_unlabeled)
                        confidence, pseudo_labels = torch.max(probs, dim=1)
                        mask_confident = confidence > self.ssl_confidence_threshold_emo
                    elif self.emotion_loss_type == 'CE':
                        probs = torch.softmax(pred_emotion_unlabeled, dim=1)
                        confidence, pseudo_labels = torch.max(probs, dim=1)
                        mask_confident = confidence > self.ssl_confidence_threshold_emo
                    else:
                        mask_confident = torch.zeros(unlabeled_mask.sum(), dtype=torch.bool, device=pred_emotion_all.device)

                    if mask_confident.any():
                        pred_emotion_confident = pred_emotion_unlabeled[mask_confident]
                        pseudo_labels_confident = pseudo_labels[mask_confident]
                        if self.emotion_loss_type == 'BCE':
                            num_c = pred_emotion_confident.size(1)
                            pseudo_labels_confident = pseudo_labels_confident.float()
                            loss += self.ssl_weight_emotion * self.emotion_loss(pred_emotion_confident, F.one_hot(pseudo_labels_confident.long(), num_classes=num_c).float())
                        else:
                            loss += self.ssl_weight_emotion * self.emotion_loss(pred_emotion_confident, pseudo_labels_confident)

        per_mask = labels.get('valid_per', None)
        pred_personality_all = outputs.get('personality_scores')
        if pred_personality_all is not None:
            if per_mask is None:
                true_personality = labels['personality']
                pred_personality = pred_personality_all
                per_valid_any = True
            else:
                per_valid_any = per_mask.any()
                if per_valid_any:
                    true_personality = labels['personality'][per_mask]
                    pred_personality = pred_personality_all[per_mask]

            if pred_personality_all is not None and (per_mask is None or per_valid_any):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0
                    valid_traits = 0
                    for i in range(5):
                        trait_mask = ~torch.isnan(true_personality[:, i])
                        if trait_mask.any():
                            loss_per += self.personality_loss(
                                true_personality[trait_mask, i],
                                pred_personality[trait_mask, i]
                            )
                            valid_traits += 1
                    if valid_traits > 0:
                        loss += (loss_per / valid_traits) * self.weight_personality
                else:
                    loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        # Полуконтролируемая потеря
        if self.ssl_weight_personality > 0.0 and pred_personality_all is not None:
            per_mask = labels.get('valid_per', None)
            if per_mask is not None:
                unlabeled_mask = ~per_mask
                if unlabeled_mask.any():
                    pred_per_unlabeled = pred_personality_all[unlabeled_mask]  # (U, 5)
                    pred_per_unlabeled = torch.clamp(pred_per_unlabeled, 0.0, 1.0)
                    pseudo_labels = (pred_per_unlabeled > 0.5).float()  # (U, 5)
                    confidence_mask = (
                        (pred_per_unlabeled > self.ssl_confidence_threshold_pt) |
                        (pred_per_unlabeled < (1 - self.ssl_confidence_threshold_pt))
                    )
                    if confidence_mask.any():
                        bce_loss_per_element = F.binary_cross_entropy(
                            pred_per_unlabeled,
                            pseudo_labels,
                            reduction='none'
                        )
                        weighted_loss = (bce_loss_per_element * confidence_mask.float()).sum()
                        total_confident = confidence_mask.sum().float()
                        if total_confident > 0:
                            ssl_loss = weighted_loss / total_confident
                            loss += self.ssl_weight_personality * ssl_loss

        if not isinstance(loss, torch.Tensor):
            device = (
                (outputs.get("emotion_logits") or outputs.get("personality_scores")).device
                if (outputs.get("emotion_logits") is not None
                    or outputs.get("personality_scores") is not None)
                else torch.device("cpu")
            )
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss
