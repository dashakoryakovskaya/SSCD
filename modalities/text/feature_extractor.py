# coding: utf-8
import torch
from transformers import AutoTokenizer, AutoModel
from .model_loader import load_fusion_model


class PretrainedTextEmbeddingExtractor:
    """
    jinaai/jina-embeddings-v → последовательный эмбеддинг (B, T, 1024) →
    Fusion-модель → логиты эмоций, оценки Big-5 и последние признаки.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "jinaai/jina-embeddings-v3",
        fusion_ckpt: str = "modalities/text/checkpoints_models/Transformer_jina_fusion.pt",
        emo_ckpt: str   = "modalities/text/checkpoints_models/Mamba_jina_emotion.pt",
        per_ckpt: str   = "modalities/text/checkpoints_models/Mamba_jina_personality.pt",
    ):
        self.device = torch.device(device)

        self.tok = AutoTokenizer.from_pretrained(model_name, code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True)
        self.enc = AutoModel.from_pretrained(model_name, code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True).to(self.device).eval()

        self.fusion, _ = load_fusion_model(
            fusion_ckpt, emo_ckpt, per_ckpt, device=self.device
        )

    @torch.no_grad()
    def extract(self, texts: list[str] | str) -> dict:
        if isinstance(texts, str):
            texts = [texts]

        batch = self.tok(texts, padding=True, truncation=True,
                        return_tensors="pt").to(self.device)

        hidden = self.enc(**batch).last_hidden_state  # (B, T, 1024)

        out = self.fusion(
            emotion_input=hidden.float(),
            personality_input=hidden.float(),
            return_features=True,
        )

        return {
            "emotion_logits": out["emotion_logits"].cpu(),
            "personality_scores": out["personality_scores"].cpu(),
            "last_emo_encoder_features": out["last_emo_encoder_features"].cpu(),
            "last_per_encoder_features": out["last_per_encoder_features"].cpu(),
        }
