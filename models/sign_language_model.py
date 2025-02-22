import torch
import torch.nn as nn
from models.encoders import VisualEncoder, GestureEncoder, EmotionEncoder
from models.decoders import TransformerDecoder

class SignLanguageTranslator(nn.Module):
    def __init__(
        self,
        gloss_vocab_size,
        text_vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6
    ):
        super().__init__()
        
        # Store vocabulary sizes as attributes
        self.gloss_vocab_size = gloss_vocab_size
        self.text_vocab_size = text_vocab_size
        
        # Encoders with desired output dimensions:
        self.visual_encoder = VisualEncoder(output_dim=512, nhead=4)
        self.gesture_encoder = GestureEncoder(d_model=256, nhead=4)
        self.emotion_encoder = EmotionEncoder(d_model=256, nhead=4)
        
        # Feature Fusion: Visual (512), Gesture (256), Emotion (256) → 1024 total.
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoders remain unchanged.
        self.gloss_decoder = TransformerDecoder(
            gloss_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers
        )
        self.translation_decoder = TransformerDecoder(
            text_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers
        )
        
    def forward(
        self,
        video_frames,
        gloss_targets=None,
        text_targets=None,
        gloss_mask=None,
        text_mask=None
    ):
        #with torch.cuda.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            visual_features = self.visual_encoder(video_frames)
            gesture_features = self.gesture_encoder(video_frames)
            emotion_features = self.emotion_encoder(video_frames)
            
            # Free up input memory
            del video_frames
            
            # Fusion with memory optimization
            combined_features = torch.cat([visual_features, emotion_features, gesture_features], dim=-1)
            del visual_features, gesture_features, emotion_features  # Free up memory
            
            memory = self.fusion_layer(combined_features)
            del combined_features  # Free up memory
            
            # Decode sequences
            gloss_output = self.gloss_decoder(
                gloss_targets,
                memory,
                tgt_mask=gloss_mask
            )
            
            text_output = self.translation_decoder(
                text_targets,
                memory,
                tgt_mask=text_mask
            )
            
            return gloss_output, text_output 