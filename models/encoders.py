import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import mediapipe as mp

# Reusable encoder block used by all three encoders
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.fc = nn.Linear(input_dim, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_proj = self.fc(x)  # (B, T, d_model)
        attn_out, _ = self.mha(x_proj, x_proj, x_proj)
        x = self.norm1(x_proj + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# Visual Encoder: Uses pretrained EfficientNet-B7 to extract deep image features.
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512, nhead=4, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b7',
            in_channels=3,
            advprop=False
        )
        self.backbone_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        if dim_feedforward is None:
            dim_feedforward = output_dim * 2
        self.temporal_reduction = nn.Sequential(
            nn.Linear(self.backbone_features, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim // 2, output_dim)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        chunk_size = 4  # Process frames in small chunks
        features_list = []
        for i in range(0, T, chunk_size):
            chunk = x[:, i:i+chunk_size].contiguous()  # (B, t, C, H, W)
            b, t, c, h, w = chunk.shape
            chunk = chunk.view(b * t, c, h, w)
            #with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            #with torch.cuda.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                chunk_features = self.backbone.extract_features(chunk)
                chunk_features = F.adaptive_avg_pool2d(chunk_features, 1)
                chunk_features = chunk_features.view(b, t, -1)
            features_list.append(chunk_features)
            if i % 8 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        features = torch.cat(features_list, dim=1)
        features = self.temporal_reduction(features)
        return features
    
# Emotion Encoder: Extracts facial emotion cues using MediaPipe FaceMesh.
class EmotionEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=None, dropout=0.1):
        super().__init__()
        
        #self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
            # Optionally, if supported, add: image_dimensions=(224,224)
        )
        # For efficiency, use a selected subset of facial landmarks (e.g. indices relevant for emotion)
        self.selected_indices = [1, 33, 61, 199, 263, 291, 467]  # 7 landmarks → 7*3 = 21 features
        d_input = len(self.selected_indices) * 3
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.encoder_block = EncoderBlock(input_dim=d_input, d_model=d_model,
                                          nhead=nhead, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        import numpy as np
        B, T, C, H, W = x.shape
        device = x.device
        m = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        features_batch = []
        for b in range(B):
            features_frames = []
            for t in range(T):
                frame = x[b, t]
                frame = frame * std + m
                frame_np = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                result = self.mp_face_mesh.process(frame_np)
                selected_features = []
                if result.multi_face_landmarks is not None and len(result.multi_face_landmarks) > 0:
                    # Use the first detected face's landmarks
                    landmarks = result.multi_face_landmarks[0].landmark
                    for idx in self.selected_indices:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            selected_features.extend([lm.x, lm.y, lm.z])
                        else:
                            selected_features.extend([0, 0, 0])
                else:
                    selected_features = [0] * (len(self.selected_indices) * 3)
                features_frames.append(selected_features)
            features_frames = torch.tensor(features_frames, dtype=torch.float32, device=device)
            features_batch.append(features_frames)
        features_batch = torch.stack(features_batch, dim=0)  # (B, T, 21)
        out = self.encoder_block(features_batch)
        return out

# Gesture Encoder: Extracts hand & body keypoints using MediaPipe Holistic.
class GestureEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=None, dropout=0.1):
        super().__init__()
        # Import mediapipe only here so that the dependency is optional outside training/inference.
        self.mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1)
        # Expected keypoints:
        # Pose: 33 landmarks, Left hand: 21, Right hand: 21 → Total = 33*3 + 21*3 + 21*3 = 225.
        d_input = 225
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.encoder_block = EncoderBlock(input_dim=d_input, d_model=d_model,
                                          nhead=nhead, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        import numpy as np
        B, T, C, H, W = x.shape
        device = x.device
        # Mean and std used during normalization (as in the dataset transform)
        m = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        features_batch = []
        for b in range(B):
            features_frames = []
            for t in range(T):
                frame = x[b, t]  # (C,H,W)
                # Undo normalization
                frame = frame * std + m
                frame_np = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                result = self.mp_holistic.process(frame_np)
                # Pose landmarks
                if result.pose_landmarks is not None:
                    pose_coords = []
                    for lm in result.pose_landmarks.landmark:
                        pose_coords.extend([lm.x, lm.y, lm.z])
                else:
                    pose_coords = [0] * (33 * 3)
                # Left-hand landmarks
                if result.left_hand_landmarks is not None:
                    left_coords = []
                    for lm in result.left_hand_landmarks.landmark:
                        left_coords.extend([lm.x, lm.y, lm.z])
                else:
                    left_coords = [0] * (21 * 3)
                # Right-hand landmarks
                if result.right_hand_landmarks is not None:
                    right_coords = []
                    for lm in result.right_hand_landmarks.landmark:
                        right_coords.extend([lm.x, lm.y, lm.z])
                else:
                    right_coords = [0] * (21 * 3)
                feature_vector = pose_coords + left_coords + right_coords  # length = 225
                features_frames.append(feature_vector)
            # Convert list to tensor for this sample (T, 225)
            features_frames = torch.tensor(features_frames, dtype=torch.float32, device=device)
            features_batch.append(features_frames)
        features_batch = torch.stack(features_batch, dim=0)  # (B, T, 225)
        out = self.encoder_block(features_batch)
        return out

