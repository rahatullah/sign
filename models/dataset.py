import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Constants for frame dimensions
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

class SignLanguageDataset(Dataset):
    def __init__(self, video_dir, annotations_dir, split='train', max_frames=8):
        """
        Args:
            video_dir (str): Base directory containing train/dev/test video folders
            annotations_dir (str): Directory containing all annotation Excel files
            split (str): One of 'train', 'dev', or 'test'
            max_frames (int): Maximum number of frames to use per video
        """
        super().__init__()
        print(f"Initializing {split} dataset...")
        
        self.video_dir = os.path.join(video_dir, split)
        self.annotations_file = os.path.join(annotations_dir, f"{split}.xlsx")
        self.max_frames = max_frames
        
        print(f"Video directory: {self.video_dir}")
        print(f"Loading annotations from: {os.path.basename(self.annotations_file)}")
        
        # Load annotations
        #self.annotations = pd.read_excel(self.annotations_file)
        self.annotations = pd.read_excel(self.annotations_file, engine='openpyxl')
        print(f"Loaded {len(self.annotations)} entries from {os.path.basename(self.annotations_file)}")
        print(f"Columns in {os.path.basename(self.annotations_file)}: {list(self.annotations.columns)}")
        
        # Clean video names - remove train/ prefix and ensure .mp4 extension
        self.annotations['video_name'] = self.annotations['name'].apply(
            lambda x: x.replace('train/', '').replace('test/', '').replace('dev/', '')
        )
        self.annotations['video_name'] = self.annotations['video_name'].apply(
            lambda x: x if x.endswith('.mp4') else x + '.mp4'
        )
        
        print("\nFirst few video names in annotations (after cleanup):")
        print(self.annotations['video_name'].head())
        
        # Get list of video files
        available_videos = set(os.listdir(self.video_dir))
        print("\nFirst few video files in directory:")
        print(list(available_videos)[:5])
        
        print("\nChecking video name formats:")
        print(f"Example annotation video name: {self.annotations['video_name'].iloc[0]}")
        print(f"Example directory video name: {next(iter(available_videos))}")
        
        # Filter annotations to only include videos that exist
        self.annotations = self.annotations[self.annotations['video_name'].isin(available_videos)]
        print(f"\nFound {len(self.annotations)} annotations for {split} split")
        
        # Build vocabularies
        self._build_vocabularies()
        
        # Define transform for video frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"\nSuccessfully loaded {len(self.annotations)} video-annotation pairs")
    '''
    def _build_vocabularies(self):
        """Build vocabularies for gloss and text"""
        # Special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        # Build gloss vocabulary
        gloss_words = set()
        for gloss in self.annotations['gloss']:
            gloss_words.update(str(gloss).split())
        gloss_words = sorted(list(gloss_words))
        
        # Build text vocabulary
        text_words = set()
        for text in self.annotations['text']:
            text_words.update(str(text).split())
        text_words = sorted(list(text_words))
        
        # Create mappings
        self.gloss_to_idx = {word: idx + len(special_tokens) for idx, word in enumerate(gloss_words)}
        self.idx_to_gloss = {idx: word for word, idx in self.gloss_to_idx.items()}
        
        self.text_to_idx = {word: idx + len(special_tokens) for idx, word in enumerate(text_words)}
        self.idx_to_text = {idx: word for word, idx in self.text_to_idx.items()}
        
        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.gloss_to_idx[token] = idx
            self.idx_to_gloss[idx] = token
            self.text_to_idx[token] = idx
            self.idx_to_text[idx] = token
        
        print(f"\nVocabulary sizes:")
        print(f"Gloss vocabulary: {len(self.gloss_to_idx)} tokens")
        print(f"Text vocabulary: {len(self.text_to_idx)} tokens")
    '''
        
    def _build_vocabularies(self):
        """Build vocabularies for gloss and text"""
        # Special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Build gloss vocabulary excluding special tokens
        gloss_words = set()
        for gloss in self.annotations['gloss']:
            words = set(str(gloss).split())
            gloss_words.update(words - set(special_tokens))
        gloss_words = sorted(list(gloss_words))

        # Build text vocabulary excluding special tokens
        text_words = set()
        for text in self.annotations['text']:
            words = set(str(text).split())
            text_words.update(words - set(special_tokens))
        text_words = sorted(list(text_words))

        # Create mappings (starting index after special tokens)
        self.gloss_to_idx = {word: idx + len(special_tokens) for idx, word in enumerate(gloss_words)}
        self.idx_to_gloss = {idx: word for word, idx in self.gloss_to_idx.items()}

        self.text_to_idx = {word: idx + len(special_tokens) for idx, word in enumerate(text_words)}
        self.idx_to_text = {idx: word for word, idx in self.text_to_idx.items()}

        # Add special tokens with fixed indices
        for idx, token in enumerate(special_tokens):
            self.gloss_to_idx[token] = idx
            self.idx_to_gloss[idx] = token
            self.text_to_idx[token] = idx
            self.idx_to_text[idx] = token

        print(f"\nVocabulary sizes:")
        print(f"Gloss vocabulary: {len(self.gloss_to_idx)} tokens")
        print(f"Text vocabulary: {len(self.text_to_idx)} tokens")

    def get_vocab_sizes(self):
        """Return the sizes of gloss and text vocabularies"""
        return len(self.gloss_to_idx), len(self.text_to_idx)
    
    def _tokenize_sequence(self, sequence, vocab):
        """Convert a sequence of words to token indices"""
        words = str(sequence).split()
        return [vocab.get(word, vocab['<unk>']) for word in words]
    
    def __getitem__(self, idx):
        # Get video path and annotations
        video_name = self.annotations.iloc[idx]['video_name']
        video_path = os.path.join(self.video_dir, video_name)
        
        gloss = self.annotations.iloc[idx]['gloss']
        text = self.annotations.iloc[idx]['text']
        
        # Load and preprocess video frames
        frames = self._load_video(video_path)
        
        # Tokenize gloss and text
        gloss_tokens = self._tokenize_sequence(gloss, self.gloss_to_idx)
        text_tokens = self._tokenize_sequence(text, self.text_to_idx)
        
        # Add SOS and EOS tokens
        gloss_tokens = [self.gloss_to_idx['<sos>']] + gloss_tokens + [self.gloss_to_idx['<eos>']]
        text_tokens = [self.text_to_idx['<sos>']] + text_tokens + [self.text_to_idx['<eos>']]
        
        return {
            'frames': frames,
            'gloss': torch.tensor(gloss_tokens, dtype=torch.long),
            'text': torch.tensor(text_tokens, dtype=torch.long)
        }
    
    def _load_video(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for transforms
            frame = Image.fromarray(frame)
            
            # Apply transforms (includes resize, to tensor, and normalize)
            frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        
        # Pad with zeros if video is shorter than max_frames
        if len(frames) == 0:
            # Create a dummy frame if video couldn't be loaded
            frames.append(torch.zeros(3, FRAME_HEIGHT, FRAME_WIDTH))
            
        while len(frames) < self.max_frames:
            frames.append(torch.zeros_like(frames[0]))
        
        # Stack frames and ensure proper shape
        frames = torch.stack(frames)  # Shape: [T, C, H, W]
        return frames
    
    def __len__(self):
        return len(self.annotations)

    def get_gloss_vocab(self):
        """Create vocabulary from all gloss annotations."""
        if not hasattr(self, '_gloss_vocab'):
            unique_words = set()
            for gloss in self.annotations['gloss']:
                words = str(gloss).split()
                unique_words.update(words)
            self._gloss_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(unique_words))
        return self._gloss_vocab

    def get_text_vocab(self):
        """Create vocabulary from all text annotations."""
        if not hasattr(self, '_text_vocab'):
            unique_words = set()
            for text in self.annotations['text']:
                words = str(text).split()
                unique_words.update(words)
            self._text_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(unique_words))
        return self._text_vocab

    @classmethod
    def get_transform(cls):
        """Return the transform pipeline used for video frames."""
        return transforms.Compose([
            transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]) 