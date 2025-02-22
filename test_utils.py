from venv import logger
import torch
import nltk
import cv2
from PIL import Image
import difflib
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu import corpus_chrf, CHRF
from rouge_score import rouge_scorer
import json
from pathlib import Path
from datetime import datetime
import os
import time
from tqdm import tqdm
import numpy as np
import math

# Add NLTK data path configuration
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')

def setup_nltk():
    """Setup NLTK data once"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Call setup at module import, but only in main process
if __name__ == '__main__':
    setup_nltk()

def decode_sequence(output_tensor, vocab):
    """
    Convert model output tensor to text using vocabulary.
    
    Args:
        output_tensor: Model output tensor of shape (batch_size, seq_len, vocab_size)
        vocab: List of vocabulary words where index matches the model output indices
    
    Returns:
        str: Decoded text
    """
    # Get the most likely token at each position
    predictions = torch.argmax(output_tensor, dim=-1)
    
    # Convert to list for easier processing
    pred_indices = predictions[0].cpu().numpy()  # Take first sequence in batch
    
    # Convert indices to words
    words = []
    for idx in pred_indices:
        word = vocab[idx]
        if word == '<eos>':
            break
        if word not in ['<pad>', '<sos>']:
            words.append(word)
    
    return ' '.join(words)

def preprocess_video(video_path, transform):
    """
    Preprocess a video for model input.
    
    Args:
        video_path: Path to video file
        transform: Torchvision transforms to apply to frames
    
    Returns:
        torch.Tensor: Preprocessed video frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    max_frames = 100  # Same as in dataset class
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    
    # Pad with zeros if video is shorter than max_frames
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))
        
    return torch.stack(frames)

def calculate_metrics(predictions, references):
    """
    Calculate BLEU, CHRF, and ROUGE scores.
    
    Args:
        predictions: List of predicted sentences
        references: List of lists of reference sentences
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # BLEU scores
    bleu1 = corpus_bleu(references, predictions, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))
    
    # CHRF score
    chrf = CHRF()
    chrf_score = chrf.corpus_score(predictions, references)
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(' '.join(predictions[0]), ' '.join(references[0][0]))
    
    metrics = {
        'BLEU-1': bleu1 * 100,
        'BLEU-2': bleu2 * 100,
        'BLEU-3': bleu3 * 100,
        'BLEU-4': bleu4 * 100,
        'CHRF': chrf_score.score * 100,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure * 100,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure * 100,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure * 100
    }
    
    # Add warning if BLEU-4 is below threshold
    if metrics['BLEU-4'] < 28:
        metrics['warning'] = 'BLEU-4 score is below the target threshold of 28'
    
    return metrics

def translate_video(model, video_path, transform, device):
    """
    Translate a single video.
    
    Args:
        model: Trained SignLanguageTranslator model
        video_path: Path to video file
        transform: Torchvision transforms for preprocessing
        device: Device to run the model on
    
    Returns:
        tuple: (gloss_prediction, text_prediction)
    """
    model.eval()
    with torch.no_grad():
        # Preprocess video
        frames = preprocess_video(video_path, transform)
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension
        
        # Generate translation
        gloss_output, text_output = model(frames)
        
        # Convert outputs to text
        gloss_pred = decode_sequence(gloss_output, model.gloss_vocab)
        text_pred = decode_sequence(text_output, model.text_vocab)
        
        return gloss_pred, text_pred 

def load_test_config(config_path):
    """Load test configuration from JSON file"""
    if not os.path.exists(config_path):
        return {
            'batch_size': 1,
            'num_workers': 0,
            'image_size': 8
        }
    with open(config_path, 'r') as f:
        return json.load(f)

def log_test_results(model, config, metrics, predictions, references, log_file):
    """Log detailed test results"""
    log_file.write("\nTest Configuration\n")
    log_file.write("="*50 + "\n")
    log_file.write(f"Model Checkpoint: {config['model_path']}\n")
    log_file.write(f"Batch Size: {config['batch_size']}\n")
    log_file.write(f"Device: {config['device']}\n")
    log_file.write(f"Number of Test Samples: {len(predictions)}\n\n")

    log_file.write("Model Architecture\n")
    log_file.write("="*50 + "\n")
    log_file.write(f"Model Type: {model.__class__.__name__}\n")
    log_file.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")
    log_file.write(f"Gloss Vocabulary Size: {model.gloss_vocab_size}\n")
    log_file.write(f"Text Vocabulary Size: {model.text_vocab_size}\n\n")

    log_file.write("Test Metrics\n")
    log_file.write("="*50 + "\n")
    log_file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
    log_file.write(f"Precision: {metrics['precision']:.4f}\n")
    log_file.write(f"Recall: {metrics['recall']:.4f}\n")
    log_file.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")

    log_file.write("Translation Metrics\n")
    log_file.write("="*50 + "\n")
    log_file.write("BLEU Scores:\n")
    log_file.write(f"  BLEU-1: {metrics['BLEU-1']:.2f}\n")
    log_file.write(f"  BLEU-2: {metrics['BLEU-2']:.2f}\n")
    log_file.write(f"  BLEU-3: {metrics['BLEU-3']:.2f}\n")
    log_file.write(f"  BLEU-4: {metrics['BLEU-4']:.2f}\n")
    log_file.write(f"CHRF: {metrics['CHRF']:.2f}\n")
    log_file.write("ROUGE Scores:\n")
    log_file.write(f"  ROUGE-1: {metrics['ROUGE-1']:.2f}\n")
    log_file.write(f"  ROUGE-2: {metrics['ROUGE-2']:.2f}\n")
    log_file.write(f"  ROUGE-L: {metrics['ROUGE-L']:.2f}\n\n")

    # Log sample predictions
    log_file.write("Sample Predictions\n")
    log_file.write("="*50 + "\n")
    for i in range(min(10, len(predictions))):  # Log first 10 examples
        log_file.write(f"\nExample {i+1}:\n")
        log_file.write(f"Reference: {references[i]}\n")
        log_file.write(f"Prediction: {predictions[i]}\n")
        
def find_best_checkpoint():
    """Find the best checkpoint in the models directory"""
    models_dir = 'models'
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith('best_model_')]
    if not checkpoints:
        raise FileNotFoundError("No best model checkpoint found")
    return os.path.join(models_dir, sorted(checkpoints)[-1])

def calculate_wer(hypotheses, references):
    """Calculate Word Error Rate"""
    def levenshtein(a, b):
        if not a: return len(b)
        if not b: return len(a)
        return min(levenshtein(a[1:], b[1:])+(a[0] != b[0]),
                  levenshtein(a[1:], b)+1,
                  levenshtein(a, b[1:])+1)

    total_wer = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        distance = levenshtein(hyp_words, ref_words)
        total_wer += distance / len(ref_words)
    
    return total_wer / len(references)

def get_alignment(ref, hyp):
    """Generate a simple diff alignment between reference and hypothesis."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    diff = difflib.ndiff(ref_words, hyp_words)
    return '\n'.join(diff)


'''
def evaluate_model(model, test_loader, config):
    """Evaluate model performance with detailed metrics.
       For each sample it prints:
         - Gloss reference, hypothesis, and alignment.
         - Text reference, hypothesis, and alignment.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Lists for metrics on text (the original code computes text-level metrics)
    text_hypotheses = []
    text_references = []
    
    # Also accumulate gloss outputs (if needed for separate reporting)
    gloss_hypotheses = []
    gloss_references = []

    frame_times = []
    translation_times = []
    
    # Initialize scorers
    chrf_scorer = CHRF()
    rouge_scorer_fn = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Measure frame processing time
            frame_start = time.time()
            frames = batch['frames'].to(device)
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Measure translation time
            trans_start = time.time()
            #gloss_output, text_output = model(frames, None, None)
            device = frames.device
            sos_gloss = torch.full((frames.size(0), 1), test_loader.dataset.gloss_to_idx['<sos>'], device=device, dtype=torch.long)
            sos_text = torch.full((frames.size(0), 1), test_loader.dataset.text_to_idx['<sos>'], device=device, dtype=torch.long)
            gloss_output, text_output = model(frames, sos_gloss, sos_text)
            translation_time = time.time() - trans_start
            translation_times.append(translation_time)
            
            # Get predictions for both gloss and text
            pred_gloss_indices = torch.argmax(gloss_output, dim=-1)
            pred_text_indices = torch.argmax(text_output, dim=-1)

            # For each sample in the batch, decode and print details
            batch_size = pred_text_indices.shape[0]
            for i in range(batch_size):
                # Decode gloss
                gloss_hyp = ' '.join([test_loader.dataset.idx_to_gloss[idx.item()]
                                      for idx in pred_gloss_indices[i] if idx.item() > 0])
                gloss_ref = ' '.join([test_loader.dataset.idx_to_gloss[idx.item()]
                                      for idx in batch['gloss'][i] if idx.item() > 0])
                
                # Decode text
                text_hyp = ' '.join([test_loader.dataset.idx_to_text[idx.item()]
                                     for idx in pred_text_indices[i] if idx.item() > 0])
                text_ref = ' '.join([test_loader.dataset.idx_to_text[idx.item()]
                                     for idx in batch['text'][i] if idx.item() > 0])
                
                gloss_hypotheses.append(gloss_hyp)
                gloss_references.append(gloss_ref)
                text_hypotheses.append(text_hyp)
                text_references.append(text_ref)
                
                # Compute alignments
                gloss_alignment = get_alignment(gloss_ref, gloss_hyp)
                text_alignment = get_alignment(text_ref, text_hyp)

                # Print sample evaluation
                logger.info(f"\n==========================================")
                logger.info(f"Sample Evaluation:")
                logger.info(f"---------- GLOSS ----------")
                logger.info(f"Reference: {gloss_ref}")
                logger.info(f"Hypothesis: {gloss_hyp}")
                logger.info(f"Alignment:\n{gloss_alignment}")
                logger.info(f"---------- TEXT ----------")
                logger.info(f"Reference: {text_ref}")
                logger.info(f"Hypothesis: {text_hyp}")
                logger.info(f"Alignment:\n{text_alignment}")
                logger.info(f"==========================================\n")
                
            # Free up memory
            del frames, gloss_output, text_output  # Free up memory
    
    # Calculate gloss-level BLEU score
    refs_for_bleu_g = [[ref.split()] for ref in gloss_references]
    bleu_score_g = corpus_bleu(refs_for_bleu_g, [hyp.split() for hyp in gloss_hypotheses],
                             smoothing_function=smooth.method1)
    
    chrf_score_g = chrf_scorer.corpus_score(gloss_hypotheses, [[ref] for ref in gloss_references]).score

    # Calculate ROUGE-L score (for gloss)
    rouge_scores_g = [rouge_scorer_fn.score(ref, hyp)['rougeL'].fmeasure
                    for ref, hyp in zip(gloss_references, gloss_hypotheses)]
    rouge_l_g = np.mean(rouge_scores)

    # Calculate Word Error Rate (WER)
    wer_g = calculate_wer(gloss_hypotheses, gloss_references)

    
    # Calculate text-level BLEU score
    refs_for_bleu = [[ref.split()] for ref in text_references]
    bleu_score = corpus_bleu(refs_for_bleu, [hyp.split() for hyp in text_hypotheses],
                             smoothing_function=smooth.method1)
    
    chrf_score = chrf_scorer.corpus_score(text_hypotheses, [[ref] for ref in text_references]).score
    
    # Calculate ROUGE-L score (for text)
    rouge_scores = [rouge_scorer_fn.score(ref, hyp)['rougeL'].fmeasure 
                    for ref, hyp in zip(text_references, text_hypotheses)]
    rouge_l = np.mean(rouge_scores)
    
    # Calculate Word Error Rate (WER)
    wer = calculate_wer(text_hypotheses, text_references)

    
    # Calculate timing metrics
    avg_frame_time = np.mean(frame_times)
    avg_translation_time = np.mean(translation_times)
    total_time = sum(frame_times) + sum(translation_times)
    
    metrics = {
        'bleu4_g': bleu_score_g * 100,  # Convert to percentage
        'chrf_g': chrf_score_g,
        'rouge_l_g': rouge_l_g * 100,  # Convert to percentage
        'wer_g': wer_g * 100,  # Convert to percentage
        'bleu4': bleu_score * 100,  # Convert to percentage
        'chrf': chrf_score,
        'rouge_l': rouge_l * 100,  # Convert to percentage
        'wer': wer * 100,  # Convert to percentage
        'frame_processing_time': avg_frame_time,
        'total_translation_time': total_time,
        'avg_translation_time': avg_translation_time
    }
    
    return metrics 
'''


def beam_search_decode(model, frame, sos_token, eos_token, beam_width=3, max_len=50, decoder='gloss'):
    device = frame.device
    model.eval()
    with torch.no_grad():
        # Compute memory once for this sample.
        memory = model.fusion_layer(torch.cat([
            model.visual_encoder(frame),
            model.emotion_encoder(frame),
            model.gesture_encoder(frame)
        ], dim=-1))
    
    beams = [([sos_token], 0.0)]
    for step in range(max_len - 1):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == eos_token:
                new_beams.append((seq, score))
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_len = seq_tensor.size(1)
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'), device=device),
                diagonal=1
            )
            if decoder == 'gloss':
                out = model.gloss_decoder(seq_tensor, memory, tgt_mask=tgt_mask)
            else:
                out = model.translation_decoder(seq_tensor, memory, tgt_mask=tgt_mask)
            logits = out[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Once <sos> has been output, we remove its probability.
            if sos_token in seq[1:]:
                probs[sos_token] = 1e-10

            topk_probs, topk_indices = torch.topk(probs, beam_width)
            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                new_seq = seq + [idx]
                new_score = score + math.log(prob + 1e-10)
                new_beams.append((new_seq, new_score))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
        if all(seq[-1] == eos_token for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    return best_seq

def evaluate_model(model, test_loader, config):
    """
    Evaluate model performance with detailed metrics.
    For each sample, prints gloss/text reference, hypothesis, and alignment.
    """
    device = next(model.parameters()).device
    model.eval()
    
    text_hypotheses = []
    text_references = []
    gloss_hypotheses = []
    gloss_references = []

    frame_times = []
    translation_times = []
    
    chrf_scorer = CHRF()
    rouge_scorer_fn = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction()
    
    max_decoding_length = config.get('max_decoding_length', 50)
    beam_width = config.get('beam_width', 3)
    
    from venv import logger

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            frame_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                frames = batch['frames'].to(device)
                end_event.record()
                torch.cuda.synchronize()
                frame_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                start_time = time.time()
                frames = batch['frames'].to(device)
                frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            trans_start = time.time()
            batch_size = frames.size(0)
            batch_gloss = []
            batch_text = []
            for i in range(batch_size):
                sample_frame = frames[i].unsqueeze(0)  # (1, T, C, H, W)
                sos_gloss = test_loader.dataset.gloss_to_idx['<sos>']
                eos_gloss = test_loader.dataset.gloss_to_idx['<eos>']
                sos_text = test_loader.dataset.text_to_idx['<sos>']
                eos_text = test_loader.dataset.text_to_idx['<eos>']
                
                decoded_gloss = beam_search_decode(
                    model, sample_frame, sos_gloss, eos_gloss,
                    beam_width=beam_width,
                    max_len=max_decoding_length,
                    decoder='gloss'
                )
                decoded_text = beam_search_decode(
                    model, sample_frame, sos_text, eos_text,
                    beam_width=beam_width,
                    max_len=max_decoding_length,
                    decoder='text'
                )
                batch_gloss.append(decoded_gloss)
                batch_text.append(decoded_text)
            translation_times.append(time.time() - trans_start)
            
            for i in range(batch_size):
                sos_gloss = test_loader.dataset.gloss_to_idx['<sos>']
                eos_gloss = test_loader.dataset.gloss_to_idx['<eos>']
                sos_text = test_loader.dataset.text_to_idx['<sos>']
                eos_text = test_loader.dataset.text_to_idx['<eos>']
                
                gloss_tokens = batch_gloss[i][1:]
                text_tokens = batch_text[i][1:]
                if eos_gloss in gloss_tokens:
                    gloss_tokens = gloss_tokens[:gloss_tokens.index(eos_gloss)]
                if eos_text in text_tokens:
                    text_tokens = text_tokens[:text_tokens.index(eos_text)]
                
                gloss_pred = ' '.join([test_loader.dataset.idx_to_gloss.get(idx, '<unk>') for idx in gloss_tokens])
                text_pred = ' '.join([test_loader.dataset.idx_to_text.get(idx, '<unk>') for idx in text_tokens])
                gloss_hypotheses.append(gloss_pred)
                text_hypotheses.append(text_pred)
                
                gloss_ref = ' '.join([test_loader.dataset.idx_to_gloss[idx.item()] for idx in batch['gloss'][i] if idx.item() > 0])
                text_ref = ' '.join([test_loader.dataset.idx_to_text[idx.item()] for idx in batch['text'][i] if idx.item() > 0])
                gloss_references.append(gloss_ref)
                text_references.append(text_ref)
                
                gloss_alignment = get_alignment(gloss_ref, gloss_pred)
                text_alignment = get_alignment(text_ref, text_pred)
                
                logger.info(f"\n==========================================")
                logger.info("Sample Evaluation:")
                logger.info("---------- GLOSS ----------")
                logger.info(f"Reference: {gloss_ref}")
                logger.info(f"Hypothesis: {gloss_pred}")
                logger.info(f"Alignment:\n{gloss_alignment}")
                logger.info("---------- TEXT ----------")
                logger.info(f"Reference: {text_ref}")
                logger.info(f"Hypothesis: {text_pred}")
                logger.info(f"Alignment:\n{text_alignment}")
                logger.info("==========================================\n")
            del frames
        
    refs_for_bleu_g = [[ref.split()] for ref in gloss_references]
    bleu_score_g = corpus_bleu(refs_for_bleu_g, [hyp.split() for hyp in gloss_hypotheses],
                               smoothing_function=smooth.method1)
    chrf_score_g = chrf_scorer.corpus_score(gloss_hypotheses, [[ref] for ref in gloss_references]).score
    rouge_scores_g = [rouge_scorer_fn.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(gloss_references, gloss_hypotheses)]
    rouge_l_g = np.mean(rouge_scores_g)
    wer_g = calculate_wer(gloss_hypotheses, gloss_references)
    
    refs_for_bleu = [[ref.split()] for ref in text_references]
    bleu_score = corpus_bleu(refs_for_bleu, [hyp.split() for hyp in text_hypotheses],
                             smoothing_function=smooth.method1)
    chrf_score = chrf_scorer.corpus_score(text_hypotheses, [[ref] for ref in text_references]).score
    rouge_scores = [rouge_scorer_fn.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(text_references, text_hypotheses)]
    rouge_l = np.mean(rouge_scores)
    wer = calculate_wer(text_hypotheses, text_references)
    
    avg_frame_time = np.mean(frame_times)
    avg_translation_time = np.mean(translation_times)
    total_time = sum(frame_times) + sum(translation_times)
    
    metrics = {
        'bleu4_g': bleu_score_g * 100,
        'chrf_g': chrf_score_g,
        'rouge_l_g': rouge_l_g * 100,
        'wer_g': wer_g * 100,
        'bleu4': bleu_score * 100,
        'chrf': chrf_score,
        'rouge_l': rouge_l * 100,
        'wer': wer * 100,
        'frame_processing_time': avg_frame_time,
        'total_translation_time': total_time,
        'avg_translation_time': avg_translation_time
    }
    
    return metrics