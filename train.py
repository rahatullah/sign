import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from models.sign_language_model import SignLanguageTranslator
from models.dataset import SignLanguageDataset
import os
import platform
import time
from test_utils import load_test_config, evaluate_model, find_best_checkpoint
import nltk
import sys
import logging
from datetime import datetime
import json
import psutil

# Only import NLTK-related modules in the main process
nltk_imports = None
def get_nltk_imports():
    global nltk_imports
    if nltk_imports is None:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from sacrebleu import CHRF
        from rouge_score import rouge_scorer
        nltk_imports = {
            'corpus_bleu': corpus_bleu,
            'SmoothingFunction': SmoothingFunction,
            'CHRF': CHRF,
            'rouge_scorer': rouge_scorer
        }
    return nltk_imports

def setup_nltk():
    if __name__ == '__main__':  # Only run in main process
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

# Set a fixed NLTK data directory so that every process sees the same data
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')

def worker_init_fn(worker_id):
    import nltk
    # Ensure workers use the same fixed NLTK data directory
    os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')
    # Override nltk.download so that no download attempts are made in worker processes
    nltk.download = lambda *args, **kwargs: None

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Setup logging configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def get_vocab_config(dataset):
    """Get vocabulary configuration from dataset"""
    return {
        'gloss_to_idx': dataset.gloss_to_idx,
        'idx_to_gloss': dataset.idx_to_gloss,
        'text_to_idx': dataset.text_to_idx,
        'idx_to_text': dataset.idx_to_text,
    }

def run_prechecks():
    """Run all pre-training checks"""
    checks = []
    
    # Check CUDA - but don't fail if not available
    if torch.cuda.is_available():
        checks.append(("CUDA available", True))
        try:
            # Test CUDA operations
            x = torch.randn(1, device='cuda')
            y = x * 2
            del x, y
            checks.append(("CUDA operations", True))
        except Exception as e:
            checks.append(("CUDA operations", False, str(e)))
            logging.warning("CUDA available but operations failed. Falling back to CPU.")
    else:
        checks.append(("CUDA available", False, "Training will proceed on CPU (slower but functional)"))
    
    # Check directories
    all_dirs_exist = True
    for dir_name in ['logs', 'models', 'data/data/videos', 'data/data/annotations']:
        exists = os.path.exists(dir_name)
        checks.append((f"Directory {dir_name}", exists))
        if not exists:
            all_dirs_exist = False
    
    # Check disk space
    try:
        free_space = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail
        free_gb = free_space / (1024**3)
        if free_gb < 10:  # Need at least 10GB
            checks.append(("Disk space", False, f"Only {free_gb:.1f}GB available"))
            raise RuntimeError(f"Insufficient disk space. Need at least 10GB, but only {free_gb:.1f}GB available")
        else:
            checks.append(("Disk space", True, f"{free_gb:.1f}GB available"))
    except Exception as e:
        if "statvfs" not in str(e):  # Only add to checks if it's not the statvfs error
            checks.append(("Disk space", False, "Couldn't check disk space"))
    
    # Print results
    logging.info("\n=== Pre-training Checks ===")
    for check in checks:
        if len(check) == 2:
            name, passed = check
            msg = ""
        else:
            name, passed, msg = check
        status = "✓" if passed else "✗"
        logging.info(f"{status} {name}: {msg if msg else ('OK' if passed else 'FAILED')}")
    
    # Only fail if critical checks fail (directories and disk space)
    if not all_dirs_exist:
        raise RuntimeError("Required directories are missing. Please create them before proceeding.")
    
    logging.info("Pre-training checks completed.\n")

def save_model_and_metrics(model, epoch, avg_loss, best_metrics, vocab_config, timestamp, is_best=False):
    """Save model checkpoint and training metrics"""
    try:
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'train_loss': avg_loss,
            'best_metrics': best_metrics,
            'vocab_config': vocab_config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = f'models/checkpoint_epoch_{epoch+1}_{timestamp}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = f'models/best_model_checkpoint.pt'
            torch.save(checkpoint, best_model_path)
            logging.info(f"Saved best model to {best_model_path}")
        
        # Save training metrics
        metrics = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'best_metrics': best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_file = f'logs/metrics_{timestamp}.jsonl'
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
        return checkpoint_path
            
    except Exception as e:
        logging.error(f"Error saving model and metrics: {str(e)}")
        return None

def train(batch_size=None, epochs=50, learning_rate=1e-4, checkpoint_path=None, 
          video_dir=None, annotations_dir=None, timestamp=None):
    """
    Train the sign language translation model.
    
    Args:
        batch_size (int): Batch size for training (auto-selected based on device)
        epochs (int): Number of epochs to train (default: 50)
        learning_rate (float): Learning rate
        checkpoint_path (str): Path to checkpoint to resume training from
        video_dir (str): Directory containing video data
        annotations_dir (str): Directory containing annotation files
    """
    try:
        logging.info("Starting training with configuration:")
        logging.info(f"Epochs: {epochs}")
        logging.info(f"Learning rate: {learning_rate}")
        logging.info(f"Checkpoint path: {checkpoint_path}")
        
        # Run pre-checks
        run_prechecks()
        
        if video_dir is None or annotations_dir is None:
            video_dir = os.path.join('data', 'data', 'videos')
            annotations_dir = os.path.join('data', 'data', 'annotations')
            logging.info(f"Using default paths:\nVideos: {video_dir}\nAnnotations: {annotations_dir}")
        
        # Validate paths
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        if not os.path.exists(annotations_dir):
            raise ValueError(f"Annotations directory not found: {annotations_dir}")

        # Device configuration - with proper error handling
        device = torch.device('cpu')
        is_gpu = False
        
        if torch.cuda.is_available():
            try:
                # Test CUDA device
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                device = torch.device('cuda:0')
                is_gpu = True
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            except Exception as e:
                logging.warning(f"CUDA initialization failed: {str(e)}")
                logging.info("Falling back to CPU training")
        else:
            logging.info("No GPU available. Training on CPU")
        
        # Memory optimization settings
        if is_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.set_per_process_memory_fraction(0.75)
            
        # Dataset and model parameters
        batch_size = 1
        gradient_accumulation_steps = 32 if is_gpu else 8  # Reduced for CPU
        max_frames = 8
        
        # Dataset with reduced memory usage
        train_dataset = SignLanguageDataset(
            video_dir=video_dir,
            annotations_dir=annotations_dir,
            split='train',
            max_frames=max_frames
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )

        logging.info("Initializing minimal model configuration...")
        gloss_vocab_size, text_vocab_size = train_dataset.get_vocab_sizes()
        
        # Clear GPU memory before model creation
        if is_gpu:
            torch.cuda.empty_cache()
            logging.info(f"Memory before model: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Create minimal model
        model = SignLanguageTranslator(
            gloss_vocab_size=gloss_vocab_size,
            text_vocab_size=text_vocab_size,
            d_model=64,  # Minimal model size
            nhead=2,     # Reduced attention heads
            num_decoder_layers=2  # Minimal decoder layers
        )
        
        logging.info("Moving model to device...")
        model = model.to('cpu')  # Ensure on CPU first
        
        if is_gpu:
            # Move model to GPU carefully
            try:
                model = model.to(device)
                logging.info(f"Model loaded to GPU. Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            except RuntimeError as e:
                logging.error(f"Failed to load model to GPU. Error: {e}")
                return None
        
        # Single GPU only - disable DataParallel to save memory
        if is_gpu and torch.cuda.device_count() > 1:
            logging.info("Multiple GPUs detected but using single GPU to save memory")
        
        # Memory efficient optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if is_gpu else None

        # Training state
        start_epoch = 0
        best_metrics = {'bleu4': 0.0, 'chrf': 0.0, 'rouge_l': 0.0}

        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info("Loading checkpoint...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else: 
                    logging.warning("No optimizer_state_dict found in checkpoint; skipping optimizer load.")
                start_epoch = checkpoint['epoch'] + 1
                best_metrics = checkpoint.get('best_metrics', best_metrics)
                logging.info(f"Resumed from epoch {start_epoch}")
                del checkpoint
                if is_gpu:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                logging.error(f"Failed to load checkpoint. Error: {e}")
                return None

        logging.info("\nStarting training...")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        logging.info(f"Max frames per video: {max_frames}")
        
        # Get vocab config once before training
        vocab_config = get_vocab_config(train_dataset)
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(start_epoch, epochs):
            try:
                epoch_start_time = time.time()
                if is_gpu:
                    torch.cuda.empty_cache()
                    logging.info(f"\nGPU memory at epoch start: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                
                logging.info(f"\n{'='*40}")
                logging.info(f"Epoch {epoch+1}/{epochs}")
                logging.info(f"{'='*40}")
                
                model.train()
                epoch_loss = 0
                total_samples = 0
                optimizer.zero_grad(set_to_none=True)
                progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        # Process single frame
                        frames = batch['frames'].to(device, non_blocking=True)
                        gloss = batch['gloss'].to(device, non_blocking=True)
                        text = batch['text'].to(device, non_blocking=True)
                        
                        with torch.amp.autocast(device_type='cuda', enabled=is_gpu):
                            gloss_out, text_out = model(frames, gloss_targets=gloss, text_targets=text)
                            loss = (criterion(gloss_out.view(-1, gloss_out.size(-1)), gloss.view(-1)) +
                                   criterion(text_out.view(-1, text_out.size(-1)), text.view(-1))) / gradient_accumulation_steps
                        
                        loss_value = loss.item() * gradient_accumulation_steps
                        
                        if is_gpu:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            if is_gpu:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                        
                        # Immediate cleanup
                        del frames, gloss, text, gloss_out, text_out, loss
                        if is_gpu:
                            torch.cuda.empty_cache()
                        
                        epoch_loss += loss_value
                        total_samples += batch_size
                        progress_bar.set_postfix({
                            'loss': f"{loss_value:.4f}",
                            'gpu_mem': f"{torch.cuda.memory_allocated()/1024**2:.1f}MB" if is_gpu else "N/A"
                        })
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e) and is_gpu:
                            torch.cuda.empty_cache()
                            logging.warning(f"\nWARNING: OOM in batch {batch_idx}. Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        raise e

                # Epoch statistics
                avg_loss = epoch_loss / total_samples
                epoch_time = time.time() - epoch_start_time
                
                logging.info(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")
                logging.info(f"Epoch time: {epoch_time:.2f}s")
                
                # Save checkpoint and metrics
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                    logging.info(f"New best loss: {best_loss:.4f}")
                
                if (epoch + 1) % 5 == 0 or is_best:  # Save every 5 epochs or if best model
                    save_model_and_metrics(model, epoch, avg_loss, best_metrics, 
                                         vocab_config, timestamp, is_best)

            except Exception as e:
                logging.error(f"Error during epoch {epoch + 1}: {str(e)}")
                # Save emergency checkpoint
                save_model_and_metrics(model, epoch, epoch_loss/total_samples if total_samples > 0 else float('inf'),
                                     best_metrics, vocab_config, timestamp, is_best=False)
                raise e

        logging.info("Training completed successfully!")
        return model

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise e

def save_metrics_to_file(metrics, timestamp, phase="test"):
    """Save detailed metrics to a structured log file"""
    metrics_file = f'logs/detailed_metrics_{phase}_{timestamp}.json'
    
    formatted_metrics = {
        'timestamp': datetime.now().isoformat(),
        'phase': phase,
        'translation_metrics': {
            'bleu4': metrics.get('bleu4', 0.0),
            'chrf': metrics.get('chrf', 0.0),
            'rouge_l': metrics.get('rouge_l', 0.0),
            'wer': metrics.get('wer', 0.0)
        },
        'performance_metrics': {
            'avg_processing_time_per_frame': metrics.get('frame_processing_time', 0.0),
            'avg_translation_time': metrics.get('total_translation_time', 0.0),
            'memory_usage': {
                'gpu': metrics.get('gpu_memory_used', 0.0) if torch.cuda.is_available() else 'N/A',
                'cpu': metrics.get('cpu_memory_used', 0.0)
            }
        },
        'robustness_metrics': {
            'noise_sensitivity': metrics.get('noise_sensitivity', 'Not tested'),
            'signer_variation': metrics.get('signer_variation', 'Not tested'),
            'background_variation': metrics.get('background_variation', 'Not tested')
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(formatted_metrics, f, indent=4)
    logging.info(f"Detailed metrics saved to {metrics_file}")

def test(timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load best checkpoint
    best_checkpoint = find_best_checkpoint()
    #checkpoint = torch.load(best_checkpoint)
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    
    # Initialize model
    model = SignLanguageTranslator(
        gloss_vocab_size=len(checkpoint['vocab_config']['gloss_to_idx']),
        text_vocab_size=len(checkpoint['vocab_config']['text_to_idx']),
        d_model=64,            # Use same d_model as training
        nhead=2,               # Same number of heads
        num_decoder_layers=2   # Same number of layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test configuration from test.json
    config = load_test_config('test.json')
    
    # Create test dataset and DataLoader
    test_dataset = SignLanguageDataset(
        video_dir=os.path.join('data', 'data', 'videos'),
        annotations_dir=os.path.join('data', 'data', 'annotations'),
        split='test',
        max_frames=config['image_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )
    
    # Evaluate with timing
    start_time = time.time()
    metrics = evaluate_model(model, test_loader, config)
    total_time = time.time() - start_time
    
    # Add timing metrics
    metrics['total_translation_time'] = total_time  # Total evaluation time

    # Log and save metrics
    logging.info("\nTest Results:")
    logging.info("=" * 40)
    logging.info("Translation Metrics:")
    logging.info(f"Gloss WER Score: {metrics['wer_g']:.4f}")
    logging.info(f"Gloss BLEU-4 Score: {metrics['bleu4_g']:.4f}")
    logging.info(f"Gloss CHRF Score: {metrics['chrf_g']:.4f}")
    logging.info(f"Gloss ROUGE-L Score: {metrics['rouge_l_g']:.4f}")
    logging.info(f"Text WER Score: {metrics['wer']:.4f}")
    logging.info(f"Text BLEU-4 Score: {metrics['bleu4']:.4f}")
    logging.info(f"Text CHRF Score: {metrics['chrf']:.4f}")
    logging.info(f"Text ROUGE-L Score: {metrics['rouge_l']:.4f}")
    
    logging.info("\nPerformance Metrics:")
    logging.info(f"Avg. Processing Time per Frame: {metrics['frame_processing_time']:.4f}s")
    logging.info(f"Total Translation Time: {metrics['total_translation_time']:.4f}s")
    logging.info(f"GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A")
    logging.info(f"CPU Memory Used: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
    

    # Save detailed metrics
    save_metrics_to_file(metrics, timestamp)
    
    return metrics

def main():
    timestamp = setup_logging()
    logging.info("Starting sign language translation training")
    
    try:
        setup_nltk()
        get_nltk_imports()
        
        epochs = 4  # Reduced to 3 epochs
        model = train(
            batch_size=1,
            epochs=epochs,
            learning_rate=1e-4,
            checkpoint_path='models/best_model_checkpoint.pt',
            video_dir=os.path.join('data', 'data', 'videos'),
            annotations_dir=os.path.join('data', 'data', 'annotations'),
            timestamp=timestamp
        )
        
        if model is not None:
            # Get the dataset for vocab config
            train_dataset = SignLanguageDataset(
                video_dir=os.path.join('data', 'data', 'videos'),
                annotations_dir=os.path.join('data', 'data', 'annotations'),
                split='train',
                max_frames=8
            )
            # Save final model state
            save_model_and_metrics(model, epochs-1, float('inf'), {}, 
                                 get_vocab_config(train_dataset), timestamp, is_best=False)
            
            # Run evaluation with detailed metrics
            logging.info("\nRunning comprehensive evaluation...")
            test(timestamp=timestamp)  # Pass timestamp to test function
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise e

if __name__ == '__main__':
    # Set environment variables for CUDA
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Disable CUDA if it's causing issues
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logging.warning("CUDA initialization failed. Disabled CUDA devices.")
    
    main()