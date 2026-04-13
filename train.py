import warnings
warnings.filterwarnings("ignore")

import os
# Suppress TensorFlow logging and factory registration warnings before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Prevent PyTorch DataParallel bugs on multi-GPU Colab instances (like T4x2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import json
import pandas as pd
from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection, TrainingArguments, Trainer, TrainerCallback
from PIL import Image
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm
import albumentations as A

def get_train_transforms(size=800):
    """Augmentation pipeline tuned for small-object detection (especially Platelets).

    Key design choices:
    - RandomSizedBBoxSafeCrop with erosion_rate=0.0 ensures that bounding boxes
      are never partially cropped, which is critical for tiny platelet boxes.
    - The crop scale range (0.3, 1.0) forces frequent zoom-in events, making
      platelets appear larger relative to the image frame.
    - ShiftScaleRotate with a scale_limit up to +20% provides additional zoom
      diversity without risking box truncation (BboxParams handles filtering).
    """
    return A.Compose([
        # Zoom into a random region that is guaranteed to keep all boxes intact.
        # scale=(0.3, 1.0) means we can crop down to 30% of the original area,
        # effectively zooming in by up to ~3x — making platelets much more visible.
        A.RandomSizedBBoxSafeCrop(height=size, width=size, erosion_rate=0.0, p=0.9),

        # Standard geometric flips for rotation-invariant cell detection.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        # Photo-metric jitter to handle variation in microscope staining.
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),

        # Additional scale variation (zoom in up to 20% more).
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.4),

    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.3, min_area=64))

def build_platelet_weighted_sampler(train_dataset, platelet_cat_id=2, boost_factor=4.0):
    """Returns a WeightedRandomSampler that over-samples images containing Platelets.

    Platelets are severely under-represented (286 instances vs 3307 RBCs).
    By giving platelet images a `boost_factor` higher probability of being sampled
    each epoch, the model sees roughly balanced platelet / non-platelet batches,
    which directly improves the gradient signal for the small-object detection heads.

    Args:
        train_dataset: CocoDetection instance.
        platelet_cat_id: 0-indexed category id for Platelets (2 after shift in dataset).
        boost_factor: How many times more likely a platelet image is to be sampled.
    """
    # Note: category ids inside the dataset are already shifted to 0-indexed.
    weights = []
    for img_id in train_dataset.ids:
        ann_ids = train_dataset.coco.getAnnIds(imgIds=img_id)
        anns = train_dataset.coco.loadAnns(ann_ids)
        # Platelets have original id=3 in the BCCD dataset.
        # We check the original 'category_id' stored in the COCO database.
        has_platelet = any(a['category_id'] == 3 for a in anns)
        weights.append(boost_factor if has_platelet else 1.0)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, processor, transforms=None):
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.processor = processor
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(path).convert("RGB")

        # Clean annotations (Remove completely invalid boxes with width <= 0 or height <= 0)
        annotations = [ann for ann in annotations if ann['bbox'][2] > 0 and ann['bbox'][3] > 0]

        # Extract labels (Deformable DETR requires 0-indexed labels: 0, 1, 2)
        # We shift them into a local list to avoid corrupting the underlying dataset in-place.
        category_ids = [ann['category_id'] - 1 for ann in annotations]
        boxes = [ann['bbox'] for ann in annotations]

        # Apply Albumentations if provided
        if self.transforms is not None and len(boxes) > 0:
            image_np = np.array(image)
            transformed = self.transforms(image=image_np, bboxes=boxes, category_ids=category_ids)
            image = Image.fromarray(transformed['image'])
            
            # Rebuild annotations based on surviving bounding boxes
            new_annotations = []
            for i, (bbox, cat_id) in enumerate(zip(transformed['bboxes'], transformed['category_ids'])):
                new_annotations.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": list(bbox),
                    "id": i,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3]
                })
            annotations = new_annotations

        # ====================================================
        # FINAL SAFETY NET (Prevents CUDA GIoU Loss Assertion)
        # ====================================================
        safe_annotations = []
        w_img, h_img = image.size
        for ann in annotations:
            x, y, w, h = ann['bbox']
            
            # Clamp coordinates to physical image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            # Box must be strictly positive and physically real (> 1 pixel)
            if w >= 1.0 and h >= 1.0:
                ann['bbox'] = [x, y, w, h]
                ann['area'] = w * h
                safe_annotations.append(ann)
                
        annotations = safe_annotations

        target = {
            "image_id": torch.tensor([img_id]),
            "annotations": annotations
        }
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
        labels = encoding["labels"][0] 
        
        return {"pixel_values": pixel_values, "labels": labels}

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    """Manual collation for Deformable DETR.
    
    Pads images in the batch to the same size (max height/width) and 
    creates the pixel_mask required by the model. This avoids using 
    processor.pad which has version-specific incompatibilities on some platforms.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Find the maximum dimensions in this batch
    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)
    
    batch_size = len(pixel_values)
    # Initialize padded tensors (zeros)
    # pixel_values are already normalized floats from the processor
    padded_pixel_values = torch.zeros((batch_size, 3, max_h, max_w), dtype=pixel_values[0].dtype)
    # pixel_mask: 1 for real pixels, 0 for padded pixels
    pixel_mask = torch.zeros((batch_size, max_h, max_w), dtype=torch.long)
    
    for i, p in enumerate(pixel_values):
        h, w = p.shape[1], p.shape[2]
        padded_pixel_values[i, :, :h, :w] = p
        pixel_mask[i, :h, :w] = 1
        
    return {
        "pixel_values": padded_pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels
    }

# --- mAP Calculation Callback ---
class MAPCallback(TrainerCallback):
    def __init__(self, val_dataset, processor, id2label, save_path, threshold=0.1):
        self.val_dataset = val_dataset
        self.processor = processor
        self.id2label = id2label
        self.save_path = save_path
        self.threshold = threshold
        self.best_map = -1.0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n Calculating mAP for Epoch {state.epoch}...")
        
        # Clear training memory before the large attention-heavy validation loop
        torch.cuda.empty_cache()
        
        model.eval()
        results = []
        coco_gt = self.val_dataset.coco
        
        # --- 1. Calculate mAP ---
        for idx in tqdm(range(len(self.val_dataset))):
            # Process single image
            img_id = self.val_dataset.ids[idx]
            img_info = coco_gt.loadImgs(img_id)[0]
            path = os.path.join(self.val_dataset.img_folder, img_info['file_name'])
            image = Image.open(path).convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            post_processed = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]
            
            for score, label, box in zip(post_processed["scores"], post_processed["labels"], post_processed["boxes"]):
                box = box.tolist()
                coco_box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                results.append({
                    "image_id": img_id,
                    "category_id": label.item() + 1, # Shift back to 1-indexed for COCOeval
                    "bbox": coco_box,
                    "score": score.item()
                })
        
        if not results:
            print(" No detections found for mAP calculation.")
            return

        # Use COCOeval
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            res_file = f.name
            
        try:
            coco_dt = coco_gt.loadRes(res_file)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            # Suppress default COCO output and capture it
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                coco_eval.summarize()
            
            
            # Extract mAP (IoU=0.50:0.95, area=all, maxDets=100)
            map_value = coco_eval.stats[0]
            map_50 = coco_eval.stats[1]
            
            print(f" Epoch {state.epoch} mAP: {map_value:.4f} (mAP@50: {map_50:.4f})")
            
            # --- 2. Save Best Model ---
            if map_value > self.best_map:
                self.best_map = map_value
                best_model_dir = os.path.join(self.save_path, "best_model")
                print(f" New Best mAP! Saving model to {best_model_dir}...")
                model.save_pretrained(best_model_dir)
                self.processor.save_pretrained(best_model_dir)
            
            # Manually trigger a log event so it's saved to CSV
            trainer.log({"eval_map": map_value, "eval_map_50": map_50})
            
        finally:
            if os.path.exists(res_file):
                os.remove(res_file)

# --- CSV Logging Callback (Updated for mAP) ---
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = logs.copy()
            entry["epoch"] = round(state.epoch, 2) if state.epoch else 0
            entry["step"] = state.global_step
            
            # Merge with existing logs for same step if possible (handling eval logs)
            if self.logs and self.logs[-1]["step"] == state.global_step:
                self.logs[-1].update(entry)
            else:
                self.logs.append(entry)
            
            pd.DataFrame(self.logs).to_csv(self.log_path, index=False)
            
            # --- Live Plotting for Notebooks ---
            try:
                from IPython.display import clear_output, display
                from plot_training import plot_metrics
                
                # Plot every time evaluation metrics or a significant number of steps are logged
                if len(self.logs) > 1:
                    clear_output(wait=True)
                    plot_metrics(self.log_path)
            except ImportError:
                pass
            except Exception as e:
                # Silently skip if error occurs during live plotting (e.g. file lock)
                pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Deformable DETR on the BCCD blood cell dataset.")
    parser.add_argument(
        "--tiled", action="store_true",
        help=(
            "Train on 2×2 tiled images (generated by tile.py). "
            "Tiles double the effective size of Platelets, significantly "
            "improving small-object detection. Run tile.py first."
        ),
    )
    parser.add_argument(
        "--no_augmentation", action="store_true",
        help="Disable Albumentations data augmentation (for ablation experiments).",
    )
    args = parser.parse_args()

    USE_DATA_AUGMENTATION = not args.no_augmentation
    USE_TILING = args.tiled

    # ==========================================
    #  CONFIGURATION SETTINGS
    # ==========================================
    print(f"[{'ENABLED' if USE_DATA_AUGMENTATION else 'DISABLED'}] Data Augmentation (Albumentations)")
    print(f"[{'ENABLED' if USE_TILING else 'DISABLED'}] Image Tiling (2×2 grid)")
    if USE_DATA_AUGMENTATION:
        print(" STRATEGY: Box-Centric Zoom (Focused on Cell Dimensions)")
        print(" RESOLUTION: 960px (Ultra-High Detail Mode)")
    
    # --- GPU Verification ---
    print("-" * 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f" GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(" Training will run with Hardware Acceleration.")
    else:
        print(" WARNING: NO GPU DETECTED.")
        print("Training will run on CPU and will be extremely slow.")
        print("Please enable GPU in Colab (Runtime -> Change runtime type -> T4 GPU).")
    print("-" * 50)

    # Predefined class mapping (0-indexed for Deformable DETR)
    id2label = {0: "WBC", 1: "RBC", 2: "Platelets"}
    label2id = {v: k for k, v in id2label.items()}
    
    # Initialize processor with standard resolution (Tiling already acts as a zoom)
    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr", size=800, max_size=1333)
    
    # Datasets (Now passing the base resolution to consistent transforms)
    aug_transforms = get_train_transforms(size=960) if USE_DATA_AUGMENTATION else None

    if USE_TILING:
        train_img_dir = "blood_cell_data/train_tiled"
        train_ann_file = "blood_cell_data/train_tiled.json"
        if not os.path.exists(train_ann_file):
            raise FileNotFoundError(
                "Tiled dataset not found. Generate it first:\n"
                "  uv run python tile.py"
            )
        print(f" Using tiled training data: {train_img_dir}")
    else:
        train_img_dir = "blood_cell_data/train"
        train_ann_file = "blood_cell_data/train.json"

    train_dataset = CocoDetection(train_img_dir, train_ann_file, processor, transforms=aug_transforms)
    #  Validation MUST NEVER have arbitrary transformations applied.
    #    Evaluation always runs on full-resolution images so metrics are comparable.
    val_dataset = CocoDetection("blood_cell_data/val", "blood_cell_data/val.json", processor, transforms=None)
    
    # --- Model ---
    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Initialize weighted sampler to handle class imbalance (11:1 RBC to Platelet)
    platelet_sampler = build_platelet_weighted_sampler(train_dataset, boost_factor=4.0)
    print("[CONFIG] WeightedRandomSampler: Platelet images boosted 4× per epoch.")

    # --- SAVE PATH ---
    save_path = "blood_cell_model"
    if os.path.exists("/content/drive/MyDrive"):
        save_path = "/content/drive/MyDrive/blood_cell_models/detr_ep50_model"
        os.makedirs(save_path, exist_ok=True)
    if os.path.exists("/kaggle/working"):
        save_path = "/kaggle/working/blood_cell_model"
        os.makedirs(save_path, exist_ok=True)

    # Define training arguments with Cosine LR schedule and linear warmup
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=1,        # Reduced to 1 to guarantee it fits on a 16GB T4 GPU
        gradient_accumulation_steps=2,        # Accumulate to maintain an effective batch size of 2
        num_train_epochs=30,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",   # Smooth decay to near-zero at epoch 50
        warmup_ratio=0.05,            # Warm up over first 5% of total steps
        save_steps=500,
        logging_steps=10,
        logging_dir="logs",
        report_to="tensorboard",
        eval_strategy="no",           # Disable redundant native evaluation to save memory
        save_strategy="epoch",
        logging_strategy="steps",
        load_best_model_at_end=False, # Must be False if eval_strategy is "no"
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=True if device == "cuda" else False,
        dataloader_num_workers=2,
        per_device_eval_batch_size=1, # Safety for internal eval calls
        save_total_limit=3,           # Prevent disk exhaustion on Kaggle (keeps last 3 + best)
    )

    csv_log_path = os.path.join(save_path, "training_progress.csv")
    csv_callback = CSVLoggerCallback(csv_log_path)
    map_callback = MAPCallback(val_dataset, processor, id2label, save_path)

    # Trainer — pass the sampler via train_dataloader override
    class TrainerWithSampler(Trainer):
        def _get_train_sampler(self, dataset=None):
            return platelet_sampler

    trainer = TrainerWithSampler(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[csv_callback, map_callback],
    )

    print(f"Starting training for 50 Epochs...")
    print(f"Logs will be saved to: {csv_log_path}")
    trainer.train()

    print(f"--- Saving Final Model to {save_path} ---")
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    print(f"--- Model Saved Successfully to '{save_path}' ---")
