import os
os.environ['MPLBACKEND'] = 'Agg'
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from tile import get_tiles          # reuse tile geometry from tile.py

def nms(boxes_xyxy, scores, labels, iou_threshold=0.5):
    """
    Class-aware Pure-Python NMS. 
    Only suppresses a box if it overlaps with a higher-scoring box of the SAME class.
    Returns indices of kept boxes sorted by score descending.
    """
    if len(boxes_xyxy) == 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        remaining = []
        for j in order:
            # Only suppress if labels are the same
            if labels[i] == labels[j]:
                b1, b2 = boxes_xyxy[i], boxes_xyxy[j]
                ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                iou = inter / (a1 + a2 - inter + 1e-6)
                if iou < iou_threshold:
                    remaining.append(j)
            else:
                # Different classes never suppress each other
                remaining.append(j)
        order = remaining
    return kept


def tiled_inference(image_path, model, processor, img_id,
                    score_threshold=0.1, nms_iou=0.5):
    """
    Run inference on a full image by splitting it into 2x2 overlapping tiles,
    converting predictions back to full-image coordinates, and merging with NMS.
    Returns a list of COCO-format prediction dicts ready for COCOeval.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    tiles = get_tiles(img_w, img_h)

    all_boxes, all_scores, all_labels = [], [], []

    for (tx1, ty1, tx2, ty2) in tiles:
        tile_img = image.crop((tx1, ty1, tx2, ty2))
        inputs = processor(images=tile_img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        tile_h_px = ty2 - ty1
        tile_w_px = tx2 - tx1
        target_sizes = torch.tensor([[tile_h_px, tile_w_px]]).to(model.device)
        preds = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        for score, label, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
            bx1, by1, bx2, by2 = box.tolist()
            # Shift from tile-local → full-image coordinates
            all_boxes.append([bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1])
            all_scores.append(score.item())
            all_labels.append(label.item())

    if not all_boxes:
        return []

    # Class-aware NMS across all tile predictions
    kept = nms(all_boxes, all_scores, all_labels, iou_threshold=nms_iou)

    results = []
    for i in kept:
        bx1, by1, bx2, by2 = all_boxes[i]
        results.append({
            "image_id": img_id,
            "category_id": all_labels[i] + 1,   # shift back to 1-indexed for COCOeval
            "bbox": [bx1, by1, bx2 - bx1, by2 - by1],
            "score": all_scores[i],
        })
    return results


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x_min, y_min, x_max, y_max]
    box1_min_max = [x1, y1, x1+w1, y1+h1]
    box2_min_max = [x2, y2, x2+w2, y2+h2]
    
    # Intersection
    xA = max(box1_min_max[0], box2_min_max[0])
    yA = max(box1_min_max[1], box2_min_max[1])
    xB = min(box1_min_max[2], box2_min_max[2])
    yB = min(box1_min_max[3], box2_min_max[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    # Union
    box1Area = w1 * h1
    box2Area = w2 * h2
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def visualize_prediction(image_path, model, processor, id2label, output_path, gt_boxes=None, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    
    draw = ImageDraw.Draw(image)
    
    # Draw Ground Truth in Green
    if gt_boxes:
        for box, cat_id in gt_boxes:
            # GT format is [x, y, w, h]
            x, y, w, h = box
            draw.rectangle([x, y, x+w, y+h], outline="green", width=3)
            draw.text((x, max(0, y-15)), f"GT: {id2label[cat_id]}", fill="green")
            
    # Draw Predictions in Red
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{id2label[label.item()]}: {round(score.item(), 2)}", fill="red")
        
    image.save(output_path)

def generate_pr_curve(coco_eval, class_names, output_path):
    """Extract precision matrix from COCOeval and plot PR curve at IoU=0.50 for each class."""
    # precision matrix shape: [T, R, K, A, M]
    # T=iou (first index 0 is 0.50), R=recall levels (101), K=categories (3), A=areas (first index 0 is all), M=maxDets (index 2 is 100)
    try:
        precision = coco_eval.eval['precision'][0, :, :, 0, 2]
        
        plt.figure(figsize=(10, 6))
        recall_levels = np.linspace(0, 1, 101)
        for i, name in enumerate(class_names):
            plt.plot(recall_levels, precision[:, i], label=f'{name}')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (IoU=0.50)')
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved PR Curve to {output_path}")
    except Exception as e:
        print(f" Could not generate PR curve: {e}")

def generate_confusion_matrix(coco_gt, predictions, class_names, num_classes, output_path, iou_thresh=0.50):
    """
    Build an object detection confusion matrix (including Background class for False Positives/Negatives).
     predictions: list of dicts {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    """
    # Matrix shape: (num_classes + 1) x (num_classes + 1)
    # The extra row/col is 'Background'
    # Rows: Ground Truth (0..num_classes-1, num_classes=Background)
    # Cols: Predictions (0..num_classes-1, num_classes=Background)
    
    bg_idx = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    img_ids = coco_gt.getImgIds()
    
    for img_id in img_ids:
        # Get Ground Truths for this image
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = [{"bbox": a['bbox'], "category_id": a['category_id'], "matched": False} for a in anns]
        
        # Get Predictions for this image (sort by score descending)
        preds = [p for p in predictions if p['image_id'] == img_id]
        preds = sorted(preds, key=lambda k: k['score'], reverse=True)
        
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if not gt['matched']:
                    iou = calculate_iou(p['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                        
            if best_iou >= iou_thresh:
                # Match
                gt_boxes[best_gt_idx]['matched'] = True
                gt_class = gt_boxes[best_gt_idx]['category_id'] - 1 # 1-indexed to 0-indexed
                pred_class = p['category_id'] - 1
                cm[gt_class][pred_class] += 1
            else:
                # False Positive (Background actual, predicted class X)
                pred_class = p['category_id'] - 1
                cm[bg_idx][pred_class] += 1
                
        # Any unmatched GT is a False Negative (Actual class X, predicted Background)
        for gt in gt_boxes:
            if not gt['matched']:
                gt_class = gt['category_id'] - 1
                cm[gt_class][bg_idx] += 1
                
    # Plotting
    labels = class_names + ["Background"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('Predicted')
    plt.title(f'Object Detection Confusion Matrix (IoU >= {iou_thresh})')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved Confusion Matrix to {output_path}")

if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Evaluate Deformable DETR on the BCCD validation set.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Model_trained_epoch_20_model_deformable_DETR_data_augm/best_model",
        help="Path to the saved model directory (default: best Deformable DETR model).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation outputs (PR curve, confusion matrix, visualizations).",
    )
    parser.add_argument(
        "--tiled", action="store_true",
        help=(
            "Run tiled inference: split each val image into 2×2 overlapping tiles, "
            "infer on each tile, shift predictions to full-image coordinates, then "
            "merge with NMS. Significantly improves small-object (Platelet) recall."
        ),
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for predictions (default: 0.1). Increase to reduce False Positives.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train.py first, or pass --model_path to point to your saved model."
        )

    # Evaluation always runs on the VALIDATION set (as required by the assignment)
    val_ann = "./blood_cell_data/val.json"
    val_img_dir = "./blood_cell_data/val"

    if not os.path.exists(val_ann):
        raise FileNotFoundError(
            "Validation annotations not found. Run preprocess.py first:\n"
            "  uv run python preprocess.py https://github.com/Shenggan/BCCD_Dataset.git ./blood_cell_data"
        )

    output_dir = args.output_dir
    vis_dir = os.path.join(output_dir, "val_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Loading Model from: {model_path}")
    
    # 0-Indexed for DeformableDETR
    id2label = {0: "WBC", 1: "RBC", 2: "Platelets"}
    class_names = ["WBC", "RBC", "Platelets"]
    
    try:
        processor = DeformableDetrImageProcessor.from_pretrained(model_path)
    except OSError:
        processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

    # --- Compatibility fix for newer transformers versions ---
    # The saved config has backbone=None but backbone_config set as a full
    # ResNet config dict. transformers >= 4.40 rejects this combination.
    # Fix: extract the backbone name string from inside backbone_config,
    # promote it to top-level 'backbone', and drop 'backbone_config'.
    from transformers import DeformableDetrConfig
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    backbone_cfg = config_dict.get("backbone_config")
    if backbone_cfg is not None:
        # Extract the backbone name from inside backbone_config (e.g. "resnet50")
        backbone_name = backbone_cfg.get("backbone") if isinstance(backbone_cfg, dict) else None
        config_dict.pop("backbone_config")
        if not config_dict.get("backbone") and backbone_name:
            config_dict["backbone"] = backbone_name
            print(f"  Migrated backbone='{backbone_name}' from backbone_config "
                  "to top-level backbone for transformers >= 4.40 compatibility.")

    # Strip metadata keys that aren't valid constructor args
    config_dict.pop("_name_or_path", None)
    config_dict.pop("transformers_version", None)

    cfg = DeformableDetrConfig(**config_dict)
    cfg.id2label = id2label
    cfg.label2id = {v: k for k, v in id2label.items()}

    model = DeformableDetrForObjectDetection.from_pretrained(
        model_path,
        config=cfg,
        ignore_mismatched_sizes=True
    )
    model.eval()

    # Inference loop to gather all predictions
    USE_TILED = args.tiled
    SCORE_THRESH = args.score_threshold
    mode_label = "TILED (2×2 tiles + NMS)" if USE_TILED else "STANDARD (full image)"
    print(f"Running Inference on Validation Dataset  [{mode_label}] at threshold {SCORE_THRESH}...")

    coco_gt = COCO(val_ann)
    img_ids = coco_gt.getImgIds()
    results = []

    for img_id in tqdm(img_ids, desc="Inferencing"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_img_dir, img_info["file_name"])

        if USE_TILED:
            # Tile the image, infer per-tile, merge with NMS
            preds = tiled_inference(img_path, model, processor, img_id,
                                    score_threshold=SCORE_THRESH, nms_iou=0.5)
            results.extend(preds)
        else:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(model.device)
            post = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=SCORE_THRESH
            )[0]

            for score, label, box in zip(post["scores"], post["labels"], post["boxes"]):
                box = box.tolist()
                results.append({
                    "image_id": img_id,
                    "category_id": label.item() + 1,  # 1-indexed for COCOeval
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": score.item(),
                })
            
    # Save predictions
    res_file = os.path.join(output_dir, 'val_predictions.json')
    if len(results) > 0:
        with open(res_file, 'w') as f:
            json.dump(results, f)

        # 1. Standard COCO mAP Metrics
        print("\n--- COCO mAP Metrics ---")
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 2. Precision-Recall Curve Deliverable
        generate_pr_curve(coco_eval, class_names, os.path.join(output_dir, 'pr_curve.png'))
        
        # 3. Confusion Matrix Deliverable
        generate_confusion_matrix(coco_gt, results, class_names, num_classes=3, output_path=os.path.join(output_dir, 'confusion_matrix.png'))
    else:
        print("Model predicted absolutely 0 bounding boxes. Cannot generate PR curve or matrix.")

    # 4. Visualization Deliverable (10 Validation Images)
    print("\nGenerating 10 Validation Image Visualizations...")
    random.seed(42)  # Fixed seed for reproducibility
    sample_ids = random.sample(img_ids, min(10, len(img_ids)))
    
    for img_id in sample_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        # Shift GT boxes from 1-indexed to 0-indexed for visualization
        gt_boxes = [(ann['bbox'], ann['category_id'] - 1) for ann in anns]
        
        out_path = os.path.join(vis_dir, f'vis_{img_info["file_name"]}')
        visualize_prediction(
            os.path.join(val_img_dir, img_info['file_name']),
            model, processor, id2label,
            out_path, gt_boxes=gt_boxes, threshold=SCORE_THRESH
        )
        
    print(f" Saved 10 Visualizations to {vis_dir}")
