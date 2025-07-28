from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import RFDETR, compute_loss # Import updated compute_loss
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset

def custom_collate_fn(batch):
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    
    images = torch.stack(images)
    
    return images, targets

if __name__ == '__main__':
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
    TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"

    # Ensure dataset is downloaded and extracted
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    if not TRAIN_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"Annotation file not found at '{TRAIN_ANNOTATION_FILE.resolve()}'")

    print(">>> Preparing dataset for training...")
    dataset = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    print(f"Dataset size: {len(dataset)}")

    num_classes = len(dataset.cat_id_to_name) + 1 # +1 for 'no-object' class
    print(f"Number of classes (including no-object): {num_classes}")

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RFDETR(num_classes=num_classes, num_queries=100)
    model.to(device)
    model.train()

    # --- DIAGNOSTIC CHANGE: Temporarily lower learning rate ---
    # If 1e-4 is still too high, try 1e-5 or 1e-6 to see if it stabilizes.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    # You can try: optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    # Or even: optimizer = optim.AdamW(model.parameters(), lr=1e-6) 
    # If it stabilizes at a very low LR, then 1e-4 is indeed too aggressive.

    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0

        for images, targets in train_loader:
            optimizer.zero_grad()
            
            images = images.to(device)
            # Ensure target boxes are float32 and labels are int64
            targets_processed = []
            for t in targets:
                targets_processed.append({
                    "boxes": t["boxes"].to(device).float(),
                    "labels": t["labels"].to(device).long(),
                    "image_id": t["image_id"].to(device)
                })

            pred_logits, pred_boxes = model(images)
            
            loss = compute_loss(pred_logits, pred_boxes, targets_processed, num_classes, device)
            
            # Check if loss is NaN/Inf before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch+1}, Batch: Loss is NaN or Inf. Skipping backward pass.")
                # You might want to break here or handle it more gracefully
                continue 

            loss.backward()
            
            # --- CRUCIAL CHANGE: Add Gradient Clipping ---
            # This prevents gradients from exploding, which often leads to NaN/Inf.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm can be adjusted (e.g., 0.1, 5.0)
            # ---------------------------------------------

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")
    
    torch.save(model.state_dict(), 'rfd_et-r_model.pth')
    print("Model saved to rfd_et-r_model.pth")

