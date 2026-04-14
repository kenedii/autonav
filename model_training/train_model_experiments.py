import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import time
from tqdm import tqdm
from datetime import timedelta

# ==================== CONFIGURATION ====================
IMG_HEIGHT = 120
IMG_WIDTH = 160
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRELOAD_DATA = True # Load dataset into RAM for max speed

AUGMENTED_CSV = r"combined_augmented_dataset.csv"
CLEANED_CSV   = r"combined_cleaned_dataset.csv"

# Root path for images relative to the workspace
# Since the CSVs have paths like "runs_rgb_depth/run_.../file.png"
# and they exist under ./
DATA_ROOT = r"./"

EXPERIMENTS_DIR = 'training_experiments'
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Define the 6 experimental configurations
EXPERIMENTS = [
    {"id": 1, "desc": "Front+Back + all sensors (Augmented)", "csv": AUGMENTED_CSV, "features": ['rgb_path', 'cam1_path', 'ir_path', 'depth_path']},
    {"id": 2, "desc": "Front only + all sensors (Augmented)", "csv": AUGMENTED_CSV, "features": ['rgb_path', 'ir_path', 'depth_path']},
    {"id": 3, "desc": "Front only RGB only (Augmented)",      "csv": AUGMENTED_CSV, "features": ['rgb_path']},
    {"id": 4, "desc": "Front+Back RGB only (Augmented)",      "csv": AUGMENTED_CSV, "features": ['rgb_path', 'cam1_path']},
    {"id": 5, "desc": "Front+Back + all sensors (Cleaned)",   "csv": CLEANED_CSV,   "features": ['rgb_path', 'cam1_path', 'ir_path', 'depth_path']},
    {"id": 6, "desc": "Front+Back RGB only (Cleaned)",        "csv": CLEANED_CSV,   "features": ['rgb_path', 'cam1_path']}
]

MODELS = ['resnet34', 'resnet101', 'resnet152']

# ==================== DATASET ====================

def ensure_full_path(rel_path):
    if pd.isna(rel_path) or not str(rel_path): return None
    p = str(rel_path)
    # The relative path starts with "runs_rgb_depth/"
    # We join it with the data root ./ (current directory)
    # Wait, the dataset folder structure is jetracer\train\runs_rgb_depth\runs_rgb_depth\run...
    # The CSV path is "runs_rgb_depth/run_..."
    # So we need to ensure it matches the actual disk location.
    
    # If path is already absolute, return it
    if os.path.isabs(p): return p
    
    # Otherwise, join with the root of the train folder
    root = r"./"
    return os.path.join(root, p.replace('/', os.sep))

class SensorDataset(Dataset):
    def __init__(self, df, features, preload_to_ram=True):
        self.df = df.reset_index(drop=True)
        self.features = features
        # Predict both steer_norm and throttle_norm
        self.targets = self.df[['steer_norm', 'throttle_norm']].values.astype(np.float32)
        self.preload_to_ram = preload_to_ram
        self.cache = {}

        if self.preload_to_ram:
            print(f"Preloading {len(self.df)} samples into RAM for maximum training speed...")
            for idx in tqdm(range(len(self.df)), desc="Preloading"):
                self.cache[idx] = self._load_sample(idx)
        
    def _load_sample(self, idx):
        row = self.df.iloc[idx]
        tensors = []
        for feat in self.features:
            path = ensure_full_path(row.get(feat))
            img = None
            if path and os.path.exists(path):
                if 'depth' in feat or 'ir' in feat:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        img = img.astype(np.float32) / 255.0
                        img = np.expand_dims(img, axis=0)
                else:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))
            
            if img is None:
                channels = 1 if ('depth' in feat or 'ir' in feat) else 3
                img = np.zeros((channels, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            tensors.append(img)
        return np.concatenate(tensors, axis=0)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if self.preload_to_ram:
            final_tensor = self.cache[idx]
        else:
            final_tensor = self._load_sample(idx)
        return torch.tensor(final_tensor), torch.tensor(self.targets[idx])

# ==================== MODEL DEFINITION ====================

def get_resnet_model(arch_name, in_channels):
    if arch_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        out_features = 512
    elif arch_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        out_features = 2048
    elif arch_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        out_features = 2048
    elif arch_name == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        out_features = 2048
        
    # Modify first layer if in_channels != 3
    if in_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                kernel_size=original_conv.kernel_size, 
                                stride=original_conv.stride, 
                                padding=original_conv.padding, 
                                bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
    features = nn.Sequential(*list(model.children())[:-2])
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(out_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2), # Output 2 values: steering and throttle
        nn.Tanh()
    )
    
    return nn.Sequential(features, head)

# ==================== TRAINING LOOP ====================

def run_experiment(exp, model_net, arch_name, df):
    # Calculate input channels
    in_channels = sum(1 if ('depth' in f or 'ir' in f) else 3 for f in exp['features'])
    
    exp_name = f"exp{exp['id']}_{arch_name}"
    save_path = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}\nStarting Experiment: {exp_name}\nDescription: {exp['desc']}\nChannels: {in_channels}\n{'='*60}")
    
    from sklearn.model_selection import train_test_split
    # Proper Train (70%), Validation (15%), Test (15%) split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Preload full data using pre-emptive loader
    train_ds = SensorDataset(train_df, exp['features'], preload_to_ram=PRELOAD_DATA)
    val_ds = SensorDataset(val_df, exp['features'], preload_to_ram=PRELOAD_DATA)
    test_ds = SensorDataset(test_df, exp['features'], preload_to_ram=PRELOAD_DATA)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = get_resnet_model(arch_name, in_channels).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_loss = float('inf')
    best_preds = None
    best_targets = None
    
    history_log = open(os.path.join(save_path, "metrics.txt"), "w")
    history_log.write(f"Experiment: {exp['desc']} | Model: {arch_name}\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_loss += criterion(out, y).item()
                all_preds.extend(out.cpu().numpy())
                all_tgts.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        scheduler.step(val_loss)
        
        info = f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(info)
        history_log.write(info + "\n")
        history_log.flush()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_preds = np.array(all_preds)
            best_targets = np.array(all_tgts)
            # Save the best model weights into the experiment folder
            weights_path = os.path.join(save_path, f"best_model_{exp_name}.pth")
            torch.save(model.state_dict(), weights_path)
            # Maintain a symbolic 'best_model.pth' for the evaluation step below
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            
    # Final Evaluation on the held-out TEST set
    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
    model.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            test_preds.extend(out.cpu().numpy())
            test_tgts.extend(y.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_tgts = np.array(test_tgts)
    
    # Regression Metrics
    test_mae = mean_absolute_error(test_tgts, test_preds)
    test_r2 = r2_score(test_tgts, test_preds, multioutput='uniform_average')
    
    # Per-output MAE
    steer_mae = mean_absolute_error(test_tgts[:, 0], test_preds[:, 0])
    throttle_mae = mean_absolute_error(test_tgts[:, 1], test_preds[:, 1])

    # Calculate pseudo-classification metrics (Left, Straight, Right) for steering
    def classify_steer(val):
        if val < -0.15: return 0   # Left
        elif val > 0.15: return 2  # Right
        else: return 1             # Straight
        
    cls_preds = np.array([classify_steer(p[0]) for p in test_preds])
    cls_targets = np.array([classify_steer(t[0]) for t in test_tgts])
    
    acc = accuracy_score(cls_targets, cls_preds)
    cm = confusion_matrix(cls_targets, cls_preds, labels=[0, 1, 2])
    
    history_log.write(f"\n{'='*20} FINAL TEST RESULTS {'='*20}\n")
    history_log.write(f"Test Multi-Output MSE: {best_loss:.4f}\n")
    history_log.write(f"Test Combined MAE: {test_mae:.4f}\n")
    history_log.write(f"  - Steering MAE: {steer_mae:.4f}\n")
    history_log.write(f"  - Throttle MAE: {throttle_mae:.4f}\n")
    history_log.write(f"Test R2 Score: {test_r2:.4f}\n")
    history_log.write(f"Test Steering Pseudo-Accuracy: {acc*100:.2f}%\n")
    history_log.write(f"Steering Confusion Matrix (Left/Straight/Right):\n{cm}\n")
    history_log.close()
    
    # Save Confusion Matrix plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Left', 'Straight', 'Right'], yticklabels=['Left', 'Straight', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({exp_name})')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()
    
    return acc, best_loss, exp_name

# ==================== MAIN SCRIPT ====================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    results = []
    
    for exp in EXPERIMENTS:
        print(f"\nLoading dataset: {exp['csv']}")
        df = pd.read_csv(exp['csv'])
        
        for model_arch in MODELS:
            acc, loss, name = run_experiment(exp, None, model_arch, df)
            results.append({"Experiment": name, "Accuracy": acc*100, "Val_MSE": loss})
            
    # Final Output
    print("\n" + "="*60)
    print("                  ALL EXPERIMENTS COMPLETED")
    print("="*60)
    
    best_acc = 0.0
    best_model_name = ""
    
    for res in results:
        print(f"Model: {res['Experiment']:<25} | Accuracy: {res['Accuracy']:>6.2f}% | MSE: {res['Val_MSE']:.4f}")
        if res['Accuracy'] > best_acc:
            best_acc = res['Accuracy']
            best_model_name = res['Experiment']
            
    print("-" * 60)
    print(f"🏆 Highest Accuracy Model: {best_model_name} with {best_acc:.2f}%")
    print(f"Results and metrics saved in './{EXPERIMENTS_DIR}/'")
    print("=" * 60)
