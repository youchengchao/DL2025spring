import torch
import random
import numpy as np
from tqdm import tqdm
import time

class GapLR_scheduler:
    """
    如果val_loss下降的比train_loss慢，則更新lr，直到變得太小了
    patience: val_loss比train_loss慢幾次才更新lr
    """
    def __init__(self, optimizer, factor=0.2, patience=2, min_lr=1e-6):
        self.optimizer = optimizer
        self.starting_lr = optimizer.param_groups[0]['lr']
        self.factor = factor
        self.previous_train_loss = None
        self.previous_val_loss = None
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0

    def step(self, train_loss, val_loss):
        if self.previous_val_loss is not None and self.previous_train_loss is not None:
            val_loss_gap  = self.previous_val_loss - val_loss
            train_loss_gap = self.previous_train_loss - train_loss
            print(f"val_loss gap:{self.previous_val_loss}-{val_loss}={val_loss_gap}")
            print(f"train_loss gap:{self.previous_train_loss}-{train_loss}={train_loss_gap}")
            if val_loss_gap < train_loss_gap:
                self.counter += 1
            else:
                self.counter = 0
        else:
            self.counter = 0
            print("first epoch finished")
        if self.counter >= self.patience:                     # 更新lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
                if param_group['lr'] < self.min_lr:
                    param_group['lr'] = self.starting_lr
                    print("Learning rate is too small, reset to starting lr")
            self.counter = 0

        self.previous_train_loss = train_loss
        self.previous_val_loss = val_loss

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

class HumanLR_scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(input(f"current learning rate:{param_group['lr']}\nEnter new learning rate:"))
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def train_model(model, model_name,
                train_loader, val_loader, 
                num_epochs,
                optimizer=None, scheduler=None,
                patience = 3, device=None):

    torch.backends.cudnn.benchmark = True # cuDNN啟用最快速的

    if device is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    # ealy stopping要使用到的變數
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses, val_losses, LearningRate = [], [], []  # log檔要存的東西
    
    start_time = time.time()
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # 將每個batch依序輸入
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.long().to(device)
            optimizer.zero_grad()   # 梯度初始化設為0
            outputs = model(inputs) # 得到前向傳播的output
            loss = loss_func(outputs, labels) # 計算loss
            loss.backward()         # 反向傳播
            optimizer.step()        # 更新參數
            train_loss += loss.item() * inputs.size(0) # 因為loss_func會回傳每個batch的average loss，所以乘上batch size代表整個batch的總loss

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct/train_total) # 更新進度條的顯示

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # 把梯度關閉，提升效能
        with torch.no_grad():
            # 把每個batch丟給model進行前向傳播計算loss
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)  # 將預測的類別設為機率最高的類別
                # predicted = torch.argmax(outputs, dim=1) # 另一種寫法
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loop.set_postfix(loss=loss.item(), acc=correct / total)  # 顯示即時 loss 與 accuracy

            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            val_losses.append(val_loss)
            current_lr = scheduler.get_last_lr()[0]
            LearningRate.append(current_lr)

            print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Accuracy: {val_accuracy:.4f}, '
                f'LR: {current_lr:.6f}')
            
            if val_loss < best_val_loss:    # 如果目前的validation loss比目前的最佳loss小，則儲存目前的參數為最佳模型
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f'{model_name}_best_model.pth')
                print("保存最佳模型")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Stop!! Loss does not improve in {patience} epochs")
                    break
            if isinstance(scheduler, HumanLR_scheduler):
                scheduler.step()
            elif isinstance(scheduler, GapLR_scheduler):
                scheduler.step(train_loss, val_loss)
            elif scheduler is not None: 
                scheduler.step(val_loss)
    end_time = time.time()
    return end_time - start_time, train_losses, val_losses, LearningRate