import torch
import random
import numpy as np
from tqdm import tqdm
import time
def train_model(model, model_name,
                train_loader, val_loader, 
                num_epochs, 
                learning_rate, weight_decay, 
                lr_patience,  factor,
                patience, device=None, seed=None):
    '''
    input:
            model 
            dataloader(train, validation)
            num_epochs
            learning_rate, weight_decay for Adam optimizer
            patience for early stopping
            device
    output: 
            best_model
            validation_losses
            training_losses
    '''
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 把模型搬到cpu或是gpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    model = model.to(device)

    # 設定一些訓練工具
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = learning_rate, 
                                 weight_decay = weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = 'min',
                                                           patience = lr_patience, 
                                                           factor = factor)
    '''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size = step_size, 
                                                gamma = factor)
    '''
    loss_func = torch.nn.CrossEntropyLoss()

    # early stopping needed parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    # record the training process
    train_losses, val_losses, LearningRate = [], [], []
    
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

        scheduler.step(val_loss) # 更新learning rate

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, '
              f'LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss:    # 如果目前的validation loss比目前的最佳loss小，則儲存目前的參數為最佳模型
            best_val_loss = val_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'{model_name}_best_model.pth')
            print("保存最佳模型")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Stop!! Loss does not improve in {patience} epochs")
                break
    end_time = time.time()
    return end_time - start_time, train_losses, val_losses, LearningRate