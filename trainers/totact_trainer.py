import torch
import torch.nn as nn
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class BaselineTrainer:
    def __init__(self, model, cfg, train_loader, val_loader, task_name):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name
        self.device = cfg.Project.device
        self.model.to(self.device)
        
        # config에서 베이스라인 전용 하이퍼파라미터 사용
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=cfg.BaselineTask.lr, 
            weight_decay=cfg.BaselineTask.weight_decay
        )
        self.criterion = nn.MSELoss(reduction='none') # 마스킹을 위해 reduction='none'

        if task_name == 'tot':
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else: # ACT
            self.criterion = nn.MSELoss(reduction='none')

    def _tot_logits(self, pred):
        if isinstance(pred, dict):
            return pred.get('logits')
        return pred

    def _act_preds(self, pred):
        if isinstance(pred, dict):
            pred = pred.get('act_preds', pred.get('preds'))
        if pred is None:
            raise KeyError("ACT model output must contain 'act_preds' or be a tensor.")
        if pred.dim() == 3 and pred.size(-1) > 1:
            return pred[:, :, 1]
        if pred.dim() == 3 and pred.size(-1) == 1:
            return pred.squeeze(-1)
        return pred

    # def _calculate_loss(self, pred, true_batch):
    #     if self.task_name == 'tot':
    #         # 베이스라인 batch의 라벨 키는 'label' 입니다.
    #         true_labels = true_batch['label'] 
    #         return self.criterion(pred, true_labels)
    #     else: # ACT
    #         # 1. 모델의 3개 예측값 중 첫 번째 채널만 사용합니다.
    #         pred_veh1_act = pred[:, :, 0]  # Shape: (B, T, 3) -> (B, T)

    #         # 2. 베이스라인 batch의 'label' 키를 사용하고, 마지막 차원을 제거합니다.
    #         true_veh1_act = true_batch['label'].squeeze(-1) # Shape: (B, T, 1) -> (B, T)

    #         # 3. 패딩(-100)을 제외하기 위한 마스크를 생성합니다.
    #         mask = true_veh1_act != -100.0
    #         if not mask.any():
    #             return torch.tensor(0.0, device=self.device, requires_grad=True)

    #         # 4. 마스크를 적용하여 유효한 값들만으로 MSE Loss를 계산합니다.
    #         loss = self.criterion(pred_veh1_act[mask], true_veh1_act[mask])
    #         return loss.mean()
        
    def _calculate_loss(self, pred, true_batch):
        if self.task_name == 'tot':
            # 베이스라인 batch의 라벨 키는 'label' 입니다.
            true_labels = true_batch['label'] 
            logits = self._tot_logits(pred)
            return self.criterion(logits, true_labels)
        else: # ACT
            pred_min_act = self._act_preds(pred)

            true_veh1_act = true_batch['label'].squeeze(-1)
            mask = true_veh1_act != -100.0
            if not mask.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # [변경] pred_min_act 변수를 사용하여 손실 계산
            loss = self.criterion(pred_min_act[mask], true_veh1_act[mask])
            return loss.mean()
        
    def _run_epoch(self, loader, is_train):
        self.model.train(is_train)
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] {self.task_name.upper()}")
        for batch in pbar:
            # batch의 텐서들을 device로 이동
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            with torch.set_grad_enabled(is_train):
                preds = self.model(batch)
                loss = self._calculate_loss(preds, batch)
            
            if is_train and loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(mse_loss=loss.item())
            
        return total_loss / len(loader)

    def train(self): 
        print(f"--- Start Baseline Training for {self.task_name.upper()} ---")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.cfg.BaselineTask.epochs):
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            val_loss = self._run_epoch(self.val_loader, is_train=False)
            
            val_acc = 0.0
            if self.task_name == 'tot':
                val_acc = self.evaluate_accuracy()

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(self.cfg.Project.weights_dir, f"best_baseline_{self.task_name}.pt")
                os.makedirs(self.cfg.Project.weights_dir, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.BaselineTask.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(f"--- Finished Baseline Training for {self.task_name.upper()} ---")

    
    @torch.no_grad()
    def evaluate_accuracy(self):
        self.model.eval()
        all_preds = []
        all_trues = []
        for batch in self.val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            logits = self._tot_logits(self.model(batch))
            preds = torch.argmax(logits, dim=1)
            
            mask = batch['label'] != -100
            all_preds.append(preds[mask].cpu())
            all_trues.append(batch['label'][mask].cpu())
            
        if not all_trues: return 0.0
        
        all_preds = torch.cat(all_preds).numpy()
        all_trues = torch.cat(all_trues).numpy()
        
        return accuracy_score(all_trues, all_preds)

class EnhancerTrainer:
    def __init__(self, enhancer_model, cfg, train_loader, val_loader, task_name):
        self.model = enhancer_model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name
        self.device = cfg.Project.device
        self.model.to(self.device)
        
        trainable_module = getattr(self.model, 'fusion_head', None)
        if trainable_module is None:
            trainable_module = getattr(self.model, 'stack_head', None)
        if trainable_module is None:
            trainable_module = getattr(self.model, 'delta_head', None)
        if trainable_module is None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = list(trainable_module.parameters())
        if not trainable_params:
            raise ValueError("No trainable enhancer parameters found.")

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=cfg.EnhancerTask.lr, 
            weight_decay=cfg.EnhancerTask.weight_decay
        )
        
        if task_name == 'tot':
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else: # ACT
            self.criterion = nn.MSELoss(reduction='none')

    def _tot_logits(self, pred):
        if isinstance(pred, dict):
            return pred.get('logits')
        return pred

    def _act_preds(self, pred):
        if isinstance(pred, dict):
            pred = pred.get('act_preds', pred.get('preds'))
        if pred is None:
            raise KeyError("ACT enhancer output must contain 'act_preds' or be a tensor.")
        if pred.dim() == 3 and pred.size(-1) > 1:
            return pred[:, :, 1]
        if pred.dim() == 3 and pred.size(-1) == 1:
            return pred.squeeze(-1)
        return pred

    # def _calculate_loss(self, pred, true_batch):
    #     if self.task_name == 'tot':
    #         # 베이스라인 batch의 라벨 키는 'label' 입니다.
    #         true_labels = true_batch['label'] 
    #         return self.criterion(pred, true_labels)
    #     else: # ACT
    #         # [수정된 부분 시작]
    #         # 1. 실제값(label) 텐서의 마지막 차원을 제거하여 (B, T) 형태로 만듭니다.
    #         true_veh1_act = true_batch['label'].squeeze(-1)

    #         # 2. 패딩 값(-100.0)을 제외하기 위한 마스크를 생성합니다.
    #         mask = true_veh1_act != -100.0
    #         if not mask.any():
    #             return torch.tensor(0.0, device=self.device, requires_grad=True)

    #         # 3. 마스크를 적용하여 유효한 실제값만 추출합니다. 결과 shape: (N,)
    #         valid_trues = true_veh1_act[mask]
            
    #         # 4. 모델 예측값(pred)에서 마스크에 해당하는 유효한 시간 스텝의 값들만 먼저 추출합니다. 결과 shape: (N, 3)
    #         valid_preds_all_channels = pred[mask]
            
    #         # 5. [핵심 수정] 그 다음, 필요한 첫 번째 채널의 예측값만 선택합니다. 결과 shape: (N,)
    #         valid_preds = valid_preds_all_channels[:, 0]

    #         # 6. 이제 (N,) vs (N,)로 shape가 동일해진 두 텐서로 손실을 계산합니다.
    #         loss = self.criterion(valid_preds, valid_trues)
    #         return loss.mean()

    def _calculate_loss(self, pred, true_batch):
        if self.task_name == 'tot':
            # 베이스라인 batch의 라벨 키는 'label' 입니다.
            true_labels = true_batch['label'] 
            logits = self._tot_logits(pred)
            return self.criterion(logits, true_labels)
        else: # ACT
            true_veh1_act = true_batch['label'].squeeze(-1)
            mask = true_veh1_act != -100.0
            if not mask.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            valid_trues = true_veh1_act[mask]
            pred_min_act = self._act_preds(pred)
            valid_preds = pred_min_act[mask]

            loss = self.criterion(valid_preds, valid_trues)
            return loss.mean()
                
    def _run_epoch(self, loader, is_train):
        self.model.train(is_train)
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] Enhancer-{self.task_name.upper()}")
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            with torch.set_grad_enabled(is_train):
                preds = self.model(batch)
                loss = self._calculate_loss(preds, batch)
            
            if is_train and loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(loader)

    def train(self, save_path):
        print(f"--- Start Enhancer Training for {self.task_name.upper()} ---")
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.EnhancerTask.epochs):
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            val_loss = self._run_epoch(self.val_loader, is_train=False)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Best Enhancer model saved with Val Loss: {best_val_loss:.4f}")
        print(f"--- Finished Enhancer Training for {self.task_name.upper()} ---")
