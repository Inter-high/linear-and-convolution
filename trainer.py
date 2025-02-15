import torch
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device, logger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.lowest_loss = float('inf')
    
    def train(self, train_dataloader):
        self.model.train()
        train_loss = 0.0
        for (x, y) in train_dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            y_hat = self.model(x)
            
            loss = self.criterion(y_hat, y)
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        return train_loss

    def valid(self, valid_dataloader):
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for (x, y) in valid_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x)
                
                loss = self.criterion(y_hat, y)
                valid_loss += loss.item()
            
            valid_loss /= len(valid_dataloader)

        return valid_loss
    
    def test(self, test_dataloader):
        predict_score = 0
        num_samples = 0
        num_classes = 10  # MNIST 클래스 개수
        class_total = torch.zeros(num_classes, dtype=torch.int32)
        class_wrong = torch.zeros(num_classes, dtype=torch.int32)
        
        # 혼동 행렬 초기화
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for (x, y) in test_dataloader:
            num_samples += y.size(0)
            x = x.to(self.device)

            output = self.model(x)
            output_idx = output.argmax(dim=1).cpu()

            sum_correct = (output_idx == y).sum().item()
            predict_score += sum_correct

            # 클래스별 오류 집계 및 혼동 행렬 업데이트
            for true, pred in zip(y.numpy(), output_idx.numpy()):
                class_total[true] += 1
                if true != pred:
                    class_wrong[true] += 1
                confusion_matrix[true, pred] += 1

        accuracy = predict_score / num_samples

        # 클래스별 오류율 계산 (백분율 변환)
        class_error_rate = (class_wrong / class_total).numpy() * 100  

        return accuracy, class_error_rate, confusion_matrix

    def training(self, train_dataloader, valid_dataloader, epochs, valid_interval, weight_path):
        train_losses = []
        valid_losses = []

        # validation을 10번만 실행하도록 설정
        validation_interval = max(1, epochs // valid_interval)  # 최소 1회는 실행되도록 설정

        for epoch in range(epochs + 1):
            train_loss = self.train(train_dataloader)
            train_losses.append(train_loss)

            if epoch % validation_interval == 0:  # 10번만 실행되도록
                valid_loss = self.valid(valid_dataloader)
                valid_losses.append(valid_loss)

                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"New best model saved at epoch {epoch} with Valid Loss: {valid_loss:.4f}")

                self.logger.info(f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        return train_losses, valid_losses
    