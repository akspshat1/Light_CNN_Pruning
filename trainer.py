import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import resnet50, resnet18
import torch.nn.utils.prune as prune
import random

class Trainer:
    def __init__(self, model, train_algorithm, is_data_augmentation, is_pretrained, is_pruning, model_type):
        self.model = model
        self.train_algorithm = train_algorithm
        self.is_data_augmentation = is_data_augmentation
        self.data_augmentation = None
        self.is_pretrained = is_pretrained
        self.is_pruning = is_pruning
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
                
    def prune_model(self, model, prune_percent=0.2):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_percent)
                prune.remove(module, 'weight') 
                
    def print_flops(self, model, input_size):
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(*input_size)
        dummy_input = dummy_input.to(self.device)
        
        with torch.autograd.profiler.profile() as prof:
            model(dummy_input)
        
        flops = prof.key_averages().total_average().cpu_time
        model.train()
        return flops

    def regrow_model(self, new_params, old_params, regrowth_percent):
        new_flat_params = torch.cat([param.flatten() for param in new_params])
        old_flat_params = torch.cat([param.flatten() for param in old_params])
        
        num_params_to_regrow = int(regrowth_percent * len(new_flat_params))
        
        pruned_indices = (new_flat_params == 0).nonzero().squeeze()
        
        regrow_indices = random.sample(pruned_indices.tolist(), num_params_to_regrow)
        
        for idx in regrow_indices:
            new_flat_params[idx] = old_flat_params[idx]
        
        start_idx = 0
        for param in new_params:
            end_idx = start_idx + param.numel()
            param.data = new_flat_params[start_idx:end_idx].reshape(param.shape)
            start_idx = end_idx
            
    def prune_and_regrow(self, model, epoch, prune_percent=0.2, regrowth_percent=0.1):
        regrowth_percent = regrowth_percent * (1 - (epoch/100))
        old_params = [param.clone().detach() for param in model.parameters()]
        self.prune_model(model, prune_percent)
        new_params = [param.clone().detach() for param in model.parameters()]
        self.regrow_model(new_params, old_params, regrowth_percent)
        return

    def extract_layer_output(self, model, layer_name, input_image):
        """
        Extracts the output of a specific layer from a CNN model.
        """
        outputs = {}
        def hook(module, input, output):
            outputs[layer_name] = output
        
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
            
        hook_handle = target_layer.register_forward_hook(hook)
        
        model(input_image)
        
        hook_handle.remove()
        
        return outputs[layer_name]

    def pre_train(self, model, train_loader, epochs=10, verbose=True):
        print("Pre-Training Model...")
        new_dataset = []
        for i, (img, _) in enumerate(train_loader):
            for x in img:
                x_0 = x
                x_1 = torch.rot90(x, 1, [1, 2])
                x_2 = torch.rot90(x, 2, [1, 2])
                x_3 = torch.rot90(x, 3, [1, 2])
                new_dataset.append((x_0, 0))
                new_dataset.append((x_1, 1))
                new_dataset.append((x_2, 2))
                new_dataset.append((x_3, 3))
        
        new_train_dataset, new_val_dataset = random_split(new_dataset, [int(0.8 * len(new_dataset)), int(0.2 * len(new_dataset))])
        new_train_loader = DataLoader(new_train_dataset, batch_size=32, shuffle=True)
        new_val_loader = DataLoader(new_val_dataset, batch_size=32, shuffle=True)
        
        model.fc = None
        if self.model_type == "heavy":
            model.fc = nn.Linear(512, 4)
        else:
            model.fc = nn.Linear(128*8*8, 4)
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            for i, (images, labels) in enumerate(new_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_hat = model(images)
                loss = criterion(y_hat, labels)
                loss.backward()
                optimizer.step()
            
            acc = self.test_model(model, new_val_loader, self.device)
            
            if verbose:
                print(f"Epoch => {epoch+1} | Train Loss => {loss.item()} | Validation Accuracy => {acc}")
        model.fc = None
        if self.model_type == "heavy":
            model.fc = nn.Linear(512, 10)
        else:
            model.fc = nn.Sequential(
                nn.Linear(128*8*8, 1024),
                nn.ReLU(),
                nn.Linear(1024, 10)
            )
        print("")
        return
    
    def test_model(self, model, loader, device):
        """
        Test the model on the test set.
        """
        model.eval()
        correct = 0
        total = 0
        # criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                images = images.cpu()
                labels = labels.cpu()
        accuracy = 100 * correct / total
        
        return accuracy

    def simpleTrain(self, model, train_loader, val_loader, epochs=100, learning_rate=0.001, verbose=False):
        """
        Train the model using the simple train algorithm.
        """
        print("Training model using Simple Train Algorithm...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()
        model.to(self.device)

        if self.is_pruning:
            print(f"FLOPs before pruning: {self.print_flops(model, (1, 3, 32, 32))}")
            print("")
        
        for epoch in range(epochs):
            running_loss = 0.0
            iters = 0
            
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                if self.is_data_augmentation:
                    images = self.data_augmentation(images)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_hat = model(images)
                loss = criterion(y_hat, labels)
                
                running_loss += loss.item()
                iters += 1
                
                loss.backward()
                optimizer.step()
                images = images.cpu()
                labels = labels.cpu()

            if self.is_pruning:
                self.prune_and_regrow(model, epoch)
            
            acc = self.test_model(model, val_loader, self.device)
            
            if verbose:
                print(f"Epoch => {epoch+1} | Loss => {running_loss/iters} | Validation Accuracy => {acc}")
                
        if self.is_pruning:
            print(f"FLOPs after pruning: {self.print_flops(model, (1, 3, 32, 32))}")
        print("")
        return

    def knowledgeDistillationLogits(self, student, teacher, train_loader, val_loader, epochs=100, learning_rate=0.001, T=2, verbose=False):
        """
        Train the student model with knowledge distillation.
        """
        print("Training model using Knowledge Distillation on Logits Algorithm...")
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        student.train()
        teacher.eval()
        student.to(self.device)
        teacher.to(self.device)
        
        if self.is_pruning:
            print(f"FLOPs before pruning: {self.print_flops(student, (1, 3, 32, 32))}")
            print("")
        
        for epoch in range(epochs):
            running_loss = 0.0
            iters = 0
            
            student.train()
            for i, (images, labels) in enumerate(train_loader):
                if self.is_data_augmentation:
                    images = self.data_augmentation(images)
                images = images.to(self.device)
                student_logits = student(images)
                teacher_logits = teacher(images)

                teacher_probs = nn.functional.softmax(teacher_logits / T, dim=1)

                kl_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_logits / T, dim=1), teacher_probs)

                running_loss += kl_loss.item()
                iters += 1
                
                optimizer.zero_grad()
                kl_loss.backward()
                optimizer.step()
                
            if self.is_pruning:
                self.prune_and_regrow(student, epoch)
            
            acc = self.test_model(student, val_loader, self.device)
            
            if verbose:
                print(f"Epoch => {epoch+1} | Loss => {running_loss/iters} | Validation Accuracy => {acc}")
                
        if self.is_pruning:
            print(f"FLOPs after pruning: {self.print_flops(student, (1, 3, 32, 32))}")
        print("")
        return  

    def knowledgeDistillationFeats(self, student, teacher, train_loader, val_loader, epochs=100, learning_rate=0.001, verbose=False):
        """
        Trains a student model with feature-level knowledge distillation (KD) loss.
        """
        print("Training model using Knowledge Distillation on Features Algorithm...")

        student.train()
        teacher.eval()
        student.to(self.device)
        teacher.to(self.device)
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        if self.is_pruning:
            print(f"FLOPs before pruning: {self.print_flops(student, (1, 3, 32, 32))}")
            print("")
        
        
        for epoch in range(epochs):
            running_loss = 0.0
            iters = 0
            student.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.is_data_augmentation:
                    images = self.data_augmentation(images)
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                teacher_outputs = self.extract_layer_output(teacher, 'relu', images)
                student_outputs = self.extract_layer_output(student, 'pool1', images)
                
                loss1 = criterion(student_outputs, teacher_outputs)
                loss2 = nn.CrossEntropyLoss()(student(images), labels)
                
                loss = loss1 + loss2
                output = student(images)
                
                running_loss += loss.item()
                iters += 1
                
                loss.backward()
                optimizer.step()

            if self.is_pruning:
                self.prune_and_regrow(student, epoch)

            acc = self.test_model(student, val_loader, self.device)
                
            if verbose:
                print(f"Epoch => {epoch+1} | Loss => {running_loss/iters} | Validation Accuracy => {acc}")
                
        if self.is_pruning:
            print(f"FLOPs after pruning: {self.print_flops(student, (1, 3, 32, 32))}")
        print("")
        return
    
    def train(self, train_loader, val_loader):
        if self.is_data_augmentation:
            self.data_augmentation = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomGrayscale(p=0.2)
        )
        
        student = self.model
        teacher = resnet50()
        teacher.fc = nn.Linear(teacher.fc.in_features, 10)
        teacher.load_state_dict(torch.load('models/strong_cnn.pth'))

        if self.is_pretrained:
            self.pre_train(self.model, train_loader)

        if self.train_algorithm == "simple":
            self.simpleTrain(self.model, train_loader, val_loader, verbose=True)
        elif self.train_algorithm == "knowledge_distillation_logits":
            self.knowledgeDistillationLogits(student, teacher, train_loader, val_loader, verbose=True)
        elif self.train_algorithm == "knowledge_distillation_feats":
            self.knowledgeDistillationFeats(student, teacher, train_loader, val_loader, verbose=True)
        else:
            raise ValueError("Invalid train algorithm.")
        
        return
