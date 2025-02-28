import pathlib
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import AutoModel
import poutyne as pt
import warnings
from deeplib.training import HistoryCallback
import pandas as pd
import numpy as np
from poutyne.framework.metrics.metrics_registering import register_metric_func
import torchvision.transforms.functional as TF
import random
import tqdm
import pickle
import os


# Augments dataset with rotated images containing rotation angle label
def prepare_rotated_dataset(original_dataset, is_train):
    rotated_dataset = []
    
    # Define rotation angles
    rotation_angles = [0, 90, 180]

    for img, label in original_dataset:
        if(is_train):
            # Randomly select rotation angle
            angle = random.choice(rotation_angles)
            rotated_img = TF.rotate(img, angle)
        else:
            angle = 0
            rotated_img = img
        
        # Append rotated image and label to the dataset
        rotated_dataset.append((rotated_img, label, angle // 90))  # Assign labels 0 to 3
            
    return rotated_dataset


# Define custom loss function for city and rotation prediction
def custom_loss(outputs, targets):
    city_logits, rotation_logits = outputs

    loss_city = criterion_city(city_logits, targets[0])
    loss_rotation = criterion_rotation(rotation_logits, targets[3])

    return loss_city + loss_rotation


# Register the acc function as a metric under the names 'acc' and 'accuracy'    
@register_metric_func('acc', 'accuracy')
def acc(y_pred, y_true, *, ignore_index=-100, reduction='mean'):
    # Calculate accuracy based only on the city prediction (not the rotation)
    y_pred = y_pred[0].argmax(1)
    y_true = y_true[0]
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = (y_pred == y_true).float() * weights

    if reduction in ['mean', 'sum']:
        acc_pred = acc_pred.sum()

    if reduction == 'mean':
        acc_pred = acc_pred / num_labels

    return acc_pred * 100


# Define a custom accuracy metric
class Accuracy(pt.BatchMetric):

    def __init__(self, *, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'acc'
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        # Compute accuracy using the acc function.
        return acc(y_pred, y_true, ignore_index=self.ignore_index, reduction=self.reduction)
    

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        tmp = transform.transforms
        transform1 = []
        transform2 = []
        flip = 0
        for t in tmp:
            if (str(t)=="ToTensor()"):
                flip = 1
            if(not flip):
                transform1.append(t)
            else:
                transform2.append(t)
        
        self.transform1 = transforms.Compose(transform1)
        self.transform2 = transforms.Compose(transform2)
        self.transform3 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, city_label, rotation_label = self.dataset[idx]
        
        if self.transform1:
            image = self.transform1(image)

        grayscale_image = transforms.functional.to_grayscale(image)

        if self.transform2:
            grayscale_image = self.transform3(grayscale_image)
            image = self.transform2(image)
            colorization_label = image
            masked_autoencoder_label = image
        
        return image, (city_label, grayscale_image, colorization_label, rotation_label, masked_autoencoder_label)


# Define test dataset class
class TestDataset(Dataset):
    def __init__(self, test_path, transform=None):
        
        self.image_paths = list(pathlib.Path(test_path).glob('*.jpg'))
        self.image_paths.sort()
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.image_paths[index]
        img_name = img_path.name[:-4]
        
        img = Image.open(img_path)
        
        # apply transforms (if any)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_name

    def __len__(self):
        return len(self.image_paths)


# Modify the model for multi-task learning (city and rotation prediction).
# Even though only city prediction is desired, adding rotation prediction 
# improves city prediction accuracy.
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.city_classifier = nn.Linear(backbone.config.hidden_size, num_classes)
        self.colorization_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output 3 channels for RGB color
            nn.Sigmoid()  # Sigmoid to ensure outputs are in [0, 1] range
        )
        self.rotation_classifier = nn.Linear(backbone.config.hidden_size, 3)  # Predict 4 rotations (0, 90, 180, 270 degrees)
        self.masked_autoencoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        last_hidden_state = outputs.last_hidden_state

        city_logits = self.city_classifier(last_hidden_state[:, 0])  
        rotation_logits = self.rotation_classifier(last_hidden_state[:, 0])

        return city_logits, rotation_logits


# Define data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define normalization for validation and test
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root='./code/where-am-i-dataset/train', transform=None)
test_dataset = TestDataset('./code/where-am-i-dataset/test', transform=test_transform)

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply transforms and add labels for auxiliary tasks.
# Pickle transformed train and val datasets to save time
# Run once, then set to 0
if 1:
    train_dataset = prepare_rotated_dataset(train_dataset, 1)
    val_dataset = prepare_rotated_dataset(val_dataset, 0)
    train_dataset = CustomDataset(train_dataset, transform=train_transform)
    val_dataset = CustomDataset(val_dataset, transform=test_transform)

    # Open a file and use dump() 
    with open('./code/results/rotated_train_dataset.pkl', 'wb') as file: 
        pickle.dump(train_dataset, file)

    # Open a file and use dump() 
    with open('./code/results/rotated_val_dataset.pkl', 'wb') as file: 
        pickle.dump(val_dataset, file)

else:
    with open('./code/results/rotated_train_dataset.pkl', 'rb') as handle:
        train_dataset = pickle.load(handle)
    with open('./code/results/rotated_val_dataset.pkl', 'rb') as handle:
        val_dataset = pickle.load(handle)


# Define data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Dictionary to map city to index
class_to_idx = {'Boston': 0, 'London': 1, 'Montreal': 2, 'Paris': 3, 'Quebec': 4}
print('class -> idx : ', class_to_idx)

# Dictionary to map index to city
idx_to_class = {0: 'Boston', 1: 'London', 2: 'Montreal', 3: 'Paris', 4: 'Quebec'}
print('idx -> class : ', idx_to_class)

# Define number of cities
num_classes = 5


# Select desired operations
train_model = True
eval_model = False
confusion_matrix = False

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train model
if train_model:

    # Load desired pretrained model
    model = AutoModel.from_pretrained('facebook/dinov2-large')

    # Freeze first layers for finetuning
    for name, param in model.named_parameters():
        if(not ("layer.23" in name or "layer.22" in name or "layer.21" in name or "layer.20" in name or "layer.19" in name or "layer.18" in name or "layer.17" in name or "layernorm.weight" in name or "layernorm.bias" in name)):
            param.requires_grad = False

    # Create the multi-task model
    model = MultiTaskModel(model, num_classes).to(device)

    # Define loss functions for all tasks
    criterion_city = nn.CrossEntropyLoss()
    criterion_rotation = nn.CrossEntropyLoss()

    # Training parameters
    n_epoch = 150
    weight_decay_bool = False
    wd = 0.01
    lr = 2e-5
    p = 2
    name = "dinov2"
    save_path = "./code/results/models/dinov2_large_finetuned/"

    # Create folder if save_path does not exist
    os.makedirs(save_path, exist_ok=True)
    
    save_path = save_path + name
    scheduler = pt.ReduceLROnPlateau(monitor='val_acc', mode='max', patience=p, factor=0.5)
    early_stopping = pt.EarlyStopping(monitor='val_acc', mode='max', min_delta=1e-5, patience=7, verbose=True)
    model_checkpoint_last = pt.ModelCheckpoint(save_path+"_last_epoch.ckpt")
    model_checkpoint = pt.ModelCheckpoint(filename=save_path+'_best_epoch_{epoch}.ckpt', monitor="val_acc", mode="max", verbose=True, save_best_only=True, keep_only_last_best=True, restore_best=True)
    callbacks=[scheduler, early_stopping, model_checkpoint_last, model_checkpoint]

    

    # Weight decay
    if weight_decay_bool:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create poutyne model
    poutyne_model = pt.Model(model, optimizer, loss_function=custom_loss, device=device, batch_metrics=['accuracy'])

    if torch.cuda.is_available():
        poutyne_model.cuda()
    else:
        warnings.warn("No GPU available")

    history_callback = HistoryCallback()
    callbacks = [history_callback] if callbacks is None else [history_callback] + callbacks

    # Train
    poutyne_model.fit_generator(train_loader, val_loader, epochs=n_epoch, callbacks=callbacks)

    # Save training history figure
    history = history_callback.history
    history.save_fig(save_path +".png")

    # Also save data for later
    data = pd.DataFrame(history.history)
    data.to_csv(save_path +".csv")


# Evaluate model
if eval_model:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choose finetuned model to evaluate
    model = AutoModel.from_pretrained('facebook/dinov2-large')
    model = MultiTaskModel(model, num_classes)
    model.load_state_dict(torch.load("./code/results/models/dinov2_large_finetuned/dinov2_best_epoch_27.ckpt"))
    model.to(device)
    model.eval()

    # Generate predictions
    label_predictions = []
    image_names = []
    with torch.no_grad():
        for batch, im_names in tqdm.tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch)[0]
            preds = output.max(dim=1)[1]
            label_predictions.extend(preds.detach().cpu().numpy()) 
            image_names.extend(im_names) 

    assert len(label_predictions) == len(image_names) 
    assert len(label_predictions) == len(test_dataset)
    print(f'There are {len(label_predictions)} test examples')

    # Store predictions in a csv file to upload to Kaggle competition
    predictions_df = pd.DataFrame(data=zip(image_names, label_predictions), columns=['image_name', 'class_label'])
    predictions_df['class'] = predictions_df['class_label'].map(idx_to_class)
    predictions_df = predictions_df.drop(labels=['class_label'], axis=1)
    predictions_df.to_csv('./code/results/models/dinov2_large_finetuned/predictions.csv', index=None)


# Create confusion matrix to see where model struggles
if confusion_matrix:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choose finetuned model
    model = AutoModel.from_pretrained('facebook/dinov2-large')
    model = MultiTaskModel(model, num_classes)
    model.load_state_dict(torch.load("./code/results/models/dinov2_large_finetuned/dinov2_best_epoch_27.ckpt"))
    model.to(device)
    model.eval()

    label_predictions = []
    image_names = []
    with torch.no_grad():
        for batch, (im_names, _, _, _, _) in tqdm.tqdm(val_loader):
            batch = batch.to(device)
            output = model(batch)[0]
            preds = output.max(dim=1)[1]
            label_predictions.extend(preds.detach().cpu().numpy()) 
            image_names.extend(im_names.detach().cpu().numpy()) 

    assert len(label_predictions) == len(image_names) 
    assert len(label_predictions) == len(val_dataset.dataset)
    print(f'There are {len(label_predictions)} test examples')

    # Store predictions in a csv file to upload to Kaggle competition
    predictions_df = pd.DataFrame(data=zip(image_names, label_predictions), columns=['true_label', 'prediction_label'])
    predictions_df['class_prediction_label'] = predictions_df['prediction_label'].map(idx_to_class)
    predictions_df['class_true_label'] = predictions_df['true_label'].map(idx_to_class)
    predictions_df.to_csv('./code/results/models/dinov2_large_finetuned/confusion_matrix.csv', index=None)


    ###################################################################################################
    ################ WINNER OF KAGGLE COMPETITION WITH ACCURACY OF 96.149 % ON TEST SET ###############
    ###################################################################################################