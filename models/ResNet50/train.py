import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torchvision import datasets, models, transforms

NUM_CLASSES = 4
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-4



def run_experiment(model, model_name, train_dir, test_dir, output_dir, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    os.makedirs(output_dir, exist_ok=True)
    train_losses, train_accuracies = [], []
    eval_losses, eval_accuracies = [], []

    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # The held-out test partition is evaluated after each epoch, following
        # the experiment protocol. A separate validation split can be used in future
        # experiments if strict train/validation/test separation is required.
        model.eval()
        eval_loss, correct_eval, total_eval = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total_eval += labels.size(0)
                correct_eval += (predicted == labels).sum().item()
        eval_loss /= len(test_loader)
        eval_acc = 100.0 * correct_eval / total_eval
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_acc)
        print(f"{model_name} epoch {epoch+1:02d}/{EPOCHS}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"eval_loss={eval_loss:.4f}, eval_acc={eval_acc:.2f}%")

    # Final evaluation on the held-out test partition.
    model.eval()
    all_labels, all_predictions, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    all_labels = np.asarray(all_labels)
    all_predictions = np.asarray(all_predictions)
    all_probs = np.asarray(all_probs)
    accuracy = 100.0 * (all_predictions == all_labels).mean()
    class_names = train_dataset.classes

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)

    y_bin = label_binarize(all_labels, classes=range(len(class_names)))
    roc_auc = {}
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[class_name] = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc[class_name]:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), eval_losses, label="Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_model.pt"))

    with open(os.path.join(output_dir, "training_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Start: {start_dt}\n")
        f.write(f"Training End: {end_dt}\n")
        f.write(f"Elapsed Time: {elapsed:.2f} seconds\n")
        f.write(f"Final Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
        f.write("\nAUC Scores:\n")
        for class_name in class_names:
            f.write(f"{class_name}: {roc_auc[class_name]:.4f}\n")

    print(f"Final test accuracy: {accuracy:.2f}%")
    print(f"Elapsed time: {elapsed:.2f} s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="ImageFolder-compatible training directory")
    parser.add_argument("--test-dir", required=True, help="ImageFolder-compatible held-out test directory")
    parser.add_argument("--output-dir", default="results_new", help="Directory for a new run")
    return parser.parse_args()


def main():
    args = parse_args()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet50(weights=None, num_classes=NUM_CLASSES)
    run_experiment(model, "ResNet50", args.train_dir, args.test_dir, args.output_dir, transform)


if __name__ == "__main__":
    main()
