# Updated code to read the given string and create graphs
#Epoch 0, Validation Loss: 2.2432, Tesseract CER: 0.3809, TrOCR CER: 0.3908, Accuracy: 0.54%, Word Error Rate: 0.7404, F1 Score: 0.4609, Precision: 0.4231, Recall: 0.5300
#Epoch 0, Validation Loss: 2.2607, Tesseract CER: 0.3809, TrOCR CER: 0.3998, Tesseract CER patch: 0.2647, Accuracy: 0.54%, Word Error Rate: 0.7404, F1 Score: 0.4609, Precision: 0.4231, Recall: 0.5300
#sroie
# Epoch 0, Validation Loss: 3.1802, Tesseract CER: 0.3802, TrOCR CER: 0.4804, Tesseract CER patch: 0.7479, Accuracy: 0.14%, Word Error Rate: 0.6593, F1 Score: 0.3671, Precision: 0.3553, Recall: 0.3804
log_data = """
Epoch 1, Prep Model Loss: 2.3798
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:50<00:00,  3.32it/s]
Epoch 1, Approximation Train Loss: 2.6582
Epoch 1, Validation Loss: 2.5888, Tesseract CER: 0.3874, TrOCR CER: 0.4569, Tesseract CER patch: 0.7473, Accuracy: 0.15%, Word Error Rate: 0.6747, F1 Score: 0.3554, Precision: 0.3459, Recall: 0.3671
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.95it/s]
Epoch 2, Prep Model Loss: 2.1827
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:50<00:00,  3.33it/s]
Epoch 2, Approximation Train Loss: 3.2181
Epoch 2, Validation Loss: 2.4130, Tesseract CER: 0.3489, TrOCR CER: 0.4317, Tesseract CER patch: 0.7359, Accuracy: 0.15%, Word Error Rate: 0.6674, F1 Score: 0.3645, Precision: 0.3541, Recall: 0.3769
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.97it/s]
Epoch 3, Prep Model Loss: 2.0563
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:47<00:00,  3.38it/s]
Epoch 3, Approximation Train Loss: 3.1902
Epoch 3, Validation Loss: 2.2605, Tesseract CER: 0.3101, TrOCR CER: 0.4055, Tesseract CER patch: 0.7168, Accuracy: 0.16%, Word Error Rate: 0.6474, F1 Score: 0.3727, Precision: 0.3636, Recall: 0.3833
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.96it/s]
Epoch 4, Prep Model Loss: 1.9465
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:43<00:00,  3.47it/s]
Epoch 4, Approximation Train Loss: 2.9850
Epoch 4, Validation Loss: 2.1751, Tesseract CER: 0.2770, TrOCR CER: 0.3866, Tesseract CER patch: 0.7091, Accuracy: 0.16%, Word Error Rate: 0.6231, F1 Score: 0.3864, Precision: 0.3781, Recall: 0.3958
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.96it/s]
Epoch 5, Prep Model Loss: 1.8356
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:40<00:00,  3.54it/s]
Epoch 5, Approximation Train Loss: 2.8689
Epoch 5, Validation Loss: 2.0828, Tesseract CER: 0.2641, TrOCR CER: 0.3743, Tesseract CER patch: 0.7221, Accuracy: 0.16%, Word Error Rate: 0.5920, F1 Score: 0.4118, Precision: 0.4033, Recall: 0.4217
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.95it/s]
Epoch 6, Prep Model Loss: 1.7483
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:39<00:00,  3.55it/s]
Epoch 6, Approximation Train Loss: 2.7359
Epoch 6, Validation Loss: 2.0155, Tesseract CER: 0.2595, TrOCR CER: 0.3640, Tesseract CER patch: 0.7080, Accuracy: 0.17%, Word Error Rate: 0.5679, F1 Score: 0.4235, Precision: 0.4163, Recall: 0.4318
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.95it/s]
Epoch 7, Prep Model Loss: 1.6715
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:40<00:00,  3.53it/s]
Epoch 7, Approximation Train Loss: 2.7310
Epoch 7, Validation Loss: 1.9555, Tesseract CER: 0.2455, TrOCR CER: 0.3458, Tesseract CER patch: 0.7044, Accuracy: 0.17%, Word Error Rate: 0.5708, F1 Score: 0.4246, Precision: 0.4185, Recall: 0.4318
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.96it/s]
Epoch 8, Prep Model Loss: 1.6264
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:38<00:00,  3.57it/s]
Epoch 8, Approximation Train Loss: 2.6593
Epoch 8, Validation Loss: 1.9231, Tesseract CER: 0.2299, TrOCR CER: 0.3377, Tesseract CER patch: 0.7113, Accuracy: 0.17%, Word Error Rate: 0.5501, F1 Score: 0.4347, Precision: 0.4296, Recall: 0.4407
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.97it/s]
Epoch 9, Prep Model Loss: 1.5652
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:38<00:00,  3.57it/s]
Epoch 9, Approximation Train Loss: 2.6327
Epoch 9, Validation Loss: 1.8857, Tesseract CER: 0.2319, TrOCR CER: 0.3353, Tesseract CER patch: 0.7070, Accuracy: 0.17%, Word Error Rate: 0.5481, F1 Score: 0.4454, Precision: 0.4382, Recall: 0.4538
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.98it/s]
Epoch 10, Prep Model Loss: 1.5097
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:38<00:00,  3.57it/s]
Epoch 10, Approximation Train Loss: 2.5289
Epoch 10, Validation Loss: 1.8467, Tesseract CER: 0.2267, TrOCR CER: 0.3307, Tesseract CER patch: 0.6947, Accuracy: 0.18%, Word Error Rate: 0.5411, F1 Score: 0.4481, Precision: 0.4430, Recall: 0.4542
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.99it/s]
Epoch 11, Prep Model Loss: 1.4696
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:36<00:00,  3.61it/s]
Epoch 11, Approximation Train Loss: 2.4571
Epoch 11, Validation Loss: 1.8077, Tesseract CER: 0.2194, TrOCR CER: 0.3227, Tesseract CER patch: 0.7011, Accuracy: 0.18%, Word Error Rate: 0.5350, F1 Score: 0.4498, Precision: 0.4465, Recall: 0.4539
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.97it/s]
Epoch 12, Prep Model Loss: 1.4201
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:37<00:00,  3.60it/s]
Epoch 12, Approximation Train Loss: 2.5423
Epoch 12, Validation Loss: 1.7751, Tesseract CER: 0.2149, TrOCR CER: 0.3112, Tesseract CER patch: 0.6869, Accuracy: 0.18%, Word Error Rate: 0.5320, F1 Score: 0.4525, Precision: 0.4493, Recall: 0.4565
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.96it/s]
Epoch 13, Prep Model Loss: 1.3751
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:36<00:00,  3.61it/s]
Epoch 13, Approximation Train Loss: 2.4263
Epoch 13, Validation Loss: 1.7367, Tesseract CER: 0.2172, TrOCR CER: 0.3061, Tesseract CER patch: 0.6950, Accuracy: 0.18%, Word Error Rate: 0.5217, F1 Score: 0.4556, Precision: 0.4503, Recall: 0.4615
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.98it/s]
Epoch 14, Prep Model Loss: 1.3442
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:36<00:00,  3.62it/s]
Epoch 14, Approximation Train Loss: 2.3692
Epoch 14, Validation Loss: 1.7277, Tesseract CER: 0.2112, TrOCR CER: 0.3119, Tesseract CER patch: 0.6954, Accuracy: 0.19%, Word Error Rate: 0.5284, F1 Score: 0.4514, Precision: 0.4468, Recall: 0.4569
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.98it/s]
Epoch 15, Prep Model Loss: 1.3113
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:37<00:00,  3.60it/s]
Epoch 15, Approximation Train Loss: 2.3732
Epoch 15, Validation Loss: 1.6950, Tesseract CER: 0.2215, TrOCR CER: 0.3049, Tesseract CER patch: 0.6910, Accuracy: 0.18%, Word Error Rate: 0.5298, F1 Score: 0.4563, Precision: 0.4509, Recall: 0.4626
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.97it/s]
Epoch 16, Prep Model Loss: 1.2715
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:38<00:00,  3.58it/s]
Epoch 16, Approximation Train Loss: 2.3942
Epoch 16, Validation Loss: 1.7170, Tesseract CER: 0.2500, TrOCR CER: 0.3194, Tesseract CER patch: 0.7092, Accuracy: 0.17%, Word Error Rate: 0.5325, F1 Score: 0.4455, Precision: 0.4396, Recall: 0.4522
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.95it/s]
Epoch 17, Prep Model Loss: 1.2493
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:37<00:00,  3.59it/s]
Epoch 17, Approximation Train Loss: 2.3426
Epoch 17, Validation Loss: 1.6571, Tesseract CER: 0.2072, TrOCR CER: 0.3037, Tesseract CER patch: 0.6897, Accuracy: 0.18%, Word Error Rate: 0.5147, F1 Score: 0.4628, Precision: 0.4570, Recall: 0.4692
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.96it/s]
Epoch 18, Prep Model Loss: 1.2268
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:37<00:00,  3.59it/s]
Epoch 18, Approximation Train Loss: 2.3435
Epoch 18, Validation Loss: 1.6527, Tesseract CER: 0.2196, TrOCR CER: 0.3003, Tesseract CER patch: 0.6917, Accuracy: 0.18%, Word Error Rate: 0.5181, F1 Score: 0.4589, Precision: 0.4530, Recall: 0.4654
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.95it/s]
Epoch 19, Prep Model Loss: 1.1996
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:37<00:00,  3.58it/s]
Epoch 19, Approximation Train Loss: 2.3303
Epoch 19, Validation Loss: 1.6506, Tesseract CER: 0.2318, TrOCR CER: 0.3012, Tesseract CER patch: 0.7035, Accuracy: 0.17%, Word Error Rate: 0.5177, F1 Score: 0.4579, Precision: 0.4520, Recall: 0.4645
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [01:21<00:00,  6.97it/s]
Epoch 20, Prep Model Loss: 1.1752
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 566/566 [02:36<00:00,  3.61it/s]
Epoch 20, Approximation Train Loss: 2.2512
Epoch 20, Validation Loss: 1.6271, Tesseract CER: 0.2210, TrOCR CER: 0.2996, Tesseract CER patch: 0.6980, Accuracy: 0.18%, Word Error Rate: 0.5143, F1 Score: 0.4589, Precision: 0.4545, Recall: 0.4638
"""
import matplotlib.pyplot as plt

# Initialize lists for storing metrics separately for training and validation
train_prep_losses = []
train_approx_losses = []
# train_losses = []
# train_cers = []
# train_trocrcers = []
# train_patches_cer = []
# train_accuracies = []
# train_wers = []
# train_f1_scores = []
# train_precisions = []
# train_recalls = []

validation_losses = []
validation_cers = []
validation_trocrcers = []
validation_patches_cer = []
validation_accuracies = []
validation_wers = []
validation_f1_scores = []
validation_precisions = []
validation_recalls = []

# Split the log data into lines
lines = log_data.strip().split('\n')

for line in lines:
    if 'Epoch' in line:
        parts = line.split(',')

        # Check for Prep Model Loss in training data
        if 'Prep Model Loss' in line:
            prep_loss = float(line.split(':')[1].strip())
            train_prep_losses.append(prep_loss)
        
        # Check for Approximation Train Loss in training data
        if 'Approximation Train Loss' in line:
            approx_loss = float(line.split(':')[1].strip())
            train_approx_losses.append(approx_loss)

        # # Check for Train Loss or Validation Loss
        # if 'Train Loss' in line:
        #     train_loss = float(parts[1].split(':')[1].strip())
        #     train_losses.append(train_loss)

        #     for part in parts[1:]:
        #         if 'Tesseract CER patch' in part:
        #             cer_patch = float(part.split(':')[1].strip())
        #             train_patches_cer.append(cer_patch)
        #         elif 'Tesseract CER full document' in part:
        #             cer = float(part.split(':')[1].strip())
        #             train_cers.append(cer)
        #         elif 'TrOCR CER full document' in part:
        #             trocrcer = float(part.split(':')[1].strip())
        #             train_trocrcers.append(trocrcer)
        #         elif 'Accuracy' in part:
        #             accuracy = float(part.split(':')[1].strip().rstrip('%')) / 100  # Convert percentage to decimal
        #             train_accuracies.append(accuracy)
        #         elif 'Word Error Rate' in part:
        #             wer = float(part.split(':')[1].strip())
        #             train_wers.append(wer)
        #         elif 'F1 Score' in part:
        #             f1_score = float(part.split(':')[1].strip())
        #             train_f1_scores.append(f1_score)
        #         elif 'Precision' in part:
        #             precision = float(part.split(':')[1].strip())
        #             train_precisions.append(precision)
        #         elif 'Recall' in part:
        #             recall = float(part.split(':')[1].strip())
        #             train_recalls.append(recall)

        elif 'Validation Loss' in line:
            validation_loss = float(parts[1].split(':')[1].strip())
            validation_losses.append(validation_loss)

            for part in parts[1:]:
                if 'Tesseract CER patch' in part:
                    cer_patch = float(part.split(':')[1].strip())
                    validation_patches_cer.append(cer_patch)
                elif 'Tesseract CER' in part:
                    cer = float(part.split(':')[1].strip())
                    validation_cers.append(cer)
                elif 'TrOCR CER' in part:
                    trocrcer = float(part.split(':')[1].strip())
                    validation_trocrcers.append(trocrcer)
                elif 'Accuracy' in part:
                    accuracy = float(part.split(':')[1].strip().rstrip('%')) / 100  # Convert percentage to decimal
                    validation_accuracies.append(accuracy)
                elif 'Word Error Rate' in part:
                    wer = float(part.split(':')[1].strip())
                    validation_wers.append(wer)
                elif 'F1 Score' in part:
                    f1_score = float(part.split(':')[1].strip())
                    validation_f1_scores.append(f1_score)
                elif 'Precision' in part:
                    precision = float(part.split(':')[1].strip())
                    validation_precisions.append(precision)
                elif 'Recall' in part:
                    recall = float(part.split(':')[1].strip())
                    validation_recalls.append(recall)

# Create epoch range based on lengths of metrics
length = max(len(train_prep_losses), len(validation_losses))  # Use the max of train/validation data
epochs = list(range(1, length + 1))

# Plotting function
def plot_metric(epochs, values, label, ylabel, dataset):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{label} vs. Epoch ({dataset})')
    plt.grid(True)
    plt.legend([label])
    plt.savefig(f'graphs/{label.replace(" ", "_").lower()}_vs_epoch_{dataset.lower()}.png')
    plt.close()



# Function to plot comparison between train and validation metrics
def plot_comparison(epochs, train_values, val_values, label, ylabel):
    plt.figure(figsize=(10, 5))  # Set figure size (width, height)
    plt.plot(epochs, val_values, label=f'TrOCR CER', color='dodgerblue')
    plt.plot(epochs, train_values, label=f'Tesseract CER', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('CER')
    plt.title(f'TrOCR and Tesseract CER Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graphs/train_vs_validation_{label.replace(" ", "_").lower()}.png')
    plt.close()  # Close the figure to free up memory


# Debugging prints to check the lengths of all lists
print(f'Epochs Length: {len(epochs)}')
print(f'Train Prep Losses Length: {len(train_prep_losses)}')
print(f'Train Approximation Losses Length: {len(train_approx_losses)}')
# print(f'Train Losses Length: {len(train_losses)}')
# print(f'Train Tesseract CERs Length: {len(train_cers)}')
# print(f'Train TrOCR CERs Length: {len(train_trocrcers)}')
# print(f'Train Tesseract CER Patches Length: {len(train_patches_cer)}')
# print(f'Train Accuracies Length: {len(train_accuracies)}')
# print(f'Train Word Error Rates Length: {len(train_wers)}')
# print(f'Train F1 Scores Length: {len(train_f1_scores)}')
# print(f'Train Precisions Length: {len(train_precisions)}')
# print(f'Train Recalls Length: {len(train_recalls)}')

print(f'Validation Losses Length: {len(validation_losses)}')
print(f'Validation Tesseract CERs Length: {len(validation_cers)}')
print(f'Validation TrOCR CERs Length: {len(validation_trocrcers)}')
print(f'Validation Tesseract CER Patches Length: {len(validation_patches_cer)}')
print(f'Validation Accuracies Length: {len(validation_accuracies)}')
print(f'Validation Word Error Rates Length: {len(validation_wers)}')
print(f'Validation F1 Scores Length: {len(validation_f1_scores)}')
print(f'Validation Precisions Length: {len(validation_precisions)}')
print(f'Validation Recalls Length: {len(validation_recalls)}')
# Plot metrics for training data
plot_metric(epochs, train_prep_losses, 'Prep Model Loss', 'Loss', 'Train')
plot_metric(epochs, train_approx_losses, 'Approximation Train Loss', 'Loss', 'Train')
# plot_metric(epochs, train_cers, 'Tesseract CER', 'CER', 'Train')
# plot_metric(epochs, train_trocrcers, 'TrOCR CER', 'CER', 'Train')
# plot_metric(epochs, train_patches_cer, 'Tesseract CER Patch', 'CER', 'Train')
# plot_metric(epochs, train_accuracies, 'Accuracy', 'Accuracy', 'Train')
# plot_metric(epochs, train_wers, 'Word Error Rate', 'WER', 'Train')
# plot_metric(epochs, train_f1_scores, 'F1 Score', 'F1 Score', 'Train')
# plot_metric(epochs, train_precisions, 'Precision', 'Precision', 'Train')
# plot_metric(epochs, train_recalls, 'Recall', 'Recall', 'Train')

# Plot metrics for validation data
plot_metric(epochs, validation_losses, 'Validation Loss', 'Loss', 'Validation')
plot_metric(epochs, validation_cers, 'Tesseract CER', 'CER', 'Validation')
plot_metric(epochs, validation_trocrcers, 'TrOCR CER', 'CER', 'Validation')
plot_metric(epochs, validation_patches_cer, 'Tesseract CER Patch', 'CER', 'Validation')
plot_metric(epochs, validation_accuracies, 'Accuracy', 'Accuracy', 'Validation')
plot_metric(epochs, validation_wers, 'Word Error Rate', 'WER', 'Validation')
plot_metric(epochs, validation_f1_scores, 'F1 Score', 'F1 Score', 'Validation')
plot_metric(epochs, validation_precisions, 'Precision', 'Precision', 'Validation')
plot_metric(epochs, validation_recalls, 'Recall', 'Recall', 'Validation')



# Check if train and validation lists match the epochs length before plotting
if len(epochs) == len(validation_cers) == len(validation_trocrcers):
    plot_comparison(epochs, validation_cers, validation_trocrcers, 'tesseract cer', 'trocr cer')

# if len(epochs) == len(train_accuracies) == len(validation_accuracies):
#     plot_comparison(epochs, train_accuracies, validation_accuracies, 'Accuracy', 'Accuracy')

# if len(epochs) == len(train_wers) == len(validation_wers):
#     plot_comparison(epochs, train_wers, validation_wers, 'Word Error Rate', 'Word Error Rate (WER)')

# if len(epochs) == len(train_cers) == len(validation_cers):
#     plot_comparison(epochs, train_cers, validation_cers, 'Tesseract CER', 'CER')

# if len(epochs) == len(train_trocrcers) == len(validation_trocrcers):
#     plot_comparison(epochs, train_trocrcers, validation_trocrcers, 'TrOCR CER', 'CER')


