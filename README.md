
# Brain Tumor Detection Using CNN and MAML

This project aims to develop a system for brain tumor detection using Convolutional Neural Networks (CNN) for binary classification to identify the presence of a tumor (yes/no). The system is enhanced by implementing a Model-Agnostic Meta-Learning (MAML) approach to classify tumor types such as glioma and meningioma. 

## Features

- **Binary Tumor Classification**: The initial model uses a CNN to classify brain images into two categories: tumor present (yes) or no tumor (no).
- **MAML for Generalization**: Model-Agnostic Meta-Learning (MAML) is applied to improve the model’s generalization ability, allowing it to adapt quickly to new, unseen tasks such as classifying specific tumor types (e.g., glioma vs meningioma).
- **Data Preprocessing**: The dataset is preprocessed to resize images and convert them to grayscale for better model performance.
- **Model Training and Evaluation**: The CNN is trained and evaluated using both standard and meta-learning techniques, with visualizations for accuracy and loss curves.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Python Imaging Library)
- Seaborn
- torchvision

You can install the necessary dependencies using `pip`:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow seaborn
```

## Project Structure

- **`main.py`**: The main script to train and evaluate the brain tumor detection model.
- **`data/`**: Contains the dataset for training and testing the model.
  - **`yes/`**: Images of brain scans with tumors.
  - **`no/`**: Images of brain scans without tumors.
- **`models/`**: Contains the CNN and MAML implementation.
- **`results/`**: Folder where the trained models and plots (accuracy and loss curves) will be saved.

## Dataset

The dataset consists of images of brain scans, where each image is labeled either with a tumor (yes) or without a tumor (no). The images are resized to 224x224 pixels for training.

The tumor types are later categorized into:
- Glioma
- Meningioma

You can replace the dataset path in the script with your dataset directory.

## Training the Model

1. **Prepare the Dataset**: 
   - Ensure your dataset is in the correct directory format. The images should be inside two subdirectories (`yes/` and `no/`) for the binary classification task.
   - Optionally, use a dataset with tumor types such as `glioma` and `meningioma`.

2. **Run the Training Script**:
   - The script will preprocess the images, train the CNN, and then apply the MAML method for better generalization.
   - To start training, simply run the following:

   ```bash
   python main.py
   ```

3. **Monitor the Progress**:
   - The training process includes both the CNN and MAML phases, with plots for training/validation accuracy and loss curves.
   - The model will save the best-performing weights and store them in `tumor_classifier.pth`.

## MAML (Meta-Learning)

The Model-Agnostic Meta-Learning (MAML) approach is implemented for this project, enabling the model to quickly adapt to new tasks (e.g., classifying different types of brain tumors). The MAML framework trains the model on multiple tasks, and the model learns to adapt to these tasks efficiently with minimal updates.

## Results

- After training, the best model will be saved as `best_glioma_meningioma_model.pt`.
- Training curves (accuracy and loss) will be displayed and saved in `training_curves.png`.
- The final model can be used for tumor classification on unseen data.

## Evaluation

The model is evaluated using a test set and reports metrics such as:

- **Accuracy**
- **Loss**
- **Confusion Matrix**

You can adjust the model parameters, such as learning rate or number of epochs, by modifying the training function in the script.

## Future Improvements

- Expand the dataset to include more tumor types.
- Integrate more advanced image augmentation techniques.
- Fine-tune the MAML hyperparameters to improve task-specific adaptation.


