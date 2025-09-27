# Bird Species Classification with CNN

## Project Objective
The main objective of this project is to develop a Convolutional Neural Network (CNN)-based model for bird species classification. The project focuses on building, training, and evaluating a deep learning model to accurately classify images of birds into their respective species.

## Dataset Information
The dataset contains images of different bird species with varying numbers of samples per class. The dataset was divided into:
- **Train set:** 70%
- **Validation set:** 15%
- **Test set:** 15%

Data augmentation techniques such as rotation, flipping, and color jitter were applied to increase the dataset's diversity and reduce overfitting.

## Methods Used
1. **Data Preprocessing:**  
   - Dataset splitting (70/15/15)  
   - Data augmentation techniques  

2. **Model Architecture:**  
   - Convolutional Layers 
   - Pooling Layers   
   - Dropout Layers   
   - Fully Connected Layers  

3. **Model Training:**  
   - Cross-entropy loss function  
   - Adam optimizer  
   - Hyperparameter tuning for learning rate, dropout, and batch size  

4. **Model Evaluation:**  
   - Accuracy & Loss curves  
   - Confusion Matrix & Classification Report  
   - Grad-CAM visualizations for interpretability
  

## Results Summary

The experiments conducted throughout the project provided valuable insights into the performance of our CNN-based bird species classification model. During the training phase, we monitored both the training and validation accuracy across multiple epochs. The model achieved a validation accuracy of approximately **71.7%**, which served as an important indicator of its ability to generalize to unseen data.  

To further evaluate the model, we used the test set to compute a detailed **classification report**. This report revealed that while certain classes achieved high precision and recall scores, others suffered due to the **class imbalance** in the dataset. Some species with very few samples were particularly challenging for the model, leading to lower F1-scores in those categories.  

We also utilized **Grad-CAM** visualizations to better interpret the model’s decision-making process. These heatmaps highlighted the regions of images that most influenced the model’s predictions, helping us understand whether the model was focusing on the correct bird features or irrelevant background areas.  

Finally, **hyperparameter optimization** experiments were conducted over learning rate, dropout rate, and batch size. The best configuration was found to be a learning rate of **0.003**, a dropout rate of **0.2**, and a batch size of **16**, resulting in improved validation accuracy and reduced overfitting.  

Overall, the project demonstrated a clear performance gain through data augmentation, hyperparameter tuning, and interpretability techniques, laying the groundwork for potential improvements such as transfer learning or more balanced datasets.

  ## Kaggle Notebook Link
You can access the full notebook with code, outputs, and analysis on Kaggle:  
[**Kaggle Notebook Link**](https://www.kaggle.com/code/ozgurdenizhincal/bird-classification)
