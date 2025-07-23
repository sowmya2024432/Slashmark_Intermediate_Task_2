# Slashmark_Intermediate_Task_2

🔢 Digit Recognition with K-Nearest Neighbors
This project demonstrates a simple handwritten digit classification model using K-Nearest Neighbors (KNN) and Scikit-learn's digits dataset. The model is trained to recognize digits (0–9) based on 8x8 grayscale images.

📌 Features
Uses Scikit-learn's built-in digits dataset

Visualizes digits with Matplotlib

Applies KNN classification (k=3)

Reports model accuracy

Displays predictions on sample test images

⚙️ Used Technologies
Python 3

Scikit-learn – For loading data, model training, and evaluation

Matplotlib – For displaying grayscale images of digits

KNeighborsClassifier – KNN algorithm for classification

train_test_split – To split data into training and test sets


🧠 How It Works
Load digit images and labels using datasets.load_digits().

Flatten each 8×8 image into a 64-element vector.

Split into training and testing datasets.

Train a KNeighborsClassifier (k=3).

Evaluate the model and visualize predictions.

📜 License
This project is licensed under the MIT License.
