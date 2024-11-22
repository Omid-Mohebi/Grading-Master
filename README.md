# GPA Prediction Using Machine Learning

This project predicts students' GPA based on demographic, academic, and extracurricular features. It includes two implementations:
1. A deep learning model (ANN) using TensorFlow/Keras.
2. A simpler, more accurate implementation using Linear Regression.

The project explores both approaches to demonstrate the strengths and trade-offs of each method.

---

## Features
- **Deep Learning Model (ANN)**:
  - Customizable architecture with fully connected layers and dropout for regularization.
  - Tracks MSE, MAE, and R-squared for evaluation.
  - Implements early stopping to prevent overfitting.

- **Linear Regression**:
  - Simpler, interpretable model achieving **95% accuracy** on the training set.
  - Best suited for datasets with strong linear relationships between features and target.

---

## Dataset
The dataset contains the following columns:

| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| `StudentID`        | Unique identifier for each student (not used).  |
| `Age`              | Student's age.                                  |
| `Gender`           | Male/Female/Other.                              |
| `Ethnicity`        | Categorical representation of ethnicity.        |
| `ParentalEducation`| Highest education level of parents.             |
| `StudyTimeWeekly`  | Hours spent studying weekly.                    |
| `Absences`         | Number of absences.                             |
| `Tutoring`         | Boolean: Whether the student attends tutoring.  |
| `ParentalSupport`  | Boolean: Whether parental support is provided.  |
| `Extracurricular`  | Boolean: Participation in extracurriculars.     |
| `Sports`           | Boolean: Participation in sports activities.    |
| `Music`            | Boolean: Involvement in music programs.         |
| `Volunteering`     | Boolean: Participation in volunteering.         |
| `GPA`              | The target variable: Grade Point Average.       |

---

## Implementations
### 1. Deep Learning (ANN)
This approach builds a multi-layer perceptron with dropout layers and uses advanced optimization techniques like Adam and early stopping.

Key Features:
- **Architecture**: Fully connected layers with ReLU activation.
- **Performance**: Achieves a Mean Absolute Error (MAE) of ~0.79, equivalent to ~20% of the GPA scale.
- **Best Use Case**: Suitable for exploring non-linear relationships or experimenting with deep learning techniques.

### 2. Simple Linear Regression
A streamlined implementation using `scikit-learn`'s `LinearRegression` model.

Key Features:
- **Performance**: Achieves ~95% accuracy, demonstrating the suitability of linear regression for this dataset.
- **Advantages**:
  - Faster and more interpretable.
  - Requires fewer computational resources.

### Why Two Versions?
The ANN implementation combines popular deep learning techniques but is not the best choice for data with a strong linear relationship. The simpler linear regression version is more accurate and efficient, making it the better choice for this dataset.

---

## File Structure
- **`src.ipynb`**: Contains the ANN implementation.
- **`simple_v.py`**: Contains the Linear Regression implementation.
- **`data/`**: Folder containing `train.csv` and `test.csv`.
- **`submission.csv`**: Predictions generated for the test set.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gpa-prediction.git
   cd gpa-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add the training and test data (`train.csv` and `test.csv`) to the `data/` directory.

---

## Usage
1. **To run the ANN model**:
   ```bash
   python ann_model.py
   ```
   - Outputs predictions to `submission.csv`.

2. **To run the simpler Linear Regression model**:
   ```bash
   python simple_v.py
   ```
   - Outputs predictions to `submission.csv`.

---

## Results
| Model               | Accuracy | MAE  | Comments                                         |
|---------------------|----------|------|-------------------------------------------------|
| **Linear Regression** | 95%      | ~0.3 | Best-performing model for this dataset.         |
| **ANN (Deep Learning)** | ~80%     | ~0.79 | Useful for exploring non-linear techniques.     |

---

## Insights
- **Linear Regression**: Simpler, faster, and better suited for this data.
- **Deep Learning**: Over-engineered for the current dataset but provides flexibility for more complex, non-linear data.

---

## Contributing
Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

