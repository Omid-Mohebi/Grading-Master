# GPA Prediction using Deep Learning

This project predicts students' GPA using a deep learning model implemented in TensorFlow/Keras. It processes student demographic data, academic habits, and extracurricular involvement to provide a regression model capable of making accurate GPA predictions.

## Features
- **Custom ANN Architecture**: A multi-layer perceptron (MLP) with adjustable hidden layer sizes, dropout, and regularization.
- **Preprocessing Pipeline**:
  - One-hot encoding for categorical data.
  - StandardScaler for feature normalization.
- **Metrics**: Tracks Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared during training.
- **Early Stopping**: Prevents overfitting by halting training when validation loss stops improving.

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

## Workflow
1. **Data Preprocessing**:
   - One-hot encoding of categorical features (`Gender`, `Ethnicity`, `ParentalEducation`, etc.).
   - Feature normalization using `StandardScaler`.

2. **Model Architecture**:
   - Fully connected layers with ReLU activation.
   - Dropout for regularization.
   - Adam optimizer for faster convergence.

3. **Training**:
   - 300 epochs (default) with early stopping based on validation loss.
   - Evaluation metrics: MSE, MAE, and R-squared.

4. **Prediction**:
   - Generates GPA predictions for unseen test data.

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

3. Place your datasets in the `data/` directory:
   - `train.csv`
   - `test.csv`

## Usage
1. Run the main script to train the model and generate predictions:
   ```bash
   python main.py
   ```

2. Predictions will be saved in a `submission.csv` file.

## Results
- The current model achieves a Mean Absolute Error (MAE) of ~0.79 on the training dataset, indicating good predictive accuracy.

## Future Improvements
- Hyperparameter tuning for hidden layers, dropout rates, and learning rates.
- Feature engineering to identify the most significant predictors of GPA.
- Experimenting with alternative models, such as ensemble methods or transformers.

## Contributing
Feel free to open issues or submit pull requests for improvements and suggestions.
