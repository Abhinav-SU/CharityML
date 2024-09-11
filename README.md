
# Data Scientist Nanodegree
# Supervised Learning
## Project: Finding Donors for CharityML

### Objective
The objective of this project is to build a model that predicts whether an individual earns more than $50,000 based on demographic data. This model is built using supervised learning techniques to assist CharityML in identifying potential donors.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Jupyter Notebook](http://jupyter.org/)

If you don’t have Python installed, it’s recommended to install the [Anaconda distribution](https://www.anaconda.com/products/individual), which comes with the necessary packages pre-installed.

### Code

The main code for the project is provided in the `finding_donors.ipynb` notebook file. Additionally, the following files are included:
- `visuals.py`: A helper file for generating visualizations.
- `census.csv`: The dataset used in this project.

### Data

The dataset is a modified version of the 1994 U.S. Census database and contains approximately 32,000 data points with the following features:

**Features:**
- `age`: Age of the individual
- `workclass`: Type of work (e.g., Private, Self-emp, Government)
- `education_level`: Highest level of education completed
- `education-num`: Number of years of education completed
- `marital-status`: Marital status (e.g., Married, Divorced)
- `occupation`: Type of job (e.g., Sales, Tech-support)
- `relationship`: Family relationship (e.g., Husband, Wife, Not-in-family)
- `race`: Race of the individual
- `sex`: Gender (Male/Female)
- `capital-gain`: Capital gain from investments
- `capital-loss`: Capital loss from investments
- `hours-per-week`: Average working hours per week
- `native-country`: Country of origin

**Target Variable:**
- `income`: Whether the individual's income exceeds $50,000 (`<=50K` or `>50K`)

### Project Steps

1. **Data Preprocessing**:
   - Handled missing data.
   - One-hot encoded categorical variables (e.g., workclass, education).
   - Normalized continuous features (e.g., age, capital-gain).

2. **Model Selection**:
   - Implemented multiple models: Decision Tree, Random Forest, and Support Vector Machine (SVM).
   - Evaluated models using accuracy, precision, recall, and F1-score.
   - Used cross-validation and grid search to fine-tune model hyperparameters.

3. **Evaluation Metrics**:
   - **Accuracy**: Overall correctness of the model's predictions.
   - **Precision**: Proportion of positive identifications that were actually correct.
   - **Recall**: Proportion of actual positives that were identified correctly.
   - **F1-score**: Harmonic mean of precision and recall.

### Running the Project

1. In a terminal, navigate to the top-level project directory (`finding_donors/`), where this README file is located.

2. Run one of the following commands to start the Jupyter Notebook and view the project:
   ```bash
   jupyter notebook finding_donors.ipynb
   ```

3. Follow the instructions in the notebook to run the project step-by-step.

### Results

The model achieved an F1-score of approximately X% on the test set, and Random Forest was found to be the best-performing model after hyperparameter tuning. The model was able to correctly identify potential donors for CharityML.

### License

This project is licensed under the MIT License.
