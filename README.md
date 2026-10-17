# ✈️ Predicta — Predictive Maintenance ML 🤖

### Predicting engine failure before it happens, one data point at a time. 💡

Hi there! 👋 Welcome to **Predicta**, a machine learning project where I explore how data and AI can help predict when an aircraft engine might need maintenance.

As a Computer Science student passionate about Machine Learning, Data Science, and AI, I wanted to build a project that goes beyond just training a model. I wanted to understand the entire process — from exploring real-world-style data to engineering features, evaluating predictions, and building something that actually works. 🧠✨

That's how **Predicta** came to life! 🚀

---

## 💭 What's the idea behind Predicta?

Imagine being able to estimate how many operational cycles an aircraft engine has left before it fails.

Instead of waiting for equipment to break down, predictive maintenance uses data to help anticipate potential problems and make maintenance planning smarter.

That's the idea behind this project! 🔧

Using the NASA C-MAPSS turbofan engine dataset, I developed a machine learning model that predicts the **Remaining Useful Life (RUL)** of an engine based on its sensor measurements and operating conditions.

The goal? Turn raw sensor data into useful predictions. 📊

### 🎯 Project goals

- 🔍 Explore and understand engine sensor data.
- 🧹 Prepare the dataset and generate RUL labels.
- ⚙️ Engineer features that capture changes in sensor behavior.
- 🤖 Train and compare machine learning models.
- 📈 Evaluate model performance using appropriate metrics.
- 💻 Build a simple command-line interface for making predictions.
- 🧪 Test the prediction pipeline and validate its behavior.

---

## 🛰️ The Dataset

For this project, I used the **NASA C-MAPSS FD001 turbofan engine dataset**.

The dataset contains simulated engine degradation data, including operational cycles, operating settings, and 21 sensor measurements.

Here are the main files used:

| File | What's inside? |
|---|---|
| `train_FD001.txt` | Training sensor measurements |
| `test_FD001.txt` | Test sensor measurements |
| `RUL_FD001.txt` | Ground-truth RUL labels for test engines |

💡 **What does RUL mean?**

Remaining Useful Life (RUL) estimates how many operational cycles remain before an engine reaches its end of useful life.

For the training data, I calculated RUL by subtracting each engine's current cycle from its final recorded cycle.

This gave the model a target to learn from! 📚

---

## 🛠️ How I built it

I approached this project as a complete machine learning workflow rather than jumping straight into model training.

### 🔎 1. Exploratory Data Analysis

First, I explored the dataset to understand its structure, investigate sensor measurements, and look for patterns related to engine degradation.

This helped me understand the data before deciding how to build the model.

📓 Notebook: `notebooks/01_data_exploration.ipynb`

### ⚙️ 2. Feature Engineering

Raw sensor readings don't always tell the whole story.

To help the model capture changes over time, I created rolling averages using windows of **5 and 10 cycles** for the sensor measurements.

These features use only the current and previous observations, so future sensor readings aren't used to predict the past. ✅

### 🤖 3. Model Development

I started with a Ridge Regression baseline and then explored Random Forest regression.

I compared model performance using:

- 📉 Mean Absolute Error (MAE)
- 📊 Root Mean Squared Error (RMSE)
- 🎯 R² Score

I also split the training and validation data by engine rather than randomly splitting individual rows. This helps prevent data from the same engine appearing in both sets and gives a more meaningful validation evaluation.

📓 Notebook: `notebooks/02_baseline_model.ipynb`

### 📈 4. Model Evaluation

After developing the model, I evaluated its performance on both a validation set and the official FD001 test set.

Here are the results! 👇

#### 🧪 Validation Results

| Metric | Result |
|---|---:|
| MAE | 23.912 cycles |
| RMSE | 33.719 cycles |
| R² Score | 0.736 |

#### 🚀 Official Test Set Results

The final test evaluation used all available training engines to train a model, then evaluated predictions for **100 test engines** against their provided ground-truth RUL labels.

| Metric | Result |
|---|---:|
| MAE | 23.520 cycles |
| RMSE | 33.324 cycles |
| R² Score | 0.357 |
| Test engines | 100 |

💡 **What do these numbers mean?**

The test MAE of approximately 23.5 cycles means that the model's predictions differed from the true RUL values by an average absolute error of about 23.5 cycles.

The test R² score also reminds me that there's still room to improve how well the model explains differences in RUL across engines.

And honestly, that's part of the process! Machine learning isn't just about getting a good score — it's about understanding the results, recognizing limitations, and figuring out what to improve next. 🌱

---

## 💻 Try it yourself!

Want to explore the project? Here's how to get started. 👇

### 📦 1. Clone the repository

```bash
git clone https://github.com/RanimRebai/predictive-maintenance-ml.git
cd predictive-maintenance-ml
```

### 🐍 2. Create a virtual environment
On Windows:
python -m venv .venv
.venv\Scripts\activate

### 📚 3. Install the dependencies
python -m pip install -r requirements.txt

Make sure the three FD001 dataset files are available in the data/ directory. If they aren't included in the repository, obtain the dataset separately.
### 🏋️ 4. Train the model
python src/train.py

This trains the Random Forest model and saves a model artifact in the models/ directory.
### 📊 5. Evaluate the model
python src/evaluate.py

This trains a model on the complete training dataset and evaluates it against the official test labels.
### 🔮 6. Make a prediction
python src/app.py

The command-line interface asks for an engine ID and a cycle cutoff, then estimates the remaining useful life using the engine's available history.
### 🧪 7. Run the tests
python -m pytest src/test_predict.py -v

The tests check normal prediction behavior and validate that invalid inputs are handled appropriately.
### 🗂️ Project Structure
Here's how everything is organized! 📁
predictive-maintenance-ml/
│
├── data/                   # Dataset files
│
├── models/                 # Saved model artifacts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_model.ipynb
│
├── src/
│   ├── train.py            # Model training
│   ├── predict.py          # Prediction logic
│   ├── app.py              # Command-line interface
│   ├── evaluate.py         # Official test evaluation
│   └── test_predict.py     # Prediction tests
│
├── requirements.txt
├── requirements-lock.txt
└── README.md

### 🧰 Technologies I used
Here's the toolkit behind Predicta! 🛠️
Technology	Purpose
🐍 Python	Main programming language
🐼 pandas	Data manipulation and analysis
🔢 NumPy	Numerical computing
🤖 scikit-learn	Machine learning and model evaluation
📊 Matplotlib	Data visualization
🎨 Seaborn	Statistical visualizations
💾 Joblib	Saving and loading trained models
📓 Jupyter Notebook	Exploratory analysis and experimentation
🧪 pytest	Automated testing
🐙 Git & GitHub	Version control and project hosting


# 🌱 What I learned
Building Predicta helped me work through several important parts of a machine learning project:
- Understanding a dataset before choosing a model.
- Creating target labels for a regression problem.
- Engineering time-dependent features from sensor data.
- Avoiding data leakage through engine-level validation splitting.
- Comparing regression models using multiple evaluation metrics.
- Evaluating predictions on a separate test dataset.
- Building a reusable prediction pipeline.
- Writing tests to check how the system handles invalid inputs.
It also reinforced an important lesson: a machine learning project doesn't end when the model trains successfully. Understanding its performance and limitations matters just as much. 💡
# 🔮 What's next?
Predicta is a work in progress, and there are plenty of things I'd love to explore next! ✨
- 🔍 Investigate engines with the largest prediction errors.
- 🧠 Experiment with additional feature engineering approaches.
- 🤖 Compare other regression algorithms.
- 📊 Create more visualizations of predicted versus actual RUL.
- 🖥️ Explore an interactive dashboard for engine health predictions.
The goal is to keep improving the project while strengthening my understanding of applied machine learning.
## ⚠️ A quick note about limitations
This project uses simulated engine data from NASA C-MAPSS FD001. Its performance has not been validated on real-world aircraft engines.
RUL predictions are estimates, not guarantees of when an engine will fail. Real maintenance applications would require much more extensive validation.
# 👩🏻‍💻 A little about me
Hi! I'm Ranim Rebai, a Computer Science student pursuing my B.Sc. at Constructor University in Bremen, Germany. 🎓
I'm interested in Data Science, Machine Learning, and Artificial Intelligence, and I'm using projects like Predicta to turn what I learn into practical experience.
I believe the best way to grow is to keep building, stay curious, learn from mistakes, and never stop asking questions. 🌱
Predicta is one step in that journey, and I'm excited to keep learning and creating! 🚀
# 💌 Thanks for stopping by!
If you've made it this far, thank you for checking out my project! 💗
I'm always learning, experimenting, and looking for ways to become a better developer and machine learning practitioner.
One project at a time. One lesson at a time. One step closer to the future I'm building. ✨
⭐ If you find this project interesting, feel free to explore the repository!
