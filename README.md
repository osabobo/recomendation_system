# Customer Recommendation System

This project is a Streamlit-based web application that provides personalized product recommendations for customers using a machine learning model.

## Features

- **Product Recommendation**: Recommends products based on customer purchase history and preferences.
- **Streamlit Interface**: User-friendly web interface powered by Streamlit.
- **Machine Learning Model**: Built with TensorFlow/Keras for generating recommendations.

## Installation

### Requirements

To run this project, ensure you have the following:
- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

### Step-by-Step Setup

1. **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download or Add Data Files**:
    - Place your dataset (`PREDICTIVE2.csv` or `PREDICTIVE.xlsx`) in the project folder.

4. **Run the Application**:
    ```bash
    streamlit run main.py
    ```

    This will start a local server. Open the displayed URL (usually http://localhost:8501) in a browser to use the app.

## Project Structure

- `main.py`: The main script that runs the Streamlit app.
- `requirements.txt`: Lists all necessary Python packages.
- `PREDICTIVE2.csv` / `PREDICTIVE.xlsx`: Data files used for generating recommendations.

## Cloud Deployment

To deploy this app to the cloud (e.g., Streamlit Cloud, Heroku), upload the entire project, including `requirements.txt`, `main.py`, and any data files required by the app. Configure the cloud environment to use `main.py` as the entry point.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
