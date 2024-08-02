# 🏡 Real Estate Dashboard

Welcome to the Real Estate Dashboard project! This interactive dashboard provides insights and visualizations for real estate data in El Salvador. 🌍

## 🚀 Features

- 📊 **Interactive Visualizations**: Easily explore real estate trends with interactive charts and maps.
- 📈 **Data Analysis**: Perform in-depth analysis on property prices, locations, and other key metrics.
- 🌐 **Geographical Mapping**: Visualize property distributions across different regions.

## 📂 Project Structure

Here's an overview of the project structure:

```
.
├── assets/
├── data/
├── env/
├── geo-map/
├── models/
├── .gitignore
├── application.py
├── Procfile
├── README.md
├── requirements.txt
├── wsgi.py
```

- **assets/**: Contains static files such as images and styles.
- **data/**: Directory for data storage.
- **env/**: Virtual environment files (ignored by Git).
- **geo-map/**: Contains geographical data and map-related files.
- **models/**: Machine learning models and related files.
- **application.py**: Main application script for running the dashboard.
- **Procfile**: File for deployment configuration.
- **requirements.txt**: List of dependencies required for the project.
- **wsgi.py**: WSGI entry point for the application.

## 🛠️ Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Manuel-ventura/Real-Estate-Dashboard.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Real-Estate-Dashboard
   ```

3. **Create a virtual environment:**

   ```bash
   python -m venv env
   ```

4. **Activate the virtual environment:**

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source env/bin/activate
     ```

5. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

Run the application locally with the following command:

```bash
python application.py
```

## 📈 Data Sources

The data used in this project is scraped from real estate listings in El Salvador. The dataset includes information such as location, property title, description, price, square meters, bedrooms, parking spaces, and bathrooms.

## 🌍 Deployment

This project is configured for deployment on AWS Elastic Beanstalk. Ensure you have the AWS CLI configured before deploying.

To deploy, run:

```bash
eb init -p python-3.7 real-estate-dashboard
eb create real-estate-env
```

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit pull requests with your changes.

## 📜 License

This project is licensed under the MIT License.

## 👥 Author

- **Manuel Ventura** - [GitHub](https://github.com/Manuel-ventura)
