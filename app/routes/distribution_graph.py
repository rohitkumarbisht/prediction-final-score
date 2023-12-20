from flask_classful import FlaskView
from flask import render_template
from app.routes.upload_csv import UploadCSV
from app.utils.file_open import save_image
import pandas as pd
import config
import psycopg2
import os
import matplotlib.pyplot as plt

class DistributionGraph(FlaskView):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DistributionGraph, cls).__new__(cls)
            cls._instance.csv_data = None
        return cls._instance
    
    def fetch_csv_data(self):
        upload_csv_instance = UploadCSV()
        uploaded_data = upload_csv_instance.fetch_data()
        if uploaded_data is None:
            self.csv_data = self.fetch_data_from_postgresql()
        else:
            self.csv_data = uploaded_data
        return self.csv_data
    
    # connect with the PostgreSQL Server
    def fetch_data_from_postgresql(self):
        try:
            conn = psycopg2.connect(
                host=config.db_host,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name
            )
            cursor = conn.cursor()

            # Fetch all data from the table
            cursor.execute(
                f'SELECT * FROM {config.db_name}.{config.schema_name}."{config.table_name}"')
            data = cursor.fetchall()

            # Get column names from the table
            cursor.execute(
                'SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s', (config.schema_name, config.table_name,))
            columns = [col[0] for col in cursor.fetchall()]

            df = pd.DataFrame(data, columns=columns)

            cursor.close()
            conn.close()
            updated_df = df.drop(columns= 'Student_ID')
            return updated_df
        
        except Exception as e:
            return ({f"Error fetching data from PostgreSQL: {str(e)}"},500)
        
    def generate_distribution_graph(self, csv_data):
        try:
            num_columns = len(csv_data.columns)
            num_rows = (num_columns + 4) // 5
            num_columns_last_row = num_columns % 5
            # Create a grid of subplots
            fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
            # Plot histograms for each column
            for i, col in enumerate(csv_data.columns):
                row_index = i // 5
                col_index = i % 5
                ax = axes[row_index, col_index]
                if i < num_columns:
                    ax.hist(csv_data[col], bins=10)
                    ax.set_title(col)
                else:
                    ax.axis('off')
            if num_columns_last_row > 0:
                for j in range(num_columns_last_row, 5):
                    axes[num_rows - 1, j].axis('off')
            plt.tight_layout()

        except Exception as e:
            return e
    
    def get(self):
        csv_df = self.fetch_csv_data()
        try:
            if csv_df is None:
                # 403 Forbidden
                return ({"error": "No data available"}, 403)
                # If no CSV data is available,  return a 403 Forbidden Error status code
            else:
                self.generate_distribution_graph(csv_df)
                png_path = save_image('distribution.png')
                if os.path.exists(png_path):
                    return render_template('distribution_graph.html',columns=list(csv_df.columns), fields = config.fields)
                else:
                    return ({"error": "Distribution graph not found"}, 404)
        except Exception as e:
            return ({"error": f"An error occurred: {str(e)}"})