import pandas as pd
import os
import pandas as pd

# class DataIngestionClass:
class DataIngestionClass:
    def __init__(self, source_path: str, directory: str, filename: str):
        self.source_path = source_path
        self.destination_path = destination_path
        self.directory = directory
        self.filename = filename

    def read_csv(self):
        # Read the CSV file and return the DataFrame
        df = pd.read_csv(self.source_path)
        print(df)
        return df

    def Save_File(self, df):
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"Directory '{self.directory}' was created.")
        else:
            print(f"Directory '{self.directory}' already exists.")

        # Corrected to use 'self.directory' for the file path construction
        file_path = os.path.join(self.directory, self.filename)
        
        # Save the DataFrame to the file
        df.to_csv(file_path, index=False)  # index=False to avoid writing row indices
        print(f"File has been saved to {file_path}")

# This block will only execute if this script is run directly
if __name__ == "__main__":
    source_path = 'C:\\Users\\SHIVRAJ SHINDE\\JupiterWorking\\XL_ML\\Z_DataSets\\01_AirlineData\\Airline.csv'
    destination_path = "Data\\01_RawData\\Airline.csv"
    directory = "Data\\01_RawData\\"
    filename = "Airline.csv"

    # Pass all required parameters to the constructor
    file_handler = DataIngestionClass(source_path, directory, filename)

    df = file_handler.read_csv()  # Read the CSV file
    file_handler.Save_File(df)  # Save the DataFrame to the destination
