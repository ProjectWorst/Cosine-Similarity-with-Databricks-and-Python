import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
from delta.tables import *
from pyspark.sql.functions import *
from pyspark.sql.types import StructType
import string
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the path to the stopwords file
stop_words_path = '/filelocation/stopwords.txt'
with open(stop_words_path, 'r') as file:
    stopwords_list = file.read().splitlines()

# Drop existing table if it exists
%sql
DROP TABLE IF EXISTS data_base_location.table_name

# Database and sheet configuration
database_name = 'data_base_name'
sheets = ['sheet1', 'sheet2', 'sheet3', 'sheet4', 'sheet5', 'sheet6', 'sheet7', 'sheet8', 'sheet9', 'sheet10', 'sheet11', 'sheet12']

# Preprocessing function to clean and filter text
def preprocess_text(text):
    """Lowercase the text, remove leading/trailing spaces, and filter out stopwords."""
    text = text.translate(str.maketrans('', '', string.punctuation)
    text = text.strip().lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_list]
    return ' '.join(tokens)

# Load and preprocess data from each sheet
data_frames = {}
for sheet in sheets:
    # Assuming data_frames is a dictionary containing DataFrames for each sheet
    data_frames[sheet] = pd.read_excel(f"{sheet}.xlsx")  # Example of loading data
    data_frames[sheet]['processed_column_header'] = data_frames[sheet]['column_header'].apply(preprocess_text)

# Initialize and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
all_text = []
for df in data_frames.values():
    all_text.extend(df['processed_column_header'].tolist())
tfidf_vectorizer.fit(all_text)

# Transform text data into TF-IDF matrices
tfidf_matrices = {sheet: tfidf_vectorizer.transform(data_frames[sheet]['processed_column_header']) for sheet in sheets}

# Mapping sheet names to custom labels
sheet_change = {'sheet1': 'DATA1', 'sheet2': 'DATA2', 'sheet3': 'DATA3', 'sheet4': 'DATA4', 'sheet5': 'DATA5', 'sheet6': 'DATA6', 'sheet7': 'DATA7', 'sheet8': 'DATA8', 'sheet9': 'DATA9', 'sheet10': 'DATA10', 'sheet11': 'DATA11', 'sheet12': 'DATA12'}

# Calculate cosine similarity and format the data
formatted_data = []
for idx1, row1 in data_frames[sheets[0]].iterrows():
    first_sheet_text = row1['column_header']
    first_tfidf_matrix = tfidf_matrices[sheets[0]][idx1]
    
    for sheet in sheets[1:]:
        for idx2, row2 in data_frames[sheet].iterrows():
            other_sheet_text = row2['column_header']
            other_tfidf_matrix = tfidf_matrices[sheet][idx2]
            
            similarity = cosine_similarity(first_tfidf_matrix, other_tfidf_matrix.reshape(1, -1))[0][0]
            formatted_data.append({
                'Column1': similarity,
                'Column2': idx1,
                'Column3': 'custom',
                'column_header': first_sheet_text,
            })     

            formatted_data.append({
                'Column1': similarity,
                'Column2': idx2,
                'Column3': sheet_change[sheet],
                'column_header': other_sheet_text,
            })  

# Enable Delta Lake preview feature
spark.sql("SET spark.databricks.delta.preview.enabled = true")

# Convert formatted data to Spark DataFrame and sort by similarity score
formatted_df = pd.DataFrame(formatted_data)
formatted_spark_df = spark.createDataFrame(formatted_df)
sorted_formatted_spark_df = formatted_spark_df.orderBy("Column1", col("Column3").asc())

# Save the results to a Delta table
full_similarity_table_name = f'data_base_location.table_name'
sorted_formatted_spark_df.write.format('delta').mode('overwrite').saveAsTable(full_similarity_table_name)

# Print the schema and show a sample of the stored data
print(f'Similarity scores for {full_similarity_table_name} have been calculated and stored in the database.')
print(formatted_spark_df.printSchema())
ver_df = spark.table(full_similarity_table_name)
ver_df.show(5)
