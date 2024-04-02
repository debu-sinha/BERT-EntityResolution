# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Author
# MAGIC
# MAGIC **Author**: Debu Sinha  
# MAGIC **Contact**: [debusinha2009@gmail.com](mailto:debusinha2009@gmail.com)
# MAGIC
# MAGIC ## Environment and Runtime Specifications
# MAGIC
# MAGIC - **Tested Runtime**: Databricks ML Runtime for GPU 13.3 LTS or above
# MAGIC - **Cloud Provider**: AWS
# MAGIC - **Cluster Node Type**: g5.8xlarge
# MAGIC     - **GPU Specs**: 1 A10 GPU
# MAGIC

# COMMAND ----------

# DBTITLE 1,Python Environment Refresh Script
# MAGIC %pip install --upgrade transformers mlflow datasets accelerate
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import torch
import gc
# Additionally, calling the garbage collector can help in freeing up memory
gc.collect()

# Set the current device to GPU and clear its memory
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Sample Data Generation

# COMMAND ----------

# DBTITLE 1,Entity Pair Augmentation Generator
import pandas as pd
import random

# Sample data
entities = [
    ("Google LLC", "Google", 1),
    ("Amazon.com, Inc.", "Amazon", 1),
    ("Facebook, Inc.", "Facebook Corporation", 1),
    ("Twitter, Inc.", "Twitter", 1),
    ("Microsoft Corporation", "Microsoft Corp.", 1),
    ("Google LLC", "Amazon", 0),
    ("Facebook, Inc.", "Twitter", 0),
    ("Microsoft Corporation", "Amazon", 0),
    ("Twitter, Inc.", "Google", 0),
    ("Amazon.com, Inc.", "Facebook Corporation", 0),
]

# Generate more data by duplicating and slightly modifying the existing pairs
additional_entities = [(a if random.random() > 0.5 else a[:-1], b if random.random() > 0.5 else b[:-1], label) for a, b, label in entities for _ in range(10)]
entities += additional_entities

# Shuffle the dataset
random.shuffle(entities)

# Split the dataset into training and validation
split_index = int(len(entities) * 0.8)
train_entities = entities[:split_index]
validation_entities = entities[split_index:]

# Convert to DataFrame
train_df = pd.DataFrame(train_entities, columns=["text_a", "text_b", "label"])
validation_df = pd.DataFrame(validation_entities, columns=["text_a", "text_b", "label"])

# Display the first few rows of the training dataset
print(train_df)

# COMMAND ----------

train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)

# COMMAND ----------

# DBTITLE 1,BERT Text Pair Classification
from datasets import load_dataset, DatasetDict

dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples['text_a'], examples['text_b'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load a pre-trained BERT model for sequence classification with 2 labels
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=6,
    weight_decay=0.01,
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# Train the model
trainer.train()

# COMMAND ----------

# DBTITLE 1,PyTorch Model Tracking Session
import mlflow
from mlflow import pytorch

# Start an MLflow run
with mlflow.start_run() as run:
    # Log parameters (optional)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("epochs", 3)
    
    # Log the PyTorch model to MLflow
    mlflow.pytorch.log_model(model, "bert_entity_resolution_model")
    
    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 1,Dummy DataFrame Generator
# Generating a dummy DataFrame with 50 rows for inference
data = pd.DataFrame({
    "description_1": ["Google LLC."] * 50,
    "description_2": ["Google ."] * 50
})

# COMMAND ----------

data.head()

# COMMAND ----------

# Convert the pandas DataFrame to a Spark DataFrame
sdf = spark.createDataFrame(data)

# Show the first few rows of the Spark DataFrame
sdf.show()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import torch
from transformers import AutoTokenizer
from mlflow.pytorch import load_model

# Global variables for caching the model and tokenizer
model = None
tokenizer = None
device = None

# Load the model and tokenizer once per executor
def load_model_and_tokenizer(model_uri, model_checkpoint):
    global model, tokenizer, device
    if model is None or tokenizer is None:
        model = load_model(model_uri)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # Move model to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

# COMMAND ----------

# DBTITLE 1,BERT Entity Resolution Loader
model_uri = f"runs:/{run_id}/bert_entity_resolution_model"
model_checkpoint = "bert-base-uncased"
load_model_and_tokenizer(model_uri, model_checkpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDF 

# COMMAND ----------

# DBTITLE 1,Entity Match Predictor UDF
@pandas_udf("string", PandasUDFType.SCALAR)
def predict_entity_match(description_1: pd.Series, description_2: pd.Series) -> pd.Series:
    global model, tokenizer, device
    
    predictions = []
    for desc1, desc2 in zip(description_1, description_2):
        inputs = tokenizer(desc1, desc2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(-1).item()
            predictions.append("Match" if prediction == 1 else "Not a Match")
    return pd.Series(predictions)


# COMMAND ----------

# DBTITLE 1,Entity Match Prediction Generator
from pyspark.sql.functions import col

# Assuming sdf is your Spark DataFrame
result_df = sdf.withColumn("Prediction", predict_entity_match(col("description_1"), col("description_2")))
result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDF iterator of series

# COMMAND ----------

from pyspark import Broadcast

# Broadcast the model and tokenizer
broadcasted_model: Broadcast = sc.broadcast(model)
broadcasted_tokenizer: Broadcast = sc.broadcast(tokenizer)

# COMMAND ----------

import pandas as pd
from typing import Iterator, Tuple
from pyspark.sql.functions import pandas_udf
import torch

@pandas_udf("string")
def predict_entity_match2(batch_iter: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    global device

    local_model = broadcasted_model.value
    local_tokenizer = broadcasted_tokenizer.value

    for description_1, description_2 in batch_iter:

        predictions = []
        for desc1, desc2 in zip(description_1, description_2):
            inputs = local_tokenizer(desc1, desc2, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = local_model(**inputs)
                prediction = outputs.logits.argmax(-1).item()
                predictions.append("Match" if prediction == 1 else "Not a Match")

        yield pd.Series(predictions)


# COMMAND ----------

# DBTITLE 1,Entity Match Predictor
from pyspark.sql.functions import struct

# Apply the pandas UDF to the DataFrame
result_df = sdf.withColumn("prediction", predict_entity_match2("description_1", "description_2"))

result_df.show()

# COMMAND ----------


