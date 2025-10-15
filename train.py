# example_factorization_machine.py

from pyspark.sql import SparkSession
from pyspark.ml.regression import FMRegressor
from spark_preprocessor import build_preprocess_pipeline, transform_with_pipeline

spark = SparkSession.builder.getOrCreate()

# Example schema
cat_cols = ["banner_id", "device_type", "region"]
num_cols = ["age", "tenure_days"]

# Load your train/test (replace with your sources)
train_df = spark.read.parquet("path/to/train")
test_df  = spark.read.parquet("path/to/test")

# Optionally compute known vocab (per column) from train_df
# known_values = {c: set(x[c] for x in train_df.select(c).distinct().collect()) for c in cat_cols}
known_values = None

# 1) Build + fit preprocessing (index+OHE with UNKNOWN)
pp_model, train_vec, train_unknowns = build_preprocess_pipeline(
    df_train=train_df,
    cat_cols=cat_cols,
    num_cols=num_cols,
    encoding_strategy="index_ohe",  # or "hashing"
    known_values=known_values
)
print("Train unknown counts:", train_unknowns)

# 2) Train FM
fm = FMRegressor(
    featuresCol="features",
    labelCol="clicked",
    stepSize=0.1,
    factorSize=20
)
fm_model = fm.fit(train_vec)

# 3) Safe inference
test_vec, test_unknowns = transform_with_pipeline(
    fitted_pipeline=pp_model,
    df_infer=test_df,
    cat_cols=cat_cols,
    known_values=known_values
)
print("Test unknown counts:", test_unknowns)

preds = fm_model.transform(test_vec)
preds.select("banner_id", "prediction").show(10, truncate=False)

# 4) (Optional) Save artifacts
pp_model.write().overwrite().save("artifacts/pipeline_model")
fm_model.write().overwrite().save("artifacts/fm_model")
