from typing import List, Optional
import os
import shutil
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

class SparkFeaturePreprocessorML:
    """
    Spark-native preprocessor that uses StringIndexer + StandardScaler in a Pipeline.
    * Categorical columns -> <col>_idx via StringIndexer.
      - extendable_cols use handleInvalid='keep'  (unseen -> unknown bucket)
      - strict cols use handleInvalid='error'     (unseen -> error)
    * Numerical columns -> scaled to <col>_scaled via StandardScaler
    """
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        extendable_cols: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        artifact_dir: str = "/tmp/spark_feature_preprocessor_pipeline"
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.extendable_cols = extendable_cols or []
        if not set(self.extendable_cols).issubset(set(self.categorical_cols)):
            raise ValueError("extendable_cols must be a subset of categorical_cols")
        self.with_mean = with_mean
        self.with_std = with_std
        self.pipeline_model: Optional[PipelineModel] = None
        self.artifact_dir = artifact_dir

        # names produced by the pipeline
        self.cat_idx_cols = [f"{c}_idx" for c in self.categorical_cols]
        self.num_vec_col = "__num_vec__"
        self.num_scaled_vec_col = "__num_scaled_vec__"

    def _build_pipeline(self, df):
        # 1) StringIndexers (label encoders)
        indexers = []
        for c in self.categorical_cols:
            handle = "keep" if c in self.extendable_cols else "error"
            indexers.append(
                StringIndexer(
                    inputCol=c,
                    outputCol=f"{c}_idx",
                    handleInvalid=handle
                )
            )

        stages = indexers[:]

        # 2) StandardScaler for numeric columns
        if self.numerical_cols:
            vec_assembler = VectorAssembler(
                inputCols=self.numerical_cols,
                outputCol=self.num_vec_col
            )
            scaler = StandardScaler(
                inputCol=self.num_vec_col,
                outputCol=self.num_scaled_vec_col,
                withMean=self.with_mean,
                withStd=self.with_std
            )
            stages.extend([vec_assembler, scaler])

        return Pipeline(stages=stages)

    def fit(self, df):
        """
        Fit StringIndexers and StandardScaler using Spark estimators.
        """
        pipe = self._build_pipeline(df)
        self.pipeline_model = pipe.fit(df)
        return self

    def transform(self, df):
        """
        Apply the fitted PipelineModel; split the scaled vector back into <col>_scaled.
        Keeps original raw columns by default; you can .select() afterward if needed.
        """
        if self.pipeline_model is None:
            raise RuntimeError("Call fit() before transform().")

        out = self.pipeline_model.transform(df)

        # Split numeric scaled vector back to individual columns
        if self.numerical_cols:
            arr = F.vector_to_array(F.col(self.num_scaled_vec_col))
            for i, c in enumerate(self.numerical_cols):
                out = out.withColumn(f"{c}_scaled", arr[i])

        return out

    def save(self, dir_path: Optional[str] = None):
        """
        Save the fitted PipelineModel to disk (Spark-native).
        """
        if self.pipeline_model is None:
            raise RuntimeError("Nothing to save; fit() first.")
        target = dir_path or self.artifact_dir
        if os.path.exists(target):
            shutil.rmtree(target)
        self.pipeline_model.save(target)
        return target

    @classmethod
    def load(cls, dir_path, numerical_cols, categorical_cols, extendable_cols=None,
             with_mean=True, with_std=True):
        """
        Re-create wrapper and load PipelineModel from disk.
        """
        inst = cls(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            extendable_cols=extendable_cols,
            with_mean=with_mean,
            with_std=with_std,
            artifact_dir=dir_path
        )
        inst.pipeline_model = PipelineModel.load(dir_path)
        return inst

#####

# --- Configuration ---
NUMERICAL_COLS = ['customer_age', 'avg_spend']
CATEGORICAL_COLS = ['banner_id', 'customer_country']
EXTENDABLE_COLS = ['banner_id']  # extendable → keep; strict → error
PIPELINE_ARTIFACT_DIR = "/tmp/preprocessor_pipeline_dir"  # a folder

# 1) Fit on training data
preproc = SparkFeaturePreprocessorML(
    numerical_cols=NUMERICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
    extendable_cols=EXTENDABLE_COLS,
    with_mean=True,   # mean=0
    with_std=True     # std=1
).fit(train_df)

transformed_train_df = preproc.transform(train_df)
print("\nTransformed training data (Spark transformers):")
transformed_train_df.select(
    *CATEGORICAL_COLS,
    *[c+"_idx" for c in CATEGORICAL_COLS],
    *NUMERICAL_COLS,
    *[c+"_scaled" for c in NUMERICAL_COLS]
).show(truncate=False)

# 2) Save via your (mock) MLflow: log the directory
with mlflow.start_run() as run:
    saved_dir = preproc.save(PIPELINE_ARTIFACT_DIR)
    # In your mock mlflow, we log the folder by zipping or copying; here we just copy a marker file
    # Better: tar/zip the directory. For the mock, copy a marker for simplicity:
    marker = os.path.join(saved_dir, "_SPARK_PIPELINE_OK_")
    open(marker, "w").close()
    mlflow.log_artifact(marker, artifact_path="preprocessor_pipeline_marker")

# 3) Inference: load and transform
print("\n=== Inference with strict checking (customer_country is strict) ===")
loaded_preproc = SparkFeaturePreprocessorML.load(
    dir_path=PIPELINE_ARTIFACT_DIR,
    numerical_cols=NUMERICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
    extendable_cols=EXTENDABLE_COLS
)

try:
    transformed_inf = loaded_preproc.transform(inference_df)  # 'GER' will ERROR (strict)
    transformed_inf.show()
except Exception as e:
    print(f"\nSUCCESSFULLY caught expected error from StringIndexer (strict col): {e}")

# 4) Partial retraining case (new banner allowed)
print("\n=== Partial retrain (new banner_id allowed via handleInvalid='keep') ===")
transformed_partial = loaded_preproc.transform(partial_retrain_df)
transformed_partial.select(
    *CATEGORICAL_COLS,
    *[c+"_idx" for c in CATEGORICAL_COLS],
    *NUMERICAL_COLS,
    *[c+"_scaled" for c in NUMERICAL_COLS]
).show(truncate=False)
