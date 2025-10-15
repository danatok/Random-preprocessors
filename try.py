# spark_preprocessor.py

from typing import List, Literal, Dict
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, FeatureHasher

EncodingStrategy = Literal["index_ohe", "hashing"]

DEFAULT_UNKNOWN_TOKEN = "__UNKNOWN__"

def _apply_unknown_guard(
    df: DataFrame,
    cat_cols: List[str],
    known_values: Dict[str, set] | None = None,
    unknown_token: str = DEFAULT_UNKNOWN_TOKEN,
) -> DataFrame:
    """
    Optionally map values not in known_values[col] to unknown_token.
    If known_values is None, just coalesce nulls to unknown_token.
    This keeps behavior explicit and lets us log unknown hits.
    """
    out = df
    for c in cat_cols:
        if known_values and c in known_values:
            out = out.withColumn(
                f"{c}__guarded",
                when(col(c).isin(list(known_values[c])), col(c)).otherwise(lit(unknown_token))
            )
        else:
            out = out.withColumn(
                f"{c}__guarded",
                when(col(c).isNull(), lit(unknown_token)).otherwise(col(c))
            )
    return out

def _build_index_ohe_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    """
    StringIndexer + OHE with handleInvalid='keep' so unseen → UNKNOWN bucket.
    """
    indexers = [
        StringIndexer(
            inputCol=f"{c}__guarded",       # guarded column
            outputCol=f"{c}__idx",
            handleInvalid="keep"            # unseen → last index
        )
        for c in cat_cols
    ]
    ohe = OneHotEncoder(
        inputCols=[f"{c}__idx" for c in cat_cols],
        outputCols=[f"{c}__ohe" for c in cat_cols],
        handleInvalid="keep"               # be extra-safe
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}__ohe" for c in cat_cols] + num_cols,
        outputCol="features"
    )
    return Pipeline(stages=indexers + [ohe, assembler])

def _build_hashing_pipeline(cat_cols: List[str], num_cols: List[str], num_features: int = 1<<18) -> Pipeline:
    """
    Hashing trick: no vocab maintenance, unseen are naturally handled.
    """
    # Use guarded columns to coalesce nulls → __UNKNOWN__
    hasher = FeatureHasher(
        inputCols=[f"{c}__guarded" for c in cat_cols] + num_cols,
        outputCol="features",
        numFeatures=num_features
    )
    return Pipeline(stages=[hasher])

def build_preprocess_pipeline(
    df_train: DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    encoding_strategy: EncodingStrategy = "index_ohe",
    known_values: Dict[str, set] | None = None,
) -> tuple[Pipeline, DataFrame, Dict[str, int]]:
    """
    Returns: (fitted_pipeline, transformed_train_df, unknown_counts)
    - unknown_counts: per-categorical-column count of values mapped to UNKNOWN on train (diagnostic)
    """
    # 1) Guard (explicit UNKNOWN mapping + metrics)
    guarded = _apply_unknown_guard(df_train, cat_cols, known_values)

    # 2) Unknown metrics on TRAIN (helps pick promotion thresholds)
    unknown_counts = {}
    for c in cat_cols:
        unknown_counts[c] = guarded.filter(col(f"{c}__guarded") == DEFAULT_UNKNOWN_TOKEN).count()

    # 3) Build & fit pipeline
    if encoding_strategy == "hashing":
        pl = _build_hashing_pipeline(cat_cols, num_cols)
    else:
        pl = _build_index_ohe_pipeline(cat_cols, num_cols)

    fitted = pl.fit(guarded)
    train_vec = fitted.transform(guarded)

    return fitted, train_vec, unknown_counts

def transform_with_pipeline(
    fitted_pipeline: Pipeline,
    df_infer: DataFrame,
    cat_cols: List[str],
    known_values: Dict[str, set] | None = None,
) -> tuple[DataFrame, Dict[str, int]]:
    """
    Applies the same UNKNOWN guard to inference data and transforms it.
    Returns transformed df plus metrics of unknown hits by column.
    """
    guarded = _apply_unknown_guard(df_infer, cat_cols, known_values)

    unknown_counts = {}
    for c in cat_cols:
        unknown_counts[c] = guarded.filter(col(f"{c}__guarded") == DEFAULT_UNKNOWN_TOKEN).count()

    out = fitted_pipeline.transform(guarded)
    return out, unknown_counts
