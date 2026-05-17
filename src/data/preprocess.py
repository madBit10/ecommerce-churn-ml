import pandas as pd 
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocess raw data for training

    Args: 
        df: Raw Dataframe

    Returns:
        dataframe 

    """

    # 1. drop duplicates

    df = df.drop_duplicates()
    logger.info(f"After dropping the duplicates: {df.shape}")

    # 2. Drop rows with NA Customer ID

    df = df.dropna(subset=["Customer ID"])
    logger.info(f"Dropped rows with missing Customer ID. Remaining: {len(df):,}")

    #3. Drop rows with NA Description

    df = df.dropna(subset=["Description"])
    logger.info(f"Dropped rows with missing Description.  Remaining: {len(df):,}")

    #4. Converting InvoiceData -> datetime

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    logger.info("InvoiceDate converted to datetime")

    #5. Converting Customer ID -> str

    df["Customer ID"] = df["Customer ID"].astype(str)
    logger.info("Customer ID converted to str (Categorical key)")

    #6. drop cancellations (Invoice starts with "C")

    before = len(df)
    df = df.loc[~df["Invoice"].str.startswith("C")]
    logger.info(f"Removed cancellations: {before - len(df):,} rows dropped, {len(df):,} remaining")

    # 7. Drop rows where Quantity <= 0 or Price <=0

    before = len(df)
    df = df.loc[(df["Quantity"] > 0) & (df["Price"] > 0)]
    logger.info(f"Dropped: {before - len(df):,} rows with Quantity ≤ 0 or Price ≤ 0, remaining: {len(df):,} rows" )

    #8. Stripping the white spaces
    str_cols = ["Description", "Country", "StockCode"]
    for col in str_cols:
        df[col] = df[col].str.strip()

    logger.info(f"Stripped whitespace from: {str_cols}")

    return df