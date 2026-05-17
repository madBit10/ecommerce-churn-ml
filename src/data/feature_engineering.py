import pandas as pd 
from src.utils.logger import get_logger

logger = get_logger(__name__)

def feature_engineer(df: pd.DataFrame, churn_window_days: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Feature Engineering, developing the churn labels from the cleaned data

    Args:
        df: Data input
        churn_window_days: the churn window days which will decide the time frame after which the customers will churn

    Returns:
        tuple with dataframe and the series

    """

    # performing the churn window split

    snapshot_date = df['InvoiceDate'].max()
    churn_window_start = snapshot_date - pd.Timedelta(days=churn_window_days)
    features_df = df.loc[df['InvoiceDate'] < churn_window_start].copy()
    label_df = df.loc[df['InvoiceDate'] >= churn_window_start].copy()

    logger.info(f"Snapshot date: {snapshot_date}, churn window start: {churn_window_start}")
    logger.info(f"features_df: {features_df.shape}, label_df: {label_df.shape}")

    # building churn labels

    active_customers = set(label_df['Customer ID'].unique())
    all_customers = features_df['Customer ID'].unique()

    churn_labels = pd.DataFrame({
        'Customer ID': all_customers,
        'Churn': (~pd.Series(all_customers).isin(active_customers)).astype(int)
    })

    logger.info(f"Churn labels built are as follows:\n {churn_labels['Churn'].value_counts()}")

    # building the rfm features from the features_df

    features_df['Revenue'] = features_df['Quantity'] * features_df['Price']

    rfm = features_df.groupby('Customer ID').agg(
        Recency = ('InvoiceDate', 'max'),
        Frequency = ('Invoice', 'nunique'),
        Monetary = ('Revenue', 'sum')
    ).reset_index()

    rfm['Recency'] = (snapshot_date - rfm['Recency']).dt.days

    logger.info(f"The rfm shape is: {rfm.shape}")
    logger.info(f"The rfm data is: \n {rfm.head()}")

    # avg basket size

    rfm['avg_basket_size'] = rfm['Monetary'] / rfm['Frequency']
    logger.info(f"avg_basket_size — mean={rfm['avg_basket_size'].mean():.2f}, median={rfm['avg_basket_size'].median():.2f}")

    # is the customer uk based

    is_uk = features_df.groupby('Customer ID')['Country'].first().reset_index()
    is_uk['is_uk'] = (is_uk['Country'] == 'United Kingdom').astype(int)

    rfm = rfm.merge(is_uk[['Customer ID', 'is_uk']], on='Customer ID')
    rfm = rfm.merge(churn_labels, on='Customer ID')

    logger.info(f"Churn labels merged into the rfm df")
    logger.info(f"The rfm shape is: {rfm.shape}")
    logger.info(f"The rfm data is: \n {rfm.head()}")

    # split the dataset

    X = rfm.drop(columns=['Customer ID', 'Churn'])
    y = rfm['Churn']



    return X,y










