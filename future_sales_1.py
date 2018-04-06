'''
Kaggle competition: Future Sales
'''

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def fit_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    print("RMSE:", np.sqrt(mean_squared_error(
        y_val.clip(0., 20.), y_pred.clip(0., 20.))))
    return model


print("Reading data ...")
train = pd.read_csv('data/sales_train_v2.csv')
test = pd.read_csv('data/test.csv')
items = pd.read_csv('data/items.csv')
item_cats = pd.read_csv('data/item_categories.csv')
shops = pd.read_csv('data/shops.csv')

print("\nCreating item features ...")
items['item_len'] = items['item_name'].map(len)
items['item_wc'] = items['item_name'].map(lambda x: len(x.split()))

print("\nCreating item category features ...")
item_cats['item_cat_len'] = item_cats['item_category_name'].map(len)
item_cats['item_cat_wc'] = item_cats['item_category_name'].map(
    lambda x: len(x.split()))

print("\nMerging item and item_category dataframes ...")
items = items.merge(item_cats, how='left', on='item_category_id')

print("\nCreating shop features ...")
shops['shop_len'] = shops['shop_name'].map(len)
shops['shop_wc'] = shops['shop_name'].map(lambda x: len(x.split()))

'''
In train, create features
shop_item_frac - percentage of total items carried by each shop
item_shop_frac - percentage of total shops an item is sold in
'''

# Number of unique items sold by each shop
print("\nCreating feature - number of unique items sold by each shop")
train_shop_unique_item = train.groupby(
    ['shop_id'])['item_id'].nunique().reset_index(name='shop_unique_item')

# Number of shops an item is sold in
print("\nCreating feature - number of unique shops a product is sold in")
train_item_unique_shop = train.groupby(
    ['item_id'])['shop_id'].nunique().reset_index(name='item_unique_shop')

shops = shops.merge(train_shop_unique_item, how='left', on='shop_id')
items = items.merge(train_item_unique_shop, how='left', on='item_id')

# fill missing values in items with -999
items['item_unique_shop'] = items['item_unique_shop'].fillna(-999)

# find number of unique items sold by shops each month
train_shop_unique_items_month = train.groupby(['date_block_num', 'shop_id'])[
    'item_id'].nunique().reset_index(name='unique_item_per_month')

# find median price of item sold at a shop each month
# groupby(month,shop,item)[price].median()

train2 = train.groupby(['date_block_num', 'shop_id', 'item_id'])[
    'item_cnt_day'].sum().reset_index()
train_med_price = train.groupby(['date_block_num', 'shop_id', 'item_id'])[
    'item_price'].median().reset_index()
train2 = train2.merge(train_med_price, how='left', on=[
                      'date_block_num', 'shop_id', 'item_id'])
train2.rename(columns={'item_price': 'item_median_price'},  inplace=True)

train2 = train2.merge(shops, how='left', on='shop_id')
train2.drop(['shop_name'], axis=1, inplace=True)

items_cat_cols = items.select_dtypes(exclude=['object']).columns.tolist()
train2 = train2.merge(items[items_cat_cols], how='left', on='item_id')
train2.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)


train_df = train2[train2['date_block_num'] < 33]
val_df = train2[train2['date_block_num'] == 33]
print("train2 shape:", train2.shape)
print("train_df shape:", train_df.shape)
print("val_df shape:", val_df.shape)

y_train = train_df['item_cnt_month']
y_val = val_df['item_cnt_month']

train_df.drop(['item_cnt_month'], axis=1, inplace=True)
val_df.drop(['item_cnt_month'], axis=1, inplace=True)


tscv = TimeSeriesSplit(n_splits=5)
gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, verbose=1)

params = {'learning_rate': [0.05],
          'n_estimators': [100]}
gbr = GradientBoostingRegressor(verbose=1)
model = GridSearchCV(gbr, param_grid=params, cv=tscv, n_jobs=-1)
model = fit_model(model, train_df, y_train, val_df, y_val)
