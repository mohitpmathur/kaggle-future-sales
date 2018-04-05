'''
Kaggle competition: Future Sales
'''

import pandas as pd

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
#items['item_unique_shop'] = items['item_unique_shop'].fillna(-999)

# find number of unique items sold by shops each month
train_shop_unique_items_month = train.groupby(['date_block_num', 'shop_id'])[
    'item_id'].nunique().reset_index(name='unique_item_per_month')

# find median price of item sold at a shop each month
# groupby(month,shop,item)[price].median()
