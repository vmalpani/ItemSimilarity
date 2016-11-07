# ItemSimilarity
Given item description, find other similar items using tf-idf vectorization.
We use pysparnn for fast approximate nearest neighbor lookup in sparse data.

Uncompress the data files:
```
unzip data.zip
```
Run the script with one of the four categories (cases, cell_phones, laptops, mp3_players):
```
python src/collection.py -c mp3_players
```
