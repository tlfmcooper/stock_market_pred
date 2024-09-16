import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the datasets
reddit_news = pd.read_csv(
    r".\stocks\RedditNews.csv",
    parse_dates=True,
)
combined_news_djia = pd.read_csv(
    r".\stocks\Combined_News_DJIA.csv",
    parse_dates=True,
)
djia_table = pd.read_csv(
    r".\stocks\upload_DJIA_table.csv",
    parse_dates=True,
)

# Step 1: Preprocessing and cleaning
reddit_news["Date"] = pd.to_datetime(reddit_news["Date"])
combined_news_djia["Date"] = pd.to_datetime(combined_news_djia["Date"])
djia_table["Date"] = pd.to_datetime(djia_table["Date"])

# Step 2: Merge reddit_news with combined_news_djia on 'Date'
# Group reddit_news to concatenate the top 25 headlines for each date
reddit_grouped = (
    reddit_news.groupby("Date")["News"].apply(lambda x: " | ".join(x)).reset_index()
)
combined_grouped = combined_news_djia.assign(
    Combined_Headlines=lambda x: x.loc[:, "Top1":"Top25"].apply(
        lambda row: " | ".join(row.values.astype(str)), axis=1
    )
)

# Merge reddit and combined on date
merged_data = pd.merge(
    reddit_grouped,
    combined_grouped[["Date", "Label", "Combined_Headlines"]],
    on="Date",
    how="inner",
)

# Step 3: Merge the DJIA OHCV data
merged_data = pd.merge(
    merged_data,
    djia_table[["Date", "Open", "High", "Close", "Volume"]],
    on="Date",
    how="inner",
)

# Step 4: Create the instruction dataset in JSONL format
instruction_dataset = []
for index, row in merged_data.iterrows():
    instruction = {
        "instruction": "Predict the stock market movement based on the following news headlines.",
        "input": f"{row['News']} | {row['Combined_Headlines']} | Open: {row['Open']}, High: {row['High']}, Close: {row['Close']}, Volume: {row['Volume']}",
        "output": str(row["Label"]),
    }
    instruction_dataset.append(instruction)


def save_to_jsonl(file_path, dataset):  #
    with open(file_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")


def create_and_split_dataset(dataset=instruction_dataset):

    # Step 5: Split the dataset into training and testing sets
    train_set, test_set = train_test_split(
        instruction_dataset, test_size=0.2, random_state=42
    )

    # Save the train and test sets as JSONL
    save_to_jsonl(r"stocks\train_dataset.jsonl", train_set)
    save_to_jsonl(r"stocks\test_dataset.jsonl", test_set)

    print("Train and test datasets successfully created and saved.")


# Call the function to execute the dataset creation and splitting

if __name__ == "__main__":
    create_and_split_dataset()
