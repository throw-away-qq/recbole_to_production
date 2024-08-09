# RecBole: Implementing Sequential Models as API

## Introduction
Recently, I've been constantly working with RecBole (a tool that allows you to try many recommendation models) as mentioned in this article. It's fun to discover various things by applying different models to different data. Now, while RecBole is primarily designed for experimentation, it doesn't have extensive documentation for production use. 
This guide demonstrates how to adapt RecBole's Sequential Models for production use through API implementation. While RecBole excels in experimental settings, it lacks comprehensive documentation for production deployment. We bridge this gap by offering insights and code examples for implementing Sequential Models like SHAN or SINE as production-ready APIs.

## Background

Sequential Models in RecBole, such as SHAN or SINE, fundamentally require only item history for predictions, not user IDs. However, RecBole's built-in functions are designed with experimentation in mind, introducing unnecessary complexity for API implementation in production environments.

## Implementation Overview

This approach bypasses RecBole's full_sort_topk function, instead directly utilizing the model's full_sort_predict method. This allows us to make predictions using only item history, enhancing flexibility and efficiency in API contexts.

## Key Steps for Production Implementation
1. **Model Loading**: Efficiently load the trained model without unnecessary data dependencies.
2. **Data Preparation**: Prepare input data in the format required by the model.
3. **Prediction**: Use the model to generate predictions based on item history.
4. **Result Processing**: Convert internal IDs to external IDs for meaningful output.




## Prerequisite

Let's say you've trained a Sequential Model using RecBole and want to use it. Models like SHAN or SINE.
Considering the principle of these models, they don't need user IDs; they only require item history. However, the `full_sort_topk` function provided by RecBole, which outputs top-k items, requires `uid_series` as an argument, making it behave as if it needs user IDs.

```python
def full_sort_topk(uid_series, model, test_data, k, device=None):
    """Calculate the top-k items' scores and ids for each user in uid_series.
    Note:
        The score of [pad] and history items will be set into -inf.
    Args:
        uid_series (numpy.ndarray): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        k (int): The top-k items.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.
    Returns:
        tuple:
            - topk_scores (torch.Tensor): The scores of topk items.
            - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
    """
    scores = full_sort_scores(uid_series, model, test_data, device)
    return torch.topk(scores, k)
```

The reason for this is simply to maintain a consistent interface with other models like General Recommender that require IDs. More precisely, it's probably intended for use only when you want to see the top-k for each user...
If we look at the model implementation code for SINE, we can see that only item_sequence is passed to the forward function.

```python
def full_sort_predict(self, interaction):
    item_seq = interaction[self.ITEM_SEQ]
    item_seq_len = interaction[self.ITEM_SEQ_LEN]
    seq_output = self.forward(item_seq, item_seq_len)
    test_items_emb = self.item_embedding.weight
    scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
    return scores
```
Let's Implement Sequential Model as an API
Sequential Models can make predictions even for unknown users (user IDs not seen during training) as long as there's some behavioral history. So, providing this model as an API could have some practical value.
Therefore, let's consider code that returns predictions by passing only item history instead of user_id, without using the full_sort_topk function provided by RecBole.

### Populating Data into Interaction
Looking at SINE.full_sort_predict, we see that the input is a class called Interaction. This is a data storage class used within RecBole. It can be initialized as follows:
```python
input_interaction = Interaction(
    {
        "variable": ["example"],
    }
)
```
It seems we can pass the necessary data to model.full_sort_predict by filling this Interaction.

SINE requires item_list and item_length, so preparing these would look like this:

```python
item_sequence = ["item1", "item2", "item3"]
item_length = len(item_sequence)
pad_length = 50  # pre-defined by recbole

padded_item_sequence = torch.nn.functional.pad(
    torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
    (0, pad_length - item_length),
    "constant",
    0,
)

input_interaction = Interaction(
    {
        "item_list": padded_item_sequence.reshape(1, -1),
        "item_length": torch.tensor([item_length]),
    }
)
```

We're zero-padding to make it the same length as the maximum item history length of 50 used when training SINE.
The dataset here comes with the RecBole model when restored and has various useful methods. Here, it's converting item strings to internal_id (int) used in the SINE model according to the item_field.

### Loading the Model
The only model loading function provided by RecBole is load_data_and_model, which also restores the dataset used during training with the same settings. This process prepares the aforementioned dataset.
However, load_data_and_model throws a ValueError: Neither [dataset/your_dataset] exists in the devicenor [your_dataset] a known dataset name. error and stops if the raw_data used during training isn't placed in exactly the same location as during training...

It's too painful to deploy data to the same instance as the API just to run the API, so let's work around this.
Looking into the contents of load_data_and_model:
```python
def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
```

What we need here is just the model file, and looking at the last three lines or so, it seems that dataset is only required in model = get_model(config['model'])(config, train_data.dataset).to(config['device']), and it doesn't seem that the training data itself is necessary to load the weights.

Looking further, we see that when loading SINE, dataset is only used to pass user_num and item_num, so it doesn't seem that the training data itself is necessary here either.

```python
import cloudpickle
from recbole.data import create_dataset
import torch

checkpoint = torch.load(model_file_path)
config = checkpoint["config"]
dataset = create_dataset(config)
cloudpickle.dump(dataset, open("output/dataset.pkl", "wb"))
```

So, for now, I decided to create a dataset from the config like this, dump it to pkl, and place it somewhere accessible during API execution.
The dataset created from the config doesn't hold all interactions, but only basic information like the correspondence table between external id and internal id for users and items, user_num, item_num, etc., so it's relatively lightweight and should not be a problem.
This script itself needs to be executed where the training data is located.

Considering all of this, the slimmed-down model loading function looks like this:

```python
def load_model(model_file: str, dataset_file: str) -> Tuple[{Your Model Class}, SequentialDataset]:
    with open(dataset_file, "rb") as f:
        dataset = cloudpickle.load(f)

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    # Set model to evaluation mode
    model.eval()
    return model, dataset
```

With this, as long as we have the checkpoint file of the trained model and dataset.pkl, we can restore the model data necessary for prediction.

### Implementing the Entire API
Now we just need to use this to do the following:

1. Load the model
2. Receive item sequence and topk from the request as model input
3. Pack the data from step 2 into the Interaction class
4. Pass it to model.full_sort_predict to get scores for all items
5. Argsort according to topk and get the top k items
6. Convert from internal id to external id and return the prediction as actual item IDs
  
 Here's an implementation example using FastAPI. The final overall picture looks like this:
```python
from typing import List, Tuple

import numpy as np
import torch
from fastapi.applications import FastAPI
from pydantic import BaseModel
from recbole.data import create_dataset
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender.sine import SINE
from recbole.utils import get_model, init_seed

app = FastAPI(docs_url=None, redoc_url=None)


def load_model(model_file: str) -> Tuple[SINE, SequentialDataset]:
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    model = get_model(config["model"])(config, dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return model, dataset


model, dataset = load_model(
    model_file="saved/{your_model_checkpoint}.pth"
    dataset_file="outout/dataset.pkl"
)


class ItemHistory(BaseModel):
    sequence: List[str]
    topk: int


class RecommendedItems(BaseModel):
    score_list: List[float]
    item_list: List[str]


@app.get("/hello")
def health_check() -> str:
    """
    Health check endpoint
    """

    return "Hello Sequential Recommendation api"


@app.post("/v1/sine/user_to_item", response_model=RecommendedItems)
def sine_user_to_item(item_history: ItemHistory):
    item_history_dict = item_history.dict()
    item_sequence = item_history_dict["sequence"]
    item_length = len(item_sequence)
    pad_length = 50  # pre-defined by recbole

    padded_item_sequence = torch.nn.functional.pad(
        torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
        (0, pad_length - item_length),
        "constant",
        0,
    )

    input_interaction = Interaction(
        {
            "item_list": padded_item_sequence.reshape(1, -1),
            "item_length": torch.tensor([item_length]),
        }
    )

    # Use torch.no_grad() for inference
    with torch.no_grad():
        scores = model.full_sort_predict(input_interaction.to(model.device))
        scores = scores.view(-1, dataset.item_num)
        scores[:, 0] = -np.inf  # Set pad item score to -inf
        topk_score, topk_iid_list = torch.topk(scores, item_history.topk)

    predicted_score_list = topk_score.tolist()[0]
    predicted_item_list = dataset.id2token(
        dataset.iid_field, topk_iid_list.tolist()
    ).tolist()

    recommended_items = {
        "score_list": predicted_score_list,
        "item_list": predicted_item_list,
    }
    return recommended_items

```

While RecBole's main purpose is for experimentation, looking into its internals reveals that it's neatly organized and what it's doing isn't all that complex. So, if you look at the contents, it's pretty easy to understand how to do what you want to do.
So, this was content that required looking at and tweaking the internal implementation when trying to do something that wasn't quite worth submitting a PR to the main project.

I hope this is helpful to someone.

## Few Considerations/Best Practises
1. Model Evaluation Mode: Always set the model to evaluation mode (model.eval()) before inference.
2. Gradient Computation: Disable gradient computation during inference using torch.no_grad() to optimize performance and resource usage.
3. Dataset Preservation: Save crucial parameters using cloudpickle during training and place them in the relevant location while serving.
4. Error Handling: Implement robust error handling and logging for production scenarios.


## Contributing
Contributions to improve this blog or extend it are welcome. Please feel free to submit issues or pull requests.
