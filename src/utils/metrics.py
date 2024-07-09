import numpy as np
import faiss
import faiss.contrib.torch_utils

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def compute_recalls(
                descriptors,
                num_references,
                num_queries,
                ground_truth,
                k_values=[1, 5, 10],
    ):
    """
    Compute recall@K scores for a given dataset and descriptors using FAISS.

    Parameters
    ----------
    descriptors : numpy.ndarray
        descriptors of both reference and query images. Shape is (num_images, embedding_dim).
        Note that the first `num_references` descriptors are reference images (num_images = num_references + num_queries).
    num_references : int
        Number of reference images.
    num_queries : int
        Number of query images.
    ground_truth : list of lists
        Ground truth labels for each query image. Each list contains the indices of relevant reference images.
    k_values : list, optional
        List of 'K' values for which recall@K scores will be computed, by default [1, 5, 10].

    Returns
    -------
    dict
        A dictionary mapping each 'K' value to its corresponding recall@K score.
    """

    assert num_references + num_queries == len(
        descriptors
    ), "Number of references and queries do not match the number of descriptors. THERE IS A BUG!"

    embed_size = descriptors.shape[1]
    faiss_index = faiss.IndexFlatL2(embed_size)
    # faiss_index = faiss.IndexFlatIP(embed_size)
    
    # add references
    faiss_index.add(descriptors[:num_references])

    # search for queries in the index
    _, predictions = faiss_index.search(descriptors[num_references:], max(k_values))

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}
    return d


# a function that prints the recall@k for each dataset
def print_recalls(recalls_list, val_set_names, title="Recall@k Performance"):
    if len(recalls_list) == 0:
        return

    console = Console()
    console.print('\n') # for better formatting in console
    table = Table(title=None, box=box.SIMPLE, header_style="bold")

    k_values = list(
        recalls_list[0].keys()
    )  # get the k values from first dict (they're the same for all datasets)
    table.add_column("Dataset", justify="left", style="")
    for k in k_values:
        table.add_column(f"R@{str(k)}", justify="center")

    for i, recalls in enumerate(recalls_list):
        table.add_row(val_set_names[i], *[f"{100*v:.2f}" for v in recalls.values()])

    console.print(Panel(table, expand=False, title=title))