# This interface will contain a generic class for the loss functions
# We instantiate a loss and perform online mining
# we will use pytorch_metric_learning library for this
# we can also develop our own loss function
# and online mining process and call then here

from typing import Optional, Callable, Tuple, Any
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)

class VPRLossFunction(torch.nn.Module):
    def __init__(
        self,
        loss_fn_name: str = "MultiSimilarityLoss",
        miner_name: str = "MultiSimilarityMiner",
    ):
        super().__init__()

        self.loss_fn = self._get_loss(loss_fn_name)
        self.miner = self._get_miner(miner_name)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # batch_accuracy corresponds to how many correct pairs/triplets we have in the batch
        # (e.g., how many non-informative sample we have in the batch)
        # we consider a sample non-informative if it has not been mined as an anchor
        # this is for debugging purposes only, this is not used in the loss function
        # nor is it used to compare the performance of different loss functions
        # this is purely for tracking if the model is learning anything
        batch_accuracy = 0.0
        if self.miner is not None:
            # indices is either a tuple of 3 tensors: (anchors, positives, negatives) when mining triplets
            # or a tuple of 4 tensors: (anchors, positives, anchors, negatives) when mining pairs
            indices = self.miner(embeddings, labels)
            loss_value = self.loss_fn(embeddings, labels, indices)

            # calculating batch accuracy
            nb_samples = embeddings.shape[0]  # number of samples in the batch
            unique_labels_mined = len(
                set(indices[0].detach().cpu().numpy())
            )  # number of unique labels mined
            batch_accuracy = 1 - (unique_labels_mined / nb_samples)  # batch accuracy
        else:
            loss_value = self.loss_fn(embeddings, labels)
        return loss_value, batch_accuracy




    def _get_loss(self, loss_name):
                
        if "SupCon" in loss_name:
            return losses.SupConLoss(temperature=0.07)

        if "CircleLoss" in loss_name:
            return losses.CircleLoss(
                m=0.4, gamma=80
            )  # these are params for image retrieval

        if "MultiSimilarityLoss" in loss_name:
            return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0)
            # return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0)
            # return losses.MultiSimilarityLoss()

        if "ContrastiveLoss" in loss_name:
            return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

        if "LiftedLoss" in loss_name:
            return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1)

        if "FastAPLoss" in loss_name:
            return losses.FastAPLoss(num_bins=30)

        if "NTXentLoss" in loss_name:
            return losses.NTXentLoss(
                temperature=0.07
            )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.

        if "TripletLoss" in loss_name:
            return losses.TripletMarginLoss(
                margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
            )  # or an int, for example 100

        if "CentroidTripletLoss" in loss_name:
            return losses.CentroidTripletLoss(
                margin=0.05,
                swap=False,
                smooth_loss=False,
                triplets_per_anchor="all",
            )

        # if you develop your own loss function, you can call it here
        # it's better to implement it in a separate file and import it here
        # for example UniLoss is implemented in src/losses/UniLoss.py
        # if loss_name == "UniLoss":
        #     return UniLoss()
        raise NotImplementedError(
            f"Sorry, <{loss_name}> loss function is not implemented!"
        )

    def _get_miner(self, miner_name, **kwargs):
        """
        Returns a miner object based on the name passed.
        This is used to do the online mining step.
        We mainly use pytorch_metric_learning library for this
        but you can implement your own miner and call it here

        Args:
            miner_name (str):
        """
        if miner_name is None:
            return None

        if "MultiSimilarityMiner" in miner_name:
            return miners.MultiSimilarityMiner(epsilon=0.1, **kwargs)

        if "TripletMarginMiner" in miner_name:  # all, hard, semihard, easy
            return miners.TripletMarginMiner(
                margin=0.1, type_of_triplets="semihard", **kwargs
            )

        if "PairMarginMiner" in miner_name:
            return miners.PairMarginMiner(
                pos_margin=0.2,
                neg_margin=0.8,
            )
        raise NotImplementedError(f"Sorry, <{miner_name}> miner  is not implemented!")
