import numpy as np
import torch
import lightning as L
from src.utils import metrics
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import v2 as T2


class VPRFramework(L.LightningModule):
    def __init__(
        self,
        backbone,
        aggregator,
        loss_function,
        lr=1e-4,
        optimizer="adamw",
        weight_decay=1e-3,
        warmup_steps=1500,
        milestones=[5, 10, 15],
        lr_mult=0.25,
        verbose=True,
    ):
        """
        Initializes the VPRFramework class.

        Args:
            backbone: The backbone model.
            aggregator: The aggregator model.
            loss_function: The loss function.
            lr (float, optional): The learning rate. Defaults to 1e-4.
            optimizer (str, optional): The optimizer algorithm. Defaults to "adamw".
            weight_decay (float, optional): The weight decay. Defaults to 1e-3.
            warmup_steps (int, optional): The number of warmup steps. Defaults to 1500.
            milestones (list, optional): The milestones for learning rate scheduling. Defaults to [5, 10, 15].
            lr_mult (float, optional): The learning rate multiplier. Defaults to 0.25.
            verbose (bool, optional): Whether to print verbose information. Defaults to True.
        """
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.verbose = verbose
        
        # save the hyperparameters except the classes
        self.save_hyperparameters(ignore=["loss_function", "backbone", "aggregator", "verbose"])
        
    def forward(self, x):
        """
        Forward pass through the backbone then the aggregator.

        Args:
            x: Input tensor.

        Returns:
            Tensor (or list of tensors) after passing through the backbone and aggregator.
        """
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns:
            List of optimizers and schedulers that will be used by the Lightning trainer.
        """
        optimizer_params = [
            {"params": self.backbone.parameters(), "lr": self.lr * 0.1, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(optimizer_params)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Define how a single optimization step is executed.

        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            optimizer: Optimizer instance.
            optimizer_closure: Closure for the optimizer.
        """
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg["initial_lr"]

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    @torch.compiler.disable() # do not run the compiler on this function
    def compute_loss(self, descriptors, labels):
        """
        Compute the loss.

        Args:
            descriptors: Descriptor tensors.
            labels: Corresponding labels.

        Returns:
            Loss value and batch accuracy.
        """
        # NOTE: in this framework, the loss also returns a batch_accuracy value 
        # which represents the fraction of valid positve pairs in the batch (after mining)
        # this is useful for debugging and monitoring the training process
        # but it is not used in the loss computation nor for comparing models.
        loss, batch_accuracy = self.loss_function(descriptors, labels)
        return loss, batch_accuracy
    
    
    
    def on_train_start(self):
        """
        Actions to perform at the start of training.
        """
        # you can do something here before the training starts
        pass
    
    ########################################################
    ################ Training loop starts here #############
    ########################################################
    def on_train_epoch_start(self):
        """
        Actions to perform at the start of each training epoch.
        """
        pass
    
    # This is the main training loop
    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Loss value for the batch.
        """
        images, labels = batch
        P, K, c, h, w = images.shape # P: number of places, K: number of views
        images = images.view(P * K, c, h, w) # so B = P * K 
        labels = labels.view(-1)
        
        model_output = self(images)
        
        # sometimes the model returns a list, sometimes a single tensor
        # for example, BoQ returns (descriptors, attentions)
        # but netvlad, mixvpr and many others return only descriptors
        # so we check if the model output is a list or a single tensor
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            descriptors = model_output[0]
        else:
            descriptors = model_output
        
        loss, batch_accuracy = self.compute_loss(descriptors, labels)

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """
        Actions to perform at the end of each training epoch.
        """
        pass
    
    ########################################################
    ################ Validation loop starts here ###########
    ########################################################
    def on_validation_epoch_start(self):
        """
        Actions to perform at the start of each validation epoch.
        """
        # we init an empty dictionary to store the descriptors for each dataloader
        self.validation_step_outputs = {}

    # At each iteration, we compute the output descriptors
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for each batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Index of the dataloader.

        Returns:
            None
        """
        images, labels = batch
        model_output = self(images)
        
        # sometimes the model returns a list, sometimes a single tensor
        # for example, BoQ returns [descriptors, attentions]
        # but netvlad, mixvpr and many others return only descriptors
        # so we check if the model output is a list or a single tensor
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            descriptors = model_output[0]
        else:
            descriptors = model_output
            
        descriptors = descriptors.detach().cpu().numpy()

        if dataloader_idx not in self.validation_step_outputs:
            # initialize the list of descriptors for this dataloader
            self.validation_step_outputs[dataloader_idx] = []
        # save the descriptors to compute the recall@k at the end of the validation epoch
        self.validation_step_outputs[dataloader_idx].append(descriptors)

    # At the end of the validation epoch, we compute the recall@k
    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of each validation epoch.
        """
        dm = self.trainer.datamodule
        list_of_recalls = [] # one list for each validation set
        for dataloader_idx, descriptors_list in self.validation_step_outputs.items():
            descriptors = np.concatenate(descriptors_list, axis=0)
            dataset = dm.val_datasets[dataloader_idx]

            if self.trainer.fast_dev_run:
                # skip the recall computation for fast dev runs
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
            else:
                # we will use the descriptors, the number of references, number of queries, and the ground truth
                # NOTE: make sure these are available in the dataset object and ARE IN THE RIGHT ORDER.
                # meaning that the first `num_references` descriptors are reference images and the rest are query images
                recalls_dict = metrics.compute_recalls(
                        descriptors, 
                        dataset.num_references,
                        dataset.num_queries,
                        dataset.ground_truth,
                        k_values=[1, 5, 10, 15]
                )
                recalls_log = {
                    f"{dm.val_set_names[dataloader_idx]}/R1": recalls_dict[1],
                    f"{dm.val_set_names[dataloader_idx]}/R5": recalls_dict[5],
                }
                self.log_dict(recalls_log, prog_bar=False, logger=True)
                list_of_recalls.append(recalls_dict)

        if self.verbose:
            metrics.print_recalls(list_of_recalls, dm.val_set_names)
        self.validation_step_outputs.clear()