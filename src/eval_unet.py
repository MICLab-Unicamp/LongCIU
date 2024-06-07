'''
Evaluate simple UNet baseline 
'''
import os
import torch
import numpy as np
import SimpleITK as sitk
import json
from pytorch_lightning import LightningDataModule, LightningModule
from monai.metrics.meandice import DiceMetric
from train_unet import LongCIUNet
from longciu import LongCIUDataModule, DataMaskSlicer
from typing import List, Callable, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt


def int_to_onehot(ints: torch.Tensor, C: int):
    '''
    ints: Tensor only with integer values. Can be of float dtype.
    C: number of channels (classes) that output should have.
    '''
    try:
        B, Y, X = ints.shape
    except Exception as e:
        raise Exception(f"int_to_onehot input not of expect shape, should be [B, Y, X] without a channel dimension: {e}")
    assert ints.max() < C, f"ints has more classes than {C}"

    onehot = torch.zeros((B, C, Y, X), dtype=ints.dtype, device=ints.device)

    for c in range(C):
        onehot[:, c] = ints == c

    onehot = onehot.float()

    assert onehot.sum(dim=1).min() == 1, f"Onehot not generated correctly. Is your {ints.shape}/{ints.dtype} only composed of integer values?"

    return onehot


def eval_model_in_data_with_metrics(model: LightningModule, 
                                    data: LightningDataModule, 
                                    metrics: List[Callable],
                                    num_classes: int,
                                    debug: bool = False) -> Dict[str, float]:
    '''
    model: LightningModule that returns one-hot probabilities
    data: LightningDataModule abstracting the data
    metrics: List of Callable metrics, should return a metric number 
    num_classes: number of segmentation classes in the problem, example: 3 for BG, GGO and consolidation
    '''
    if len(metrics) > 1:
        raise NotImplementedError("Currently assuming metric is only DiceMetric")
    
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()
    
    dataloaders = [("val", val_dataloader),
                   ("test", test_dataloader)]
    evaluation_results = {"val_per_item": [], "test_per_item": [], "val_IDs": [], "test_IDs": []}
    for mode, dataloader in dataloaders:
        print(f"Starting {mode} evaluation...")
        for i, batch in enumerate(dataloader):
            # y are saved as integer labels, convert to one hot
            x, y, m = batch
            
            plt.subplot(1, 3, 1)
            plt.imshow(x.squeeze(1).cpu().numpy()[0], cmap="gray")
            plt.title("x")
            plt.subplot(1, 3, 2)
            plt.imshow(int_to_onehot(y.squeeze(1), 3).squeeze().cpu().numpy().transpose(1, 2, 0))
            plt.title("y")

            x = x.cuda()
            y = y.cuda()
            idx = m["idx"][0]
            evaluation_results[f"{mode}_IDs"].append(m["ID"][0])
            
            # model output is logits, softmax, argmax and convert to one hot for metrics
            if isinstance(model, LightningModule):
                y_hat: torch.Tensor = model(x)                        
                y_hat = int_to_onehot(y_hat.softmax(dim=1).argmax(dim=1), C=num_classes)
            else:
                # Armengue
                _, y_hat, m_hat = model[mode][i]
                y_hat = y_hat - 1
                y_hat[y_hat < 0] = 0
                y_hat = torch.from_numpy(y_hat).cuda()
                y_hat = int_to_onehot(y_hat, C=num_classes)
                print(m_hat["idx"])
                assert m["idx"][0] == m_hat["idx"], f"idxs are not paired between target and prediction dataset {m} vs {m_hat}"
            
            plt.subplot(1, 3, 3)
            plt.imshow(y_hat.squeeze().cpu().numpy().transpose(1, 2, 0))
            plt.title("y_hat")
            
            for metric in metrics:
                result = metric(y_hat, y).squeeze().tolist()
                evaluation_results[f"{mode}_per_item"].append(result)
                plt.suptitle(str(result))
                if debug:
                    plt.show()
                else:
                    plt.close("all")
                print(result)
        
        evaluation_results[f"{mode}_mean"] = torch.tensor(evaluation_results[f"{mode}_per_item"]).mean(dim=0).tolist()
        evaluation_results[f"{mode}_std"] = torch.tensor(evaluation_results[f"{mode}_per_item"]).std(dim=0).tolist()

    print(evaluation_results)
    return evaluation_results


def medpseg_output_wrapping(mode):
    '''
    Instead of reproducing MEDPseg, just get the output we already have
    '''
    with open(os.path.join("../data", "longciu_splits.json"), 'r') as splits_file:
        splits = json.load(splits_file)

    data_path = os.path.join("../data", "longciu_img.nii.gz")
    mask_path = os.path.join("../data", "longciu_medpseg_output.nii.gz")
    data, mask = sitk.GetArrayFromImage(sitk.ReadImage(data_path)), sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    
    # You can customize this preprocessing if you want!
    MIN, MAX = -1024, 600
    data = np.clip(data, MIN, MAX)
    data = (data - MIN)/(MAX - MIN)
    data, mask = data.astype(np.float32), mask.astype(np.float32)

    return DataMaskSlicer(f"{mode}_medpseg_prediction", data=data, mask=mask, idxs=splits[mode], transform=None)



if __name__ == "__main__":
    from train_unet import TrainTransform
    NUM_CLASSES = 3

    metric = DiceMetric(num_classes=NUM_CLASSES, reduction="mean", ignore_empty=False)
    model = LongCIUNet.load_from_checkpoint(os.path.join("logs", "baseline_unet_epoch=4932-val_loss=0.48.ckpt"))
    
    # For proper metric statistics, keeping original eval_batch_size
    model.hparams.mutated_eval_batch_size = 1 

    data = LongCIUDataModule(data_dir=model.hparams.data_dir,
                             num_workers=model.hparams.num_workers,
                             train_batch_size=model.hparams.train_batch_size,
                             eval_batch_size=model.hparams.mutated_eval_batch_size,
                             train_transform=TrainTransform(),
                             eval_transform=None)
    data.prepare_data()
    data.setup(None)

    with open("eval_results/trained_unet_results.json", 'w') as trained_unet_results:
        json.dump(eval_model_in_data_with_metrics(model, data, [metric], num_classes=NUM_CLASSES, debug=False), trained_unet_results)


    metric = DiceMetric(num_classes=NUM_CLASSES, reduction="mean", ignore_empty=False)
    model = {m: medpseg_output_wrapping(m) for m in ["val", "test"]}

    with open("eval_results/blind_medpseg_results.json", 'w') as blind_medpseg_results:
        json.dump(eval_model_in_data_with_metrics(model, data, [metric], num_classes=NUM_CLASSES, debug=False), blind_medpseg_results)