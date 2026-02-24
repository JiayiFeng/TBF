from mmq.data.causal_lm_data_loader import get_batch_dataset_v2 as get_batch_dataset
from mmq.utils.byte_tensor import tensor_to_object
from mmq.utils.arguments import get_args
from mmq.data.causal_lm_data_loader import get_train_data_loader
from mmq.data.causal_lm_data import CausalMicroBatchSplitterByHint, CausalCollatorV2
from mmq.data.var_len_batch import VarLenBatch
import torch
import dataclasses

from torch.utils.data import DataLoader

data_args = get_args().data.train

def dataloader_start_at_batch_id(batch_id: int) -> DataLoader:
    dataset = get_batch_dataset(data_args, start_batch_id=batch_id)
    return DataLoader(dataset, collate_fn=lambda x: tensor_to_object(x[0]), shuffle=False, num_workers=0)

splitter = CausalMicroBatchSplitterByHint()
collator = CausalCollatorV2(
    splitter,
    need_text=True,
    need_label=True,
    varlen_batch_concat=data_args.varlen_batch_concat,
)

def batch_data_to_dict(batch) -> list[dict[str, torch.Tensor]]:
    res = []
    for micro_batch in batch:
        dd = {}
        for key, value in micro_batch.items():
            value: VarLenBatch
            sub_dict = dataclasses.asdict(value)
            if "max_size" in sub_dict and sub_dict["max_size"] is not None:
                sub_dict["max_size"] = torch.tensor(sub_dict["max_size"])
            if "varlen" in sub_dict and sub_dict["varlen"] is not None:
                sub_dict["varlen"] = torch.tensor(sub_dict["varlen"])
            for k in list(sub_dict.keys()):
                sub_dict[f"{key}#{k}"] = sub_dict[k]
                del sub_dict[k]
            dd.update(sub_dict)
        res.append(dd)
    return res



def to_records(global_batch: object) -> list[dict[str, torch.Tensor]]:
    batch = collator(global_batch)
    return batch_data_to_dict(batch)
    

