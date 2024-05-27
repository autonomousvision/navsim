# Dataset splits vs. filtered training / test splits

The NAVSIM framework utilizes several dataset splits for standardized training and evaluating agents. 
All of them use the OpenScene dataset that is divided into the dataset splits `mini`,`trainval`,`test`,`private_test_e2e`, which can all be downloaded separately.

It is possible to run trainings and evaluations directly on these sets (see `Standard` in table below). 
Alternatively, you can run trainings and evaluations on training and validation splits that were filtered for challenging scenarios (see `NAVSIM` in table below), which is the recommended option for producing comparable and competitive results efficiently.
In contrast to the dataset splits which refer to a downloadable set of logs, the training / test splits are implemented as scene filters, which define how scenes are extracted from these logs.

The NAVSIM training / test splits subsample the OpenScene dataset splits.
Moreover, the NAVSIM splits include overlapping scenes, while the Standard splits are non-overlapping.
Specifically, `navtrain` is based on the `trainval` data and `navtest` on the `test` data.

As the `trainval` sensor data is very large, we provide a separate download link, which loads only the frames needed for `navtrain`. 
This eases access for users that only want to run the `navtrain` split and not the `trainval` split. If you already downloaded the full `trainval` sensor data, it is **not necessary** to download the `navtrain` frames as well.
The logs are always the complete dataset split.

## Overview
The Table belows offers an overview on the training and test splits supported by NAVSIM. 
In Navsim-v1.1, the training/test split can bet set with a single config parameter given in the table.

<table border="0">
    <tr>
        <th></th>
        <th>Name</th>
        <th>Description</th>
        <th>Logs</th>
        <th>Sensors</th>
        <th>Config parameters</th>
    </tr>
    <tr>
        <td rowspan="3">Standard</td>
        <td>trainval</td>
        <td>Large split for training and validating agents with regular driving recordings. Corresponds to nuPlan and downsampled to 2HZ.</td>
        <td>14GB</td>
        <td>&gt;2000GB</td>
        <td>
        train_test_split=trainval
        </td>
    </tr>
    <tr>
        <td>test</td>
        <td>Small split for testing agents with regular driving recordings. Corresponds to nuPlan and downsampled to 2HZ.</td>
        <td>1GB</td>
        <td>217GB</td>
        <td>
        train_test_split=test
        </td>
    </tr>
    <tr>
        <td>mini</td>
        <td>Demo split for with regular driving recordings. Corresponds to nuPlan and downsampled to 2HZ.</td>
        <td>1GB</td>
        <td>151GB</td>
        <td>
        train_test_split=mini
        </td>
    </tr>
    <tr>
        <td rowspan="2">NAVSIM</td>
        <td>navtrain</td>
        <td>Standard split for training agents in NAVSIM with non-trivial driving scenes. Sensors available separately in <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain.sh">download_navtrain.sh</a>.</td>
        <td>-</td>
        <td>445GB*</td>
        <td>
        train_test_split=navtrain
        </td>
    </tr>
    <tr>
        <td>navtest</td>
        <td>Standard split for testing agents in NAVSIM with non-trivial driving scenes. Available as a filter for test split.</td>
        <td>-</td>
        <td>-</td>
        <td>
        train_test_split=navtest
        </td>
    </tr>
    <tr>
        <td rowspan="2">Competition</td>
        <td>warmup_test_e2e</td>
        <td>Warmup test split to validate submission on hugging face. Available as a filter for mini split.</td>
        <td>-</td>
        <td>-</td>
        <td>
        train_test_split=warmup_test_e2e
        </td>
    </tr>
    <tr>
        <td>private_test_e2e</td>
        <td>Private test split for the challenge leaderboard on hugging face.</td>
        <td>&lt;1GB</td>
        <td>25GB</td>
        <td>
        train_test_split=private_test_e2e
        </td>
    </tr>
</table>

(*300GB without history)

## Splits

The standard splits `trainval`, `test`, and `mini` are from the OpenScene dataset. Note that the data corresponds to the nuPlan dataset with a lower frequency of 2Hz. You can download all standard splits over Hugging Face with the bash scripts in [download](../download)

NAVSIM provides a subset and filter of the `trainval` split, called `navtrain`. The `navtrain` split facilitates a standardized training scheme and requires significantly less sensor data storage than `travel` (445GB vs. 2100GB). If your agents don't need historical sensor inputs, you can download `navtrain` without history, which requires 300GB of storage. Note that `navtrain` can be downloaded separately via [download_navtrain.sh](https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain.sh) but still requires access to the `trainval` logs. Similarly, the `navtest` split enables a standardized set for testing agents with a provided scene filter. Both `navtrain` and `navtest` are filtered to increase interesting samples in the sets. 

For the challenge on Hugging Face, we provide the `warmup_test_e2e` and `private_test_e2e` for the warm-up and challenge track, respectively. Note that `private_test_e2e` requires you to download the data, while `warmup_test_e2e` is a scene filter for the `mini` split.
