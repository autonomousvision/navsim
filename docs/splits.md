# Dataset splits vs. filtered training / test splits

The NAVSIM framework utilizes several dataset splits for standardized training and evaluating agents.
All of them use the OpenScene dataset that is divided into the dataset splits `mini`, `trainval`,` test`, which can all be downloaded separately.

It is possible to run trainings and evaluations directly on these sets (see `OpenScene` in table below).
Alternatively, you can run trainings and evaluations on training and validation splits that were filtered for challenging scenarios (see `NAVSIM` in table below), which is the recommended option for producing comparable and competitive results efficiently.
In contrast to the dataset splits which refer to a downloadable set of logs, the training / test splits are implemented as scene filters, which define how scenes are extracted from these logs.

The NAVSIM training / test splits subsample the OpenScene dataset splits.
Moreover, the NAVSIM splits include overlapping scenes, while the Standard splits are non-overlapping.
Specifically, `navtrain` is based on the `trainval` data and `navtest` and `navhard_two_stage` on the `test` data.

As the `trainval` sensor data is very large, we provide a separate download link, which loads only the frames needed for `navtrain`.
This eases access for users that only want to run the `navtrain` split and not the `trainval` split. If you already downloaded the full `trainval` sensor data, it is **not necessary** to download the `navtrain` frames as well.
The logs are always the complete dataset split.

## Overview

The Table belows offers an overview on the training and test splits supported by NAVSIM.

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
        <td rowspan="3">OpenScene</td>
        <td>trainval</td>
        <td>Large split for training and validating agents with regular driving recordings. Corresponds to nuPlan and downsampled to 2HZ.</td>
        <td>14GB</td>
        <td>>2000GB</td>
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
        <td rowspan="3">NAVSIM</td>
        <td>navtrain</td>
        <td>Standard split for training agents in NAVSIM with non-trivial driving scenes. Sensors available separately in <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain_aws.sh">download_navtrain_aws.sh</a> (AWS download) or <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain_hf.sh">download_navtrain_hf.sh</a> (HuggingFace download).</td>
        <td>14GB</td>
        <td>445GB*</td>
        <td>
        train_test_split=navtrain
        </td>
    </tr>
    <tr>
        <td>navtest</td>
        <td>Standard split for testing agents in NAVSIM with non-trivial driving scenes. Available as a filter for test split.</td>
        <td>983MB</td>
        <td>223GB</td>
        <td>
        train_test_split=navtest
        </td>
    </tr>
    <tr>
        <td>navhard_two_stage</td>
        <td>Standard split for testing agents in NAVSIM v2 with real and synthetic driving scenes. Synthetic frames downloadable via <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navhard_two_stage.sh">download_navhard_two_stage.sh</a>.</td>
        <td>892MB</td>
        <td>31GB</td>
        <td>
        train_test_split=navhard_two_stage
        </td>
    </tr>
    <tr>
        <td rowspan="2">Competition</td>
        <td>warmup_two_stage</td>
        <td>Warmup test split to validate submission on hugging face. Synthetic frames downloadable via <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_warmup_two_stage.sh">download_warmup_two_stage.sh</a>.</td>
        <td>27MB</td>
        <td>1.2G</td>
        <td>
        train_test_split=warmup_two_stage
        </td>
    </tr>
    <tr>
        <td>private_test_hard_two_stage</td>
        <td>Private test split for the challenge leaderboard on hugging face. Original and synthetic frames downloadable via <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_private_test_hard_two_stage.sh">download_private_test_hard_two_stage.sh</a></td>
        <td>14MB</td>
        <td>11GB</td>
        <td>
        train_test_split=private_test_hard_two_stage
        </td>
    </tr>
</table>

(*300GB without history)

## Splits

The standard splits `trainval`, `test`, and `mini` are from the OpenScene dataset. Note that the data corresponds to the nuPlan dataset with a lower frequency of 2Hz. You can download all standard splits over Hugging Face with the bash scripts in [download](../download)

NAVSIM provides a subset and filter of the `trainval` split, called `navtrain`. The `navtrain` split facilitates a standardized training scheme and requires significantly less sensor data storage than `travel` (445GB vs. 2100GB). If your agents don't need historical sensor inputs, you can download `navtrain` without history, which requires 300GB of storage. Note that the sensor data for `navtrain` can be downloaded separately via <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain_aws.sh">download_navtrain_aws.sh</a> or <a href="https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain_hf.sh">download_navtrain_hf.sh</a> but it still requires access to the `trainval` logs.

The `navtest` split enables a standardized set for testing agents in NAVSIM v1 with a provided scene filter. Similarly, the `navhard_two_stage` split split facilitates pseudo closed-loop simulation for evaluation in NAVSIM v2. `navtrain`, `navtest` and `navhard_two_stage` are filtered to increase interesting samples in the sets.

For the challenge on Hugging Face, we provide the `warmup_two_stage` and `private_test_e2e` for the warm-up and challenge track, respectively.

## Troubleshooting

As previous users reported missing files when downloading `navtrain`, we provide MD5 checksums for the `.tgz` files to identify corrupted downloads. We recommend to re-download `navtrain` without deleting the `.tgz` files (i.e. removing Line 12 and 22 in [download_navtrain_aws.sh](https://github.com/autonomousvision/navsim/blob/main/download/download_navtrain_aws.sh)) and running:

```bash
echo "6f92f38d5f03ed852da7872a7122bdd2  navtrain_current_1.tgz" | md5sum -c -
echo "7a72f0a758b5df6cbe4c565920a4869f  navtrain_current_2.tgz" | md5sum -c -
echo "b083fce1428308abb5682a1a150cc1af  navtrain_current_3.tgz" | md5sum -c -
echo "68354ac2c993aa1ebbfac59547fdb840  navtrain_current_4.tgz" | md5sum -c -
echo "dc46ed34d92d5ab9cc1464d67b72fbf6  navtrain_history_1.tgz" | md5sum -c -
echo "fab177bdb79c0c9536da1566d13e5995  navtrain_history_2.tgz" | md5sum -c -
echo "71ed9a2387edc3849921861d7873c7f0  navtrain_history_3.tgz" | md5sum -c -
echo "2cc13aced2f458e50fe4cc2f26d18e07  navtrain_history_4.tgz" | md5sum -c -
```

<details>
<summary>Expected output:</summary>

```bash
navtrain_current_1.tgz: OK
navtrain_current_2.tgz: OK
navtrain_current_3.tgz: OK
navtrain_current_4.tgz: OK
navtrain_history_1.tgz: OK
navtrain_history_2.tgz: OK
navtrain_history_3.tgz: OK
navtrain_history_4.tgz: OK
```

</details>
</br>
