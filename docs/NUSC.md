## Getting Started with CenterPoint on nuScenes
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Prepare data

#### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuScenes # rename to nuScenes
```
Remember to change the DATA_ROOT to the actual path in your system. 


#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

### Train & Evaluate in Command Line

**Now we only support training and evaluation with gpu. Cpu only mode is not supported.**

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH

```
**Training one GPU**
```bash
python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py /home/rares/repos/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py
```

For distributed testing with 4 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```

**class-agnostic CenterPoint - 1 GPU**
```bash
python ./tools/dist_test.py configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py --work_dir work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms/ --checkpoint work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms/latest.pth --speed_test

```

The pretrained models and configurations are in [MODEL ZOO](../configs/nusc/README.md).

### Tracking

You can find the detection files are in the [MODEL ZOO](../configs/nusc/README.md). After downloading the detection files, you can simply run 

```bash 
# val set 
python tools/nusc_tracking/pub_test.py --work_dir WORK_DIR_PATH  --checkpoint DETECTION_PATH  

# test set 
python tools/nusc_tracking/pub_test.py --work_dir WORK_DIR_PATH  --checkpoint DETECTION_PATH  --version v1.0-test  --root data/nuScenes/v1.0-test    
```

### Test Set 

Organize your dataset as follows 

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
                     └── v1.0-test <-- main test folder 
                            ├── samples       <-- key frames
                            ├── sweeps        <-- frames without annotation
                            ├── maps          <-- unused
                            |── v1.0-test <-- metadata and annotations
                            |── infos_test_10sweeps_withvelo.pkl <-- test info
```

Download the ```centerpoint_voxel_1440_flip``` [here](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/EhgzjwV2EghOnHFKyRgSadoBr2kUo7yPu52N-I3dG3c5dA?e=EP9G6L), save it into ```work_dirs/nusc_0075_flip```, then run the following commands in the main folder to get detection prediction 

```bash
python tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_flip.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_0075_flip/voxelnet_converted.pth  --testset 
```


Debugging:
1. Open your project in Visual Studio Code.
2. Click on the Debugging icon in the View Bar on the side of the editor, or use the Ctrl + Shift + D keyboard shortcut to open the Debugging panel.
3. In the Debugging panel, click on the gear icon to create a new launch configuration.
4. In the launch configuration, select Python as the configuration type, then select Python File as the configuration option.
5. Add the following lines to the launch configuration:
    "program": "${file}",
    "args": ["-m", "torch.distributed.launch", "--nproc_per_node=1", "./tools/train.py", "/home/rares/repos/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py"],
    "pythonPath": "${config:python.pythonPath}"

6. In the terminal, navigate to the folder containing your Python file and run the following command to start the debugger:
    python -m ptvsd --host localhost --port 5678 --wait -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py /home/rares/repos/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py

7. In Visual Studio Code, press the F5 key or click on the green play button in the Debugging panel to start the debugger.
