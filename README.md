# 📝 proof-of-concept rail marking detections for autonomous train system #
***

This project implements rail-track detection using fast semantic segmentation for high-resolution images from [bisenetv2 algorithm](https://arxiv.org/abs/2004.02147).

The author of [bisenetv2](https://arxiv.org/abs/2004.02147) has not made the official implementation public so the implementation in this project might yeild different performance with the network introduced in the original paper.

This project trains [bisenetv2](https://arxiv.org/abs/2004.02147) on a modified version of [RailSem19 dataset](https://ieeexplore.ieee.org/document/9025646) with only three labels ("rail-raised", "rail-track", "background"). Please follow [here](https://wilddash.cc/railsem19) if you want to download the original dataset.

## :tada: TODO
***

- [x] Implement bisenetv2 and train on modified railsem19 dataset
- [ ] Refine the semantic mask using connected components algorithm
- [ ] Cluster the mask to obtain seperate railtrack

## 🎛  Dependencies
***

- create rail_marking conda environment

```bash
    conda env create --file environment.yml
```

- activate conda environment
```bash
    conda activate rail_marking
```

## 🔨 How to Build ##
***
YYYYnprocYYnprocYYY

## :running: How to Run ##
***

## :gem: References ##
***

- [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)
- [RailSem19: A Dataset for Semantic Rail Scene Understanding](https://openaccess.thecvf.com/content_CVPRW_2019/html/WAD/Zendel_RailSem19_A_Dataset_for_Semantic_Rail_Scene_Understanding_CVPRW_2019_paper.html)
