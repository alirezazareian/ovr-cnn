## Webcam and Jupyter notebook demo

This folder contains several Jupyter notebooks that create offline video demos for our open-vocabulary object detector. It also contains the original webcam demo of the `maskrcnn_benchmark` repository, which we haven't tested on our models yet. Note the notebooks should be run with a kernel that has this repository and its requirements installed (from [here](../INSTALL.md)). 

Each notebook loads a list of video files and feeds into two models (ours vs. baseline), visualizes the results of both and shows side-by-side in an output video. Here is a list of the files and what each does:

* [`demo-01.ipynb`](demo-01.ipynb): our model trained on 48 COCO classes, tested on 65 classes (48 seen + 17 unseen), compared to a regular Faster R-CNN trained and tested on the 48 seen classes.
* [`demo-02.ipynb`](demo-02.ipynb): our model trained on 48 COCO classes, tested on all 80 COCO classes, compared to a regular Faster R-CNN trained and tested on the 48 seen classes.
* [`demo-03.ipynb`](demo-03.ipynb): our model trained on 48 COCO classes, tested on 1200 frequent words from COCO captions, compared to a regular Faster R-CNN trained and tested on the 48 seen classes.
* [`demo-04.ipynb`](demo-04.ipynb): our model trained on 48 COCO classes, tested on 600 Open Images classes, compared to a regular Faster R-CNN trained and tested on the 48 seen classes.
* [`demo-05.ipynb`](demo-05.ipynb): our model trained on all 80 COCO classes, tested on 1200 frequent words from COCO captions, compared to a regular Faster R-CNN trained and tested on 80 COCO classes.
* [`demo-06.ipynb`](demo-06.ipynb): our model trained on all 80 COCO classes, tested on 600 Open Images classes, compared to a regular Faster R-CNN trained and tested on 80 COCO classes.
* [`demo-07.ipynb`](demo-07.ipynb): same as `demo-01.ipynb` but newer version that can generate GIF, and solo demo with emphasized unseen labels.
* [`demo-08.ipynb`](demo-08.ipynb): same as `demo-03.ipynb` but newer version that can generate GIF, and solo demo with emphasized unseen labels.




