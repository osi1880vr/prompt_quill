# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.


import re
import os, shutil, gc, time

import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

#####  This script will predict the aesthetic score for this image file:

img_path = "test.jpg"


models={}


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class aestetic_score:

    def __init__(self):
        self.models = {}

    def normalized(self, a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)


    def load_aestetics_prediction_model(self):
        if 'apm' not in self.models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

            s = torch.load("pq/generators/aesthetic/aesthetic-model.pth")   # load the model you trained previously or the model available in this repo

            model.load_state_dict(s)

            model.to(device)
            model.eval()
            self.models['apm'] = model
            del model
            self.models['clip'] , self.prediction = clip.load("ViT-L/14", device=device)


    def get_single_aestetics_score(self, pil_image):
        score = self.get_aestetics_score(pil_image)
        yield "{:.5f}".format(float(score[0][0]))
        self.models = {}
        gc.collect()


    def get_aestetics_score(self, pil_image):
        self.load_aestetics_prediction_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64

        image = self.prediction(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = self.models['clip'].encode_image(image)

        im_emb_arr = self.normalized(image_features.cpu().detach().numpy() )

        prediction = self.models['apm'](torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

        return prediction


    def run_aestetic_prediction(self, fileList, min_aestetics_level, aesthetics_keep_folder_structure, aestetics_output_folder):
        yield 'Aestetics calculation started'
        start_time = time.time()
        outfolder = os.path.join('api_out','scored')
        if len(fileList) > 0:
            for file in fileList:
                pil_image = Image.open(file)
                score = self.get_aestetics_score(pil_image)
                if score[0][0] > min_aestetics_level:

                    out_folder = os.path.join(outfolder, "{:.1f}".format(float(score[0][0])))
                    if aesthetics_keep_folder_structure:
                        out_folder = os.path.join(outfolder,aestetics_output_folder, "{:.1f}".format(float(score[0][0])))

                    os.makedirs(out_folder, exist_ok=True)
                    filename = os.path.basename(file)
                    dst = os.path.join(out_folder, "{:.5f}".format(float(score[0][0])) + '_' + filename)
                    shutil.copyfile(file, dst)
            end_time = time.time()
            elapsed_time = end_time - start_time
            yield f'Aestetics calculation finished, it took {elapsed_time} seconds'
        else:
            yield 'No Files found'
        self.models = {}
        gc.collect()


