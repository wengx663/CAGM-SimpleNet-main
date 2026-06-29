# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images

LOGGER = logging.getLogger(__name__)

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class ECABlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, hidden, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, channels, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.avg_pool(x).flatten(1)
        weights = self.fc(weights).view(x.shape[0], x.shape[1], 1, 1)
        return x * weights


class CBAMBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        padding = (kernel_size - 1) // 2
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, hidden, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = torch.nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3))
        max_values = torch.amax(x, dim=(2, 3))
        channel_weights = self.sigmoid(self.channel_mlp(avg) + self.channel_mlp(max_values))
        x = x * channel_weights.view(x.shape[0], x.shape[1], 1, 1)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        spatial_weights = self.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial_weights


class FeatureAttention(torch.nn.Module):
    def __init__(self, layer_names, feature_dimensions, attention_type="none", reduction=16, kernel_size=3):
        super().__init__()
        self.attention_type = attention_type.lower()
        self.layer_to_key = {}
        self.blocks = torch.nn.ModuleDict()

        if self.attention_type == "none":
            return

        for layer_name, channels in zip(layer_names, feature_dimensions):
            key = layer_name.replace(".", "_")
            self.layer_to_key[layer_name] = key
            if self.attention_type == "eca":
                self.blocks[key] = ECABlock(channels, kernel_size=kernel_size)
            elif self.attention_type == "se":
                self.blocks[key] = SEBlock(channels, reduction=reduction)
            elif self.attention_type == "cbam":
                self.blocks[key] = CBAMBlock(channels, reduction=reduction)
            else:
                raise ValueError(f"Unsupported feature attention: {attention_type}")

    @property
    def enabled(self):
        return len(self.blocks) > 0

    def forward(self, features, layer_names):
        if not self.enabled:
            return features

        enhanced_features = []
        for feature, layer_name in zip(features, layer_names):
            if feature.ndim == 4:
                enhanced_features.append(self.blocks[self.layer_to_key[layer_name]](feature))
            else:
                enhanced_features.append(feature)
        return enhanced_features


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        feature_attention="none",
        attention_reduction=16,
        attention_kernel_size=3,
        feature_l2_norm=False,
        score_top_k=0,
        global_branch=False,
        global_loss_weight=0.2,
        global_score_weight=0.3,
        global_noise_std=0,
        global_nn_branch=False,
        global_nn_weight=0.2,
        global_nn_k=5,
        resume=False,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, top_k=score_top_k, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.feature_attention = FeatureAttention(
            self.layers_to_extract_from,
            feature_dimensions,
            attention_type=feature_attention,
            reduction=attention_reduction,
            kernel_size=attention_kernel_size,
        )
        self.feature_attention.to(self.device)
        self.feature_attention_opt = None
        if self.feature_attention.enabled:
            self.feature_attention_opt = torch.optim.AdamW(
                self.feature_attention.parameters(), lr=lr * 0.1, weight_decay=1e-5
            )
        self.feature_l2_norm = feature_l2_norm
        self.resume = resume
        self.global_branch = global_branch
        self.global_loss_weight = global_loss_weight
        self.global_score_weight = global_score_weight
        self.global_noise_std = noise_std if global_noise_std <= 0 else global_noise_std
        self.global_nn_branch = global_nn_branch
        self.global_nn_weight = global_nn_weight
        self.global_nn_k = max(1, global_nn_k)
        self.global_memory = None
        self.global_nn_mean = torch.tensor(0.0, device=self.device)
        self.global_nn_std = torch.tensor(1.0, device=self.device)

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        scheduler_steps = max(1, (meta_epochs - aed_meta_epochs) * gan_epochs)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, scheduler_steps, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 
        self.global_discriminator = None
        self.global_dsc_opt = None
        self.global_dsc_schl = None
        if self.global_branch:
            self.global_discriminator = Discriminator(
                self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden
            )
            self.global_discriminator.to(self.device)
            self.global_dsc_opt = torch.optim.Adam(
                self.global_discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
            )
            self.global_dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.global_dsc_opt, scheduler_steps, self.dsc_lr*.4
            )

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _aggregate_embeddings(self, features):
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if self.feature_l2_norm:
            features = F.normalize(features, p=2, dim=1, eps=1e-12)
        return features

    def _make_global_features(self, features):
        pooled_features = []
        for feature in features:
            if feature.ndim == 4:
                pooled_features.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
            elif feature.ndim == 3:
                pooled_features.append(feature.mean(dim=1))
            else:
                pooled_features.append(feature.reshape(len(feature), -1))
        return self._aggregate_embeddings(pooled_features)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False, return_global=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        features = self.feature_attention(features, self.layers_to_extract_from)

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        global_features = self._make_global_features(features) if return_global else None

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self._aggregate_embeddings(features)

        if return_global:
            return features, patch_shapes, global_features
        return features, patch_shapes

    
    def test(self, training_data, test_data, save_segmentation_images):

        ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)


            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
                # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                            norm_segmentations)
        else:
            full_pixel_auroc = -1 
            pro = -1

        return auroc, full_pixel_auroc, pro

    def _make_fake_features(self, true_feats, noise_std):
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)
        noise = torch.stack([
            torch.normal(0, noise_std * 1.1**(k), true_feats.shape)
            for k in range(self.mix_noise)], dim=1).to(self.device)
        return true_feats + (noise * noise_one_hot.unsqueeze(-1)).sum(1)

    def _discriminator_loss(self, discriminator, true_feats, noise_std):
        fake_feats = self._make_fake_features(true_feats, noise_std)
        scores = discriminator(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(true_feats):]

        th = self.dsc_margin
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)
        return true_loss.mean() + fake_loss.mean(), p_true, p_fake

    def _global_knn_scores(self, query_features, memory_features=None, exclude_self=False):
        memory_features = self.global_memory if memory_features is None else memory_features
        if memory_features is None or len(memory_features) == 0:
            return torch.zeros(len(query_features), device=query_features.device)

        distances = torch.cdist(query_features, memory_features)
        if exclude_self and distances.shape[0] == distances.shape[1]:
            if distances.shape[0] == 1:
                return torch.zeros(1, device=query_features.device)
            distances.fill_diagonal_(float("inf"))

        available_neighbors = distances.shape[1] - (1 if exclude_self and distances.shape[0] == distances.shape[1] else 0)
        k = min(self.global_nn_k, max(1, available_neighbors))
        return distances.topk(k, largest=False, dim=1).values.mean(dim=1)

    def _fit_global_memory(self, training_data):
        if not self.global_nn_branch:
            return

        was_attention_training = self.feature_attention.training
        was_projection_training = self.pre_projection.training if self.pre_proj > 0 else False
        if self.feature_attention.enabled:
            self.feature_attention.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()

        memory_features = []
        with torch.no_grad():
            for data_item in tqdm.tqdm(
                training_data,
                desc="Fitting global NN memory...",
                leave=False,
            ):
                img = data_item["image"].to(torch.float).to(self.device)
                global_features = self._embed(
                    img, evaluation=True, return_global=True
                )[2]
                if self.pre_proj > 0:
                    global_features = self.pre_projection(global_features)
                global_features = F.normalize(global_features, p=2, dim=1, eps=1e-12)
                memory_features.append(global_features.detach())

        self.global_memory = torch.cat(memory_features, dim=0)
        train_scores = self._global_knn_scores(
            self.global_memory,
            self.global_memory,
            exclude_self=True,
        )
        finite_scores = train_scores[torch.isfinite(train_scores)]
        if len(finite_scores) > 1:
            self.global_nn_mean = finite_scores.mean()
            self.global_nn_std = finite_scores.std(unbiased=False).clamp_min(1e-6)
        else:
            self.global_nn_mean = torch.tensor(0.0, device=self.device)
            self.global_nn_std = torch.tensor(1.0, device=self.device)

        if self.feature_attention.enabled and was_attention_training:
            self.feature_attention.train()
        if self.pre_proj > 0 and was_projection_training:
            self.pre_projection.train()
        
    
    def train(self, training_data, test_data):

        
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path) and not self.resume:
            LOGGER.info(f"Existing checkpoint ignored because resume=False: {ckpt_path}")
        if os.path.exists(ckpt_path) and self.resume:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if self.feature_attention.enabled and "feature_attention" not in state_dict:
                LOGGER.warning(
                    "Checkpoint does not contain feature_attention weights; "
                    "ignoring it and retraining the enhanced model."
                )
                state_dict = {}
            elif self.global_branch and "global_discriminator" not in state_dict:
                LOGGER.warning(
                    "Checkpoint does not contain global_discriminator weights; "
                    "ignoring it and retraining the global-branch model."
                )
                state_dict = {}
            else:
                if 'discriminator' in state_dict:
                    self.discriminator.load_state_dict(state_dict['discriminator'])
                    if "pre_projection" in state_dict:
                        self.pre_projection.load_state_dict(state_dict["pre_projection"])
                    if "feature_attention" in state_dict and self.feature_attention.enabled:
                        self.feature_attention.load_state_dict(state_dict["feature_attention"])
                    if "global_discriminator" in state_dict and self.global_branch:
                        self.global_discriminator.load_state_dict(state_dict["global_discriminator"])
                else:
                    self.load_state_dict(state_dict, strict=False)

                self._fit_global_memory(training_data)
                self.predict(training_data, "train_")
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
                auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
                
                return auroc, full_pixel_auroc, anomaly_pixel_auroc

        def update_state_dict(d):
            
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})
            if self.feature_attention.enabled:
                state_dict["feature_attention"] = OrderedDict({
                    k:v.detach().cpu()
                    for k, v in self.feature_attention.state_dict().items()})
            if self.global_branch:
                state_dict["global_discriminator"] = OrderedDict({
                    k:v.detach().cpu()
                    for k, v in self.global_discriminator.state_dict().items()})

        best_record = None
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data)
            self._fit_global_memory(training_data)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro 
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
        
        torch.save(state_dict, ckpt_path)
        
        return best_record
            

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        if self.feature_attention.enabled:
            self.feature_attention.train()
        self.discriminator.train()
        if self.global_branch:
            self.global_discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                all_global_loss = []
                all_global_p_true = []
                all_global_p_fake = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.global_dsc_opt is not None:
                        self.global_dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    if self.feature_attention_opt is not None:
                        self.feature_attention_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    embed_result = self._embed(
                        img, evaluation=False, return_global=self.global_branch
                    )
                    true_feats = embed_result[0]
                    global_true_feats = embed_result[2] if self.global_branch else None

                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(true_feats)
                        if global_true_feats is not None:
                            global_true_feats = self.pre_projection(global_true_feats)

                    patch_loss, p_true, p_fake = self._discriminator_loss(
                        self.discriminator, true_feats, self.noise_std
                    )
                    loss = patch_loss
                    if self.global_branch:
                        global_loss, global_p_true, global_p_fake = self._discriminator_loss(
                            self.global_discriminator,
                            global_true_feats,
                            self.global_noise_std,
                        )
                        loss = loss + self.global_loss_weight * global_loss
                        self.logger.logger.add_scalar(
                            "global_loss", global_loss, self.logger.g_iter
                        )
                        self.logger.logger.add_scalar(
                            "global_p_true", global_p_true, self.logger.g_iter
                        )
                        self.logger.logger.add_scalar(
                            "global_p_fake", global_p_fake, self.logger.g_iter
                        )
                        all_global_loss.append(global_loss.detach().cpu().item())
                        all_global_p_true.append(global_p_true.cpu().item())
                        all_global_p_fake.append(global_p_fake.cpu().item())

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                    self.logger.logger.add_scalar("patch_loss", patch_loss, self.logger.g_iter)
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.feature_attention_opt is not None:
                        self.feature_attention_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()
                    if self.global_dsc_opt is not None:
                        self.global_dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.cos_lr:
                    self.dsc_schl.step()
                    if self.global_dsc_schl is not None:
                        self.global_dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_global_loss) > 0:
                    pbar_str += (
                        f" g_loss:{round(sum(all_global_loss) / len(input_data), 5)}"
                        f" g_true:{round(sum(all_global_p_true) / len(input_data), 3)}"
                        f" g_fake:{round(sum(all_global_p_fake) / len(input_data), 3)}"
                    )
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        if self.feature_attention.enabled:
            self.feature_attention.eval()
        self.discriminator.eval()
        if self.global_branch:
            self.global_discriminator.eval()
        with torch.no_grad():
            needs_global_features = self.global_branch or self.global_nn_branch
            embed_result = self._embed(
                images,
                provide_patch_shapes=True,
                evaluation=True,
                return_global=needs_global_features,
            )
            features, patch_shapes = embed_result[0], embed_result[1]
            global_features = embed_result[2] if needs_global_features else None
            if self.pre_proj > 0:
                features = self.pre_projection(features)
                if global_features is not None:
                    global_features = self.pre_projection(global_features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            if self.global_branch:
                global_scores = -self.global_discriminator(global_features)
                global_scores = global_scores.detach().cpu().numpy().reshape(-1)
                image_scores = image_scores + self.global_score_weight * global_scores
            if self.global_nn_branch and self.global_memory is not None:
                nn_features = F.normalize(global_features, p=2, dim=1, eps=1e-12)
                global_nn_scores = self._global_knn_scores(nn_features)
                global_nn_scores = (global_nn_scores - self.global_nn_mean) / self.global_nn_std
                global_nn_scores = global_nn_scores.detach().cpu().numpy().reshape(-1)
                image_scores = image_scores + self.global_nn_weight * global_nn_scores

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
