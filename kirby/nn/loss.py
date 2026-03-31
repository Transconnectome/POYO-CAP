import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchmetrics import R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from kirby.taxonomy import OutputType
import numpy as np


class FFTLoss(nn.Module):
    """
    Frequency Domain (FFT) Loss.
    
    Computes the loss in the frequency domain between the predicted and target images.
    It encourages the model to match the high-frequency details of the target image.
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean'):
        """
        Args:
            loss_type (str): The type of loss to use ('l1' or 'l2'). Default is 'l1'.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(FFTLoss, self).__init__()

        if loss_type.lower() == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type.lower() == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1' or 'l2'.")
            
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): The predicted image tensor, shape [B, C, H, W].
            true (torch.Tensor): The ground truth image tensor, shape [B, C, H, W].

        Returns:
            torch.Tensor: The calculated FFT loss as a scalar tensor.
        """
        # --- Defensive checks ---
        if pred.shape != true.shape:
            raise ValueError(f"Input shapes must be the same. Got {pred.shape} and {true.shape}")
        
        # --- Compute FFT ---
        # Apply N-dimensional FFT. Using `fftn` is general for 2D/3D.
        # The result is a complex tensor.
        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        true_fft = torch.fft.fftn(true, dim=(-2, -1))
        
        # --- Compute loss on the magnitude of the spectrum ---
        # The phase information is usually discarded. We compute loss on the absolute values (magnitudes).
        pred_fft_mag = torch.abs(pred_fft)
        true_fft_mag = torch.abs(true_fft)

        # Calculate the loss between the frequency spectrums
        loss = self.loss_fn(pred_fft_mag, true_fft_mag, reduction=self.reduction)
        
        return loss

class ImprovedFFTLoss(nn.Module):
    """
    FFT Loss with high-frequency emphasis
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean',
                 high_freq_weight: float = 2.0):
        """
        Args:
            loss_type (str): 'l1' or 'l2'
            reduction (str): 'none' | 'mean' | 'sum'
            high_freq_weight (float): high-frequency emphasis weight (1.0 = uniform, 2.0 = 2x emphasis)
        """
        super().__init__()
        
        if loss_type.lower() == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type.lower() == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        self.reduction = reduction
        self.high_freq_weight = high_freq_weight
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")
        
        # FFT
        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        true_fft = torch.fft.fftn(true, dim=(-2, -1))
        
        pred_fft_mag = torch.abs(pred_fft)
        true_fft_mag = torch.abs(true_fft)
        
        # generate high-frequency weight map
        h, w = pred.shape[-2], pred.shape[-1]

        # generate frequency coordinates (distance from center)
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=pred.device).float() - cy
        x = torch.arange(w, device=pred.device).float() - cx

        # 2D grid
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # distance from center (normalized)
        freq_distance = torch.sqrt(yy**2 + xx**2)
        freq_distance = freq_distance / freq_distance.max()

        # weight map: center (low freq) = 1.0, edge (high freq) = high_freq_weight
        # Linear interpolation: 1.0 → high_freq_weight
        weight_map = 1.0 + (self.high_freq_weight - 1.0) * freq_distance

        # reshape: [1, 1, H, W] → broadcast
        weight_map = weight_map.view(1, 1, h, w)

        # apply weights
        weighted_pred = pred_fft_mag * weight_map
        weighted_true = true_fft_mag * weight_map

        # compute loss
        loss = self.loss_fn(weighted_pred, weighted_true, reduction=self.reduction)
        
        return loss

# class GradientDifferenceLoss(nn.Module):
#     """
#     Gradient Difference Loss (GDL)
    
#     Computes the L1 loss between the gradients of the predicted and target images.
#     The gradients are computed using Sobel filters.
#     """
#     def __init__(self, channels: int, loss_type: str = 'l1'):
#         """
#         Args:
#             channels (int): The number of channels in the input images (e.g., 3 for RGB).
#             loss_type (str): The type of loss to use ('l1' or 'l2'). Default is 'l1'.
#         """
#         super(GradientDifferenceLoss, self).__init__()
#         self.channels = channels
        
#         # Define Sobel filters for x and y gradients
#         kernel_x = torch.from_numpy(np.array([
#             [-1, 0, 1],
#             [-2, 0, 2],
#             [-1, 0, 1]
#         ])).float().unsqueeze(0).unsqueeze(0)

#         kernel_y = torch.from_numpy(np.array([
#             [-1, -2, -1],
#             [0, 0, 0],
#             [1, 2, 1]
#         ])).float().unsqueeze(0).unsqueeze(0)

#         # Expand kernels to match the number of input channels
#         kernel_x = kernel_x.repeat(self.channels, 1, 1, 1)
#         kernel_y = kernel_y.repeat(self.channels, 1, 1, 1)

#         # Create convolutional layers for applying the Sobel filters
#         # `groups=channels` applies the filter independently to each channel
#         self.conv_x = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
#                                 kernel_size=3, stride=1, padding=1, bias=False, 
#                                 groups=self.channels)
        
#         self.conv_y = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
#                                 kernel_size=3, stride=1, padding=1, bias=False, 
#                                 groups=self.channels)

#         # Set the weights of the convolutional layers to the Sobel kernels
#         self.conv_x.weight.data = kernel_x
#         self.conv_y.weight.data = kernel_y
        
#         # Freeze the weights
#         self.conv_x.weight.requires_grad = False
#         self.conv_y.weight.requires_grad = False

#         # Define the loss function
#         if loss_type.lower() == 'l1':
#             self.loss_fn = F.l1_loss
#         elif loss_type.lower() == 'l2':
#             self.loss_fn = F.mse_loss
#         else:
#             raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1' or 'l2'.")

#     def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred (torch.Tensor): The predicted image tensor, shape [B, C, H, W].
#             true (torch.Tensor): The ground truth image tensor, shape [B, C, H, W].

#         Returns:
#             torch.Tensor: The calculated gradient difference loss as a scalar tensor.
#         """
#         # Ensure the conv layers are on the same device as the input tensors
#         self.conv_x.to(pred.device)
#         self.conv_y.to(pred.device)

#         # Calculate gradients in x and y directions for both images
#         pred_grad_x = self.conv_x(pred)
#         true_grad_x = self.conv_x(true)
        
#         pred_grad_y = self.conv_y(pred)
#         true_grad_y = self.conv_y(true)

#         # Compute the loss for each direction
#         loss_x = self.loss_fn(pred_grad_x, true_grad_x)
#         loss_y = self.loss_fn(pred_grad_y, true_grad_y)
        
#         # Total loss is the sum of the losses in both directions
#         total_loss = loss_x + loss_y
        
#         return total_loss

# ===== 2. Gradient Loss (edge sharpness) =====
class GradientLoss(nn.Module):
    """
    Edge sharpness loss based on first-order derivatives
    """
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()

        if loss_type.lower() == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type.lower() == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge loss using Sobel gradient
        """
        # gradient in X direction
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # gradient in Y direction
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Loss
        loss_x = self.loss_fn(pred_dx, target_dx)
        loss_y = self.loss_fn(pred_dy, target_dy)
        
        return loss_x + loss_y

class LaplacianLoss(nn.Module):
    """
    Laplacian (edge) loss between predicted and target images.
    Applies a Laplacian filter to both images and computes L1 or L2 distance between the filtered outputs.
    
    Args:
        loss_type (str): 'l1' or 'l2'. Specifies L1 or L2 on Laplacian responses.
        reduction (str): 'mean' or 'sum' or 'none' for reduction over batch.
        normalize (bool): If True, normalize per-sample Laplacian maps by their max abs value to reduce scale sensitivity.
        eps (float): Small epsilon for numerical stability in normalization.
        multi_scale (bool): If True, compute Laplacian loss on an image pyramid (scales factor 1, 1/2, 1/4) and average.
    """
    def __init__(self, loss_type='l1', reduction='mean', normalize=False, eps=1e-6, multi_scale=False):
        super().__init__()
        assert loss_type in ('l1', 'l2'), "loss_type must be 'l1' or 'l2'"
        assert reduction in ('mean', 'sum', 'none'), "reduction must be 'mean','sum' or 'none'"
        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize = normalize
        self.eps = eps
        self.multi_scale = multi_scale
        
        # 3x3 discrete Laplacian kernel (common choice)
        # [[0,  1, 0],
        #  [1, -4, 1],
        #  [0,  1, 0]]
        lap = torch.tensor([[0.0,  1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0,  1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('kernel_3x3', lap.unsqueeze(0).unsqueeze(0))  # shape (1,1,3,3)
    
    def _apply_lap(self, x):
        """
        x: tensor (N, C, H, W)
        returns: tensor (N, C, H, W) of Laplacian response
        """
        N, C, H, W = x.shape
        # expand kernel to per-channel groups conv: shape (C,1,k,k)
        kernel = self.kernel_3x3.repeat(C, 1, 1, 1).to(x.device)
        # padding=1 for same spatial size
        # use groups=C to apply same kernel independently per channel
        lap = F.conv2d(x, kernel, bias=None, stride=1, padding=1, groups=C)
        return lap

    def _single_scale_loss(self, pred, target):
        lap_pred = self._apply_lap(pred)
        lap_target = self._apply_lap(target)
        
        if self.normalize:
            # normalize per-sample (per N) by max abs value in target Lap response to reduce scale sensitivity
            # shape: (N,1,1,1)
            max_vals = lap_target.abs().reshape(lap_target.shape[0], -1).max(dim=1)[0].clamp(min=self.eps)
            max_vals = max_vals.view(-1, 1, 1, 1)
            lap_pred = lap_pred / max_vals
            lap_target = lap_target / max_vals
        
        if self.loss_type == 'l1':
            loss_map = (lap_pred - lap_target).abs()
        else:
            loss_map = (lap_pred - lap_target) ** 2
        
        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        else:
            return loss_map  # no reduction
    
    def forward(self, pred, target):
        """
        pred, target: tensors (N, C, H, W), expected float, same device.
        """
        assert pred.shape == target.shape, "pred and target must have same shape"
        if not self.multi_scale:
            return self._single_scale_loss(pred, target)
        else:
            # multi-scale: factors 1.0, 0.5, 0.25 (if image large enough)
            scales = [1.0, 0.5, 0.25]
            losses = []
            for s in scales:
                if s == 1.0:
                    p = pred
                    t = target
                else:
                    # use area interpolation for downsampling to keep aliasing low
                    h = max(1, int(round(pred.shape[2] * s)))
                    w = max(1, int(round(pred.shape[3] * s)))
                    p = F.interpolate(pred, size=(h, w), mode='area')
                    t = F.interpolate(target, size=(h, w), mode='area')
                losses.append(self._single_scale_loss(p, t))
            # average scales
            total = sum(losses) / len(losses)
            return total



class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()

        # load VGG19 model and use only intermediate layers
        vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.eval().to("cpu")

        # specify which layers to use (typically layers 2~5)
        if layers is None:
            layers = [0, 5, 10, 19, 28]  # default layer settings (e.g., intermediate layers of VGG19)

        # configure to use only the specified layers
        self.vgg = nn.Sequential(*[vgg[i] for i in layers])

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # freeze parameters (VGG19 is a frozen model)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred_image, target_image):
        """
        pred_image: reconstructed image (prediction)
        target_image: ground truth image

        perceptual loss mainly uses MSE to compute difference in feature space
        """
        # move pred_image and target_image to the same device as VGG19
        device = next(self.vgg.parameters()).device  # check VGG19 device
        pred_image = pred_image.detach().to("cpu").float() #.to(device).float()  # move pred_image to same device as VGG19
        target_image = target_image.detach().to("cpu").float() #.to(device).float()  # move target_image to same device as VGG19

        pred_image = (pred_image - self.mean) / self.std
        target_image = (target_image - self.mean) / self.std
        pred_features = self.vgg(pred_image)  # extract features from predicted image
        target_features = self.vgg(target_image)  # extract features from ground truth image

        # compute feature difference using MSE loss
        loss = F.mse_loss(pred_features, target_features)
        return loss


class AlexNetPerceptualLoss(nn.Module):
    def __init__(self, layer=3):
        super().__init__()
        alexnet = models.alexnet(weights='IMAGENET1K_V1').features.eval()
        self.features = nn.Sequential(*list(alexnet.children())[:layer+1])
        
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        device = next(self.features.parameters()).device
        pred = pred.to(device)
        target = target.to(device) 
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.mse_loss(self.features(pred), self.features(target))


class TVLoss(nn.Module):
    """Total Variation Loss for image smoothness"""
    
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            TV loss value
        """
        batch_size, channels, height, width = x.size()
        
        # Calculate differences along height and width dimensions
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # Vertical differences
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # Horizontal differences
        
        # Sum all differences
        tv_loss = torch.sum(diff_h) + torch.sum(diff_w)
        
        # Normalize by the number of pixels
        tv_loss = tv_loss / (batch_size * channels * height * width)
        
        return self.weight * tv_loss
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        """
        Contrastive loss function based on cosine similarity.
        Args:
            temperature (float): Scaling factor for the similarity scores.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, f_masked, f_original):
        """
        Compute contrastive loss.
        
        Args:
            f_masked (Tensor): Features from masked sequence (batch, N_dim, dim).
            f_original (Tensor): Features from original sequence (batch, N_dim, dim).
        
        Returns:
            loss (Tensor): Contrastive loss value.
        """
        # Normalize embeddings along the last dimension (dim)
        f_masked = F.normalize(f_masked, dim=-1)
        f_original = F.normalize(f_original, dim=-1)

        # Compute cosine similarity along the last dimension (dim)
        similarity = torch.sum(f_masked * f_original, dim=-1)  # Shape: (batch, N_dim)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Contrastive loss (maximize similarity across all N_dim)
        loss = torch.mean(1 - similarity)
        
        return loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, output, target):
        device = output.get_device()
        self.ssim = self.ssim.to(device)
        return 1-self.ssim(output, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, loss_type='l1'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        if self.loss_type == 'l1':
            base_loss = F.l1_loss(pred, target, reduction='none')
        
        # Calculate confidence (1 - normalized error)
        error = torch.abs(pred - target)
        max_error = error.max()
        confidence = 1 - (error / (max_error + 1e-8))
        
        # Focal weight: (1-confidence)^gamma
        focal_weight = (1 - confidence) ** self.gamma
        
        # Apply focal loss
        focal_loss = self.alpha * focal_weight * base_loss
        
        return focal_loss.mean()

class RangeLoss(nn.Module):
    """
    Loss that encourages the output image value range to cover the target range (e.g., [0, 255]).
    Penalizes when the batch min is above target_min or the max is below target_max.
    """
    def __init__(self, target_min=0.0, target_max=255.0):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        pred_min = pred.min()
        pred_max = pred.max()
        # penalize if min is above target_min or if max is below target_max
        loss_min = F.relu(pred_min - self.target_min)
        loss_max = F.relu(self.target_max - pred_max)
        return loss_min + loss_max


class MultiScaleLoss(nn.Module):
    def __init__(self, loss_type='l1', scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        """
        Multi-scale loss function
        
        Args:
            loss_type (str): Type of loss ('l1', 'l2', 'smooth_l1')
            scales (list): Scale factors for downsampling [1.0, 0.5, 0.25]
            weights (list): Weights for each scale [1.0, 0.5, 0.25]
        """
        super(MultiScaleLoss, self).__init__()
        
        assert len(scales) == len(weights), "scales and weights must have same length"
        
        self.scales = scales
        self.weights = weights
        self.loss_type = loss_type
        
        # Define loss function
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred, target):
        """
        Calculate multi-scale loss
        
        Args:
            pred (torch.Tensor): Predicted image [B, C, H, W]
            target (torch.Tensor): Target image [B, C, H, W]
            
        Returns:
            torch.Tensor: Multi-scale loss value
        """
        total_loss = 0.0
        pred = pred.float()
        target = target.float()
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1.0:
                # Original resolution
                scaled_pred = pred
                scaled_target = target
            else:
                # Downsample both pred and target
                scaled_pred = F.interpolate(pred, scale_factor=scale, 
                                          mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target, scale_factor=scale, 
                                            mode='bilinear', align_corners=False)
            
            # Calculate loss at this scale
            scale_loss = self.loss_fn(scaled_pred, scaled_target)
            total_loss += weight * scale_loss
        
        return total_loss

def compute_loss_or_metric(
    loss_or_metric: str,
    output_type: OutputType,
    output: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    decoder_id: str,
) -> torch.Tensor:
    r"""Helper function to compute various losses or metrics for a given output type.

    It supports both continuous and discrete output types, and a variety of losses
    and metrics, including mse loss, binary cross entropy loss, and R2 score.

    Args:
        loss_or_metric: The name of the metric to compute. e.g. bce, mse
        output_type: The nature of the output. One of the values from OutputType. e.g. MULTINOMIAL
        output: The output tensor.
        target: The target tensor.
        weights: The sample-wise weights for the loss computation.
    """
    if 'NATURAL_' in decoder_id and output_type == None:
        # [batch, 3, 32, 64]
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(output, target)
    else:        
        if output_type == OutputType.CONTINUOUS:
            if loss_or_metric == "mse":
                # TODO mse could be used as a loss or as a metric. Currently it fails when
                # called as a metric
                # MSE loss
                loss_noreduce = F.mse_loss(output, target, reduction="none").mean(dim=1)
                return (weights * loss_noreduce).sum() / weights.sum()
            elif loss_or_metric == "r2":
                r2score = R2Score(num_outputs=target.shape[1])
                return r2score(output, target)
            elif loss_or_metric == "frame_diff_acc":
                normalized_window = 30 / 450
                differences = torch.abs(output - target)
                correct_predictions = differences <= normalized_window
                accuracy = (
                    correct_predictions.float().mean()
                )  # Convert boolean tensor to float and calculate mean
                return accuracy
            else:
                raise NotImplementedError(
                    f"Loss/Metric {loss_or_metric} not implemented for continuous output"
                )

        if output_type in [
            OutputType.BINARY,
            OutputType.MULTINOMIAL,
            OutputType.MULTILABEL,
        ]:
            if decoder_id == 'CRE_LINE':
                unique_classes = torch.unique(target)
                class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
                device = target.get_device()
                target = torch.tensor([class_to_index[cls.item()] for cls in target])
                target = target.to(device)

            if loss_or_metric == "bce":
                target = target.to(torch.long).flatten()
                # target = target.squeeze(dim=1)
                loss_noreduce = F.cross_entropy(output, target , reduction="none")
                if loss_noreduce.ndim > 1:
                    loss_noreduce = loss_noreduce.mean(dim=1)
                return (weights * loss_noreduce).sum() / weights.sum()
            elif loss_or_metric == "mallows_distance":
                num_classes = output.size(-1)
                output = torch.softmax(output, dim=-1).view(-1, num_classes)
                target = target.view(-1, 1)
                weights = weights.view(-1)
                # Mallow distance
                target = torch.zeros_like(output).scatter_(1, target, 1.0)
                # we compute the mallow distance as the sum of the squared differences
                loss = torch.mean(
                    torch.square(
                        torch.cumsum(target, dim=-1) - torch.cumsum(output, dim=-1)
                    ),
                    dim=-1,
                )
                loss = (weights * loss).sum() / weights.sum()
                return loss
            # elif loss_or_metric == "accuracy":
            #     pred_class = torch.argmax(output, dim=1)
            #     return (pred_class == target.squeeze()).sum() / len(target)

            elif loss_or_metric == "accuracy":
                pred_class = torch.argmax(output, dim=1)

                # --- Before ---
                # return (pred_class == target.squeeze()).sum() / len(target)

                # --- After ---
                # 1. tensor of correct predictions on GPU
                correct_predictions = (pred_class == target.flatten()).sum()

                # 2. explicitly convert Python int total count to a GPU tensor
                total_count = torch.tensor(target.numel(), device=correct_predictions.device)

                # 3. perform division between GPU tensors to guarantee result is on GPU
                return correct_predictions / total_count

            elif loss_or_metric == "frame_diff_acc":
                pred_class = torch.argmax(output, dim=1)
                difference = torch.abs(pred_class - target.flatten())
                correct_predictions = difference <= 30
                return correct_predictions.float().mean()
            else:
                raise NotImplementedError(
                    f"Loss/Metric {loss_or_metric} not implemented for binary/multilabel "
                    "output"
                )

        raise NotImplementedError(
            "I don't know how to handle this task type. Implement plis"
        )
