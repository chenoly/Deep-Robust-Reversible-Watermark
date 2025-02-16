import os
import re
import torch.nn.functional as F
import numpy as np
import torch
import kornia
from torch import nn, Tensor
import lpips


class Sample(nn.Module):
    def __init__(self, img_size: int, input_dim: int, bit_length: int):
        super().__init__()
        self.bit_length = bit_length
        self.bit_plane_size = int(bit_length ** 0.5)
        scale_f = img_size // self.bit_plane_size
        self.noise = nn.Parameter(1 + torch.abs(torch.round(torch.randn(size=(1, input_dim, img_size, img_size)))))
        self.upsample = nn.Upsample(scale_factor=scale_f, mode='nearest')
        self.downsample = nn.Upsample(scale_factor=1. / scale_f, mode='nearest')

    def __call__(self, img_tensor: Tensor, reverse=False):
        if not reverse:
            img_tensor = img_tensor.view(-1, 1, self.bit_plane_size, self.bit_plane_size)
            out = self.upsample(img_tensor) * self.noise
        else:
            out = self.downsample(img_tensor / self.noise).mean(dim=1)
            out = out.view(-1, self.bit_length)
        return out


class LPIPSLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.lpips = lpips.LPIPS(net="alex", verbose=False)

    def __call__(self, input_image, target_image):
        if input_image.shape[1] == 1:
            input_image = input_image.repeat(1, 3, 1, 1)
            target_image = target_image.repeat(1, 3, 1, 1)
        normalized_input = input_image.clamp(0, 1.) * 2 - 1
        normalized_encoded = target_image.clamp(0, 1.) * 2 - 1
        lpips_loss = self.lpips(normalized_input, normalized_encoded).mean()
        return lpips_loss


class EdgeLoss(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        self.im_size = im_size
        self.mse = nn.MSELoss()
        self.falloff_im = self.falloff_im()

    def falloff_im(self):
        size = (self.im_size, self.im_size)
        l2_edge_gain = 10
        falloff_speed = 4
        falloff_im = np.ones(size)
        for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
            falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
            falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
        for j in range(int(falloff_im.shape[1] / falloff_speed)):
            falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
            falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im = 1. - falloff_im
        falloff_im = torch.from_numpy(falloff_im)
        falloff_im *= l2_edge_gain
        return falloff_im

    def forward_org(self, input_tensor, target_tensor):
        im_diff = input_tensor - target_tensor
        im_diff += im_diff * self.falloff_im.to(input_tensor.device)
        image_loss = torch.mean(im_diff ** 2)
        return image_loss

    def computeJND(self, input_image, kernel_size=8):
        """
        Compute JND map for an RGB image.
        Args:
            input_image: Tensor of shape (N, C, H, W)
            kernel_size: Size of the pooling kernel
        Returns:
            jnd_map: Tensor of shape (N, C, H, W) representing the JND map
        """
        # Average pooling for each channel
        mean_pool = F.avg_pool2d(input_image, kernel_size=kernel_size, stride=kernel_size, padding=0)
        mean_expanded = F.interpolate(mean_pool, size=input_image.shape[-2:], mode='bilinear', align_corners=False)
        alpha = 0.001
        jnd_luminance = alpha * mean_expanded
        if input_image.shape[1] == 3:
            sobel_kernel = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
                                         [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
                                         [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]).to(input_image.device)
            edges = F.conv2d(input_image, sobel_kernel, padding=1, groups=3)  # Apply per-channel convolution
        else:
            sobel_kernel = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]).to(input_image.device)
            edges = F.conv2d(input_image, sobel_kernel, padding=1, groups=1)  # Apply per-channel convolution
        jnd_contrast = 0.5 * edges.abs()
        jnd_map = jnd_luminance + jnd_contrast
        return jnd_map + input_image

    def __call__(self, input_tensor, target_tensor):
        """
        Forward pass to compute the YUV loss with edge detection influencing the Y channel.
        :param input_tensor: The predicted or generated image (RGB format).
        :param target_tensor: The target or original image (RGB format).
        :return: Loss value combining YUV differences and edge strength.
        """
        # Compute edges on the target image using the Sobel operator
        cover_jnd = self.computeJND(target_tensor)
        loss = self.mse(input_tensor, cover_jnd)
        return loss


# Stochastic Round class for stochastic rounding operations
class StochasticRound:
    def __init__(self, scale=1 / 255.):
        """
        Initializes the Stochastic Round operation.

        Parameters:
            scale (float): The scaling factor for the rounding operation. Default is 1/255.
        """
        super().__init__()
        self.scale = scale

    def __call__(self, x, hard_round):
        """
        Perform stochastic rounding on the input tensor.

        Parameters:
            x (Tensor): Input tensor to be rounded.
            hard_round (bool): Whether to use hard rounding or soft rounding.

        Returns:
            Tensor: The rounded tensor.
        """
        # Scale the input by the defined scaling factor
        scale_x = x / self.scale
        # Perform the rounding operation
        round_out = scale_x + (torch.round(scale_x) - scale_x).detach()
        out = round_out + torch.rand_like(x) - 0.5  # Add noise for stochastic rounding
        if hard_round:
            return round_out * self.scale  # Return scaled result for hard rounding
        return out * self.scale  # Return original tensor if no rounding is performed


# Penalty loss class to compute penalties for overflow pixels
class PenalityLoss(nn.Module):
    def __init__(self, max_value=1.):
        """
        Initializes the Penalty Loss for overflow pixels.

        Parameters:
            max_value (float): Maximum allowable pixel value (default is 1).
        """
        super().__init__()
        self.max_value = max_value
        # self.MSE = nn.MSELoss(reduce=True, size_average=False)
        self.MSE = nn.MSELoss(reduce=True)

    def __call__(self, input_tensor):
        """
        Computes the penalty loss for pixels that overflow the allowable range.

        Parameters:
            input_tensor (Tensor): Input tensor to compute penalty loss on.

        Returns:
            Tensor: The penalty loss value.
        """
        # Calculate the penalty for pixels below 0
        loss_0 = self.MSE(torch.relu(-input_tensor), torch.zeros_like(input_tensor))
        # Calculate the penalty for pixels above max_value
        loss_255 = self.MSE(torch.relu(input_tensor - self.max_value), torch.zeros_like(input_tensor))
        # Total penalty loss is the sum of both losses
        loss = loss_0 + loss_255
        return loss


# Discrete Wavelet Transform (DWT) class to decompose images
class DWT(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img_tensor: Tensor):
        """
        Performs the Discrete Wavelet Transform on an input image tensor.

        Parameters:
            img_tensor (Tensor): Input image tensor of shape (batch, channels, height, width).

        Returns:
            Tensor: The wavelet-transformed image with four sub-bands.
        """
        # Decompose image tensor into sub-bands
        x01 = img_tensor[:, :, 0::2, :] / 2
        x02 = img_tensor[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # Calculate the four sub-bands (LL, HL, LH, HH)
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        # Concatenate the four sub-bands along the channel dimension
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# Inverse Discrete Wavelet Transform (IDWT) class to reconstruct images
class IDWT(nn.Module):
    def __init__(self, r=2):
        """
        Initializes the Inverse Discrete Wavelet Transform.

        Parameters:
            r (int): Upsampling factor. Default is 2.
        """
        super().__init__()
        self.r = r

    def __call__(self, img_tensor: Tensor):
        """
        Performs the Inverse Discrete Wavelet Transform on the input image tensor.

        Parameters:
            img_tensor (Tensor): The wavelet-transformed image tensor.

        Returns:
            Tensor: The reconstructed image tensor.
        """
        batch_size, in_channel, in_height, in_width = img_tensor.size()
        out_channel = int(in_channel / (self.r ** 2))
        out_height = self.r * in_height
        out_width = self.r * in_width

        # Decompose input tensor into four sub-bands
        x1 = img_tensor[:, 0:out_channel, :, :] / 2
        x2 = img_tensor[:, out_channel:out_channel * 2, :, :] / 2
        x3 = img_tensor[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = img_tensor[:, out_channel * 3:out_channel * 4, :, :] / 2

        # Initialize the output tensor with zeros
        h = torch.zeros([batch_size, out_channel, out_height, out_width], dtype=img_tensor.dtype).to(img_tensor.device)

        # Apply inverse DWT to reconstruct the image
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


# Normalize function to scale image tensor to [0, 1] range
def normalize(input_image):
    """
    Normalize the input image tensor to the range [0, 1].

    Parameters:
        input_image (Tensor): Input image tensor to normalize.

    Returns:
        Tensor: The normalized image tensor.
    """
    min_vals = input_image.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = input_image.amax(dim=(1, 2, 3), keepdim=True)
    normalized_img = (input_image - min_vals) / (max_vals - min_vals + 1e-5)  # Prevent division by zero
    return normalized_img


# Function to extract the accuracy of a secret image
def extract_accuracy(ext_secret, secret, max_value=1.):
    """
    Extracts the accuracy of a secret image by comparing it to the expected secret.

    Parameters:
        ext_secret (Tensor): The extracted secret image.
        secret (Tensor): The ground truth secret image.

    Returns:
        float: The accuracy value.

    Parameters
    ----------
    secret
    ext_secret
    max_value
    """
    acc = 1.0 - (torch.abs(torch.round(ext_secret.clamp(0., max_value)) - secret).mean())
    return acc.item()


# x = torch.randint(0, 2, size=(4, 24))
# y = x.clone()
# y[:, 2] = 1
# y[:, 4] = 1
# print(extract_accuracy(x, y))


# Function to calculate the number of overflow pixels in a stego image
def overflow_num(stego, mode, min_value=0., max_value=1.):
    """
    Calculate the number of overflow pixels in the stego image.

    Parameters:
        stego (Tensor): The stego image tensor.
        mode (int): The overflow mode (0 for below min_value, 255 for above max_value).
        min_value (float): The minimum allowed pixel value (default is 0).
        max_value (float): The maximum allowed pixel value (default is 1).

    Returns:
        float: The average overflow pixel count.
    """
    assert mode in [0, 255]
    if mode == 0:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) < min_value, dim=(1, 2, 3)).float().mean()
    else:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) > max_value, dim=(1, 2, 3)).float().mean()
    return overflow_pixel_n.item()


# Function to compute the PSNR between the input and target images
def compute_psnr(input_image, target_image, max_value=1.):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor.
        max_value (float): The maximum allowable pixel value (default is 1).

    Returns:
        float: The average PSNR value.
    """
    # Apply stochastic rounding and clamp images to the range [0, 1]
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    target_image = StochasticRound()(target_image.clamp(0., 1.), True)
    average_psnr = kornia.metrics.psnr(input_image, target_image, max_value).mean()
    return average_psnr.item()


def quantize_image(input_image):
    """
    Quantize the input image using stochastic rounding.

    Parameters:
        input_image (Tensor): Input image tensor with values between 0 and 1.

    Returns:
        Tensor: The quantized image tensor after applying stochastic rounding.
    """
    # Apply stochastic rounding to the input image, ensuring the values are within the [0, 1] range.
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)

    return input_image


def quantize_residual_image(input_image, target_image):
    """
    Quantize the residual image, which is the difference between the input and target image.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor to compare against.

    Returns:
        Tensor: The quantized residual image after applying stochastic rounding.
    """
    # Calculate the residual (difference) between the input image and the target image.
    # Then normalize the residual and apply stochastic rounding.
    shift = (input_image - target_image) * 10.
    input_image = StochasticRound()(normalize(shift), True)

    return input_image


def find_latest_model(model_dir):
    """
    Find the .pth file with the largest epoch number in the given directory.

    Parameters:
        model_dir (str): Path to the directory containing model files.

    Returns:
        str: Path to the latest model file, or None if no .pth files are found.
    """
    # Initialize variables to track the maximum epoch and corresponding model file.
    max_epoch = -1
    latest_model_path = None

    # Regular expression pattern to match the file format: "model_{epoch}.pth"
    pattern = re.compile(r"model_(\d+)\.pth")

    # Iterate through files in the given directory.
    for file_name in os.listdir(model_dir):
        # Check if the file name matches the pattern.
        match = pattern.match(file_name)
        if match:
            # Extract the epoch number from the file name.
            epoch = int(match.group(1))
            # Update the maximum epoch and the path to the latest model if necessary.
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model_path = os.path.join(model_dir, file_name)

    return latest_model_path

# if __name__ == "__main__":
#     # Generate a sample image tensor with integer values between 0 and 255.
#     # The tensor has shape (1, 3, 256, 256), representing 1 batch, 3 color channels, and 256x256 resolution.
#     img_tensor = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)
#
#     # Initialize Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT) layers.
#     dwt = DWT()
#     idwt = IDWT()
#
#     # Perform DWT on the image to decompose it into different frequency bands.
#     dwt_output = dwt(img_tensor)
#     print(dwt_output)
#
#     # Perform IDWT to reconstruct the image from the DWT coefficients.
#     reconstructed_image = idwt(dwt_output)
#
#     # Check if the reconstructed image is identical to the original image.
#     # A perfect DWT/IDWT pair should reconstruct the original image exactly.
#     if torch.allclose(img_tensor, reconstructed_image, atol=1e-5):  # Allowing a small tolerance
#         print("The IDWT reconstruction is identical to the original image!")
#     else:
#         # Print any discrepancies between the original and reconstructed images.
#         print("There is a discrepancy between the original image and the reconstructed image.")
#         print("Original Image:\n", img_tensor)
#         print("Reconstructed Image:\n", reconstructed_image)
