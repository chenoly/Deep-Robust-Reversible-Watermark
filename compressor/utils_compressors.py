import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from .arithmeticcoder import ArithmeticEncoder


class CustomArithmeticEncoder:
    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 20):
        """
        Initialize the encoder with the specified bit lengths for levels and frequencies.

        :param level_bits_len: Bit length used for encoding integer values.
        :param freq_bits_len: Bit length used for encoding frequency information.
        """
        self.buffer_bits = 40
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """
        Convert a NumPy array into a list of strings.

        :param data: Input ndarray to convert.
        :return: List of strings where each element is a string representation of a value in the ndarray.
        """
        data_list = data.flatten().tolist()
        data_str_list = [str(data_item) for data_item in data_list]
        return data_str_list

    def strlist2ndarray(self, data_str_list: list) -> list:
        """
        Convert a list of strings back into a list of integers.

        :param data_str_list: List of strings to convert.
        :return: List of integers corresponding to the original data values.
        """
        data_list = [int(data_str_item) for data_str_item in data_str_list]
        return data_list

    def datastr2bits(self, data_str_list):
        """
        Encode each string in the input list into a fixed-length binary bit stream.

        :param data_str_list: List of strings to encode.
        :return: A flat list of integers (0s and 1s) representing the encoded bits.
        """
        encoded_bits = []
        for item in data_str_list:
            num = int(item)
            if num < 0:
                num = (1 << self.freq_bits_len) + num  # Handle negative integers
            binary_representation = bin(num)[2:]  # Convert to binary string
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # Zero-pad the string
            if len(padded_binary) > self.freq_bits_len:
                raise ValueError(f"Value {num} cannot be represented in {self.freq_bits_len} bits.")
            encoded_bits.extend(int(bit) for bit in padded_binary)  # Add each bit as an integer to the list
        return encoded_bits

    def bits2datastr(self, encoded_bits):
        """
        Decode a list of binary bits back into the original string values.

        :param encoded_bits: List of integers (0s and 1s) representing the encoded bits.
        :return: List of strings corresponding to the decoded values.
        """
        decoded_data = []
        for i in range(0, len(encoded_bits), self.freq_bits_len):
            bit_str = ''.join(map(str, encoded_bits[i:i + self.freq_bits_len]))  # Group bits into chunks
            num = int(bit_str, 2)  # Convert binary string to an integer
            if num >= (1 << (self.freq_bits_len - 1)):
                num -= (1 << self.freq_bits_len)  # Convert back to signed integer if necessary
            decoded_data.append(str(num))  # Convert back to string and add to list
        return decoded_data

    def integer2bits(self, integer: int):
        """
        Convert an integer to a binary bit list of fixed length.

        :param integer: The integer to convert.
        :return: A list of bits representing the integer.
        """
        if integer < 0:
            integer = (1 << self.level_bits_len) + integer  # Handle negative integers
        if integer >= (1 << self.level_bits_len):
            raise ValueError(f"Value {integer} cannot be represented in {self.level_bits_len} bits.")
        binary_representation = bin(integer)[2:]  # Convert to binary string
        padded_binary = binary_representation.zfill(self.level_bits_len)  # Zero-pad the string
        return [int(bit) for bit in padded_binary]  # Convert binary string to list of bits

    def bits2integer(self, bits: list):
        """
        Convert a list of bits back into an integer.

        :param bits: List of bits (0s and 1s) representing the integer.
        :return: The decoded integer value.
        """
        if len(bits) != self.level_bits_len:
            raise ValueError(f"Bits list must have length {self.level_bits_len}.")
        bit_str = ''.join(str(bit) for bit in bits)  # Convert list of bits to binary string
        num = int(bit_str, 2)  # Convert binary string to integer
        if num >= (1 << (self.level_bits_len - 1)):
            num -= (1 << self.level_bits_len)  # Handle signed integers
        return num

    def compress(self, data: ndarray, frequencies=None) -> list:
        """
        Compress an ndarray by encoding its elements as binary bit streams.

        :param frequencies:
        :param data: NumPy ndarray to compress.
        :return: List of encoded bits representing the compressed data.
        """
        data_str_list = self.ndarray2strlist(data)
        if frequencies is None:
            _frequencies = list(set(data_str_list))  # Get unique elements (frequencies)
            freqs_bits = self.datastr2bits(_frequencies)  # Encode frequencies into bits
            auxbits = freqs_bits + self.integer2bits(len(_frequencies))  # Auxiliary bits: frequency bits + length
            frequencies_input = _frequencies + ["<EOM>"]
        else:
            frequencies_input = frequencies + ["<EOM>"]
            auxbits = []
        coder = ArithmeticEncoder(frequencies=frequencies_input, bits=self.buffer_bits)  # Initialize arithmetic encoder
        data_bits = list(coder.encode(data_str_list + ["<EOM>"]))  # Encode data string list
        if frequencies is None:
            return data_bits + auxbits
        else:
            return data_bits

    def decompress(self, data_freqs_bits: list, frequencies=None) -> ndarray:
        """
        Decompress the encoded bit stream back into an ndarray.

        :param frequencies:
        :param data_freqs_bits: List of encoded bits, including frequency information.
        :return: NumPy ndarray with the decompressed data.
        """
        if frequencies is None:
            len_bits_freqs = data_freqs_bits[-self.level_bits_len:]  # Extract the length of frequencies from the end
            retain_bits = data_freqs_bits[:-self.level_bits_len]  # Remove length bits from the data stream
            len_freqs = self.bits2integer(len_bits_freqs)  # Convert length bits to integer
            freqs_bits = retain_bits[-len_freqs * self.freq_bits_len:]  # Extract frequency bits
            retain_bits = retain_bits[:-len_freqs * self.freq_bits_len]  # Remove frequency bits from data stream
            frequencies = self.bits2datastr(freqs_bits)  # Decode frequencies from bits
            data_bits = retain_bits  # Remaining bits are the actual data bits
            frequencies += ["<EOM>"]
        else:
            data_bits = data_freqs_bits
            frequencies += ["<EOM>"]
        coder = ArithmeticEncoder(frequencies=frequencies, bits=self.buffer_bits)  # Initialize arithmetic decoder
        data_str_list = list(coder.decode(data_bits))  # Decode data bits
        data = [int(data_item) for data_item in data_str_list[:-1]]  # Convert decoded strings back to integers
        return np.asarray(data)  # Return as a NumPy array


class ACCompress:
    def __init__(self, im_size, z_size, level_bits_len, freq_bits_len, device: str = "cpu"):
        """
        Initialize the CustomCoder.

        :param im_size: Tuple representing the size of the image (Height, Width, Channels).
        :param z_size: Tuple representing the size of the latent vector (z) (Height, Width).
        :param device: Torch device ('cpu' or 'cuda') where tensors are processed.
        """
        self.mark_len = 20
        self.im_size = im_size  # Image size (H, W, C)
        self.z_size = z_size  # Latent vector (drop_z) size
        self.device = device  # Device to use for tensor computations
        self.im_len = im_size[0] * im_size[1] * im_size[2]  # Total number of elements in image
        self.z_len = z_size[0] * z_size[1]  # Total number of elements in drop_z
        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list):
        """

        Parameters
        ----------
        z_bits
        stego_bits

        Returns
        -------

        """
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        combined_bits = length_bits + z_bits + stego_bits
        return combined_bits

    def split_bits(self, combined_bits: list):
        """

        Parameters
        ----------
        combined_bits

        Returns
        -------

        """
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def encode(self, stego_img, drop_z):
        """
        Encode the stego image and latent vector (drop_z) into binary bits.

        :param stego_img: A tensor of the stego image of shape (N, H, W, C), where N should be 1 (batch size).
        :param drop_z: A tensor of the latent vector z of shape (N, H, W), where N should be 1 (batch size).
        :return: A list of binary data bits.
        """
        overflow_bits_list = []
        clip_stego_img = None
        if stego_img is not None:
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            overflow = np.zeros_like(steg_img_numpy)
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)
            overflow_bits_list = self.coder.compress(overflow.flatten())

        drop_z_bits_list = []
        if drop_z is not None:
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy_clip = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy_clip.flatten())

        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)
            drop_z_bits_list, overflow_bits_list = self.split_bits(data_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decode(self, clip_stego_img: Tensor, data_bits: list):
        """
        Decode the binary data bits back into the stego image and latent vector (drop_z).

        :param clip_stego_img:
        :param data_bits: A list of binary bits representing the encoded stego image and latent vector.
        :return: Two tensors: the decoded stego image and latent vector (z).
        """
        # compute mask
        mask_0 = (clip_stego_img == 0) + 0
        mask_255 = (clip_stego_img == 255) + 0

        # Decode the data bits into a list of floating-point numbers
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        overflow_list = self.coder.decompress(overflow_bits_list)
        # Convert lists back into tensors and reshape them
        overflow = torch.as_tensor(overflow_list).reshape(self.im_size).to(self.device).permute(2, 0, 1).unsqueeze(0)
        stego_img = (clip_stego_img - mask_0 * overflow + mask_255 * overflow) / 1.
        rec_z = torch.as_tensor(drop_z_list).reshape(self.z_size).to(self.device) / 1.
        return stego_img, rec_z


class SparseTensorCompressor:
    def __init__(self, image_shape, z_shape, level_bits_len, freq_bits_len, val_bits=10):
        """
        Initialize the compression class. The input is a tuple representing the image's height, width, and number of channels.
        :param image_shape: Shape of the image (height, width, channels)
        :param value_range: The value range for tensor elements, e.g., (-255, 255)
        """
        self.height, self.width, self.channels = image_shape
        self.mark_len = 20
        self.z_shape = z_shape

        # Dynamically calculate the bit count required for row, column, and channel based on image dimensions
        self.row_bits = int(np.ceil(np.log2(self.height)))  # Number of bits required for row index
        self.col_bits = int(np.ceil(np.log2(self.width)))  # Number of bits required for column index

        # If it's a grayscale image, there is no need for channel encoding
        self.channel_bits = 0 if self.channels == 1 else int(np.ceil(np.log2(self.channels)))  # Number of bits required for channel index
        self.value_bits = int(val_bits)  # Number of bits required for value (for signed values)

        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list):
        """Combine z_bits and stego_bits into a single bitstream with a length marker."""
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        combined_bits = length_bits + z_bits + stego_bits
        return combined_bits

    def split_bits(self, combined_bits: list):
        """Split the combined bitstream into z_bits and stego_bits."""
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def compress(self, stego_img, drop_z):
        """
        Compress the stego image and optional drop_z tensor into a bitstream.
        :param stego_img: The stego image tensor to be compressed
        :param drop_z: The optional drop_z tensor to be compressed
        :return: The compressed stego image and bitstream
        """
        overflow_bits_list = []
        clip_stego_img = None
        if stego_img is not None:
            # Clip the input tensor to make sure all values are within the valid range [0, 255]
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            # Initialize the overflow tensor to store out-of-range values
            overflow = np.zeros_like(steg_img_numpy)
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)
            # Get the positions of non-zero elements (row, column, channel)
            non_zero_positions = (overflow != 0).nonzero()
            pos_1 = non_zero_positions[0]
            pos_2 = non_zero_positions[1]
            pos_3 = non_zero_positions[2]
            # Get the values corresponding to the non-zero positions
            non_zero_values = overflow[pos_1, pos_2, pos_3]
            compressed_bitstream = ''
            for row, col, channel, val in zip(pos_1.tolist(), pos_2.tolist(), pos_3.tolist(), non_zero_values.tolist()):
                # Convert row, column, channel, and value to binary strings with the respective bit lengths
                row_bin = f'{row:0{self.row_bits}b}'
                col_bin = f'{col:0{self.col_bits}b}'
                # Skip channel encoding for grayscale images
                if self.channels > 1:
                    channel_bin = f'{channel:0{self.channel_bits}b}'
                else:
                    channel_bin = ''  # No channel info for grayscale
                value_bin = f'{val:0{self.value_bits}b}'
                # Concatenate all the binary parts into the compressed bitstream
                compressed_bitstream += row_bin + col_bin + channel_bin + value_bin
            # Convert the bitstream to a list of integers (0s and 1s)
            overflow_bits_list = [int(bit) for bit in compressed_bitstream]

        drop_z_bits_list = []
        if drop_z is not None:
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy.flatten())

        # Choose the data list depending on whether drop_z or stego_img is None
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)
        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decompress(self, clip_stego_img, data_bits):
        """
        Decompress the binary bitstream and recover the original tensor.
        :param clip_stego_img: The clipped stego image
        :param data_bits: The binary bitstream as a list of integers
        :return: The recovered tensor
        """
        # Decode the data bits into a list of floating-point numbers
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        # print(rec_drop_z.detach().cpu().numpy().tolist())
        rec_z = torch.as_tensor(drop_z_list, dtype=torch.float64).reshape(self.z_shape)
        #
        # Convert the data list back into a binary string
        compressed_bitstream = ''.join(str(bit) for bit in overflow_bits_list)

        # Initialize an empty tensor with zeros for the overflow
        overflow = np.zeros(shape=(self.height, self.width, self.channels))
        index = 0

        while index < len(compressed_bitstream):
            # Extract the row, column, channel, and value from the bitstream
            row = int(compressed_bitstream[index:index + self.row_bits], 2)
            col = int(compressed_bitstream[index + self.row_bits:index + self.row_bits + self.col_bits], 2)
            channel_start = index + self.row_bits + self.col_bits
            if self.channels > 1:
                channel = int(compressed_bitstream[channel_start:channel_start + self.channel_bits], 2)
                index += self.row_bits + self.col_bits + self.channel_bits
            else:
                channel = 0  # For grayscale, just set channel to 0
                index += self.row_bits + self.col_bits  # Skip the channel bits

            value = int(compressed_bitstream[index:index + self.value_bits], 2)
            overflow[row, col, channel] = value
            # Move to the next element in the bitstream (total bits used for each element)
            index += self.value_bits

        # Compute mask for zero and 255 positions
        mask_0 = (clip_stego_img == 0.).float()
        mask_255 = (clip_stego_img == 255.).float()
        overflow_tensor = torch.as_tensor(overflow, dtype=clip_stego_img.dtype).permute(2, 0, 1).unsqueeze(0)
        # Reconstruct the stego image by adding overflow to the original image where necessary
        stego_img = clip_stego_img - mask_0 * overflow_tensor + mask_255 * overflow_tensor
        return stego_img, rec_z


class TensorCoder:
    def __init__(self, image_shape, drop_z_shape, level_bits_len, freq_bits_len):
        self.sparsetensorcompressor = SparseTensorCompressor(image_shape, drop_z_shape, level_bits_len, freq_bits_len)
        self.accompress = ACCompress(image_shape, drop_z_shape, level_bits_len, freq_bits_len)

    # def compress(self, input_tensor, drop_z, mode="a"):
    #     clip_stego_img = None
    #     data_list = []
    #     if mode == "a":
    #         clip_stego_img, data_list = self.accompress.encode(input_tensor, drop_z)
    #     if mode == "s":
    #         clip_stego_img, data_list = self.sparsetensorcompressor.compress(input_tensor, drop_z)
    #     return clip_stego_img, data_list
    #
    # def decompress(self, clip_stego_img, data_list, mode="a"):
    #     rec_img = None
    #     rec_drop_z = None
    #     if mode == "a":
    #         rec_img, rec_drop_z = self.accompress.decode(clip_stego_img, data_list)
    #     if mode == "s":
    #         rec_img, rec_drop_z = self.sparsetensorcompressor.decompress(clip_stego_img, data_list)
    #     return rec_img, rec_drop_z

    def compress(self, input_tensor, drop_z, mode="s"):
        clip_stego_img_a, data_list_a = self.accompress.encode(input_tensor, drop_z)
        clip_stego_img_s, data_list_s = self.sparsetensorcompressor.compress(input_tensor, drop_z)
        if len(data_list_a[0]) > len(data_list_s[0]):
            data_list = [0] + data_list_s[0]
            ress_tuple = (data_list, data_list_s[1], data_list_s[2])
        else:
            data_list = [1] + data_list_a[0]
            ress_tuple = (data_list, data_list_a[1], data_list_a[2])
        clip_stego_img = clip_stego_img_a
        return clip_stego_img, ress_tuple

    def decompress(self, clip_stego_img, data_list, mode="s"):
        data_list_ = data_list[1:]
        if data_list[0] == 1:
            rec_img, rec_drop_z = self.accompress.decode(clip_stego_img, data_list_)
        else:
            rec_img, rec_drop_z = self.sparsetensorcompressor.decompress(clip_stego_img, data_list_)

        return rec_img, rec_drop_z


if __name__ == "__main__":
    import torch
    import numpy as np

    # Define the image and latent vector sizes
    im_size = (400, 400, 3)  # Example image size (H, W, C)
    z_size = (1, 100)  # Example latent vector size (H, W)

    # Define the device ('cpu' or 'cuda')
    device = 'cpu'  # or torch.device('cuda') if using GPU

    # Create random stego image and latent vector (drop_z) tensors
    stego_img = torch.randint(-10, 280, (1, *(im_size[2], im_size[0], im_size[1])), dtype=torch.float32, device=device)  # Stego image of shape (1, H, W, C)
    drop_z = torch.randint(44631230, 44631236, (1, *z_size), dtype=torch.float32, device=device)  # Latent vector (z) of shape (1, H, W)

    print(2 ** 40 > 44631236)
    # Initialize the TensorCoder
    tensor_coder = TensorCoder(im_size, z_size, 10, 40)

    # Encode the stego image and latent vector
    clip_encoded_img, data_bits = tensor_coder.compress(stego_img, drop_z)

    # Decode the binary data back into the stego image and latent vector
    decoded_img, decoded_z = tensor_coder.decompress(clip_encoded_img, data_bits[0])

    # Print only the first 10 elements of the decoded image and latent vector
    print("original stego image [first 10 values]:", stego_img.flatten()[:10].tolist())
    print("Decoded stego image [first 10 values]:", decoded_img.flatten()[:10].tolist())

    print("original latent vector [first 10 values]:", drop_z.flatten()[:10].tolist())
    print("Decoded latent vector (drop_z) [first 10 values]:", decoded_z.flatten()[:10].tolist())

    # Check if the decoded tensors match the original tensors
    assert torch.allclose(stego_img, decoded_img, atol=1e-8), "Stego image decoding failed!"
    assert torch.allclose(drop_z, decoded_z, atol=1e-8), "Latent vector decoding failed!"
    print("Encoding and decoding successful!")
