import math
import torch
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Tuple
from numpy import ndarray
from scipy.stats import mode
import matplotlib.pyplot as plt
from .arithmeticcoder import CustomArithmeticEncoder


class RDH:
    def __init__(self, img_size: Tuple[int, int, int], height_end: int = 3, bit_plane: int = 3, time_len: int = 5,
                 peak_len: int = 6,
                 flag_len: int = 2):
        """
        Initialize the RDH (Reversible Data Hiding) class.

        :param img_size: Tuple representing the dimensions of the image (height, width, channels).
        :param height_end: Height at which the image is split for embedding.
        """
        self.img_size = img_size
        self.bit_plane = bit_plane
        self.storage_len = height_end * self.img_size[1] * self.img_size[2] * self.bit_plane
        self.c_len = math.ceil(np.log2(img_size[2])) + 1  # Bits required for channel index
        self.w_len = math.ceil(np.log2(img_size[1])) + 1  # Bits required for width index
        self.h_len = math.ceil(np.log2(img_size[0])) + 1  # Bits required for height index
        self.freq_len = self.c_len + self.w_len + self.h_len
        self.time_len = time_len
        self.peak_len = peak_len
        self.flag_len = flag_len
        self.map_len = self.freq_len * 2  # Total bits for location map
        self.height_end = height_end  # Height end index for image splitting
        self.mask_o, self.mask_x = self.set_mask()
        self.customAC = CustomArithmeticEncoder()

    def set_mask(self):
        """

        :return:
        """
        base_mask = np.fromfunction(lambda h, w: (h + w) % 2, (self.img_size[0] - self.height_end, self.img_size[1]), dtype=int)
        mask = np.repeat(base_mask[:, :, np.newaxis], self.img_size[2], axis=2)
        return mask, 1 - mask

    def prediect(self, cover_img: ndarray, h: int, w: int, c: int):
        """


        :param cover_img: The cover image as a NumPy array.
        :param h: Height index of the pixel.
        :param w: Width index of the pixel.
        :param c: Channel index of the pixel.
        :return: Predicted value based on the MED algorithm.
        """
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        block = cover_img[h - 1: h + 2, w - 1: w + 2, c]
        predict_value = np.round(np.sum(mask * block) / 4.)
        return predict_value

    def predicting_error(self, cover_img: ndarray):
        """
        Calculate the prediction error of the cover image.

        :param cover_img: The cover image as a NumPy array.
        :return: A tuple containing the prediction error and predicted values.
        """
        H, W, C = cover_img.shape
        pv = np.copy(cover_img)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    pv[h, w, c] = self.prediect(cover_img, h, w, c)  # Compute predicted values

        pv_1 = self.mask_o * pv
        pv_2 = self.mask_x * pv
        pe_1 = cover_img * self.mask_o - pv_1
        pe_2 = cover_img * self.mask_x - pv_2
        return pe_1, pe_2, pv_1, pv_2

    def split_img(self, cover_img: ndarray):
        """
        Split the cover image into two parts: the part for embedding and the location map.

        :param cover_img: The cover image as a NumPy array.
        :return: A tuple containing the image for embedding and the location map.
        """
        img4locmap = cover_img[:self.height_end, :, :]  # Image section for embedding
        img4embed = cover_img[self.height_end:, :, :]  # Image section for location mapping
        return img4embed, img4locmap

    def merge_img(self, watermarked_img4embed: ndarray, marked_img4locmap: ndarray):
        """
        Merge the embedded image and the location map into a single watermarked image.

        :param watermarked_img4embed: The watermarked image section for embedding.
        :param marked_img4locmap: The modified location map.
        :return: The final merged watermarked image.
        """
        watermarked_img = np.zeros(shape=self.img_size)  # Initialize the watermarked image
        watermarked_img[:self.height_end, :, :] = marked_img4locmap  # Place location map in the appropriate section
        watermarked_img[self.height_end:, :, :] = watermarked_img4embed  # Place embedded image in the appropriate section
        return watermarked_img

    def reversible_embed(self, shifted_pe: ndarray, min_v, max_v, set_mask: ndarray, wm_list: list):
        """
        Embed the watermark into the image using the prediction error and predicted values.

        :param set_mask:
        :param max_v:
        :param min_v:
        :param shifted_pe: The shifted prediction error.
        :param pv: The predicted values.
        :param wm_list: The list of watermark bits to embed.
        :return:
        """
        wm_index = 0  # Initialize watermark index
        H, W, C = shifted_pe.shape
        embedded_pe = shifted_pe.copy()
        stopcoordinate = (0, 0, 0)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):  # Iterate over channels
                    stopcoordinate = (h, w, c)
                    if set_mask[h, w, c] == 1:
                        if shifted_pe[h, w, c] == max_v:  # Only embed watermark if prediction error is zero
                            embedded_pe[h, w, c] = shifted_pe[h, w, c] + wm_list[wm_index]
                            wm_index += 1
                        elif shifted_pe[h, w, c] == min_v:  # Embed if the prediction error is -1
                            embedded_pe[h, w, c] = shifted_pe[h, w, c] - wm_list[wm_index]
                            wm_index += 1
                        if wm_index == len(wm_list):
                            return embedded_pe, stopcoordinate, None  # Successfully embedded all watermark bits
        # If loop finishes without completing the embedding, return failure
        return embedded_pe, stopcoordinate, wm_list[wm_index:]

    def get_top_two_frequent_values(self, pe: ndarray, set_mask: ndarray):
        """
        Get the top two most frequent values and their counts from the input array.

        :param set_mask:
        :param pe: Input ndarray, can contain NaN values.
        :return: A tuple containing the two most frequent values and their corresponding counts.
                 Returns (first_most, first_count, second_most, second_count).
        """

        pe_flat = pe[1:-1, 1:-1, :][set_mask[1:-1, 1:-1, :] == 1]  # Remove NaN values
        unique, counts = np.unique(pe_flat, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
        top_two_indices = sorted_indices[:2]  # Get the top two indices
        first_most = unique[top_two_indices[0]] if len(top_two_indices) > 0 else None
        first_count = counts[top_two_indices[0]] if len(top_two_indices) > 0 else None
        second_most = unique[top_two_indices[1]] if len(top_two_indices) > 1 else None
        second_count = counts[top_two_indices[1]] if len(top_two_indices) > 1 else None
        return first_most, first_count, second_most, second_count

    def shift_predicting_error(self, pe: ndarray, set_mask: ndarray):
        """
        Shift the predicting error to ensure it remains within valid pixel value range.

        :param set_mask:
        :param pe: The prediction error.
        :param pv: The predicted values.
        :return: The shifted prediction error and the location map.
        """
        first_most, first_count, second_most, second_count = self.get_top_two_frequent_values(pe, set_mask)
        capacity = first_count + second_count
        min_value = min(first_most, second_most)
        max_value = max(first_most, second_most)
        shifted_pe = np.copy(pe)  # Copy prediction error
        H, W, C = pe.shape
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1:
                        if pe[h, w, c] < min_value:  # Shift positive prediction errors
                            shifted_pe[h, w, c] = pe[h, w, c] - 1
                        if pe[h, w, c] > max_value:  # Shift positive prediction errors
                            shifted_pe[h, w, c] = pe[h, w, c] + 1
        return shifted_pe, int(min_value), int(max_value), capacity

    def compute_stego_img_and_location_map(self, pe_1: ndarray, pe_2: ndarray, pv_1: ndarray, pv_2: ndarray):
        """

        :param pe_1:
        :param pe_2:
        :param pv_1:
        :param pv_2:
        :return:
        """
        stego_img = (pe_1 + pv_1) * self.mask_o + (pe_2 + pv_2) * self.mask_x
        location_map = ((stego_img > 255) | (stego_img < 0)).astype(np.uint8)
        return stego_img, location_map

    def shift_predicting_error_reversibly(self, pe: ndarray, min_v: int, max_v: int):
        """
        Shift the predicting error to ensure it remains within valid pixel value range.

        :param max_v:
        :param min_v:
        :param pe: The prediction error.
        :param pv: The predicted values.
        :return: The shifted prediction error and the location map.
        """
        recovered_pe = pe.copy()
        H, W, C = pe.shape
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if pe[h, w, c] < min_v:  # Shift positive prediction errors
                        recovered_pe[h, w, c] = pe[h, w, c] + 1
                    if pe[h, w, c] > max_v:  # Shift positive prediction errors
                        recovered_pe[h, w, c] = pe[h, w, c] - 1
        return recovered_pe

    def extract_lsb0img(self, img4locmap: ndarray) -> list:
        """
        Extract the least significant bits (LSB) from the location map image.

        :param img4locmap: The image containing the location map.
        :return: A list of extracted least significant bits.
        """
        # img4locmap uint 8
        img4locmap_uint8 = img4locmap.astype(np.uint8)

        # Create a mask to extract the N least significant bits
        mask = (1 << self.bit_plane) - 1

        # Extract the N least significant bits
        lsb_bits = img4locmap_uint8 & mask

        # Convert each bit position into a separate axis for better representation
        bitstream = np.unpackbits(lsb_bits[..., None], axis=-1, bitorder='big')[:, :, :, -self.bit_plane:]
        return bitstream.flatten().tolist()

    def encodeIntegerbyGivenLength(self, n: int, length: int) -> list:
        """
        Encode an integer as a binary list of a given length, supporting both positive and negative integers.

        :param n: The integer to encode.
        :param length: The desired length of the binary representation.
        :return: A list representing the binary encoding of the integer.
        """
        # Check if the number is negative
        if n < 0:
            n = (1 << length) + n  # Convert to 2's complement for negative numbers
        binary_representation = bin(n)[2:]  # Get binary representation without '0b' prefix
        if len(binary_representation) < length:
            binary_representation = binary_representation.zfill(length)  # Pad with zeros to the left
        elif len(binary_representation) > length:
            raise ValueError("The number cannot be represented in the given length")
        return [int(bit) for bit in binary_representation]  # Convert to a list of integers

    def decodeIntegerbyGivenBits(self, bits: list) -> int:
        """
        Decode a binary list back into an integer, considering the possibility of negative numbers.

        :param bits: The binary representation as a list of bits.
        :return: The decoded integer.
        """
        bit_string = ''.join(map(str, bits))  # Convert list of bits to a string
        value = int(bit_string, 2)  # Convert binary string to integer
        # Check if the number is negative (if the sign bit is set)
        if bits[0] == 1:
            value -= (1 << len(bits))  # Convert to negative value
        return value  # Return the decoded integer

    def encode_auxiliary_information(self, location_map: ndarray, time4embedding: int,
                                     stopcoordinate: Tuple[int, int, int],
                                     min_v_1: int, max_v_1: int, min_v_2: int = 0, max_v_2: int = 0, flag: int = 0):
        """
        Compute the auxiliary information required for the reversible data hiding process.

        :param time4embedding:
        :param location_map: The location map indicating where data has been embedded.
        :param stopcoordinate: The coordinates indicating where the embedding stops.
        :return: A list of bits representing the auxiliary information.
        """
        # Compress the location map.
        bits4locmap = self.customAC.compress(location_map, ["1", "0"])
        # time for embedding
        bits4time = self.encodeIntegerbyGivenLength(time4embedding, self.time_len)  # Encode frequencies

        # value for embedding
        bits4minpeak1 = self.encodeIntegerbyGivenLength(min_v_1, self.peak_len)  # Encode frequencies
        bits4maxpeak1 = self.encodeIntegerbyGivenLength(max_v_1, self.peak_len)  # Encode frequencies
        bits4minpeak2 = self.encodeIntegerbyGivenLength(min_v_2, self.peak_len)  # Encode frequencies
        bits4maxpeak2 = self.encodeIntegerbyGivenLength(max_v_2, self.peak_len)  # Encode frequencies
        bits4peak = bits4minpeak1 + bits4maxpeak1 + bits4minpeak2 + bits4maxpeak2

        # Convert the stop coordinate to bits.
        bits4stopheight = self.encodeIntegerbyGivenLength(stopcoordinate[0], self.h_len)  # Height
        bits4stopwidth = self.encodeIntegerbyGivenLength(stopcoordinate[1], self.w_len)  # Width
        bits4stopchannel = self.encodeIntegerbyGivenLength(stopcoordinate[2], self.c_len)  # Channel
        bits4stopcoordinate = bits4stopheight + bits4stopwidth + bits4stopchannel  # Merge stop coordinate bits

        # Convert the stop coordinate to bits.
        bits4flag = self.encodeIntegerbyGivenLength(flag, self.flag_len)  # Height

        # Merge all bits into a final list, ensuring sufficient capacity.
        final_merged_list = bits4stopcoordinate + bits4time + bits4peak + bits4flag + bits4locmap + [1]  # Append end marker
        supp_zeros = [0 for _ in range(self.storage_len - len(final_merged_list))]  # Fill with zeros if necessary
        bits4auxinfo = final_merged_list + supp_zeros  # Create final auxiliary information list
        return bits4auxinfo

    def decode_auxiliary_information(self, lsb_bits: list):
        """
        Decomposing auxiliary information
        :param lsb_bits:
        :return:
        """
        # Remove redundant information (0)
        _bits4auxinfo = lsb_bits.copy()
        index = len(_bits4auxinfo) - 1 - _bits4auxinfo[::-1].index(1)
        bits4auxinfo = _bits4auxinfo[:index]

        # recovery stop coordinate
        len0stopcoordinate = self.h_len + self.w_len + self.c_len
        bits4stopcoordinate = bits4auxinfo[:len0stopcoordinate]
        h = self.decodeIntegerbyGivenBits(bits4stopcoordinate[:self.h_len])
        w = self.decodeIntegerbyGivenBits(bits4stopcoordinate[self.h_len:self.h_len + self.w_len])
        c = self.decodeIntegerbyGivenBits(bits4stopcoordinate[self.h_len + self.w_len:])
        stopcoordinate = (h, w, c)
        remaining_bit = bits4auxinfo[len0stopcoordinate:]

        # time for embedding
        bits4time = remaining_bit[:self.time_len]
        time4embedding = self.decodeIntegerbyGivenBits(bits4time)
        remaining_bit = remaining_bit[self.time_len:]

        # recovery peak points
        bits4peak = remaining_bit[:self.peak_len * 4]
        min_v_1 = self.decodeIntegerbyGivenBits(bits4peak[:self.peak_len])  # Encode frequencies
        max_v_1 = self.decodeIntegerbyGivenBits(bits4peak[self.peak_len:self.peak_len * 2])  # Encode frequencies
        min_v_2 = self.decodeIntegerbyGivenBits(bits4peak[self.peak_len * 2:self.peak_len * 3])  # Encode frequencies
        max_v_2 = self.decodeIntegerbyGivenBits(bits4peak[self.peak_len * 3:self.peak_len * 4])  # Encode frequencies
        remaining_bit = remaining_bit[self.peak_len * 4:]

        # recovery flag
        bits4flag = remaining_bit[:self.flag_len]
        flag = self.decodeIntegerbyGivenBits(bits4flag)  # Encode frequencies
        remaining_bit = remaining_bit[self.flag_len:]

        # freq
        location_map = self.customAC.decompress(remaining_bit, ["1", "0"])
        # location_map = np.uint8(_location_map).reshape(reshape_shape)
        return list(location_map), time4embedding, stopcoordinate, min_v_1, max_v_1, min_v_2, max_v_2, flag

    def embed_bits2imgbylsb(self, img4locmap: ndarray, wm_bits: list):
        """
        Embed the watermark bits into the least significant bits of the location map image.

        :param img4locmap: The image containing the location map.
        :param wm_bits: The list of watermark bits to embed.
        :return: The image with embedded watermark bits.
        """
        wm_bits_shape = img4locmap.shape + (self.bit_plane,)
        max_len = wm_bits_shape[0] * wm_bits_shape[1] * wm_bits_shape[2] * wm_bits_shape[3]
        if len(wm_bits) != wm_bits_shape[0] * wm_bits_shape[1] * wm_bits_shape[2] * wm_bits_shape[3]:
            print("the capacity for location map is insufficient, reversible embedding is fail!")
            wm_bits_array = np.array(wm_bits[:max_len], dtype=np.uint8).reshape(wm_bits_shape)  # Reshape watermark bits to match image
            isfail = True
        else:
            wm_bits_array = np.array(wm_bits, dtype=np.uint8).reshape(wm_bits_shape)  # Reshape watermark bits to match image
            isfail = False
        decimal_values = np.sum(wm_bits_array * (2 ** np.arange(self.bit_plane)[::-1]), axis=-1)
        img4locmap_uint8 = np.array(img4locmap, dtype=np.uint8)
        # Step 1: Clear the least significant two bits of img
        mask = (0b11111111 << self.bit_plane) & 0b11111111  # Keep mask as an integer
        img_cleared = img4locmap_uint8 & mask  # Perform bitwise AND operation
        # Step 3: Embed aux_2bit into the cleared img
        embedded_img = img_cleared + decimal_values
        return isfail, embedded_img

    def embed(self, cover_img: ndarray, watermark_list: list):
        """
        Embed a watermark into the cover image through iterative embedding.

        :param cover_img: The original cover image to embed the watermark into (H, W, C).
        :param watermark_list: A list of bits representing the watermark to embed.
        :return:
            True: Indicating the embedding process has completed.
            marked_img: The image with the embedded watermark (H, W, C).
        """
        isDone = False  # Flag to track when embedding is complete
        time4embedding = 0  # Counter to track the number of embedding iterations
        marked_img = cover_img.astype(np.float32)  # Convert cover image to float64 to avoid overflow/underflow during embedding
        if cover_img.ndim == 2:
            marked_img = marked_img.reshape((*marked_img.shape, 1))

        watermark_list_next = watermark_list.copy()
        all_len = len(watermark_list)
        retain_watermark_len = all_len

        # Iteratively embed watermark bits into the cover image
        while not isDone:
            # Embed watermark bits in the image in one pass
            # 'isDone' indicates if the entire watermark has been embedded
            # 'watermark_list' is updated as bits are embedded
            watermark_list_now = watermark_list_next.copy()
            isDone, marked_img, watermark_list_next = self.embed_once(marked_img, watermark_list_now, time4embedding)
            retain_watermark_len = len(watermark_list_next)
            if len(watermark_list_now) < len(watermark_list_next):
                break

            print(f"{time4embedding}-th embedding, retained bit length: {len(watermark_list_next)}")
            # Increment the embedding iteration counter
            time4embedding += 1
        capacity = (all_len - retain_watermark_len)
        if marked_img.shape[2] == 1:
            marked_img = marked_img[:, :, 0]
        # Return True to indicate successful embedding and convert the marked image back to uint8
        return capacity, np.uint8(np.clip(marked_img, 0, 255))

    def embed_once(self, cover_img: ndarray, watermark_list: list, time4embedding: int = 0):
        """
        Embed a watermark into the cover image using reversible data hiding techniques.

        :param cover_img: The cover image as a NumPy array.
        :param watermark_list: The list of watermark bits to embed.
        :return: A tuple indicating success and the resulting watermarked image.
        """
        img4embed, img4locmap = self.split_img(cover_img)
        bits4lsb0img = self.extract_lsb0img(img4locmap)
        pe_1, pe_2, pv_1, pv_2 = self.predicting_error(img4embed)
        shifted_pe_1, min_v_1, max_v_1, capacity = self.shift_predicting_error(pe_1, self.mask_o)
        embedded_pe_1, stopcoordinate_1, rest_wm_1 = self.reversible_embed(shifted_pe_1, min_v_1, max_v_1, self.mask_o, bits4lsb0img + watermark_list)
        print(f"{time4embedding}-th embedding, the capacity is {capacity}...")
        stego_img4embed_1, location_map_1 = self.compute_stego_img_and_location_map(embedded_pe_1, pe_2, pv_1, pv_2)
        pe_1, pe_2, pv_1, pv_2 = self.predicting_error(stego_img4embed_1)
        if rest_wm_1 is None:
            # ndarray_map = self.remove_redundant_nonoverflow(stopcoordinate_1, location_map_1, None)
            # location_map_1_ = self.remove_redundant_nonoverflow_reversibly(ndarray_map)
            # print(np.array_equal(location_map_1_, location_map_1))
            print("Embed Auxiliary information:", time4embedding, stopcoordinate_1, min_v_1, max_v_1, 0)
            bits4auxinfo = self.encode_auxiliary_information(np.asarray(location_map_1), time4embedding, stopcoordinate_1, min_v_1, max_v_1, flag=0)
            isfail, marked_img4locmap = self.embed_bits2imgbylsb(img4locmap, bits4auxinfo)
            watermarked_img = self.merge_img(stego_img4embed_1, marked_img4locmap)
            return True, np.clip(watermarked_img, 0, 255), []
        else:
            shifted_pe_2, min_v_2, max_v_2, capacity = self.shift_predicting_error(pe_2, self.mask_x)
            embedded_pe_2, stopcoordinate_2, rest_wm_2 = self.reversible_embed(shifted_pe_2, min_v_2, max_v_2, self.mask_x, rest_wm_1)
            stego_img4embed_2, location_map_2 = self.compute_stego_img_and_location_map(pe_1, embedded_pe_2, pv_1, pv_2)
            location_map = location_map_1 | location_map_2
            print("Embed Auxiliary information:", time4embedding, stopcoordinate_2, min_v_1, max_v_1, min_v_2, max_v_2, 1)
            bits4auxinfo = self.encode_auxiliary_information(np.asarray(location_map), time4embedding, stopcoordinate_2,
                                                             min_v_1, max_v_1, min_v_2, max_v_2, flag=1)
            isfail, marked_img4locmap = self.embed_bits2imgbylsb(img4locmap, bits4auxinfo)
            watermarked_img = self.merge_img(stego_img4embed_2, marked_img4locmap)
            if rest_wm_2 is None:
                return True, np.clip(watermarked_img, 0, 255), []
            else:
                return False, np.clip(watermarked_img, 0, 255), rest_wm_2

    def extract(self, stego_img: ndarray):
        """
        Extract the embedded watermark bits from the stego image through iterative extraction.

        :param stego_img: The stego image from which the watermark is to be extracted (H, W, C).
        :return:
            stego_img: The stego image after processing (H, W, C).
            wm_list: The list of extracted watermark bits (in reverse order of extraction).
        """
        wm_list = []  # Initialize an empty list to store the extracted watermark bits
        isDone = False  # Flag to determine when the extraction process is complete
        if stego_img.ndim == 2:
            stego_img = stego_img.reshape((*stego_img.shape, 1))
        stego_img = stego_img.astype(np.float64)  # Convert the stego image to float64 to avoid data overflow/underflow
        index = 0
        # Iteratively extract watermark bits from the stego image
        while not isDone:
            # Extract watermark bits from the image in one pass
            # 'isDone' indicates if extraction is complete
            # 'extract_bits' contains the watermark bits from the current extraction pass
            print(f"{index}-th extraction")
            isDone, stego_img, extract_bits = self.extract_once(stego_img)
            index += 1
            # Prepend the extracted bits to the beginning of 'wm_list' to maintain correct bit order
            wm_list = extract_bits + wm_list

        # Return the modified stego image and the full list of extracted watermark bits
        return stego_img, wm_list

    def extract_once(self, stego_img: ndarray):
        """
        Embed a watermark into the cover image using reversible data hiding techniques.

        :param stego_img: The stego image as a NumPy array.
        :return: A tuple indicating success and the resulting watermarked image.
        """
        img0stego, img4locmap = self.split_img(stego_img)
        bits4lsb0img = self.extract_lsb0img(img4locmap)
        location_map_list, time4embedding, stopcoordinate, min_v_1, max_v_1, min_v_2, max_v_2, flag = self.decode_auxiliary_information(
            bits4lsb0img)
        print("Extract Auxiliary information:", time4embedding, stopcoordinate, min_v_1, max_v_1, min_v_2, max_v_2, flag)
        recovered_img0stego = self.recovery_overflowed_stego_img(img0stego, np.asarray(location_map_list).reshape(img0stego.shape))
        pe_1, pe_2, pv_1, pv_2 = self.predicting_error(recovered_img0stego)
        isDone = True if time4embedding == 0 else False
        if flag == 1:
            wm_list_2, recovered_embedded_pe_2 = self.reversible_extract(pe_2, min_v_2, max_v_2, self.mask_x,
                                                                         stopcoordinate)
            recovered_pe_2 = self.shift_predicting_error_reversibly(recovered_embedded_pe_2, min_v_2, max_v_2)
            recovered_img4embed_2, _ = self.compute_stego_img_and_location_map(pe_1, recovered_pe_2, pv_1, pv_2)
            pe_1, pe_2, pv_1, pv_2 = self.predicting_error(recovered_img4embed_2)
            wm_list_1, recovered_embedded_pe_1 = self.reversible_extract(pe_1, min_v_1, max_v_1, self.mask_o,
                                                                         (-1, -1, -1))
            recovered_pe_1 = self.shift_predicting_error_reversibly(recovered_embedded_pe_1, min_v_1, max_v_1)
            recovered_img4embed, _ = self.compute_stego_img_and_location_map(recovered_pe_1, pe_2, pv_1, pv_2)
            wm_list = wm_list_1 + wm_list_2
        else:
            wm_list, recovered_embedded_pe_1 = self.reversible_extract(pe_1, min_v_1, max_v_1, self.mask_o,
                                                                       stopcoordinate)
            recovered_pe_1 = self.shift_predicting_error_reversibly(recovered_embedded_pe_1, min_v_1, max_v_1)
            recovered_img4embed, _ = self.compute_stego_img_and_location_map(recovered_pe_1, pe_2, pv_1, pv_2)
        _, recovered_img4locmap = self.embed_bits2imgbylsb(img4locmap, wm_list[:self.storage_len])
        recovered_cover_img = self.merge_img(recovered_img4embed, recovered_img4locmap)
        extracted_bits = wm_list[self.storage_len:]
        return isDone, recovered_cover_img, extracted_bits

    def recovery_overflowed_stego_img(self, img0stego: ndarray, location_map: ndarray):
        """
        Recover pixels in the stego image that have overflowed or underflowed.

        :param img0stego: The stego image with possible overflow/underflow (H, W, C).
        :param location_map: A binary map indicating which pixels might have overflowed/underflowed (H, W, C).
        :return: A recovered stego image where overflowed/underflowed pixels have been adjusted.
        """
        shifted_cover_img = img0stego.copy()  # Create a copy of the stego image to modify
        H, W, C = location_map.shape  # Get the dimensions of the location map

        # Loop through each pixel, avoiding the border pixels (h = 1 to H-2, w = 1 to W-2)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # Check for overflow/underflow conditions
                    if location_map[h, w, c] == 1 and img0stego[h, w, c] == 0:
                        shifted_cover_img[h, w, c] -= 1  # Recover from underflow (0 → -1)
                    if location_map[h, w, c] == 1 and img0stego[h, w, c] == 255:
                        shifted_cover_img[h, w, c] += 1  # Recover from overflow (255 → 256)

        return shifted_cover_img  # Return the recovered image

    def reversible_extract(self, pe, min_v, max_v, set_mask: ndarray, stopcoordinate):
        """
        Extract the embedded watermark bits from the modified pixel values and recover the original image.

        :param pe: The image containing embedded watermark (H, W, C).
        :param min_v: The minimum pixel value to consider for recovery.
        :param max_v: The maximum pixel value to consider for recovery.
        :param set_mask: A binary mask indicating which pixels contain embedded data (H, W, C).
        :param stopcoordinate: The pixel coordinate at which to stop the extraction process.
        :return:
            wm_list: The list of extracted watermark bits (0 or 1).
            recovered_pe: The image with recovered pixel values.
        """
        H, W, C = pe.shape  # Get the dimensions of the input image
        wm_list = []  # List to store extracted watermark bits
        recovered_pe = pe.copy()  # Create a copy of the input image to recover original pixel values

        # Loop through each pixel, avoiding the border pixels (h = 1 to H-2, w = 1 to W-2)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # Check if the current pixel contains embedded data
                    if set_mask[h, w, c] == 1:
                        # Extract watermark bit and recover original pixel value based on modified pixel value
                        if pe[h, w, c] == min_v:
                            recovered_pe[h, w, c] = pe[h, w, c]  # No change needed, bit is 0
                            wm_list.append(0)
                        elif pe[h, w, c] == min_v - 1:
                            recovered_pe[h, w, c] = pe[h, w, c] + 1  # Recover original value, bit is 1
                            wm_list.append(1)
                        elif pe[h, w, c] == max_v:
                            recovered_pe[h, w, c] = pe[h, w, c]  # No change needed, bit is 0
                            wm_list.append(0)
                        elif pe[h, w, c] == max_v + 1:
                            recovered_pe[h, w, c] = pe[h, w, c] - 1  # Recover original value, bit is 1
                            wm_list.append(1)

                    # Stop extraction when reaching the stop coordinate
                    if (h, w, c) == stopcoordinate:
                        return wm_list, recovered_pe  # Return the extracted bits and recovered image

        return wm_list, recovered_pe  # Return the watermark bits and fully recovered image

    def remove_redundant_nonoverflow(self, stop_corrdinate, location_map_1: ndarray = None,
                                     location_map_2: ndarray = None):
        """
        Remove redundant non-overflow data in the specified region up to the stop coordinate.

        :param stop_corrdinate: Coordinate at which to stop the removal process.
        :param location_map_1: The first location map (H, W, C).
        :param location_map_2: The second location map (H, W, C).
        :return:
            overflow_map: An array of overflow data up to the stop coordinate.
        """
        overflow_map = []  # List to store overflow data
        H, W, C = location_map_1.shape  # Dimensions of the location maps

        # Loop through each pixel in the image (avoiding borders)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # If the pixel is marked in mask_o, append its value from location_map_1 to overflow_map
                    if self.mask_o[h, w, c] == 1:
                        overflow_map.append(location_map_1[h, w, c])
                    # Stop the process when reaching the stop coordinate
                    if (h, w, c) == stop_corrdinate:
                        return np.asarray(overflow_map).astype(np.uint8)

        # Continue through location_map_2 if stop coordinate is not reached
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # If the pixel is marked in mask_x, append its value from location_map_2 to overflow_map
                    if self.mask_x[h, w, c] == 1:
                        overflow_map.append(location_map_2[h, w, c])
                    # Stop the process when reaching the stop coordinate
                    if (h, w, c) == stop_corrdinate:
                        return np.asarray(overflow_map).astype(np.uint8)

        # Return combined location maps if no early stopping
        return location_map_1 | location_map_2

    def remove_redundant_nonoverflow_reversibly(self, bits4location_map):
        """
        Fill in redundant data from a bit sequence into the location map in a reversible way.

        :param bits4location_map: A list of bits for filling the location map.
        :return:
            location_map: The location map filled with the given bit sequence.
        """
        index = 0  # Initialize index to iterate through bits4location_map
        location_map = np.zeros_like(self.mask_o)  # Create an empty location map of the same shape as mask_o
        H, W, C = self.mask_o.shape  # Dimensions of the mask

        # First loop through mask_o to fill location map with bits4location_map values
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # If the current pixel is marked in mask_o and bits remain in the list
                    if self.mask_o[h, w, c] == 1 and index < len(bits4location_map):
                        location_map[h, w, c] = bits4location_map[index]  # Set location map pixel to the current bit
                        index += 1  # Move to the next bit

        # Second loop through mask_o to continue filling location map, if needed
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    # Again, check mask_o and if bits are left in the sequence
                    if self.mask_o[h, w, c] == 1 and index < len(bits4location_map):
                        location_map[h, w, c] = bits4location_map[index]  # Set location map pixel to the current bit
                        index += 1  # Move to the next bit

        return location_map  # Return the fully populated location map


class CustomRDH:
    def __init__(self, img_size: Tuple[int, int, int], height_end: int = 5, device: str = "cpu"):
        """
        Initializes the CustomRDH class for reversible data hiding (RDH).

        :param img_size: The dimensions of the image as (Height, Width, Channels), e.g., (256, 256, 3).
        :param height_end: The height limit for embedding the watermark, default is 5. This controls
                           how many rows from the top will be used for embedding.
        :param device: Specifies the computation device ("cuda" for GPU or "cpu").
        """
        self.device = device
        self.rdh = RDH(img_size, height_end=height_end)

    def embed(self, cover_img: Tensor, watermark_list: list):
        """
        Embeds a watermark into the cover image using reversible data hiding.

        :param cover_img: The cover image as a tensor of shape (N, C, H, W), where N=1 (one image),
                          C=channels, H=height, and W=width.
        :param watermark_list: List of bits or data to embed within the cover image.
        :return: A tuple containing:
                 - capacity: The capacity used in the image for embedding (measured in bits or similar units).
                 - rdh_stego_img: The output stego image containing the embedded watermark.
        """
        print(f"Auxiliary information bits length: {len(watermark_list)}")
        N, C, H, W = cover_img.shape
        assert N == 1  # Ensure a single image in the batch
        # Convert the tensor to a NumPy array for RDH embedding
        cover_img_numpy = cover_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # Embed watermark using RDH method
        capacity, rdh_stego_img = self.rdh.embed(cover_img_numpy, watermark_list)
        return capacity, rdh_stego_img

    def extract(self, rdh_stego_img: ndarray):
        """
        Extracts the embedded watermark and reconstructs the cover image.

        :param rdh_stego_img: The stego image (NumPy ndarray) with the embedded watermark.
        :return: A tuple containing:
                 - rec_stego_img_tensor: The reconstructed stego image as a tensor on the specified device.
                 - wm_list: The extracted watermark list.
        """
        # Use RDH method to extract watermark and reconstruct the stego image
        rec_stego_img, wm_list = self.rdh.extract(rdh_stego_img)
        # Convert the reconstructed image to a tensor and move it to the specified device
        rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return rec_stego_img_tensor.to(self.device), wm_list


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    :param original: The original image as a numpy ndarray.
    :param compressed: The compressed/reconstructed image as a numpy ndarray.
    :return: The PSNR value in decibels (dB).
    """
    # Ensure the two images have the same shape
    assert original.shape == compressed.shape, "Input images must have the same dimensions"

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original - compressed) ** 2)

    # If MSE is zero, return infinity (no error)
    if mse == 0:
        return float('inf')

    # MAX value for 8-bit images
    max_value = 255.0

    # Calculate PSNR
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr


def test4grayimage(img_path):
    # Open the cover image in grayscale mode
    cover_img = Image.open(img_path).convert("L")  # Convert to grayscale
    cover_img = np.float32(cover_img)
    org = np.float32(Image.open("images/cover.png").convert("L"))
    cover_img_np = cover_img.reshape(cover_img.shape[0], cover_img.shape[1], 1)  # Reshape for single-channel

    # Initialize RDH with the image size
    rdh = RDH(img_size=cover_img_np.shape, height_end=20)

    # Generate a random watermark
    watermark_length = 40000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark into the cover image
    capacity, stego_img = rdh.embed(cover_img_np, watermark_list)
    # stego_img[0:20, :] = cover_img[0:20, :]
    # Save the watermarked image
    residual = np.abs(stego_img - org)
    residual_save = (residual - np.min(residual)) / (np.max(residual) - np.min(residual)) * 255
    Image.fromarray(np.uint8(stego_img)).save('images/stego_img_gray.png')
    Image.fromarray(np.uint8(residual_save)).save('images/residual_gray.png')
    reload_stego_img = np.float32(Image.open('images/stego_img_gray.png'))

    # Extract the watermark from the stego image
    recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # Check if recovered image and watermark match the originals
    print(np.array_equal(recovered_cover_img, cover_img_np), np.array_equal(watermark_list, wm_list))

    # Calculate and print PSNR
    print(f"PSNR: {calculate_psnr(cover_img, stego_img)}")
    print("Watermark embedding successful.")


def test4rgbimage(img_path):
    # Open the cover image in grayscale mode
    cover_img = Image.open(img_path)  # Convert to grayscale
    cover_img_np = np.float32(cover_img)

    # Initialize RDH with the image size
    rdh = RDH(img_size=cover_img_np.shape)

    # Generate a random watermark
    watermark_length = 60000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark into the cover image
    capacity, stego_img = rdh.embed(cover_img_np, watermark_list)

    # Save the watermarked image
    Image.fromarray(np.uint8(stego_img)).save('images/stego_img_rgb.tif')
    reload_stego_img = np.float32(Image.open('images/stego_img_rgb.tif'))

    # Extract the watermark from the stego image
    recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # Check if recovered image and watermark match the originals
    print(np.array_equal(recovered_cover_img, cover_img_np), np.array_equal(watermark_list, wm_list))

    # Calculate and print PSNR
    print(f"PSNR: {calculate_psnr(cover_img_np, stego_img)}")
    print("Watermark embedding successful.")

if __name__ == "__main__":
    # test4rgbimage('images/cover.png')
    test4grayimage('images/stego.png')
