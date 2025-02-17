<div align="center">
<h1>Deep Robust Reversible Watermarking</h1>

Jiale Chen<sup>1</sup>, Wei Wang<sup>2,3</sup>, Chongyang Shi<sup>1</sup>, Li Dong<sup>4</sup>, Yuanman Li<sup>5</sup>, Xiping Hu<sup>2,3</sup>

<sup>1</sup>School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China  
<sup>2</sup>School of Medical Technology, Beijing Institute of Technology, Beijing, China  
<sup>3</sup>Shenzhen MSU-BIT University, Shenzhen, China  
<sup>4</sup>Department of Computer Science and Technology, Ningbo University, Ningbo, China  
<sup>5</sup>College of Electronics and Information Engineering, Shenzhen University, Shenzhen, China  
</div>

![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📝 Introduction

This repository contains the official implementation of **Deep Robust Reversible Watermarking (DRRW)**, a robust and cover-recoverable invisible image watermarking method. DRRW enables perfect reconstruction of the original cover image in lossless channels and robust watermark extraction in lossy channels.

DRRW leverages an **Integer Discrete Invertible Watermark Network (IDIWN)** to achieve lossless and invertible mapping between cover-watermark pairs and stego images. It addresses the trade-off between robustness and reversibility in traditional robust reversible watermarking methods, offering significant improvements in robustness, visual quality, and computational efficiency.

Key features:
- **Robustness**: Enhanced against distortions through an Encoder-Noise Layer-Decoder framework.
- **Reversibility**: Ensures lossless recovery of both the cover image and the watermark in lossless channel.
- **Efficiency**: Reduces time complexity and auxiliary bitstream length.

---

## 🚀 Training

To train the DRRW model, use the following command:

```bash
python train_color.py
```

## 🚀 Evaluation

To test the DRRW model, use the following command:

```bash
python test.py
```

## 🚀 WatermarkLab

For evaluation and distortion layers, we utilize **WatermarkLab**, a comprehensive watermarking evaluation framework. WatermarkLab provides a wide range of distortion simulations and evaluation metrics to rigorously test the robustness and performance of watermarking algorithms.

You can find the WatermarkLab repository here: [github/chenoly/watermarklab](https://github.com/chenoly/watermarklab)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
