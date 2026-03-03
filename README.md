# WuYonghao
**Image Inpainting based on Visual-Neural-Inspired Specific Object-of-Interest Imaging Technology**

> This repository contains an implementation for image inpainting based on a visual-neural-inspired specific object-of-interest imaging pipeline.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Dataset / Download](#dataset--download)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference / Testing](#inference--testing)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
This project focuses on **image inpainting** guided by the concept of **Specific Object-of-Interest (SOI)**, inspired by visual neural mechanisms.  
The goal is to reconstruct missing or corrupted regions in an image while preserving the structure and texture consistency around the target object/region.

**Typical use cases:**
- Remove unwanted objects and fill the area naturally
- Repair damaged images
- Restore missing content while keeping the target region consistent

---

## Features
- End-to-end training pipeline (see `trainModel.py`)
- Supports dataset-driven training (download link below)
- Reproducible experiments with clear instructions

> If you add more scripts later (e.g., `infer.py`, `utils.py`), update the structure section accordingly.

---

## Dataset / Download
The full dataset (~2GB) is hosted on **Google Drive**.

### Full Dataset (Public)
- Google Drive (share page):  
  https://drive.google.com/file/d/10zQYQHpBUjdcDr2g2Zej95Aq_xSbBZtb/view?usp=drive_link

- Google Drive (direct download):  
  https://drive.google.com/uc?export=download&id=10zQYQHpBUjdcDr2g2Zej95Aq_xSbBZtb

### How to place the dataset locally
1. Download and unzip the dataset
2. Put it into this folder structure (example):
