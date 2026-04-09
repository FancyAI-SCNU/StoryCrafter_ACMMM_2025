import os
import random
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def natural_sort(items):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def sort_key(key):
        return [convert(chunk) for chunk in re.split(r"([0-9]+)", key)]

    return sorted(items, key=sort_key)


class StorySalonDataset(Dataset):
    def __init__(self, root, dataset_name):
        self.root = root
        self.dataset_name = dataset_name
        self.keys = ["African", "Bloom", "Book", "Digital", "Literacy", "StoryWeaver"]

        self.image_base = os.path.join(root, "Image_inpainted")
        self.mask_base = os.path.join(root, "Mask_ins")
        self.text_base = os.path.join(root, "Text", "Caption")

        self.test_story_ids = []
        test_split_path = os.path.join(root, "PDF_test_set.txt")
        if os.path.exists(test_split_path):
            with open(test_split_path, "r") as handle:
                self.test_story_ids = [line.strip() for line in handle if line.strip()]
        else:
            print(f"Warning: test split file not found at {test_split_path}. Treating all stories as training data.")

        self.samples = []
        self.story_sequences = []
        self._build_index()

    def _build_index(self):
        for key in self.keys:
            image_dir = os.path.join(self.image_base, key)
            mask_dir = os.path.join(self.mask_base, key)
            text_dir = os.path.join(self.text_base, key)

            if not os.path.isdir(image_dir):
                print(f"Warning: image directory {image_dir} does not exist. Skipping this category.")
                continue
            if not os.path.isdir(text_dir):
                print(f"Warning: text directory {text_dir} does not exist. Skipping this category.")
                continue

            for folder in sorted(os.listdir(image_dir)):
                is_test = folder in self.test_story_ids
                if (self.dataset_name == "train" and is_test) or (self.dataset_name == "test" and not is_test):
                    continue

                story_image_dir = os.path.join(image_dir, folder)
                story_text_dir = os.path.join(text_dir, folder)
                story_mask_dir = os.path.join(mask_dir, folder) if os.path.isdir(mask_dir) else None

                if not os.path.isdir(story_image_dir) or not os.path.isdir(story_text_dir):
                    print(f"Warning: invalid image/text folder for story {key}_{folder}. Skipping.")
                    continue

                image_files = natural_sort(
                    [
                        name
                        for name in os.listdir(story_image_dir)
                        if os.path.isfile(os.path.join(story_image_dir, name)) and name.endswith(".jpg")
                    ]
                )
                text_files = natural_sort(
                    [
                        name
                        for name in os.listdir(story_text_dir)
                        if os.path.isfile(os.path.join(story_text_dir, name)) and name.endswith(".txt")
                    ]
                )

                if not image_files or not text_files:
                    print(f"Warning: insufficient image/text files for story {key}_{folder}. Skipping.")
                    continue
                if len(image_files) != len(text_files):
                    print(
                        f"Warning: image/text count mismatch for story {key}_{folder} "
                        f"({len(image_files)} images vs. {len(text_files)} texts). Skipping."
                    )
                    continue

                story_info = {
                    "story_id": f"{key}_{folder}",
                    "image_folder": story_image_dir,
                    "mask_folder": story_mask_dir,
                    "text_folder": story_text_dir,
                    "image_files": image_files,
                    "text_files": text_files,
                    "total_frames": len(image_files),
                    "start_index": len(self.samples),
                }
                self.story_sequences.append(story_info)

                if self.dataset_name == "train":
                    for index in range(len(image_files) - 1):
                        self.samples.append(
                            {
                                "type": "frame_pair",
                                "story_id": story_info["story_id"],
                                "ref_frame_idx": index,
                                "target_frame_idx": index + 1,
                                "image_folder": story_image_dir,
                                "mask_folder": story_mask_dir,
                                "text_folder": story_text_dir,
                                "image_files": image_files,
                                "text_files": text_files,
                            }
                        )
                else:
                    self.samples.append(
                        {
                            "type": "story_sequence",
                            "story_id": story_info["story_id"],
                            "image_folder": story_image_dir,
                            "mask_folder": story_mask_dir,
                            "text_folder": story_text_dir,
                            "image_files": image_files,
                            "text_files": text_files,
                            "total_frames": len(image_files),
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample["type"] == "frame_pair":
            return self._get_train_item(sample)
        return self._get_test_item(sample)

    def _read_text(self, path, kind):
        if not os.path.exists(path):
            print(f"Warning: {kind} text {path} does not exist. Using an empty prompt.")
            return ""
        with open(path, "r") as handle:
            return handle.read()

    def _load_rgb_tensor(self, path, kind):
        try:
            image = Image.open(path).convert("RGB").resize((512, 512))
            return transforms.ToTensor()(image)
        except Exception as exc:
            print(f"Warning: failed to load {kind} image {path}. Error: {exc}")
            return torch.zeros(3, 512, 512).float()

    def _get_train_item(self, sample):
        ref_idx = sample["ref_frame_idx"]
        target_idx = sample["target_frame_idx"]

        ref_image_file = sample["image_files"][ref_idx]
        ref_image_path = os.path.join(sample["image_folder"], ref_image_file)
        ref_image_group = [ref_image_path]

        ref_subdir = os.path.join(sample["image_folder"], ref_image_file.split(".")[0])
        if os.path.isdir(ref_subdir):
            for sub_image in sorted([name for name in os.listdir(ref_subdir) if name.endswith(".jpg")]):
                ref_image_group.append(os.path.join(ref_subdir, sub_image))

        target_image_file = sample["image_files"][target_idx]
        target_image_path = os.path.join(sample["image_folder"], target_image_file)
        ref_image_group.append(target_image_path)

        mask_group = []
        if sample["mask_folder"] is not None:
            target_mask_dir = os.path.join(sample["mask_folder"], target_image_file.split(".")[0])
            if os.path.isdir(target_mask_dir):
                for sub_mask in sorted([name for name in os.listdir(target_mask_dir) if name.endswith(".png")]):
                    mask_group.append(os.path.join(target_mask_dir, sub_mask))

        ref_text_path = os.path.join(sample["text_folder"], sample["text_files"][ref_idx])
        target_text_path = os.path.join(sample["text_folder"], sample["text_files"][target_idx])
        ref_prompts = self._read_text(ref_text_path, "reference")
        prompt = self._read_text(target_text_path, "target")

        ref_images_processed = []
        for image_path in ref_image_group[:-1]:
            try:
                image = Image.open(image_path).convert("RGB").resize((512, 512))
                ref_images_processed.append(transforms.ToTensor()(image))
            except Exception as exc:
                print(f"Warning: failed to load image {image_path}. Error: {exc}")

        target_image = self._load_rgb_tensor(target_image_path, "target")

        masks_processed = []
        for mask_path in mask_group:
            try:
                mask = Image.open(mask_path).convert("RGB").resize((512, 512))
                masks_processed.append(transforms.ToTensor()(mask))
            except Exception as exc:
                print(f"Warning: failed to load mask {mask_path}. Error: {exc}")

        ref_images = torch.stack(ref_images_processed).float() if ref_images_processed else torch.zeros(1, 3, 512, 512).float()
        masks = torch.stack(masks_processed).float() if masks_processed else torch.zeros(1, 3, 512, 512).float()
        masks = self.apply_dropout_and_mosaic(masks, prob=0.3)

        if random.uniform(0, 1) < 0.05:
            prompt = ""
        if random.uniform(0, 1) < 0.1:
            ref_prompts = ""
            ref_images = ref_images * 0.0

        ref_images = ref_images * 2.0 - 1.0
        target_image = target_image * 2.0 - 1.0

        return {
            "ref_image": ref_images,
            "ref_prompt": ref_prompts,
            "image": target_image,
            "mask": masks,
            "prompt": prompt,
        }

    def _get_test_item(self, sample):
        all_gt_images = []
        all_gt_prompts = []

        for frame_idx in range(len(sample["image_files"])):
            gt_image_file = sample["image_files"][frame_idx]
            gt_image_path = os.path.join(sample["image_folder"], gt_image_file)
            try:
                gt_image = Image.open(gt_image_path).convert("RGB").resize((512, 512))
                gt_image = transforms.ToTensor()(gt_image) * 2.0 - 1.0
            except Exception as exc:
                print(f"Warning: failed to load GT image {gt_image_path}. Using a zero image. Error: {exc}")
                gt_image = torch.zeros(3, 512, 512).float()
            all_gt_images.append(gt_image)

            gt_text_path = os.path.join(sample["text_folder"], sample["text_files"][frame_idx])
            try:
                with open(gt_text_path, "r") as handle:
                    gt_prompt = handle.read()
            except Exception as exc:
                print(f"Warning: failed to load GT text {gt_text_path}. Using an empty prompt. Error: {exc}")
                gt_prompt = ""
            all_gt_prompts.append(gt_prompt)

        ref_images = torch.stack(all_gt_images).float() if all_gt_images else torch.zeros(1, 3, 512, 512).float()

        target_frames_info = []
        for frame_idx, target_image_file in enumerate(sample["image_files"]):
            target_text_path = os.path.join(sample["text_folder"], sample["text_files"][frame_idx])
            try:
                with open(target_text_path, "r") as handle:
                    target_prompt = handle.read()
            except Exception as exc:
                print(f"Warning: failed to load target text {target_text_path}. Using an empty prompt. Error: {exc}")
                target_prompt = ""

            target_frames_info.append(
                {
                    "frame_idx": frame_idx,
                    "prompt": target_prompt,
                    "image_file": target_image_file,
                }
            )

        return {
            "ref_image": ref_images,
            "ref_prompt": all_gt_prompts[0] if all_gt_prompts else "",
            "all_ref_prompts": all_gt_prompts,
            "story_id": sample["story_id"],
            "target_frames": target_frames_info,
            "image_folder": sample["image_folder"],
            "text_folder": sample["text_folder"],
        }

    def apply_dropout_and_mosaic(self, masks, prob=0.3):
        if random.uniform(0, 1) < prob:
            masks = masks * 0.0
        return masks

    def get_story_sequences(self):
        return self.story_sequences


def test_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if key == "all_ref_prompts":
            collated[key] = [item[key] for item in batch]
        elif key in {"ref_image", "image", "mask"}:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated
