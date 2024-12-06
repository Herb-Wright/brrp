import torch
from transformers import CLIPProcessor, CLIPModel


def crop_and_resize(
    image: torch.Tensor, 
    segmentation: torch.Tensor, 
    target_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """crops and resizes image"""
    segmentations = torch.nn.functional.one_hot(
        segmentation, torch.amax(segmentation) + 1
    ).permute(2, 0, 1)[1:]  # ignore the '0' class
    # Ensure the image and segmentation tensors have the correct shapes
    assert image.dim() == 3 and image.shape[0] == 3, "Image should be in shape (3, H, W)"
    assert segmentations.dim() == 3, "Segmentations should be in shape (N, H, W)"
    cropped_resized_images = []
    # Process each segmentation mask
    for segmentation in segmentations:
        mask_indices = torch.nonzero(segmentation, as_tuple=False)
        y_min, x_min = mask_indices[:, 0].min(), mask_indices[:, 1].min()
        y_max, x_max = mask_indices[:, 0].max(), mask_indices[:, 1].max()
        cropped_image = image[:, y_min:y_max + 1, x_min:x_max + 1]
        resized_image = torch.nn.functional.interpolate(
            cropped_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        )
        cropped_resized_images.append(resized_image.squeeze(0))
    # Return
    return torch.stack(cropped_resized_images)

def run_clip(
    imgs: torch.Tensor, 
    classes: list[str],
    *,
    device: str = "cpu",
    checkpoint = "openai/clip-vit-base-patch16",
):
    model = CLIPModel.from_pretrained(checkpoint).to(device)
    processor = CLIPProcessor.from_pretrained(checkpoint)
    classes = [f"a photo of a {cl}" for cl in classes]
    with torch.no_grad():
        inputs = processor(text=classes, images=imgs.to(device), return_tensors="pt", padding=True, do_rescale=False)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
    return probs
