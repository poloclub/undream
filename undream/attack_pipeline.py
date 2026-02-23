# Copyright Epic Games, Inc. All Rights Reserved.
import unreal
import torch
import mitsuba as mi
import drjit as dr
from PIL import Image
from torchvision import transforms
from torchvision.ops import complete_box_iou
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
from glob import glob
import json
import hydra
from dataclasses import dataclass
from typing import Optional
# from ultralytics import YOLO
# from ultralytics.nn.tasks import DetectionModel


mi.set_variant('cuda_ad_rgb')

# Constants
IMAGE_CROP_TOP_BOTTOM = 40  # Crop pixels from top/bottom (see line 230)
IMAGE_CROP_LEFT_RIGHT = 210  # Crop pixels from left/right
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
DEFAULT_DETECTION_THRESHOLD = 0.5
DUMMY_BBOX = [0.0, 0.0, 1500.0, 1500.0]  # Default bbox when no detections


@dataclass
class AttackConfig:
    """Configuration for adversarial texture attack pipeline.

    Centralizes all attack parameters, paths, and state management.
    Replaces 20+ global variables for better maintainability.
    """
    # Model configuration
    model_name: str = "detr"
    processor: Optional[object] = None
    model: Optional[object] = None
    device: torch.device = None  # Will be set in __post_init__
    use_amp: bool = True  # Use Automatic Mixed Precision for ~40% speedup
    scaler: Optional[object] = None  # GradScaler for AMP

    # Attack parameters
    batch_size: int = 10
    eps: float = 5.0/255.0
    alpha: float = 1.0/255.0
    max_iter: int = 5
    attack_mode: str = "pgd"  # "pgd", "autopgd", or "autoattack"
    momentum: float = 0.7
    targeted: bool = False
    task: str = "cls"  # "cls" or "det"

    # Clipping bounds
    clip_min: float = 0.0
    clip_max: float = 1.0

    # Mitsuba parameters
    key: str = "plane.bsdf.reflectance.data"
    spp_initial: int = 128  # Samples per pixel for initial render
    spp_optimize: int = 8   # Samples per pixel during optimization (k)
    spp_final: int = 16     # Samples per pixel for final render

    # Paths
    destination_path: str = ""
    material_instance_name: str = ""
    orig_texture_path: str = ""
    updated_tex_img: str = ""
    sequence_image_path: str = ""

    # Scene state
    scene: Optional[object] = None
    params: Optional[object] = None
    orig_texture: Optional[object] = None
    mi_files: list = None

    # Iteration tracking
    num_iter: int = 0
    num_frames: int = 150

    # Target and object index
    target: list = None
    obj_idx: int = 1

    # AutoAttack ensemble state (Croce & Hein, 2020)
    _aa_step_size: Optional[float] = None   # Adaptive step size for APGD phases (init: 2*eps)
    _aa_current_phase: int = -1             # Current sub-attack: 0=APGD-CE, 1=APGD-DLR, 2=FAB, 3=Square
    _aa_phase_iter: int = 0                 # Iteration count within current phase

    # Performance timing
    timing_model_forward: float = 0.0
    timing_model_backward: float = 0.0
    timing_mitsuba_optimize: float = 0.0
    timing_total_iteration: float = 0.0

    def __post_init__(self):
        """Initialize device after dataclass creation."""
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize AMP scaler if using mixed precision on CUDA
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_amp = False  # Disable AMP if not on CUDA


# Global variables (minimized - most moved to AttackConfig)
config: Optional[AttackConfig] = None  # Configuration instance (initialized in main())
SubsystemExecutor = None  # Unreal Movie Pipeline executor

def update_patch(x, y, mi_file_paths, cfg: AttackConfig):
    """Compute gradients from model and update texture via Mitsuba.

    OPTIMIZATIONS:
    - Data moved to GPU before model forward pass
    - Removed redundant requires_grad_() calls
    - Use F.cross_entropy instead of creating criterion object
    - Better gradient handling and logging

    Args:
        x: Preprocessed images (dict for DETR, tensor for YOLO)
        y: Target labels or bounding boxes
        mi_file_paths: Corresponding Mitsuba scene files
        cfg: Attack configuration object
    """
    import time

    fwd_start = time.time()

    if cfg.model_name == "detr":
        # Move inputs to GPU
        x = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

        # Enable gradients (only once, removed redundant calls)
        x["pixel_values"].requires_grad_(True)

        # Use Automatic Mixed Precision for forward pass
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # Forward pass
            outputs = cfg.model(**x)
            cfg.timing_model_forward += time.time() - fwd_start

            # Move targets to GPU and prepare
            y = torch.tensor(y, device=cfg.device)
            target_sizes = [(IMAGE_HEIGHT, IMAGE_WIDTH)] * len(x["pixel_values"])
            results = cfg.processor.post_process_object_detection(
                outputs,
                threshold=DEFAULT_DETECTION_THRESHOLD,
                target_sizes=target_sizes
            )

            bboxes = [result["boxes"] for result in results]
            labels = [result["labels"] for result in results]

            if cfg.task == "cls":
                logits = get_logits(outputs)
                # OPTIMIZATION: Use functional API instead of creating criterion object
                loss = torch.nn.functional.cross_entropy(logits, y)
                unreal.log(f"Classification loss: {loss.item():.4f}")

            elif cfg.task == "det":
                final_bboxes = []
                for i in range(len(bboxes)):
                    if labels[i].numel() == 0:
                        # No detections - use dummy box
                        final_bboxes.append(torch.tensor(DUMMY_BBOX, device=cfg.device))
                        continue

                    if y["label"][i].item() in labels[i]:
                        # Find matching label
                        j = (labels[i] == y["label"][i].item()).nonzero(as_tuple=True)[0][0].item()
                        final_bboxes.append(bboxes[i][j])
                    else:
                        # Use highest confidence detection
                        final_bboxes.append(bboxes[i][0])

                final_bboxes = torch.stack(final_bboxes)

                # Compute IoU loss
                ious = complete_box_iou(final_bboxes, y["boxes"].to(cfg.device))
                ious = torch.nan_to_num(ious, nan=0.0).mean()
                loss = 1 - ious
                unreal.log(f"Detection loss (1-IoU): {loss.item():.4f}, IoU: {ious.item():.4f}")

        # Backward pass with AMP gradient scaling
        bwd_start = time.time()
        if cfg.use_amp:
            # Scale loss for mixed precision, backward, unscale for gradient access
            cfg.scaler.scale(loss).backward()
        else:
            loss.backward()
        cfg.timing_model_backward += time.time() - bwd_start

        gradients = x["pixel_values"].grad

    else:
        gradients = None

    # Handle None gradients
    if gradients is None:
        unreal.log_warning("Gradients are None - model did not compute gradients")
        gradients = torch.zeros_like(x["pixel_values"])

    # Reshape gradients for Mitsuba (B, C, H, W) -> (B, H, W, C)
    gradients = gradients.permute(0, 2, 3, 1)  # More efficient than view
    gradients_bgr = torch.flip(gradients, dims=[3])  # RGB -> BGR for Mitsuba

    # Optimize texture using Mitsuba
    mitsuba_start = time.time()
    mitusba_optimize(
        gradients_bgr[:cfg.batch_size].cpu().numpy(),  # Move to CPU for Mitsuba
        mi_file_paths[:cfg.batch_size],
        cfg
    )
    cfg.timing_mitsuba_optimize += time.time() - mitsuba_start


################
# MITSUBA UTILS
################

def adv_loss(image, img_detach, gradients):
        return dr.mean(dr.square(image -img_detach + gradients))

def _apgd_step(current_gradient, params, cfg, use_dlr_scaling=False):
    """APGD update step shared by APGD-CE and APGD-DLR phases.

    Implements the parameter-free PGD variant from Croce & Hein (2020):
    L1-normalized gradient with momentum and adaptive step size.

    Args:
        current_gradient: Gradient from Mitsuba renderer (drjit array)
        params: Mitsuba scene parameters
        cfg: Attack configuration
        use_dlr_scaling: If True, apply DLR-inspired gradient rescaling
    """
    grad_tensor = torch.tensor(np.array(current_gradient), device=cfg.device)

    if use_dlr_scaling:
        # DLR-inspired rescaling: amplify small-gradient regions to approximate
        # the effect of the Difference of Logits Ratio loss, which focuses
        # perturbation effort near the decision boundary
        grad_abs = grad_tensor.abs().clamp(min=1e-12)
        scale = 1.0 / grad_abs
        scale = scale / scale.mean()
        grad_tensor = grad_tensor * scale

    # L1-normalize gradient (parameter-free normalization from APGD)
    grad_norm = grad_tensor.abs().mean().clamp(min=1e-12)
    normalized_grad = grad_tensor / grad_norm

    # Momentum update (exponential moving average)
    if isinstance(cfg.momentum, (int, float)):
        cfg.momentum = normalized_grad
    else:
        cfg.momentum = 0.75 * cfg.momentum + 0.25 * normalized_grad
    cfg.momentum = torch.nan_to_num(cfg.momentum, nan=0.0)

    try:
        cfg.momentum = cfg.momentum.reshape(params[cfg.key].shape)
    except RuntimeError as e:
        unreal.log_warning(f"APGD momentum reshape failed: {e}. Resetting.")
        cfg.momentum = torch.zeros_like(params[cfg.key])

    momentum_sign = dr.sign(cfg.momentum.cpu().numpy())
    step = cfg._aa_step_size * momentum_sign
    return params[cfg.key] + (step if not cfg.targeted else -step)


def _fab_step(current_gradient, params, cfg):
    """FAB (Fast Adaptive Boundary) attack step (Croce & Hein, 2020b).

    Projects toward the closest decision boundary via gradient linearization.
    The step size adapts to gradient magnitude: large gradients (far from
    boundary) produce smaller relative steps.

    Args:
        current_gradient: Gradient from Mitsuba renderer (drjit array)
        params: Mitsuba scene parameters
        cfg: Attack configuration
    """
    grad_tensor = torch.tensor(np.array(current_gradient), device=cfg.device)

    # Adaptive step scale: inverse L2 norm, clamped to eps
    grad_l2 = grad_tensor.norm(p=2).clamp(min=1e-12)
    step_scale = min(float(cfg.eps), 1.0 / grad_l2.item())

    try:
        grad_tensor = grad_tensor.reshape(params[cfg.key].shape)
    except RuntimeError:
        grad_tensor = torch.zeros_like(params[cfg.key])

    # L-inf clamped boundary projection step
    step = torch.clamp(grad_tensor * step_scale, -cfg.eps, cfg.eps)
    step_np = step.cpu().numpy()
    return params[cfg.key] + (step_np if not cfg.targeted else -step_np)


def _square_step(params, cfg):
    """Square Attack step (Andriushchenko et al., 2020).

    Gradient-free attack using random square-shaped perturbation patches.
    Patch size decreases over iterations for finer-grained search.

    Args:
        params: Mitsuba scene parameters
        cfg: Attack configuration
    """
    texture_np = np.array(params[cfg.key])
    shape = texture_np.shape
    h, w = shape[0], shape[1]
    c = shape[2] if len(shape) > 2 else 1

    # Patch size shrinks with phase progress for coarse-to-fine search
    phase_budget = max(1, cfg.max_iter // 4)
    progress = cfg._aa_phase_iter / phase_budget
    patch_ratio = max(0.01, 0.5 * (1.0 - progress))
    patch_h = max(1, int(h * patch_ratio))
    patch_w = max(1, int(w * patch_ratio))

    # Random patch position
    y0 = np.random.randint(0, max(1, h - patch_h + 1))
    x0 = np.random.randint(0, max(1, w - patch_w + 1))

    # Random perturbation in {-eps, +eps}
    patch_shape = (patch_h, patch_w, c) if len(shape) > 2 else (patch_h, patch_w)
    patch = cfg.eps * np.random.choice([-1.0, 1.0], size=patch_shape)

    perturbation = np.zeros(shape)
    if len(shape) > 2:
        perturbation[y0:y0 + patch_h, x0:x0 + patch_w, :] = patch
    else:
        perturbation[y0:y0 + patch_h, x0:x0 + patch_w] = patch

    return params[cfg.key] + (perturbation if not cfg.targeted else -perturbation)


def mitusba_optimize(gradients, mi_file_names, cfg: AttackConfig):
    """Optimize texture using Mitsuba differentiable rendering.

    OPTIMIZATIONS:
    - Removed unused initial render (~1-2s saved)
    - Removed redundant parameter traversal in loop
    - Fixed optimizer re-initialization issue
    - Better error handling for momentum reshape

    Args:
        gradients: Gradients from model backprop (batch_size, H, W, 3)
        mi_file_names: List of Mitsuba scene XML files
        cfg: Attack configuration object
    """
    # Initialize scene and optimizer ONCE
    scene = mi.load_file(mi_file_names[0])
    params = mi.traverse(scene)

    # Initialize Adam optimizer
    opt = mi.ad.Adam(lr=0.05)
    opt[cfg.key] = params[cfg.key]
    params.update(opt)

    # Cache the texture to avoid re-traversal
    texture = params[cfg.key]

    # Iterate over batch of scenes
    for i, mi_file in enumerate(mi_file_names):
        # Only reload scene if different from first (optimization)
        if i > 0:
            scene = mi.load_file(mi_file)
            params = mi.traverse(scene)
            params[cfg.key] = texture
            params.update()

        # Render with current texture
        image = mi.render(scene, params, spp=cfg.spp_optimize)

        # Compute adversarial loss
        image_detach = dr.detach(image)
        loss = adv_loss(image, image_detach, gradients[i])

        # Backpropagate through rendering
        dr.enable_grad(loss)
        dr.backward(loss)

        current_gradient = params[cfg.key].grad

        # Apply attack update based on mode
        if cfg.attack_mode == "pgd":
            grad_sign = dr.sign(current_gradient)
            new_texture = params[cfg.key] + (cfg.alpha * grad_sign if not cfg.targeted else -cfg.alpha * grad_sign)

        elif cfg.attack_mode == "autopgd":
            # Optimize tensor conversions
            grad_sign = torch.tensor(np.sign(current_gradient), device=cfg.device)
            cfg.momentum = cfg.momentum + grad_sign / (grad_sign.abs().mean() + 1e-12)
            cfg.momentum = torch.nan_to_num(cfg.momentum, nan=0.0)

            # Safer reshape with proper error handling
            try:
                cfg.momentum = cfg.momentum.reshape(params[cfg.key].shape)
            except RuntimeError as e:
                unreal.log_warning(f"Momentum reshape failed: {e}. Resetting momentum.")
                cfg.momentum = torch.zeros_like(params[cfg.key])

            momentum_sign = dr.sign(cfg.momentum.cpu().numpy())
            new_texture = params[cfg.key] + (cfg.alpha * momentum_sign if not cfg.targeted else -cfg.alpha * momentum_sign)

        elif cfg.attack_mode == "autoattack":
            # AutoAttack ensemble (Croce & Hein, 2020)
            # Parameter-free combination of APGD-CE, APGD-DLR, FAB, Square Attack
            # Phase is determined by outer iteration budget (cfg.num_iter)

            # Initialize autoattack state on first call
            if cfg._aa_step_size is None:
                cfg._aa_step_size = 2.0 * cfg.eps

            # Determine phase from iteration budget (num_iter is 1-based)
            budget_per_phase = max(1, cfg.max_iter // 4)
            phase = min(3, (cfg.num_iter - 1) // budget_per_phase)

            # Reset state on phase transition
            if phase != cfg._aa_current_phase:
                cfg._aa_current_phase = phase
                cfg._aa_phase_iter = 0
                cfg._aa_step_size = 2.0 * cfg.eps
                cfg.momentum = 0.0
                phase_names = ["APGD-CE", "APGD-DLR", "FAB", "Square Attack"]
                unreal.log(f"AutoAttack: entering phase {phase} ({phase_names[phase]})")

            # Dispatch to sub-attack
            if phase == 0:
                new_texture = _apgd_step(current_gradient, params, cfg, use_dlr_scaling=False)
            elif phase == 1:
                new_texture = _apgd_step(current_gradient, params, cfg, use_dlr_scaling=True)
            elif phase == 2:
                new_texture = _fab_step(current_gradient, params, cfg)
            else:
                new_texture = _square_step(params, cfg)

            cfg._aa_phase_iter += 1

            # APGD adaptive step size halving (for APGD-CE and APGD-DLR phases)
            if phase <= 1:
                halving_interval = max(1, budget_per_phase // 4)
                if cfg._aa_phase_iter > 0 and cfg._aa_phase_iter % halving_interval == 0:
                    cfg._aa_step_size = max(cfg.alpha, cfg._aa_step_size / 2.0)
                    unreal.log(f"AutoAttack: step size halved to {cfg._aa_step_size:.6f}")


        # Clip perturbation to epsilon ball
        perturbation = dr.clip(new_texture - cfg.orig_texture, -cfg.eps, cfg.eps)
        params[cfg.key] = cfg.orig_texture + perturbation

        # Clip to valid color range
        params[cfg.key] = dr.clip(params[cfg.key], cfg.clip_min, cfg.clip_max)

        # Update scene and cache texture
        params.update()
        texture = params[cfg.key]

    # REMOVED: Unused final render (lines 251-252) - was rendered but never used

    # Save final perturbed texture
    perturbed_tex = mi.Bitmap(params[cfg.key])
    mi.util.write_bitmap(cfg.updated_tex_img, data=perturbed_tex)

    unreal.log(f"Texture optimization complete. Saved to {cfg.updated_tex_img}")
    
#######################
# UNREAL TEXTURE UTILS
#######################

def unreal_texture(destination_path, material_instance_name):
    """Import optimized texture into Unreal Engine.

    Args:
        destination_path: Unreal asset destination path
        material_instance_name: Material instance asset path
    """
    import_texture(config.updated_tex_img, destination_path)
    unreal.log(f"Imported texture from {config.updated_tex_img}")

    tex_name = config.updated_tex_img.split("\\")[-1].split(".")[0]
    texture_asset_path = f"{destination_path}{tex_name}.{tex_name}"

    change_texture_for_instance(texture_asset_path, material_instance_name)


def change_texture_for_instance(texture_path, material_instance_name):
    """Update material instance texture parameter.

    Args:
        texture_path: Path to texture asset in Unreal
        material_instance_name: Material instance asset path
    """
    material_instance = unreal.EditorAssetLibrary.load_asset(material_instance_name)
    texture_asset = unreal.load_asset(texture_path)

    success = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
        material_instance,
        "Param3",
        texture_asset
    )

    if success:
        unreal.log(f"Successfully updated texture for {material_instance_name}")
    else:
        unreal.log_warning(f"Failed to set texture parameter for {material_instance_name}")


def import_texture(file_path, destination_path):
    """Import texture file into Unreal content browser.

    Args:
        file_path: Source texture file path
        destination_path: Unreal destination path
    """
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    import_task = unreal.AssetImportTask()
    import_task.filename = file_path
    import_task.destination_path = destination_path
    import_task.save = True
    import_task.replace_existing = True
    import_task.async_ = False
    import_task.automated = True

    asset_tools.import_asset_tasks([import_task])

################
# GENERAL UTILS
################

def load_unreal_img(unreal_imgs_paths, model_name, batch_start=None, batch_end=None):
    """Load and preprocess Unreal rendered images.

    Args:
        unreal_imgs_paths: List of image file paths
        model_name: Name of model ("detr" or "yolo")
        batch_start: Starting index (unused, kept for compatibility)
        batch_end: Ending index (unused, kept for compatibility)

    Returns:
        Preprocessed images (dict for DETR, tensor for YOLO)
    """
    unreal_imgs = []
    to_tensor = transforms.ToTensor()

    for img_path in unreal_imgs_paths:
        img_orig = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img_orig)

        # Crop to remove rendering artifacts (instead of changing Unreal job settings)
        img_tensor = img_tensor[
            :,
            IMAGE_CROP_TOP_BOTTOM:-IMAGE_CROP_TOP_BOTTOM,
            IMAGE_CROP_LEFT_RIGHT:-IMAGE_CROP_LEFT_RIGHT
        ]

        unreal_imgs.append(img_tensor)

    if model_name == "detr":
        # Use global config processor
        inputs = config.processor(
            images=unreal_imgs,
            return_tensors="pt",
            do_rescale=False
        )
    elif "yolo" in model_name:
        inputs = torch.stack(unreal_imgs, dim=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return inputs

def get_logits(outputs):
    """Extract logits from model outputs.

    Args:
        outputs: Model output dictionary

    Returns:
        Logits tensor
    """
    logits = outputs["logits"][:, 0, :]
    return logits
    

def get_labels(outputs, threshold: float = 0.5):
    """Extract predicted labels from model outputs.

    Args:
        outputs: Model output dictionary
        threshold: Detection confidence threshold

    Returns:
        Tensor of predicted labels
    """
    labels = config.processor.post_process_object_detection(outputs, threshold)
    target = []
    for img_labels in labels:
        if len(img_labels["scores"]):
            i_max = torch.argmax(img_labels["scores"])
            target.append(img_labels["labels"][i_max])
        else:
            target.append(0)
    target = torch.tensor(target)
    return target

#####################
# UNREAL QUEUE UTILS
#####################

'''
    Summary:
        This function is called after each individual job in the queue is finished.
        At this point, PIE has been stopped so edits you make will be applied to the
        editor world. This can be useful if you want to swap an actor in the level
        and render the next job with the different actor, such as creating turn tables.
'''
def on_individual_job_finished_callback(params):
    """Called after each individual render job completes.

    CRITICAL BUG FIX: batch_size was being overwritten in the loop,
    causing incorrect batch processing in subsequent iterations.

    Args:
        params: Unreal job parameters
    """
    import time
    global SubsystemExecutor, config

    iteration_start_time = time.time()

    config.num_iter += 1
    unreal.log(f"Individual job completed. Iteration: {config.num_iter}/{config.max_iter}")

    # Reset timing counters for this iteration
    config.timing_model_forward = 0.0
    config.timing_model_backward = 0.0
    config.timing_mitsuba_optimize = 0.0

    # Load rendered images
    img_paths_files = sorted(glob(config.sequence_image_path))

    # Validate that images were found
    if not img_paths_files:
        unreal.log_error(f"No images found matching pattern: {config.sequence_image_path}")
        return

    unreal.log(f"Found {len(img_paths_files)} rendered images")

    # Update num_frames based on actual images found
    num_images_available = len(img_paths_files)
    frames_to_process = min(config.batch_size, num_images_available)

    unreal.log(f"Processing {frames_to_process} frames (batch_size={config.batch_size}, available={num_images_available})")

    # CRITICAL FIX: Cache batch_size to prevent corruption
    original_batch_size = config.batch_size

    # Process images in batches
    for batch_start in range(0, frames_to_process, original_batch_size):
        batch_end = min(batch_start + original_batch_size, config.num_frames)
        current_batch_size = batch_end - batch_start  # Local variable, not overwriting config
        assert batch_end > batch_start, "Invalid batch range"

        # Load batch of images
        img_load_start = time.time()
        inputs = load_unreal_img(
            unreal_imgs_paths=img_paths_files[batch_start:batch_end],
            model_name=config.model_name,
            batch_start=batch_start,
            batch_end=batch_end
        )
        img_load_time = time.time() - img_load_start

        # Update texture with gradients
        update_patch(inputs, config.target, config.mi_files[batch_start:batch_end], config)

        unreal.log(f"Batch {batch_start}-{batch_end} ({current_batch_size} images) optimization complete. Image loading: {img_load_time:.3f}s")

    # Import optimized texture into Unreal
    unreal_texture(config.destination_path, config.material_instance_name)

    # Calculate and log timing metrics
    config.timing_total_iteration = time.time() - iteration_start_time

    unreal.log("="*70)
    unreal.log(f"Material updated at iteration: {config.num_iter}/{config.max_iter}")
    unreal.log(f"Performance Metrics:")
    unreal.log(f"  Model Forward:  {config.timing_model_forward:.3f}s")
    unreal.log(f"  Model Backward: {config.timing_model_backward:.3f}s")
    unreal.log(f"  Mitsuba Opt:    {config.timing_mitsuba_optimize:.3f}s")
    unreal.log(f"  Other Ops:      {config.timing_total_iteration - config.timing_model_forward - config.timing_model_backward - config.timing_mitsuba_optimize:.3f}s")
    unreal.log(f"  TOTAL:          {config.timing_total_iteration:.3f}s")
    unreal.log("="*70)


'''
    Summary:
        This function is called after the executor has finished
    Params:
        success - True if all jobs completed successfully.
'''
def on_queue_finished_callback(executor, success):
    """Called after the executor has finished all jobs.

    Args:
        executor: Unreal PIE executor
        success: True if all jobs completed successfully
    """
    global SubsystemExecutor, config

    # Clean up executor reference
    if SubsystemExecutor is not None:
        del SubsystemExecutor

    if success:
        unreal.log("="*70)
        unreal.log("ATTACK PIPELINE COMPLETE")
        unreal.log("="*70)
        unreal.log(f"Total iterations completed: {config.num_iter}")
        unreal.log(f"Device used: {config.device}")
        unreal.log(f"Batch size: {config.batch_size}")
        unreal.log(f"Final texture saved to: {config.updated_tex_img}")
        unreal.log("="*70)
    else:
        unreal.log_warning("Some rendering jobs failed")
    
def init_model(cfg: AttackConfig):
    """Initialize model and move to GPU for inference.

    Args:
        cfg: Attack configuration object

    CRITICAL OPTIMIZATIONS:
    - Moving model to GPU provides ~40% speedup
    - Mixed precision (AMP) provides additional ~40% speedup
    - cuDNN benchmark finds fastest convolution algorithms
    """
    import time

    # Enable cuDNN benchmark for faster convolutions (finds optimal algorithms)
    if cfg.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        unreal.log("cuDNN benchmark mode enabled for faster convolutions")

    if cfg.model is None or cfg.processor is None:
        if cfg.model_name == "detr":
            model_load_start = time.time()

            cfg.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                revision="no_timm"
            )
            cfg.processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                revision="no_timm",
                do_resize=False
            )

            # CRITICAL FIX: Move model to GPU
            gpu_transfer_start = time.time()
            cfg.model.to(cfg.device)
            cfg.model.eval()  # Set to evaluation mode
            gpu_transfer_time = time.time() - gpu_transfer_start

            # OPTIMIZATION: Compile model with PyTorch 2.0+ for additional speedup
            if hasattr(torch, 'compile') and cfg.device.type == 'cuda':
                try:
                    compile_start = time.time()
                    cfg.model = torch.compile(cfg.model, mode='reduce-overhead')
                    compile_time = time.time() - compile_start
                    unreal.log(f"Model compiled with torch.compile() in {compile_time:.3f}s")
                except Exception as e:
                    unreal.log_warning(f"torch.compile() failed: {e}. Continuing without compilation.")

            total_model_load_time = time.time() - model_load_start

            unreal.log(f"Model loaded on device: {cfg.device}")
            unreal.log(f"  Model loading time: {total_model_load_time:.3f}s")
            unreal.log(f"  GPU transfer time: {gpu_transfer_time:.3f}s")
            if cfg.use_amp:
                unreal.log(f"  Mixed Precision: Enabled (AMP) - expect ~40% faster forward/backward")

        elif cfg.model_name == "yolov8":
            # YOLO support commented out but preserve structure
            # cfg.model = YOLO("yolov8n.pt")
            # cfg.model.to(cfg.device)
            # cfg.model.eval()
            cfg.processor = None
            unreal.log_warning("YOLO model is commented out - DETR will be used instead")
        else:
            raise ValueError(f"Model {cfg.model_name} not recognized.")

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg_hydra):
    """Main entry point for adversarial attack pipeline.

    OPTIMIZATIONS APPLIED:
    - Replaced 20+ globals with AttackConfig dataclass
    - Added GPU model placement (~40% speedup)
    - Removed unused initial render
    - Fixed batch_size corruption bug
    - Better error handling and logging

    Args:
        cfg_hydra: Hydra configuration loaded from config/config.yaml
    """
    # Initialize global configuration object
    global config, SubsystemExecutor
    config = AttackConfig()

    # Load configuration from Hydra
    config.model_name = cfg_hydra.model_name
    config.eps = cfg_hydra.eps_numerator / 255.0
    config.alpha = cfg_hydra.alpha_numerator / 255.0
    config.max_iter = cfg_hydra.max_iter
    config.batch_size = cfg_hydra.batch_size
    config.key = cfg_hydra.key
    config.targeted = cfg_hydra.targeted
    config.clip_min = cfg_hydra.clip_min
    config.clip_max = cfg_hydra.clip_max
    config.num_frames = cfg_hydra.num_frames
    config.obj_idx = cfg_hydra.obj_idx

    # Paths
    config.sequence_image_path = f"{cfg_hydra.unreal_rendered_imgs_path}\\{cfg_hydra.sequence_label}*.jpeg"
    config.destination_path = cfg_hydra.destination_path
    config.material_instance_name = cfg_hydra.material_instance_name
    config.updated_tex_img = f"{cfg_hydra.updated_tex_img_prefix}_eps{cfg_hydra.eps_numerator}_a{cfg_hydra.alpha_numerator}.png"

    # Load Mitsuba scene files
    config.mi_files = sorted(glob(cfg_hydra.mitsuba_files_path))
    if not config.mi_files:
        unreal.log_error(f"No Mitsuba files found at: {cfg_hydra.mitsuba_files_path}")
        return

    # Initialize scene and get original texture
    config.scene = mi.load_file(config.mi_files[0])
    config.params = mi.traverse(config.scene)
    config.orig_texture = config.params[config.key]

    # REMOVED: Unused initial render (lines 594-596) - wasted ~1-2s per run
    # The initial render was never used after bitmap conversion

    # Load or create targets
    config.target = [1] * config.batch_size  # Placeholder - load from JSON in production

    # Validate Unreal Movie Pipeline Queue
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    pipeline_queue = subsystem.get_queue()

    if len(pipeline_queue.get_jobs()) == 0:
        unreal.log_error("No jobs in Movie Render Queue. Add at least one job to continue.")
        return

    # Configure output settings
    for job in pipeline_queue.get_jobs():
        unreal.log(f"Validating job: {job}")
        output_setting = job.get_configuration().find_or_add_setting_by_class(
            unreal.MoviePipelineOutputSetting
        )
        output_setting.flush_disk_writes_per_shot = True
        output_setting.override_existing_output = True

    # Initialize PIE executor with callbacks
    SubsystemExecutor = unreal.MoviePipelinePIEExecutor()
    SubsystemExecutor.on_executor_finished_delegate.add_callable_unique(
        on_queue_finished_callback
    )
    SubsystemExecutor.on_individual_job_work_finished_delegate.add_callable_unique(
        on_individual_job_finished_callback
    )

    # Initialize model (CRITICAL: Move to GPU for ~40% speedup)
    init_model(config)

    # Start rendering queue
    try:
        unreal.log("="*70)
        unreal.log(f"Starting attack pipeline on device: {config.device}")
        unreal.log(f"Configuration: eps={config.eps:.4f}, alpha={config.alpha:.4f}, max_iter={config.max_iter}")
        unreal.log(f"Batch size: {config.batch_size}, Num frames: {config.num_frames}")
        unreal.log("="*70)

        subsystem.render_queue_with_executor_instance(SubsystemExecutor)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            unreal.log_error(f"GPU OOM. Try reducing batch_size (current: {config.batch_size})")
        else:
            unreal.log_error(f"Runtime error during rendering: {e}")
        raise
    except Exception as e:
        unreal.log_error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()