import os
import time
import argparse
import numpy as np
import torch

from transformers import TimesformerForVideoClassification


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("FedCAV E1 — Dense TimeSformer Profiling")
    print("=" * 70)
    print("Device :", device)

    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    print("\nLoading TimeSformer...")

    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=101,
        ignore_mismatched_sizes=True,
    )

    # -----------------------------------------------------
    # Load trained E1 checkpoint
    # -----------------------------------------------------
    print("Loading checkpoint:")
    print(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # Parameters
    # -----------------------------------------------------
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"\nParameters          : "
        f"{total_params / 1e6:.3f} M"
    )

    # -----------------------------------------------------
    # Token count
    # -----------------------------------------------------
    patch_size = model.config.patch_size

    if isinstance(patch_size, (tuple, list)):
        patch_h = patch_size[0]
        patch_w = patch_size[1]
    else:
        patch_h = patch_w = patch_size

    spatial_tokens_per_frame = (
        args.image_size // patch_h
    ) * (
        args.image_size // patch_w
    )

    patch_tokens = (
        spatial_tokens_per_frame * args.frames
    )

    total_tokens = patch_tokens + 1  # CLS

    print(
        f"Frames              : {args.frames}"
    )
    print(
        f"Resolution          : "
        f"{args.image_size} x {args.image_size}"
    )
    print(
        f"Patch size          : "
        f"{patch_h} x {patch_w}"
    )
    print(
        f"Tokens/frame        : "
        f"{spatial_tokens_per_frame}"
    )
    print(
        f"Patch tokens/clip   : "
        f"{patch_tokens}"
    )
    print(
        f"Tokens incl. CLS    : "
        f"{total_tokens}"
    )

    # -----------------------------------------------------
    # Dummy input
    # TimeSformer expects:
    # [batch, frames, channels, height, width]
    # -----------------------------------------------------
    x = torch.randn(
        args.batch_size,
        args.frames,
        3,
        args.image_size,
        args.image_size,
        device=device
    )

    # -----------------------------------------------------
    # Peak GPU memory
    # -----------------------------------------------------
    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda",
                enabled=args.amp
            ):
                _ = model(pixel_values=x)

        torch.cuda.synchronize()

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

        print(
            f"\nPeak GPU allocated  : "
            f"{peak_allocated:.2f} MB"
        )
        print(
            f"Peak GPU reserved   : "
            f"{peak_reserved:.2f} MB"
        )

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------
    print(
        f"\nWarm-up iterations  : "
        f"{args.warmup}"
    )

    with torch.no_grad():

        for _ in range(args.warmup):

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    args.amp and
                    device.type == "cuda"
                )
            ):
                _ = model(pixel_values=x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -----------------------------------------------------
    # Latency
    # -----------------------------------------------------
    print(
        f"Measured iterations : "
        f"{args.iters}"
    )

    times_ms = []

    with torch.no_grad():

        for _ in range(args.iters):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    args.amp and
                    device.type == "cuda"
                )
            ):
                _ = model(pixel_values=x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times_ms.append(
                (end - start) * 1000.0
            )

    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    std_ms = float(np.std(times_ms))
    p95_ms = percentile(times_ms, 95)
    p99_ms = percentile(times_ms, 99)

    clips_per_sec = (
        args.batch_size /
        (mean_ms / 1000.0)
    )

    print("\n" + "=" * 70)
    print("E1 PROFILE RESULTS")
    print("=" * 70)

    print(
        f"Mean latency        : "
        f"{mean_ms:.3f} ms/batch"
    )
    print(
        f"Median latency      : "
        f"{median_ms:.3f} ms/batch"
    )
    print(
        f"Std latency         : "
        f"{std_ms:.3f} ms"
    )
    print(
        f"P95 latency         : "
        f"{p95_ms:.3f} ms/batch"
    )
    print(
        f"P99 latency         : "
        f"{p99_ms:.3f} ms/batch"
    )
    print(
        f"Throughput          : "
        f"{clips_per_sec:.3f} clips/s"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    os.makedirs(
        args.results_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        args.results_dir,
        "e1_dense_profile.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FedCAV E1 Dense TimeSformer Profile\n"
        )
        f.write("=" * 50 + "\n")

        f.write(
            f"Parameters_M: "
            f"{total_params / 1e6:.6f}\n"
        )
        f.write(
            f"Frames: {args.frames}\n"
        )
        f.write(
            f"Resolution: "
            f"{args.image_size}\n"
        )
        f.write(
            f"Tokens_per_frame: "
            f"{spatial_tokens_per_frame}\n"
        )
        f.write(
            f"Patch_tokens_clip: "
            f"{patch_tokens}\n"
        )
        f.write(
            f"Tokens_including_CLS: "
            f"{total_tokens}\n"
        )

        if device.type == "cuda":
            f.write(
                f"Peak_GPU_allocated_MB: "
                f"{peak_allocated:.4f}\n"
            )
            f.write(
                f"Peak_GPU_reserved_MB: "
                f"{peak_reserved:.4f}\n"
            )

        f.write(
            f"Mean_latency_ms: "
            f"{mean_ms:.6f}\n"
        )
        f.write(
            f"Median_latency_ms: "
            f"{median_ms:.6f}\n"
        )
        f.write(
            f"P95_latency_ms: "
            f"{p95_ms:.6f}\n"
        )
        f.write(
            f"P99_latency_ms: "
            f"{p99_ms:.6f}\n"
        )
        f.write(
            f"Throughput_clips_s: "
            f"{clips_per_sec:.6f}\n"
        )

    print("\nSaved:", output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=8
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=200
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results"
    )

    args = parser.parse_args()

    main(args)