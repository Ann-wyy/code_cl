#!/usr/bin/env python3
"""
Training health monitor for DINOv3.
Analyzes training logs to detect issues early.

Usage:
    # Real-time monitoring (tail logs)
    python monitor_training_health.py --log-file training.log --watch

    # One-time analysis
    python monitor_training_health.py --log-file training.log
"""

import argparse
import re
import time
from pathlib import Path
from collections import defaultdict


class TrainingHealthMonitor:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.iterations = []
        self.losses = defaultdict(list)
        self.lrs = []
        self.gpu_mem = []

    def parse_log_line(self, line):
        """Parse a single log line for training metrics."""
        # Example log format: "Iteration 100: loss=10.5, lr=0.001, gpu_mem=8.5GB"
        # Adapt this to your actual log format

        iteration_match = re.search(r'(?:Iteration|iteration|iter)[:\s]+(\d+)', line, re.IGNORECASE)
        if iteration_match:
            iteration = int(iteration_match.group(1))

            # Extract loss values
            loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if loss_match:
                loss = float(loss_match.group(1))
                self.iterations.append(iteration)
                self.losses['total'].append(loss)

            # Extract DINO loss
            dino_loss_match = re.search(r'dino_loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if dino_loss_match:
                self.losses['dino'].append(float(dino_loss_match.group(1)))

            # Extract learning rate
            lr_match = re.search(r'lr[:\s=]+([0-9.e\-+]+)', line, re.IGNORECASE)
            if lr_match:
                self.lrs.append(float(lr_match.group(1)))

            # Extract GPU memory
            mem_match = re.search(r'gpu_mem[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if mem_match:
                self.gpu_mem.append(float(mem_match.group(1)))

    def analyze_training_health(self):
        """Analyze collected metrics for training health."""
        print("\n" + "=" * 80)
        print("  Training Health Analysis")
        print("=" * 80)

        if not self.iterations:
            print("\n⚠️  No training iterations found in log file.")
            print("   Make sure the log file contains training output.")
            return False

        print(f"\nIterations analyzed: {len(self.iterations)}")
        print(f"Iteration range: {min(self.iterations)} - {max(self.iterations)}")

        all_healthy = True

        # ====================================================================
        # 1. Check initial loss value
        # ====================================================================
        print("\n" + "-" * 80)
        print("1. Initial Loss Check")
        print("-" * 80)

        if self.losses['total']:
            initial_loss = self.losses['total'][0]
            print(f"Initial loss: {initial_loss:.4f}")

            # For DINOv3 with 65536 prototypes, initial loss should be around log(65536) ≈ 11
            expected_loss = 11.0
            if 9.0 < initial_loss < 13.0:
                print("✓ Initial loss is in expected range (9-13)")
            else:
                print(f"⚠️  Initial loss {initial_loss:.4f} is outside expected range (9-13)")
                print("   This might indicate initialization issues")
                all_healthy = False

        # ====================================================================
        # 2. Check loss trend
        # ====================================================================
        print("\n" + "-" * 80)
        print("2. Loss Trend Analysis")
        print("-" * 80)

        if len(self.losses['total']) > 10:
            first_10_avg = sum(self.losses['total'][:10]) / 10
            if len(self.losses['total']) >= 100:
                iter_100_avg = sum(self.losses['total'][90:100]) / 10
                print(f"First 10 iterations avg loss: {first_10_avg:.4f}")
                print(f"Iterations 90-100 avg loss: {iter_100_avg:.4f}")

                if iter_100_avg < first_10_avg:
                    print("✓ Loss is decreasing (healthy)")
                else:
                    print("✗ Loss is NOT decreasing after 100 iterations!")
                    print("   Training might be stuck or configuration is wrong")
                    all_healthy = False
            else:
                print(f"First 10 iterations avg loss: {first_10_avg:.4f}")
                print("   (Need at least 100 iterations for trend analysis)")

            # Check for NaN or inf
            if any(loss > 1e6 or loss != loss for loss in self.losses['total']):
                print("✗ Detected NaN or extremely large loss values!")
                print("   Training has diverged. Check learning rate and gradient clipping.")
                all_healthy = False
            else:
                print("✓ No NaN or inf detected in loss")

        # ====================================================================
        # 3. Check learning rate schedule
        # ====================================================================
        print("\n" + "-" * 80)
        print("3. Learning Rate Schedule")
        print("-" * 80)

        if self.lrs:
            print(f"Initial LR: {self.lrs[0]:.6f}")
            if len(self.lrs) > 1:
                print(f"Current LR: {self.lrs[-1]:.6f}")

                # Check if LR is changing (warmup or decay)
                lr_changed = abs(self.lrs[-1] - self.lrs[0]) > 1e-10
                if lr_changed:
                    print("✓ Learning rate is being scheduled")
                else:
                    print("  Learning rate is constant (might be before warmup ends)")
        else:
            print("⚠️  No learning rate information found in logs")

        # ====================================================================
        # 4. Check GPU memory stability
        # ====================================================================
        print("\n" + "-" * 80)
        print("4. GPU Memory Usage")
        print("-" * 80)

        if self.gpu_mem:
            avg_mem = sum(self.gpu_mem) / len(self.gpu_mem)
            max_mem = max(self.gpu_mem)
            min_mem = min(self.gpu_mem)

            print(f"Average GPU memory: {avg_mem:.2f} GB")
            print(f"Max GPU memory: {max_mem:.2f} GB")
            print(f"Min GPU memory: {min_mem:.2f} GB")

            mem_variance = max_mem - min_mem
            if mem_variance < 2.0:  # Less than 2GB variance
                print("✓ GPU memory usage is stable")
            else:
                print(f"⚠️  GPU memory variance is high ({mem_variance:.2f} GB)")
                print("   This might indicate memory leaks")
        else:
            print("  No GPU memory information found in logs")

        # ====================================================================
        # 5. Summary
        # ====================================================================
        print("\n" + "=" * 80)
        if all_healthy:
            print("✓ Training appears HEALTHY!")
            print("\nGood signs:")
            print("  • Loss started at expected value")
            print("  • Loss is decreasing")
            print("  • No NaN or divergence detected")
        else:
            print("⚠️  Training shows WARNING SIGNS!")
            print("\nPlease review the issues above.")
        print("=" * 80)

        return all_healthy

    def watch_mode(self, interval=10):
        """Watch log file in real-time."""
        print(f"Watching log file: {self.log_file}")
        print(f"Update interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")

        last_size = 0
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size > last_size:
                        with open(self.log_file, 'r') as f:
                            f.seek(last_size)
                            new_lines = f.readlines()
                            for line in new_lines:
                                self.parse_log_line(line)
                        last_size = current_size

                        # Show quick status
                        if self.iterations:
                            latest_iter = self.iterations[-1]
                            latest_loss = self.losses['total'][-1] if self.losses['total'] else 'N/A'
                            print(f"[{time.strftime('%H:%M:%S')}] Iter {latest_iter}: loss={latest_loss}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nStopped watching. Running final analysis...")
            self.analyze_training_health()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor DINOv3 training health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-file", type=str, required=True,
                       help="Path to training log file")
    parser.add_argument("--watch", action="store_true",
                       help="Watch log file in real-time")
    parser.add_argument("--interval", type=int, default=10,
                       help="Update interval in seconds for watch mode (default: 10)")

    args = parser.parse_args()

    monitor = TrainingHealthMonitor(args.log_file)

    if args.watch:
        monitor.watch_mode(interval=args.interval)
    else:
        # One-time analysis
        if not Path(args.log_file).exists():
            print(f"✗ Log file not found: {args.log_file}")
            return 1

        with open(args.log_file, 'r') as f:
            for line in f:
                monitor.parse_log_line(line)

        monitor.analyze_training_health()

    return 0


if __name__ == "__main__":
    exit(main())
