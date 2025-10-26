import argparse
import time
import numpy as np

try:
    import pyopencl as cl
except Exception as e:
    raise SystemExit(
        "PyOpenCL import needed. Install with `pip install pyopencl numpy` and make sure that your OpenCL drivers are present.\n"
        f"Import error: {e}"
    )

KERNEL_SRC = r"""
__kernel void stress_kernel(__global float* out, uint iter) {
    const uint gid = get_global_id(0);
    float x = (float)(gid & 0xFFFF) + 1.2345f;
    const float y = 1.000001f;
    // Inner loop: unrolled multiply-add sequence (8 FMA per iteration)
    for (uint i = 0; i < iter; ++i) {
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
        x = x * y + 1.0000001f;
    }
    out[gid] = x; // write back to avoid being taken away
}
"""

def choose_device(index):
    plats = cl.get_platforms()
    if not plats:
        raise SystemExit("No OpenCL platforms found on your device, lil bro.")

    devs = []
    for p in plats:
        for d in p.get_devices():
            devs.append((p, d))
    if not devs:
        raise SystemExit("No OpenCL devices found, lil bro.")
    if index < 0 or index >= len(devs):
        print("Available devices:")
        for i, (p, d) in enumerate(devs):
            print(f"  {i}: {p.name.strip()} - {d.name.strip()} ({cl.device_type.to_string(d.type)})")
        raise SystemExit(f"Device index {index} out of range.")
    return devs[index]

def main():
    parser = argparse.ArgumentParser(description="GPU stress tester (OpenCL)")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds (default 120)")
    parser.add_argument("--device", type=int, default=0, help="Device index (default 0). If out of range, script prints device list.")
    parser.add_argument("--local", type=int, default=None, help="Local (work-group) size; default picks a reasonable value.")
    parser.add_argument("--global-mult", type=int, default=128, help="Multiplier for global size per compute unit (default 128).")
    args = parser.parse_args()

    plat, dev = choose_device(args.device)
    print(f"# Using platform: {plat.name.strip()}")
    print(f"# Using device:   {dev.name.strip()}")
    print(f"# Device type:    {cl.device_type.to_string(dev.type)}")
    print(f"# Compute units:  {dev.max_compute_units}")
    print(f"# Max work-group: {dev.max_work_group_size}")
    try:
        print(f"# Max clock (MHz): {dev.max_clock_frequency}")
    except Exception:
        pass
    try:
        print(f"# Global mem (MB): {dev.global_mem_size // (1024*1024)}")
    except Exception:
        pass

    ctx = cl.Context(devices=[dev])

    queue = cl.CommandQueue(ctx, device=dev, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(ctx, KERNEL_SRC).build()

    compute_units = dev.max_compute_units or 1
    max_wg = dev.max_work_group_size or 64
    local_size = args.local if args.local else min(256, max_wg)
    global_size = compute_units * args.global_mult * local_size

    if global_size % local_size != 0:
        global_size = ((global_size // local_size) + 1) * local_size

    UNROLL = 8
    FLOPS_PER_ITER = UNROLL * 2

    iter_per_launch = 1024

    out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=global_size * 4)  #float32

    print(f"# Global size:     {global_size} (local {local_size})")
    print(f"# Iter/launch:     {iter_per_launch}")
    print(f"# UNROLL:          {UNROLL} (FLOPs/iter = {FLOPS_PER_ITER})")
    print(f"# Target duration: {args.duration} s")
    print("# Starting test... (press Ctrl+C to abort)\n")

    total_launches = 0
    total_flops = 0
    t0 = time.time()
    try:
        while True:

            evt = program.stress_kernel(queue, (global_size,), (local_size,), out_buf, np.uint32(iter_per_launch))
            evt.wait()
           
            start_ns = evt.profile.start
            end_ns = evt.profile.end
            elapsed_s = (end_ns - start_ns) * 1e-9

            flops_this = global_size * iter_per_launch * FLOPS_PER_ITER
            total_launches += 1
            total_flops += flops_this
            wall = time.time() - t0
           
            if wall >= args.duration:
                break
           
            if total_launches == 1:
                if elapsed_s < 0.05:
                    #try to increase
                    factor = min(64, int(0.2 / max(elapsed_s, 1e-6)))
                    iter_per_launch = max(1, iter_per_launch * factor)
                    print(f"# Kernel too fast ({elapsed_s:.4f}s). Increasing iter_per_launch -> {iter_per_launch}")
                elif elapsed_s > 0.6:
                    iter_per_launch = max(1, int(iter_per_launch * 0.5))
                    print(f"# Kernel slow ({elapsed_s:.4f}s). Decreasing iter_per_launch -> {iter_per_launch}")
            #continue until duration
    except KeyboardInterrupt:
        print("\n# Interrupted by user.", flush=True)

    elapsed_total = time.time() - t0
    gflops = total_flops / elapsed_total / 1e9
    gflops_per_cu = gflops / max(1, compute_units)

    host_out = np.empty(1, dtype=np.float32)
    cl.enqueue_copy(queue, host_out, out_buf, device_offset=0, is_blocking=True)

    print("\n--- Performance summary ---")
    print(f"Elapsed time:      {elapsed_total:,.2f} s")
    print(f"Total kernel launches: {total_launches}")
    print(f"Total FLOPs:       {total_flops:,}")
    print(f"GFLOPS (counted):  {gflops:,.2f}")
    print(f"GFLOPS / CU:       {gflops_per_cu:,.2f}")
    print(f"Compute units:     {compute_units}")
    print(f"Global size:       {global_size}")
    print(f"Local size:        {local_size}")
    print(f"Iter/launch:       {iter_per_launch}")
    print(f"UNROLL (mul+add):  {UNROLL}")
    print(f"Device name:       {dev.name.strip()}")
    print("---------------------------")
    print("\n# Results are only approximate for comparing across devices.")

if __name__ == "__main__":
    main()
