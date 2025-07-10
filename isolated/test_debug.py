# test_debug.py
import ray
import sys
# import debugpy

# debugpy.listen(("localhost", 5678))

# Initialize Ray with debugging enabled
ray.init(
    runtime_env={
        "env_vars": {"RAY_DEBUG": "1"},
    }
)


@ray.remote
def debug_task(x):
    y = x * x
    print(f"About to hit breakpoint, y = {y}")
    breakpoint()  # This will pause for debugger
    print(f"After breakpoint, returning {y}")
    return y


if __name__ == "__main__":
    print("Starting debug task...")
    result = ray.get(debug_task.remote(10))
    print(f"Result: {result}")
